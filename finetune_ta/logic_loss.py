"""
Candle self-consistency logic loss for predictor training.

Penalties measure internal OHLC / Heikin-Ashi structure of a continuous
feature tensor — not supervised regression against labels. Soft-decode maps
predictor s1/s2 logits through a frozen tokenizer decoder so the penalty is
differentiable w.r.t. those logits (hard argmax→decode is not).

VRAM mitigations (used by the basemodel trainer):
- Contiguous timestep subsample before soft-decode (keeps HA recurrence valid).
- Activation checkpointing through frozen tokenizer decoder layers.
- Optional micro-batching over the batch dimension.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from indicators import HEIKIN_ASHI_COLUMNS


def _col_index(feature_list: Sequence[str], name: str) -> Optional[int]:
    try:
        return feature_list.index(name)
    except ValueError:
        return None


def ohlc_validity_loss(x: torch.Tensor, feature_list: Sequence[str]) -> torch.Tensor:
    """Hinge penalties when high/low violate candle envelope rules.

    Expects ``x`` shaped ``(B, T, F)`` in the same column order as ``feature_list``.
    """
    oi = _col_index(feature_list, 'open')
    hi = _col_index(feature_list, 'high')
    li = _col_index(feature_list, 'low')
    ci = _col_index(feature_list, 'close')
    if None in (oi, hi, li, ci):
        return x.new_zeros(())

    o, h, l, c = x[..., oi], x[..., hi], x[..., li], x[..., ci]
    # high >= open, close, low; low <= open, close
    viol = (
        F.relu(o - h) + F.relu(c - h) + F.relu(l - h)
        + F.relu(l - o) + F.relu(l - c)
    )
    return viol.mean()


def heikin_ashi_consistency_loss(x: torch.Tensor, feature_list: Sequence[str]) -> torch.Tensor:
    """L1 mismatch vs HA formulas (close mean, open recurrence, high/low envelopes)."""
    oi = _col_index(feature_list, 'open')
    hi = _col_index(feature_list, 'high')
    li = _col_index(feature_list, 'low')
    ci = _col_index(feature_list, 'close')
    ha_o = _col_index(feature_list, 'HA_open')
    ha_h = _col_index(feature_list, 'HA_high')
    ha_l = _col_index(feature_list, 'HA_low')
    ha_c = _col_index(feature_list, 'HA_close')
    if None in (oi, hi, li, ci, ha_o, ha_h, ha_l, ha_c):
        return x.new_zeros(())

    o, h, l, c = x[..., oi], x[..., hi], x[..., li], x[..., ci]
    ha_open, ha_high, ha_low, ha_close = (
        x[..., ha_o], x[..., ha_h], x[..., ha_l], x[..., ha_c]
    )

    # HA_close = (O+H+L+C)/4
    loss_close = (ha_close - (o + h + l + c) * 0.25).abs().mean()

    # Interior HA_open_t = 0.5 * (HA_open_{t-1} + HA_close_{t-1})
    if x.shape[1] > 1:
        expected_open = 0.5 * (ha_open[:, :-1] + ha_close[:, :-1])
        loss_open = (ha_open[:, 1:] - expected_open).abs().mean()
    else:
        loss_open = x.new_zeros(())

    # HA_high = max(H, HA_open, HA_close); HA_low = min(L, HA_open, HA_close)
    expected_high = torch.maximum(torch.maximum(h, ha_open), ha_close)
    expected_low = torch.minimum(torch.minimum(l, ha_open), ha_close)
    loss_high = (ha_high - expected_high).abs().mean()
    loss_low = (ha_low - expected_low).abs().mean()

    return loss_close + loss_open + loss_high + loss_low


def candle_logic_loss(
    x: torch.Tensor,
    feature_list: Sequence[str],
    ohlc_weight: float = 1.0,
    ha_weight: float = 1.0,
) -> torch.Tensor:
    """Weighted sum of OHLC validity and (optional) Heikin-Ashi consistency."""
    loss = x.new_zeros(())
    if ohlc_weight:
        loss = loss + ohlc_weight * ohlc_validity_loss(x, feature_list)
    if ha_weight and all(c in feature_list for c in HEIKIN_ASHI_COLUMNS):
        loss = loss + ha_weight * heikin_ashi_consistency_loss(x, feature_list)
    return loss


def build_bit_tables(
    s1_bits: int,
    s2_bits: int,
    codebook_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bipolar bit tables matching ``KronosTokenizer.indices_to_bits(..., half=True)``."""
    q_scale = 1.0 / (codebook_dim ** 0.5)

    def _table(n_bits: int) -> torch.Tensor:
        vocab = 1 << n_bits
        indices = torch.arange(vocab, device=device, dtype=torch.long)
        mask = 2 ** torch.arange(n_bits, device=device, dtype=torch.long)
        bits = (indices.unsqueeze(-1) & mask) != 0
        return (bits.to(dtype) * 2 - 1) * q_scale

    return _table(s1_bits), _table(s2_bits)


def contiguous_time_slice(seq_len: int, max_timesteps: int) -> Optional[slice]:
    """Random contiguous window of length ``max_timesteps``, or None to use all."""
    if max_timesteps is None or max_timesteps <= 0 or seq_len <= max_timesteps:
        return None
    start = int(torch.randint(0, seq_len - max_timesteps + 1, (1,)).item())
    return slice(start, start + max_timesteps)


def soft_decode_from_logits(
    tokenizer,
    s1_logits: torch.Tensor,
    s2_logits: torch.Tensor,
    s1_bit_table: torch.Tensor,
    s2_bit_table: torch.Tensor,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    """Differentiable decode: softmax(logits) @ bit tables → frozen tokenizer decoder.

    Softmax/bit matmul run in float32 for stability; decoder runs in the
    tokenizer parameter dtype. Do not wrap in ``no_grad`` / ``detach`` during
    training — that would block gradients into the predictor logits.

    When ``use_checkpoint`` is True, decoder layers are run under
    ``torch.utils.checkpoint`` to cut activation memory (recompute on backward).
    """
    # Soft expected bipolar codes (float32 for stable softmax @ large vocab).
    p1 = F.softmax(s1_logits.float(), dim=-1)
    p2 = F.softmax(s2_logits.float(), dim=-1)
    soft_s1 = p1 @ s1_bit_table.to(device=p1.device, dtype=p1.dtype)
    soft_s2 = p2 @ s2_bit_table.to(device=p2.device, dtype=p2.dtype)
    quantized = torch.cat([soft_s1, soft_s2], dim=-1)

    emb_dtype = tokenizer.post_quant_embed.weight.dtype
    z = tokenizer.post_quant_embed(quantized.to(dtype=emb_dtype))
    for layer in tokenizer.decoder:
        if use_checkpoint and z.requires_grad:
            # use_reentrant=False plays better with frozen params + AMP.
            z = checkpoint(layer, z, use_reentrant=False)
        else:
            z = layer(z)
    return tokenizer.head(z)


def logic_loss_from_logits(
    tokenizer,
    s1_logits: torch.Tensor,
    s2_logits: torch.Tensor,
    s1_bit_table: torch.Tensor,
    s2_bit_table: torch.Tensor,
    feature_list: Sequence[str],
    ohlc_weight: float = 1.0,
    ha_weight: float = 1.0,
    max_timesteps: int = 128,
    use_checkpoint: bool = True,
    microbatch_size: int = 0,
) -> torch.Tensor:
    """Soft-decode (optionally subsampled / micro-batched) then candle logic loss."""
    time_slice = contiguous_time_slice(s1_logits.shape[1], max_timesteps)
    if time_slice is not None:
        s1_logits = s1_logits[:, time_slice]
        s2_logits = s2_logits[:, time_slice]

    batch = s1_logits.shape[0]
    mb = int(microbatch_size) if microbatch_size else 0
    if mb <= 0 or mb >= batch:
        soft_x = soft_decode_from_logits(
            tokenizer, s1_logits, s2_logits, s1_bit_table, s2_bit_table,
            use_checkpoint=use_checkpoint,
        )
        return candle_logic_loss(
            soft_x, feature_list, ohlc_weight=ohlc_weight, ha_weight=ha_weight
        )

    chunk_losses = []
    for start in range(0, batch, mb):
        end = min(start + mb, batch)
        soft_x = soft_decode_from_logits(
            tokenizer,
            s1_logits[start:end],
            s2_logits[start:end],
            s1_bit_table,
            s2_bit_table,
            use_checkpoint=use_checkpoint,
        )
        chunk_losses.append(
            candle_logic_loss(
                soft_x, feature_list, ohlc_weight=ohlc_weight, ha_weight=ha_weight
            )
        )
    return torch.stack(chunk_losses).mean()


def freeze_tokenizer(tokenizer) -> None:
    """Eval mode + freeze params; forward still propagates grads to soft inputs."""
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
