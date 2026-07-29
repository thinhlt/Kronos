"""Synthetic sanity checks for candle logic loss + soft-decode grad flow."""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

sys.path.append('../')
from logic_loss import (
    build_bit_tables,
    candle_logic_loss,
    contiguous_time_slice,
    freeze_tokenizer,
    heikin_ashi_consistency_loss,
    logic_loss_from_logits,
    ohlc_validity_loss,
    soft_decode_from_logits,
)


FEATURE_LIST = [
    'open', 'high', 'low', 'close', 'volume', 'amount',
    'HA_open', 'HA_high', 'HA_low', 'HA_close',
]


def _valid_ohlc_batch(b=2, t=4):
    # open=1, high=2, low=0.5, close=1.5 — valid envelope
    x = torch.zeros(b, t, len(FEATURE_LIST))
    x[..., 0] = 1.0
    x[..., 1] = 2.0
    x[..., 2] = 0.5
    x[..., 3] = 1.5
    x[..., 4] = 0.0
    x[..., 5] = 0.0
    # HA matching formulas from OHLC (and recursive open)
    o, h, l, c = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    ha_close = (o + h + l + c) * 0.25
    ha_open = torch.empty_like(ha_close)
    ha_open[:, 0] = 0.5 * (o[:, 0] + c[:, 0])
    for i in range(1, t):
        ha_open[:, i] = 0.5 * (ha_open[:, i - 1] + ha_close[:, i - 1])
    ha_high = torch.maximum(torch.maximum(h, ha_open), ha_close)
    ha_low = torch.minimum(torch.minimum(l, ha_open), ha_close)
    x[..., 6], x[..., 7], x[..., 8], x[..., 9] = ha_open, ha_high, ha_low, ha_close
    return x


def test_ohlc_valid_near_zero():
    x = _valid_ohlc_batch()
    loss = ohlc_validity_loss(x, FEATURE_LIST)
    assert float(loss) < 1e-6, loss


def test_ohlc_invalid_positive():
    x = _valid_ohlc_batch()
    # Invert high/low
    x[..., 1], x[..., 2] = x[..., 2].clone(), x[..., 1].clone()
    loss = ohlc_validity_loss(x, FEATURE_LIST)
    assert float(loss) > 0.1, loss


def test_ha_valid_near_zero():
    x = _valid_ohlc_batch()
    loss = heikin_ashi_consistency_loss(x, FEATURE_LIST)
    assert float(loss) < 1e-5, loss


def test_ha_broken_close_positive():
    x = _valid_ohlc_batch()
    x[..., 9] = x[..., 9] + 1.0
    loss = heikin_ashi_consistency_loss(x, FEATURE_LIST)
    assert float(loss) > 0.5, loss


def test_combined_weights():
    x = _valid_ohlc_batch()
    loss = candle_logic_loss(x, FEATURE_LIST, ohlc_weight=1.0, ha_weight=1.0)
    assert float(loss) < 1e-5, loss


def test_contiguous_time_slice():
    assert contiguous_time_slice(100, 0) is None
    assert contiguous_time_slice(50, 128) is None
    sl = contiguous_time_slice(200, 128)
    assert sl is not None
    assert sl.stop - sl.start == 128


class _TinyTok(nn.Module):
    """Minimal stand-in for KronosTokenizer decode path."""

    def __init__(self, s1_bits=2, s2_bits=2, d_model=8, d_in=10):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits
        self.post_quant_embed = nn.Linear(self.codebook_dim, d_model)
        self.decoder = nn.ModuleList([nn.Identity()])
        self.head = nn.Linear(d_model, d_in)


def test_soft_decode_grad_reaches_logits_not_tokenizer():
    tok = _TinyTok()
    freeze_tokenizer(tok)
    s1_table, s2_table = build_bit_tables(tok.s1_bits, tok.s2_bits, tok.codebook_dim, device=torch.device('cpu'))
    s1_logits = torch.randn(2, 3, 2 ** tok.s1_bits, requires_grad=True)
    s2_logits = torch.randn(2, 3, 2 ** tok.s2_bits, requires_grad=True)
    soft_x = soft_decode_from_logits(
        tok, s1_logits, s2_logits, s1_table, s2_table, use_checkpoint=True
    )
    loss = candle_logic_loss(soft_x, FEATURE_LIST)
    loss.backward()
    assert s1_logits.grad is not None and s1_logits.grad.abs().sum() > 0
    assert s2_logits.grad is not None and s2_logits.grad.abs().sum() > 0
    for p in tok.parameters():
        assert p.grad is None
        assert not p.requires_grad


def test_logic_loss_from_logits_subsample_and_microbatch():
    tok = _TinyTok()
    freeze_tokenizer(tok)
    s1_table, s2_table = build_bit_tables(tok.s1_bits, tok.s2_bits, tok.codebook_dim, device=torch.device('cpu'))
    s1_logits = torch.randn(4, 20, 2 ** tok.s1_bits, requires_grad=True)
    s2_logits = torch.randn(4, 20, 2 ** tok.s2_bits, requires_grad=True)
    loss = logic_loss_from_logits(
        tok, s1_logits, s2_logits, s1_table, s2_table, FEATURE_LIST,
        max_timesteps=8, use_checkpoint=True, microbatch_size=2,
    )
    loss.backward()
    assert s1_logits.grad is not None and s1_logits.grad.abs().sum() > 0


if __name__ == '__main__':
    test_ohlc_valid_near_zero()
    test_ohlc_invalid_positive()
    test_ha_valid_near_zero()
    test_ha_broken_close_positive()
    test_combined_weights()
    test_contiguous_time_slice()
    test_soft_decode_grad_reaches_logits_not_tokenizer()
    test_logic_loss_from_logits_subsample_and_microbatch()
    print('All logic_loss sanity checks passed.')
