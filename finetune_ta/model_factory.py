"""
Constructs the tokenizer/predictor for training, centralizing the "pretrained
vs from-scratch" branching that finetune_csv duplicates across three scripts.

Architecture hyperparameters (d_model, n_heads, ...) for the from-scratch case
mirror NeoQuasar/Kronos-Tokenizer-base and NeoQuasar/Kronos-base exactly, so
that only `d_in` differs when indicators change the feature count.
"""
import os

from model import Kronos, KronosTokenizer

TOKENIZER_ARCH_DEFAULTS = dict(
    d_model=256,
    n_heads=4,
    ff_dim=512,
    n_enc_layers=4,
    n_dec_layers=4,
    ffn_dropout_p=0.0,
    attn_dropout_p=0.0,
    resid_dropout_p=0.0,
    s1_bits=10,
    s2_bits=10,
    beta=0.05,
    gamma0=1.0,
    gamma=1.1,
    zeta=0.05,
    group_size=4,
)

PREDICTOR_ARCH_DEFAULTS = dict(
    s1_bits=10,
    s2_bits=10,
    n_layers=12,
    d_model=832,
    n_heads=16,
    ff_dim=2048,
    ffn_dropout_p=0.2,
    attn_dropout_p=0.0,
    resid_dropout_p=0.2,
    token_dropout_p=0.0,
    learn_te=True,
)


def build_tokenizer_from_scratch(d_in: int) -> KronosTokenizer:
    return KronosTokenizer(d_in=d_in, **TOKENIZER_ARCH_DEFAULTS)


def build_initial_tokenizer(config) -> KronosTokenizer:
    """Tokenizer used to *start* phase-1 (tokenizer) training: pretrained
    weights if requested and d_in matches, otherwise a fresh random init."""
    if getattr(config, 'pre_trained_tokenizer', False):
        return KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    return build_tokenizer_from_scratch(config.d_in)


def load_finetuned_tokenizer(config) -> KronosTokenizer:
    """Tokenizer used for phase-2 (predictor) training: always the tokenizer
    that phase-1 actually saved to disk, regardless of how it was initialized."""
    if not os.path.exists(config.finetuned_tokenizer_path):
        raise FileNotFoundError(
            f"Finetuned tokenizer not found at {config.finetuned_tokenizer_path}. "
            "Run tokenizer training (phase 1) before predictor training (phase 2)."
        )
    return KronosTokenizer.from_pretrained(config.finetuned_tokenizer_path)


def build_predictor(config) -> Kronos:
    if getattr(config, 'pre_trained_predictor', True):
        return Kronos.from_pretrained(config.pretrained_predictor_path)
    return Kronos(**PREDICTOR_ARCH_DEFAULTS)
