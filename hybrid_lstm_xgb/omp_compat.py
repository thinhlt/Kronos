"""macOS OpenMP workaround: torch and xgboost both ship OpenMP runtimes.

Import this module before torch/xgboost on any process that loads both.
`KMP_DUPLICATE_LIB_OK` alone is not enough on recent macOS + xgboost 3.x —
without capping OpenMP threads, `xgboost.train` can segfault after torch has
already initialized its runtime.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if sys.platform == "darwin":
    # Must be set before OpenMP initializes (i.e. before torch/xgboost import).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
