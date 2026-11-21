"""SanDaHa-Zero: an AI and game client for the Hunan card game 三打哈."""
import os as _os

# Inference uses many small batched matmuls (batch 1-64). Multi-threaded BLAS spends far
# more time waking/syncing threads than computing (measured 25x slower on Windows OpenBLAS),
# so pin BLAS to a single thread before numpy is imported anywhere.
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

__version__ = "2.4.1"
