"""rankforge — package init.

Sets a few env vars to avoid OpenMP/thread-pool conflicts when faiss-cpu,
sentence-transformers, and lightgbm are imported in the same process on macOS.
Set here so they take effect before any of those C++ libs initialize.
"""
import os as _os

_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
