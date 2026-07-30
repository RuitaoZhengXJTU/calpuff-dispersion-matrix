"""Deprecated compatibility wrapper; use ``calpuff-matrix prepare``."""
from __future__ import annotations
import sys
import warnings
from calpuff_matrix.preparation import main
warnings.warn("prepare_official_sparse_calpuff.py is deprecated; use calpuff-matrix prepare", FutureWarning, stacklevel=2)
raise SystemExit(main(sys.argv[1:]))
