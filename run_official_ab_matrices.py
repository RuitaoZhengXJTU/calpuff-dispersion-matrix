"""Deprecated compatibility wrapper; use ``calpuff-matrix run``."""
from __future__ import annotations
import sys
import warnings
from calpuff_matrix.matrices import main
warnings.warn("run_official_ab_matrices.py is deprecated; use calpuff-matrix run", FutureWarning, stacklevel=2)
raise SystemExit(main(sys.argv[1:]))
