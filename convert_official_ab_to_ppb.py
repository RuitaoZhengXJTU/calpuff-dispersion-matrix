"""Deprecated compatibility wrapper; use ``calpuff-matrix convert-units``."""
from __future__ import annotations
import sys
import warnings
from calpuff_matrix.conversion import main
warnings.warn("convert_official_ab_to_ppb.py is deprecated; use calpuff-matrix convert-units", FutureWarning, stacklevel=2)
raise SystemExit(main(sys.argv[1:]))
