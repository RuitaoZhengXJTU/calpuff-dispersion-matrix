"""Deprecated compatibility wrapper; use the packaged CALPOST parser."""
from __future__ import annotations
import sys
import warnings
from calpuff_matrix.calpost import main
warnings.warn("parse_calpost_tseries.py is deprecated; use calpuff-matrix run or calpuff_matrix.calpost", FutureWarning, stacklevel=2)
raise SystemExit(main(sys.argv[1:]))
