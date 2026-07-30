"""Deprecated compatibility wrapper; use ``calpuff-matrix validate``."""
from __future__ import annotations
import sys
import warnings
from calpuff_matrix.validation import main
warnings.warn("validate_official_ab.py is deprecated; use calpuff-matrix validate", FutureWarning, stacklevel=2)
raise SystemExit(main(sys.argv[1:]))
