"""Deprecated compatibility wrapper; use ``calpuff-matrix fetch-hrrr``."""
from __future__ import annotations
import sys
import warnings
from calpuff_matrix.cli import main
warnings.warn("fetch_hrrr_selected_messages.py is deprecated; use calpuff-matrix fetch-hrrr", FutureWarning, stacklevel=2)
raise SystemExit(main(["fetch-hrrr", "--legacy-direct", *sys.argv[1:]]))
