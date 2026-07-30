"""Deprecated compatibility wrapper; use ``calpuff-matrix build-calmet``."""
from __future__ import annotations
import sys
import warnings
from calpuff_matrix.calmet_inputs import main
warnings.warn("build_hrrr_calmet_inputs.py is deprecated; use calpuff-matrix build-calmet", FutureWarning, stacklevel=2)
raise SystemExit(main(sys.argv[1:]))
