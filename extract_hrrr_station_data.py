"""Deprecated compatibility wrapper; use ``calpuff-matrix build-calmet``."""
from __future__ import annotations
import sys
import warnings
from calpuff_matrix.station_data import main
warnings.warn("extract_hrrr_station_data.py is deprecated; use calpuff-matrix build-calmet", FutureWarning, stacklevel=2)
raise SystemExit(main(sys.argv[1:]))
