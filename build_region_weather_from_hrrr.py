"""Deprecated compatibility wrapper; use ``calpuff-matrix build-weather``."""
from __future__ import annotations
import sys
import warnings
from calpuff_matrix.cli import main
warnings.warn("build_region_weather_from_hrrr.py is deprecated; use calpuff-matrix build-weather", FutureWarning, stacklevel=2)
raise SystemExit(main(["build-weather", "--legacy-direct", *sys.argv[1:]]))
