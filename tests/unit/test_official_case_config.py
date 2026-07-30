from __future__ import annotations

import argparse
from pathlib import Path

from calpuff_matrix.config import load_case_config
from calpuff_matrix.matrices import _apply_case_config, _state_equation


def test_portable_case_config_applies_paths_time_and_domain() -> None:
    config = load_case_config(Path("config/official_case_20250623_18z.yaml"))
    args = argparse.Namespace(
        case_root=None,
        partition_dir=None,
        sources=None,
        receptors=None,
        receptor_batch_dir=None,
        region_index=None,
        generators=None,
        weather=None,
        output_root=None,
        seed=None,
        calpost_template=None,
        calmet_dat=None,
        start_utc=None,
        hours=None,
        a_start_hour=1,
        a_hours=None,
        a_sparse_radius_km=None,
    )

    _apply_case_config(args, config)

    assert args.case_id == "case_20250623_18z_30sqmi"
    assert args.start_utc == "2025-06-23T18:00:00Z"
    assert args.hours == 24
    assert args.a_hours == 23
    assert args.concentration_unit == "ppb"
    assert args.molecular_weight_g_mol == 46.0055
    assert args.tracer_species == "NO2_equivalent_passive_tracer"
    assert args.partition_dir.name == "area_capped_30sqmi_population_balanced"
    assert args.calpuff_domain.nx == 79
    assert args.calpuff_domain.projected_crs.startswith("+proj=lcc")


def test_state_equation_tracks_configured_horizon() -> None:
    assert _state_equation(3) == "c1 = B0 @ emitted_mass_lb; c[h+1] = A[h] @ c[h] for h=1..2"
