from __future__ import annotations

import numpy as np
import pandas as pd

from calpuff_matrix.units import ppb_factor_array, ppb_per_g_m3


def test_no2_mass_concentration_to_ppb_uses_ideal_gas_relation() -> None:
    factor = float(ppb_per_g_m3(298.15, 101325.0, 46.0055))

    assert np.isclose(factor, 531793.0, rtol=1e-4)


def test_ppb_factor_array_orders_region_ids_and_requires_all_endpoints() -> None:
    weather = pd.DataFrame({
        "hour_index": [0, 0, 1, 1],
        "region_id": ["b", "a", "b", "a"],
        "temperature_k": [300.0, 290.0, 302.0, 292.0],
        "pressure_pa": [100000.0, 99000.0, 100000.0, 99000.0],
    })

    factors = ppb_factor_array(weather, np.array(["a", "b"]), 1, 46.0055)

    assert factors.shape == (2, 2)
    assert factors[0, 0] == ppb_per_g_m3(290.0, 99000.0, 46.0055)
    assert factors[1, 1] == ppb_per_g_m3(302.0, 100000.0, 46.0055)
