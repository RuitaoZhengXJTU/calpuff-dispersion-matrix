"""Conversions between gas mass concentration and volume mixing ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd


UNIVERSAL_GAS_CONSTANT_J_MOL_K = 8.31446261815324
DEFAULT_NO2_MOLECULAR_WEIGHT_G_MOL = 46.0055


def ppb_per_g_m3(
    temperature_k: np.ndarray | float,
    pressure_pa: np.ndarray | float,
    molecular_weight_g_mol: float,
) -> np.ndarray:
    """Return ppb per g/m3 under the ideal-gas approximation.

    The conversion is species-specific through molecular weight. It is valid
    for gaseous volume mixing ratios, not for particulate mass concentration.
    """
    if molecular_weight_g_mol <= 0:
        raise ValueError("molecular_weight_g_mol must be positive")
    temperature = np.asarray(temperature_k, dtype=float)
    pressure = np.asarray(pressure_pa, dtype=float)
    if not np.isfinite(temperature).all() or (temperature <= 0).any():
        raise ValueError("temperature_k must contain finite positive values")
    if not np.isfinite(pressure).all() or (pressure <= 0).any():
        raise ValueError("pressure_pa must contain finite positive values")
    return UNIVERSAL_GAS_CONSTANT_J_MOL_K * temperature * 1.0e9 / (
        pressure * molecular_weight_g_mol
    )


def ppb_factor_array(
    weather: pd.DataFrame,
    region_ids: np.ndarray,
    state_hours: int,
    molecular_weight_g_mol: float,
) -> np.ndarray:
    """Return conversion factors indexed by state time and regional column.

    ``state_hours`` is the number of one-hour transport intervals. Factors are
    required at the interval endpoints 0 through ``state_hours`` inclusive.
    """
    required = {"hour_index", "region_id", "temperature_k", "pressure_pa"}
    missing = required - set(weather.columns)
    if missing:
        raise ValueError(f"weather table lacks gas-unit columns: {sorted(missing)}")
    indexed = weather.assign(region_id=weather["region_id"].astype(str)).set_index(
        ["hour_index", "region_id"]
    )
    if indexed.index.has_duplicates:
        raise ValueError("weather table contains duplicate (hour_index, region_id) rows")
    factors = np.zeros((state_hours + 1, len(region_ids)), dtype=float)
    for hour_index in range(state_hours + 1):
        try:
            rows = indexed.loc[hour_index]
        except KeyError as exc:
            raise ValueError(f"weather table lacks ppb conversion data for hour {hour_index}") from exc
        try:
            ordered = rows.loc[list(region_ids)]
        except KeyError as exc:
            raise ValueError(f"weather table lacks one or more regions at hour {hour_index}") from exc
        factors[hour_index] = ppb_per_g_m3(
            ordered["temperature_k"].to_numpy(float),
            ordered["pressure_pa"].to_numpy(float),
            molecular_weight_g_mol,
        )
    return factors
