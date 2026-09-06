"""Unified catalogue of legacy labels for hypothesis discovery."""

from __future__ import annotations

import pandas as pd

from src.targets import build_targets

DISCOVERY_TARGET_FAMILIES = (
    "G0",
    "G1",
    "W0",
    "W1",
)


def build_discovery_targets(
    outcomes: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    g1_tolerances_bps: tuple[int, ...] = (25, 50, 100),
    legacy_w0_deterioration_bps: int = 75,
    legacy_w1_low_percentile: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the four original target concepts used by the project."""
    legacy_rows, legacy_registry = build_targets(
        outcomes,
        horizons=horizons,
        g1_tolerances_bps=g1_tolerances_bps,
        w0_deterioration_bps=legacy_w0_deterioration_bps,
        w1_low_percentile=legacy_w1_low_percentile,
    )
    result = legacy_rows
    registry = legacy_registry.copy()
    registry["target_concept"] = registry["family"]
    registry["target_variant"] = registry["name"]
    registry = registry.sort_values(
        ["family", "horizon", "name"]
    ).reset_index(drop=True)
    return result, registry
