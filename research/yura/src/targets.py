"""Two business labels used by Yura without altering the shared target module."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _nullable_binary(condition: pd.Series, valid: pd.Series) -> pd.Series:
    target = pd.Series(pd.NA, index=condition.index, dtype="Int8")
    target.loc[valid] = condition.loc[valid].astype("int8")
    return target


def build_yura_targets(
    outcomes: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    w1_deterioration_bps: float = 75.0,
    w1_low_percentile: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build G0 and the original low-zone-then-deterioration W1 target."""
    required = {"available_at", "currency", "rate", "percentile_90d"}
    if missing := required.difference(outcomes.columns):
        raise KeyError(f"Не хватает полей для Yura targets: {sorted(missing)}")
    if w1_deterioration_bps < 0:
        raise ValueError("w1_deterioration_bps не может быть отрицательным")
    if not 0.0 <= w1_low_percentile <= 1.0:
        raise ValueError("w1_low_percentile должен лежать в [0, 1]")

    result = outcomes.copy().sort_values(
        ["currency", "available_at"]
    ).reset_index(drop=True)
    definitions: list[dict] = []

    for horizon in horizons:
        centered_min = f"centered_min_rate_{horizon}d"
        if centered_min not in result:
            raise KeyError(f"Не хватает outcome {centered_min!r}")

        g0_name = f"target_g0_exact_min_h{horizon}d"
        g0_valid = result[centered_min].notna()
        g0_condition = pd.Series(
            np.isclose(
                result["rate"], result[centered_min], rtol=0, atol=1e-12
            ),
            index=result.index,
        )
        result[g0_name] = _nullable_binary(g0_condition, g0_valid)
        definitions.append({
            "name": g0_name,
            "family": "G0",
            "scenario": "GOOD_NOW",
            "horizon": int(horizon),
            "description": "Exact local minimum in ±h calendar days",
            "threshold_bps": np.nan,
        })

        future_return = f"future_return_{horizon}d_bps"
        if future_return not in result:
            raise KeyError(f"Не хватает outcome {future_return!r}")
        threshold_label = f"{float(w1_deterioration_bps):g}".replace(".", "p")
        percentile_label = f"{float(w1_low_percentile):g}".replace(".", "p")
        w1_name = (
            f"target_w1_lowpct_{percentile_label}_"
            f"deterioration_{threshold_label}bps_h{horizon}d"
        )
        w1_valid = result[future_return].notna() & result["percentile_90d"].notna()
        w1_condition = (
            result["percentile_90d"].le(w1_low_percentile)
            & result[future_return].gt(w1_deterioration_bps)
        )
        result[w1_name] = _nullable_binary(
            w1_condition, w1_valid
        )
        definitions.append({
            "name": w1_name,
            "family": "W1",
            "scenario": "WINDOW_CLOSING",
            "horizon": int(horizon),
            "description": "Currently in the low zone and worse after h days",
            "threshold_bps": float(w1_deterioration_bps),
            "low_percentile": float(w1_low_percentile),
        })

    result = result.sort_values(
        ["available_at", "currency"]
    ).reset_index(drop=True)
    registry = pd.DataFrame(definitions).sort_values(
        ["scenario", "family", "horizon"]
    ).reset_index(drop=True)
    return result, registry
