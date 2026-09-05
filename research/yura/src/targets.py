"""Two business labels used by Yura without altering the shared target module."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _nullable_binary(condition: pd.Series, valid: pd.Series) -> pd.Series:
    target = pd.Series(pd.NA, index=condition.index, dtype="Int8")
    target.loc[valid] = condition.loc[valid].astype("int8")
    return target


def _strict_future_median(series: pd.Series, horizon: int) -> pd.Series:
    """Median of t+1...t+h; the current observation is deliberately absent."""
    shifted = series.shift(-1)
    return (
        shifted.iloc[::-1]
        .rolling(horizon, min_periods=horizon)
        .median()
        .iloc[::-1]
    )


def build_yura_targets(
    outcomes: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    w1_forward_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build strict local-minimum G0 and forward-window deterioration W1.

    W1 is intentionally not conditioned on an observable percentile feature.
    It asks whether acting now beats the median rate over the complete future
    window. Current cheapness remains available to rule/ML engines as evidence.
    """
    required = {"available_at", "currency", "rate"}
    if missing := required.difference(outcomes.columns):
        raise KeyError(f"Не хватает полей для Yura targets: {sorted(missing)}")
    if w1_forward_bps < 0:
        raise ValueError("w1_forward_bps не может быть отрицательным")

    result = outcomes.copy().sort_values(
        ["currency", "available_at"]
    ).reset_index(drop=True)
    grouped = result.groupby("currency", sort=False)["rate"]
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

        future_median = grouped.transform(
            lambda values: _strict_future_median(values, int(horizon))
        )
        forward_advantage = (
            (future_median / result["rate"] - 1.0) * 10_000.0
        )
        result[f"future_median_rate_{horizon}d"] = future_median
        result[f"forward_median_advantage_{horizon}d_bps"] = forward_advantage
        threshold_label = str(float(w1_forward_bps)).replace(".", "p")
        w1_name = (
            f"target_w1_forward_median_ge_{threshold_label}bps_h{horizon}d"
        )
        w1_valid = forward_advantage.notna()
        result[w1_name] = _nullable_binary(
            forward_advantage.ge(w1_forward_bps), w1_valid
        )
        definitions.append({
            "name": w1_name,
            "family": "W1",
            "scenario": "WINDOW_CLOSING",
            "horizon": int(horizon),
            "description": (
                "Median rate over t+1...t+h is worse than today's rate"
            ),
            "threshold_bps": float(w1_forward_bps),
        })

    result = result.sort_values(
        ["available_at", "currency"]
    ).reset_index(drop=True)
    registry = pd.DataFrame(definitions).sort_values(
        ["scenario", "family", "horizon"]
    ).reset_index(drop=True)
    return result, registry
