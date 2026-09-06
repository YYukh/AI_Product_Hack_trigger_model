"""Causal state-space features used by the production signal engines."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRODUCTION_ADVANCED_FEATURES = (
    "kalman_level_gap_bps",
    "kalman_trend_z",
    "kalman_reversal_score",
    "return_surprise_z_60d",
)


def _kalman_group(frame: pd.DataFrame) -> pd.DataFrame:
    """Causal adaptive local-linear-trend Kalman filter for one currency."""
    values = np.log(pd.to_numeric(frame["rate"], errors="raise").to_numpy(float))
    count = len(values)
    level_gap = np.full(count, np.nan)
    trend_z = np.full(count, np.nan)

    state = np.array([values[0], 0.0], dtype=float)
    covariance = np.diag([1e-4, 1e-6])
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    observation = np.array([1.0, 0.0])
    identity = np.eye(2)
    return_variance = 1e-8
    previous_value = values[0]

    for position, value in enumerate(values):
        if position:
            observed_return = value - previous_value
            # EW variance is updated only after the previous state existed.
            return_variance = (
                0.97 * return_variance + 0.03 * observed_return ** 2
            )
        variance = max(return_variance, 1e-10)
        process_noise = np.diag([0.05 * variance, 0.005 * variance])
        measurement_noise = max(0.50 * variance, 1e-10)

        predicted_state = transition @ state
        predicted_covariance = (
            transition @ covariance @ transition.T + process_noise
        )
        innovation = value - float(observation @ predicted_state)
        innovation_variance = float(
            observation @ predicted_covariance @ observation + measurement_noise
        )
        gain = predicted_covariance @ observation / innovation_variance
        state = predicted_state + gain * innovation
        covariance = (identity - np.outer(gain, observation)) @ predicted_covariance

        current_trend_z = state[1] / np.sqrt(max(covariance[1, 1], 1e-12))
        level_gap[position] = (value - state[0]) * 10_000.0
        trend_z[position] = current_trend_z
        previous_value = value

    output = pd.DataFrame(index=frame.index)
    output["kalman_level_gap_bps"] = level_gap
    output["kalman_trend_z"] = trend_z
    previous_negative_trend = pd.Series(trend_z, index=frame.index).shift(1).clip(upper=0).abs()
    output["kalman_reversal_score"] = (
        previous_negative_trend * pd.Series(trend_z, index=frame.index).clip(lower=0)
    )
    return output


def _return_surprise_group(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardized daily return against a strictly trailing distribution."""
    log_rate = np.log(pd.to_numeric(frame["rate"], errors="raise"))
    returns = log_rate.diff() * 10_000.0
    prior_mean = returns.shift(1).rolling(60, min_periods=30).mean()
    prior_std = returns.shift(1).rolling(60, min_periods=30).std().clip(lower=1e-6)
    surprise = ((returns - prior_mean) / prior_std).clip(-8.0, 8.0)

    output = pd.DataFrame(index=frame.index)
    output["return_surprise_z_60d"] = surprise
    return output


def add_production_indicator_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add the compact causal state-space feature set used by the pipeline."""
    required = {"available_at", "currency", "rate"}
    if missing := required.difference(data.columns):
        raise KeyError(
            f"Не хватает полей для production indicators: {sorted(missing)}"
        )
    result = data.copy().sort_values(["currency", "available_at"])
    parts = []
    for _, group in result.groupby("currency", sort=False):
        state = _kalman_group(group)
        surprise = _return_surprise_group(group)
        parts.append(state.join(surprise))
    selected = pd.concat(parts).sort_index()
    for column in PRODUCTION_ADVANCED_FEATURES:
        result.loc[selected.index, column] = selected[column]
    return result.sort_values(["available_at", "currency"]).reset_index(drop=True)
