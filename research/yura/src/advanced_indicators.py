"""Causal experimental features and rule hypotheses for indicator discovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import IndicatorRule, default_indicator_rules


ADVANCED_FEATURES = (
    "kalman_level_gap_bps",
    "kalman_trend_bps",
    "kalman_trend_z",
    "kalman_trend_change_z",
    "kalman_reversal_score",
    "return_surprise_z_60d",
    "cusum_up_score",
    "cusum_down_score",
    "cusum_reversal_score",
    "mean_shift_5d_60d_z",
    "volatility_shift_log_7d_30d",
    "slope_inflection_3d_10d_bps",
    "rare_low_reversal_score",
)

# Only the discovery features with a repeatable economic role are promoted to
# the production candidate library. The wider set remains research-only.
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
    trend_bps = np.full(count, np.nan)
    trend_z = np.full(count, np.nan)
    trend_change_z = np.full(count, np.nan)

    state = np.array([values[0], 0.0], dtype=float)
    covariance = np.diag([1e-4, 1e-6])
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    observation = np.array([1.0, 0.0])
    identity = np.eye(2)
    return_variance = 1e-8
    previous_value = values[0]
    previous_trend_z = 0.0

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
        trend_bps[position] = state[1] * 10_000.0
        trend_z[position] = current_trend_z
        trend_change_z[position] = current_trend_z - previous_trend_z
        previous_trend_z = current_trend_z
        previous_value = value

    output = pd.DataFrame(index=frame.index)
    output["kalman_level_gap_bps"] = level_gap
    output["kalman_trend_bps"] = trend_bps
    output["kalman_trend_z"] = trend_z
    output["kalman_trend_change_z"] = trend_change_z
    previous_negative_trend = pd.Series(trend_z, index=frame.index).shift(1).clip(upper=0).abs()
    output["kalman_reversal_score"] = (
        previous_negative_trend * pd.Series(trend_z, index=frame.index).clip(lower=0)
    )
    return output


def _cusum_group(frame: pd.DataFrame) -> pd.DataFrame:
    """Directional standardized CUSUM whose reference distribution is trailing."""
    log_rate = np.log(pd.to_numeric(frame["rate"], errors="raise"))
    returns = log_rate.diff() * 10_000.0
    prior_mean = returns.shift(1).rolling(60, min_periods=30).mean()
    prior_std = returns.shift(1).rolling(60, min_periods=30).std().clip(lower=1e-6)
    surprise = ((returns - prior_mean) / prior_std).clip(-8.0, 8.0)

    positive = np.zeros(len(frame), dtype=float)
    negative = np.zeros(len(frame), dtype=float)
    allowance = 0.25
    for position, value in enumerate(surprise.fillna(0.0).to_numpy(float)):
        prior_positive = positive[position - 1] if position else 0.0
        prior_negative = negative[position - 1] if position else 0.0
        positive[position] = max(0.0, prior_positive + value - allowance)
        negative[position] = max(0.0, prior_negative - value - allowance)

    positive_series = pd.Series(positive, index=frame.index)
    negative_series = pd.Series(negative, index=frame.index)
    output = pd.DataFrame(index=frame.index)
    output["return_surprise_z_60d"] = surprise
    output["cusum_up_score"] = positive_series
    output["cusum_down_score"] = negative_series
    output["cusum_reversal_score"] = (
        negative_series.shift(1) * surprise.clip(lower=0)
    )
    return output


def add_advanced_indicator_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add only online-computable state-space/change/rare-state features."""
    required = {
        "available_at", "currency", "rate", "return_3d_bps",
        "rolling_std_7d_bps", "rolling_std_30d_bps",
        "slope_3d_bps_per_day", "slope_10d_bps_per_day", "zscore_90d",
    }
    if missing := required.difference(data.columns):
        raise KeyError(f"Не хватает полей для advanced indicators: {sorted(missing)}")
    result = data.copy().sort_values(["currency", "available_at"])
    advanced_parts = []
    for _, group in result.groupby("currency", sort=False):
        part = _kalman_group(group).join(_cusum_group(group))
        advanced_parts.append(part)
    advanced = pd.concat(advanced_parts).sort_index()
    for column in advanced:
        result.loc[advanced.index, column] = advanced[column]

    grouped = result.groupby("currency", sort=False)
    daily = grouped["rate"].pct_change(fill_method=None) * 10_000.0
    prior_mean_60 = daily.groupby(result["currency"], sort=False).transform(
        lambda values: values.shift(1).rolling(60, min_periods=30).mean()
    )
    prior_std_60 = daily.groupby(result["currency"], sort=False).transform(
        lambda values: values.shift(1).rolling(60, min_periods=30).std()
    ).clip(lower=1e-6)
    mean_5 = daily.groupby(result["currency"], sort=False).transform(
        lambda values: values.rolling(5, min_periods=5).mean()
    )
    result["mean_shift_5d_60d_z"] = (mean_5 - prior_mean_60) / prior_std_60
    result["volatility_shift_log_7d_30d"] = np.log(
        pd.to_numeric(result["rolling_std_7d_bps"], errors="coerce").clip(lower=1e-6)
        / pd.to_numeric(result["rolling_std_30d_bps"], errors="coerce").clip(lower=1e-6)
    )
    result["slope_inflection_3d_10d_bps"] = (
        result["slope_3d_bps_per_day"] - result["slope_10d_bps_per_day"]
    )
    result["rare_low_reversal_score"] = (
        (-pd.to_numeric(result["zscore_90d"], errors="coerce")).clip(lower=0)
        * pd.to_numeric(result["return_3d_bps"], errors="coerce").clip(lower=0)
    )
    return result.sort_values(["available_at", "currency"]).reset_index(drop=True)


def add_production_indicator_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add the small causal feature subset selected by indicator discovery."""
    required = {"available_at", "currency", "rate"}
    if missing := required.difference(data.columns):
        raise KeyError(
            f"Не хватает полей для production indicators: {sorted(missing)}"
        )
    result = data.copy().sort_values(["currency", "available_at"])
    parts = []
    for _, group in result.groupby("currency", sort=False):
        state = _kalman_group(group)
        surprise = _cusum_group(group).loc[:, ["return_surprise_z_60d"]]
        parts.append(state.join(surprise))
    selected = pd.concat(parts).sort_index()
    for column in PRODUCTION_ADVANCED_FEATURES:
        result.loc[selected.index, column] = selected[column]
    return result.sort_values(["available_at", "currency"]).reset_index(drop=True)


def discovery_indicator_rules() -> list[IndicatorRule]:
    """Broad but finite hypothesis library; thresholds are fit inside each fold."""
    rules = list(default_indicator_rules())

    def add(family: str, feature: str, operator: str, values: tuple[float, ...]) -> None:
        rules.extend(IndicatorRule(family, feature, operator, value) for value in values)

    for window in (20, 30, 60, 90, 120, 180):
        add("level_high", f"percentile_{window}d", "ge", (0.80, 0.90))
        add("bounce", f"bounce_from_prior_low_{window}d_bps", "ge", (20.0, 50.0, 100.0))
        add("zscore_low", f"zscore_{window}d", "le", (-1.5, -1.0, -0.5))
        add("zscore_high", f"zscore_{window}d", "ge", (0.5, 1.0, 1.5))
    add("acceleration_up", "acceleration_1d_bps", "ge", (0.0, 25.0, 50.0))
    add("acceleration_down", "acceleration_1d_bps", "le", (-50.0, -25.0, 0.0))
    add("calendar_preholiday", "recipient_preholiday_7", "ge", (1.0,))
    add("calendar_postholiday", "recipient_postholiday_3", "ge", (1.0,))
    add("russia_preholiday", "russia_preholiday_7", "ge", (1.0,))
    for prefix in ("usd", "eur", "cny"):
        add(f"context_{prefix}_down", f"{prefix}_return_5d_bps", "le", (-100.0, -50.0, 0.0))
        add(f"context_{prefix}_up", f"{prefix}_return_5d_bps", "ge", (0.0, 50.0, 100.0))

    add("kalman_level_low", "kalman_level_gap_bps", "le", (-25.0, -10.0, 0.0))
    add("kalman_trend_down", "kalman_trend_z", "le", (-2.0, -1.0, -0.5))
    add("kalman_trend_up", "kalman_trend_z", "ge", (0.0, 0.5, 1.0, 2.0))
    add("kalman_reversal", "kalman_reversal_score", "ge", (0.25, 0.5, 1.0, 2.0))
    add("kalman_acceleration", "kalman_trend_change_z", "ge", (0.25, 0.5, 1.0))
    add("cusum_down", "cusum_down_score", "ge", (2.0, 4.0, 6.0, 8.0))
    add("cusum_up", "cusum_up_score", "ge", (2.0, 4.0, 6.0, 8.0))
    add("cusum_reversal", "cusum_reversal_score", "ge", (1.0, 2.0, 4.0, 8.0))
    add("positive_surprise", "return_surprise_z_60d", "ge", (0.5, 1.0, 2.0))
    add("negative_surprise", "return_surprise_z_60d", "le", (-2.0, -1.0, -0.5))
    add("mean_shift_up", "mean_shift_5d_60d_z", "ge", (0.5, 1.0, 2.0))
    add("mean_shift_down", "mean_shift_5d_60d_z", "le", (-2.0, -1.0, -0.5))
    add("volatility_expansion", "volatility_shift_log_7d_30d", "ge", (0.0, 0.25, 0.5))
    add("volatility_contraction", "volatility_shift_log_7d_30d", "le", (-0.5, -0.25, 0.0))
    add("slope_inflection_up", "slope_inflection_3d_10d_bps", "ge", (0.0, 25.0, 50.0))
    add("slope_inflection_down", "slope_inflection_3d_10d_bps", "le", (-50.0, -25.0, 0.0))
    add("rare_low_reversal", "rare_low_reversal_score", "ge", (10.0, 25.0, 50.0, 100.0))

    unique = {rule.name: rule for rule in rules}
    return list(unique.values())
