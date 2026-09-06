"""Continuous future outcomes, отделённые от бинарных target."""

from __future__ import annotations

import pandas as pd


DEFAULT_HORIZONS = (1, 3, 5, 10, 20)


def _future_rolling(series: pd.Series, horizon: int, operation: str) -> pd.Series:
    reversed_series = series.iloc[::-1]
    rolling = reversed_series.rolling(
        horizon + 1,
        min_periods=horizon + 1,
    )
    transformed = getattr(rolling, operation)()
    return transformed.iloc[::-1]


def add_future_outcomes(
    features: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Добавить непрерывные outcomes на календарных горизонтах.

    Все future-поля предназначены только для разметки и исторической
    оценки. Они не являются признаками и не должны передаваться индикатору.
    """
    required = {"available_at", "currency", "rate"}
    missing = required.difference(features.columns)
    if missing:
        raise KeyError(f"Не хватает полей features: {sorted(missing)}")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("Все горизонты должны быть положительными")

    result = (
        features
        .copy()
        .sort_values(["currency", "available_at"])
        .reset_index(drop=True)
    )
    grouped = result.groupby("currency", sort=False)["rate"]

    for horizon in horizons:
        future_best = grouped.transform(
            lambda series: _future_rolling(series, horizon, "min")
        )
        future_worst = grouped.transform(
            lambda series: _future_rolling(series, horizon, "max")
        )
        future_rate = grouped.shift(-horizon)

        centered_mean = (
            grouped
            .rolling(
                2 * horizon + 1,
                center=True,
                min_periods=2 * horizon + 1,
            )
            .mean()
            .reset_index(level=0, drop=True)
        )
        centered_min = (
            grouped
            .rolling(
                2 * horizon + 1,
                center=True,
                min_periods=2 * horizon + 1,
            )
            .min()
            .reset_index(level=0, drop=True)
        )

        result[f"future_best_rate_{horizon}d"] = future_best
        result[f"future_best_regret_{horizon}d_bps"] = (
            (result["rate"] - future_best) / result["rate"] * 10_000
        )
        result[f"future_return_{horizon}d_bps"] = (
            (future_rate / result["rate"] - 1) * 10_000
        )
        result[f"local_advantage_{horizon}d_bps"] = (
            (centered_mean - result["rate"]) / centered_mean * 10_000
        )
        result[f"max_adverse_move_{horizon}d_bps"] = (
            (future_worst - result["rate"]) / result["rate"] * 10_000
        )
        result[f"centered_min_rate_{horizon}d"] = centered_min

    return (
        result
        .sort_values(["available_at", "currency"])
        .reset_index(drop=True)
    )


def outcome_columns(data: pd.DataFrame) -> list[str]:
    """Вернуть список future/outcome полей."""
    prefixes = (
        "future_",
        "local_advantage_",
        "max_adverse_move_",
        "centered_min_rate_",
    )
    return [column for column in data.columns if column.startswith(prefixes)]

