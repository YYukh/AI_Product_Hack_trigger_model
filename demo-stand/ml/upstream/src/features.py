"""Единый causal Feature Engine для FX-рядов."""

from __future__ import annotations

import numpy as np
import pandas as pd


RETURN_HORIZONS = (1, 3, 5, 10, 20)
LEVEL_WINDOWS = (20, 30, 60, 90, 120, 180)
SLOPE_WINDOWS = (3, 5, 10)
VOLATILITY_WINDOWS = (5, 7, 20, 30)
CONTEXT_CURRENCIES = ("USD", "EUR", "CNY")
RECIPIENT_COUNTRIES = {
    "AMD": "AM",
    "KZT": "KZ",
    "KGS": "KG",
    "TJS": "TJ",
    "UZS": "UZ",
}


def _percentile_of_last(values: np.ndarray) -> float:
    current = values[-1]
    below = np.count_nonzero(values < current)
    equal = np.count_nonzero(values == current)
    average_rank = below + (equal - 1) / 2
    return float(average_rank / (len(values) - 1))


def _log_slope_bps(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype="float64")
    y = np.log(values.astype("float64"))
    x_centered = x - x.mean()
    denominator = np.square(x_centered).sum()
    return float((x_centered * (y - y.mean())).sum() / denominator * 10_000)


def _days_since_min(values: np.ndarray) -> float:
    return float(len(values) - 1 - np.argmin(values))


def _streak(values: pd.Series, *, positive: bool) -> pd.Series:
    condition = values.gt(0) if positive else values.lt(0)
    groups = (~condition).cumsum()
    return condition.groupby(groups).cumsum().astype("int16")


def _holiday_distances(
    dates: pd.Series,
    *,
    country: str,
    cap_days: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Дни до следующего и после последнего государственного праздника."""
    try:
        import holidays
    except ImportError as error:
        raise ImportError(
            "Для календарных признаков установите зависимости из requirements.txt"
        ) from error

    timestamps = pd.to_datetime(dates)
    years = range(timestamps.dt.year.min() - 1, timestamps.dt.year.max() + 2)
    calendar = holidays.country_holidays(country, years=list(years))
    ordinals = np.asarray(
        sorted(day.toordinal() for day in calendar),
        dtype=np.int64,
    )
    if not len(ordinals):
        raise ValueError(f"Не найден календарь праздников для {country}")

    values = np.asarray(
        [timestamp.date().toordinal() for timestamp in timestamps],
        dtype=np.int64,
    )
    positions = np.searchsorted(ordinals, values, side="right")
    previous_positions = np.maximum(positions - 1, 0)
    next_positions = np.minimum(positions, len(ordinals) - 1)
    days_since = np.minimum(values - ordinals[previous_positions], cap_days)
    days_to = np.minimum(ordinals[next_positions] - values, cap_days)
    return days_to.astype("float64"), days_since.astype("float64")


def _add_update_streaks(frame: pd.DataFrame) -> pd.DataFrame:
    """Посчитать серии роста/снижения именно по обновлениям ЦБ."""
    result = frame.copy()
    result["consecutive_down"] = np.nan
    result["consecutive_up"] = np.nan

    for _, positions in result.groupby("currency", sort=False).groups.items():
        group = result.loc[positions]
        update_positions = group.index[group["is_update_day"]]
        update_changes = result.loc[update_positions, "rate"].pct_change()

        result.loc[update_positions, "consecutive_down"] = _streak(
            update_changes,
            positive=False,
        ).to_numpy()
        result.loc[update_positions, "consecutive_up"] = _streak(
            update_changes,
            positive=True,
        ).to_numpy()

        result.loc[positions, ["consecutive_down", "consecutive_up"]] = (
            result.loc[positions, ["consecutive_down", "consecutive_up"]]
            .ffill()
            .fillna(0)
            .to_numpy()
        )

    result["consecutive_down"] = result["consecutive_down"].astype("int16")
    result["consecutive_up"] = result["consecutive_up"].astype("int16")
    return result


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Рассчитать только causal-признаки, доступные на текущую дату."""
    required = {
        "available_at",
        "currency",
        "rate",
        "is_update_day",
        "source_available_at",
        "source",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise KeyError(f"Не хватает полей market panel: {sorted(missing)}")

    result = (
        panel
        .copy()
        .sort_values(["currency", "available_at"])
        .reset_index(drop=True)
    )
    result["available_at"] = pd.to_datetime(result["available_at"])

    expected_step = (
        result
        .groupby("currency", sort=False)["available_at"]
        .diff()
        .dropna()
    )
    if not expected_step.eq(pd.Timedelta(days=1)).all():
        raise ValueError("Feature Engine ожидает полный ежедневный panel")

    grouped_rate = result.groupby("currency", sort=False)["rate"]

    for horizon in RETURN_HORIZONS:
        result[f"return_{horizon}d_bps"] = (
            grouped_rate.pct_change(horizon, fill_method=None) * 10_000
        )

    for window in LEVEL_WINDOWS:
        rolling = grouped_rate.rolling(window, min_periods=window)
        rolling_mean = rolling.mean().reset_index(level=0, drop=True)
        rolling_std = rolling.std().reset_index(level=0, drop=True)
        rolling_low = rolling.min().reset_index(level=0, drop=True)
        rolling_high = rolling.max().reset_index(level=0, drop=True)

        result[f"percentile_{window}d"] = (
            rolling
            .apply(_percentile_of_last, raw=True)
            .reset_index(level=0, drop=True)
        )
        result[f"distance_from_low_{window}d_bps"] = (
            (result["rate"] / rolling_low - 1) * 10_000
        )
        result[f"distance_from_high_{window}d_bps"] = (
            (result["rate"] / rolling_high - 1) * 10_000
        )
        result[f"zscore_{window}d"] = (
            (result["rate"] - rolling_mean) / rolling_std
        )
        prior_low = grouped_rate.transform(
            lambda values: values.shift(1).rolling(
                window,
                min_periods=window,
            ).min()
        )
        previous_rate = grouped_rate.shift(1)
        result[f"previous_distance_from_low_{window}d_bps"] = (
            (previous_rate / prior_low - 1) * 10_000
        )
        result[f"bounce_from_prior_low_{window}d_bps"] = (
            (result["rate"] / prior_low - 1) * 10_000
        )

    for window in SLOPE_WINDOWS:
        result[f"slope_{window}d_bps_per_day"] = (
            grouped_rate
            .rolling(window, min_periods=window)
            .apply(_log_slope_bps, raw=True)
            .reset_index(level=0, drop=True)
        )

    result = _add_update_streaks(result)

    result["momentum_change_1d_5d_bps"] = (
        result["return_1d_bps"] - result["return_5d_bps"] / 5
    )
    result["acceleration_1d_bps"] = (
        result["return_1d_bps"]
        - result.groupby("currency", sort=False)["return_1d_bps"].shift(1)
    )
    result["days_since_local_min_30d"] = (
        grouped_rate
        .rolling(30, min_periods=30)
        .apply(_days_since_min, raw=True)
        .reset_index(level=0, drop=True)
    )

    daily_return_bps = grouped_rate.pct_change(fill_method=None) * 10_000
    result["absolute_return_1d_bps"] = daily_return_bps.abs()
    result["absolute_return_5d_bps"] = result["return_5d_bps"].abs()

    for window in VOLATILITY_WINDOWS:
        result[f"rolling_std_{window}d_bps"] = (
            daily_return_bps
            .groupby(result["currency"], sort=False)
            .rolling(window, min_periods=window)
            .std()
            .reset_index(level=0, drop=True)
        )

    result["volatility_ratio_7d_30d"] = (
        result["rolling_std_7d_bps"]
        / result["rolling_std_30d_bps"]
    )

    result["day_of_week"] = result["available_at"].dt.dayofweek.astype("int8")
    result["month"] = result["available_at"].dt.month.astype("int8")
    result["month_start"] = result["available_at"].dt.is_month_start
    result["month_end"] = result["available_at"].dt.is_month_end
    month_angle = 2 * np.pi * (result["month"] - 1) / 12
    result["month_sin"] = np.sin(month_angle)
    result["month_cos"] = np.cos(month_angle)

    context_columns = []
    for currency in CONTEXT_CURRENCIES:
        prefix = currency.lower()
        source_columns = [
            "available_at",
            "return_1d_bps",
            "return_5d_bps",
            "return_20d_bps",
            "rolling_std_20d_bps",
        ]
        context = result.loc[
            result["currency"].eq(currency),
            source_columns,
        ].rename(
            columns={
                "return_1d_bps": f"{prefix}_return_1d_bps",
                "return_5d_bps": f"{prefix}_return_5d_bps",
                "return_20d_bps": f"{prefix}_return_20d_bps",
                "rolling_std_20d_bps": f"{prefix}_volatility_20d_bps",
            }
        )
        context_columns.extend(context.columns.difference(["available_at"]))
        result = result.merge(
            context,
            on="available_at",
            how="left",
            validate="many_to_one",
        )

    result["russia_days_to_holiday_30"] = np.nan
    result["russia_days_since_holiday_30"] = np.nan
    result["recipient_days_to_holiday_30"] = np.nan
    result["recipient_days_since_holiday_30"] = np.nan
    for currency, country in RECIPIENT_COUNTRIES.items():
        mask = result["currency"].eq(currency)
        if not mask.any():
            continue
        recipient_to, recipient_since = _holiday_distances(
            result.loc[mask, "available_at"],
            country=country,
        )
        russia_to, russia_since = _holiday_distances(
            result.loc[mask, "available_at"],
            country="RU",
        )
        result.loc[mask, "recipient_days_to_holiday_30"] = recipient_to
        result.loc[mask, "recipient_days_since_holiday_30"] = recipient_since
        result.loc[mask, "russia_days_to_holiday_30"] = russia_to
        result.loc[mask, "russia_days_since_holiday_30"] = russia_since

    result["recipient_preholiday_7"] = (
        result["recipient_days_to_holiday_30"].between(1, 7).astype("int8")
    )
    result["recipient_postholiday_3"] = (
        result["recipient_days_since_holiday_30"].le(3).astype("int8")
    )
    result["russia_preholiday_7"] = (
        result["russia_days_to_holiday_30"].between(1, 7).astype("int8")
    )

    return (
        result
        .sort_values(["available_at", "currency"])
        .reset_index(drop=True)
    )
