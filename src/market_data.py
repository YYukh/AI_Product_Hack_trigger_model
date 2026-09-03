"""Point-in-time представление дневных валютных курсов."""

from __future__ import annotations

import pandas as pd


REQUIRED_INDEX_NAME = "available_at"


def validate_wide_rates(rates: pd.DataFrame) -> None:
    """Проверить нормированный широкий ряд курсов."""
    if not isinstance(rates.index, pd.DatetimeIndex):
        raise TypeError("Индекс rates должен быть DatetimeIndex")
    if rates.empty:
        raise ValueError("rates не должен быть пустым")
    if rates.index.has_duplicates:
        raise ValueError("Индекс rates содержит повторяющиеся даты")
    if not rates.index.is_monotonic_increasing:
        raise ValueError("Индекс rates должен быть отсортирован")
    if rates.columns.has_duplicates:
        raise ValueError("В rates есть повторяющиеся валюты")
    if rates.isna().any().any():
        raise ValueError("В исходных обновлениях rates есть пропуски")
    if not rates.gt(0).all().all():
        raise ValueError("Все курсы должны быть положительными")


def build_daily_market_panel(
    rates: pd.DataFrame,
    *,
    source: str = "CBR",
) -> pd.DataFrame:
    """Построить календарный point-in-time market panel.

    ``rates`` содержит только фактические обновления. В календарном panel
    последний известный курс переносится вперёд, но такой перенос явно
    маркируется ``is_update_day=False``. Поле ``source_available_at`` хранит
    дату появления используемого значения и защищает as-of семантику.
    """
    validate_wide_rates(rates)

    normalized = rates.copy()
    normalized.index = pd.to_datetime(normalized.index).normalize()
    normalized.index.name = REQUIRED_INDEX_NAME

    if normalized.index.has_duplicates:
        raise ValueError(
            "После нормализации времени появились повторяющиеся даты"
        )

    calendar = pd.date_range(
        normalized.index.min(),
        normalized.index.max(),
        freq="D",
        name=REQUIRED_INDEX_NAME,
    )

    observed = normalized.reindex(calendar)
    is_update = observed.notna()
    daily_rates = observed.ffill()

    calendar_values = pd.DataFrame(
        {
            currency: calendar
            for currency in normalized.columns
        },
        index=calendar,
    )
    source_available_at = calendar_values.where(is_update).ffill()

    panel = pd.concat(
        {
            "rate": daily_rates,
            "is_update_day": is_update,
            "source_available_at": source_available_at,
        },
        axis="columns",
        sort=False,
    )
    panel.columns.names = ["field", "currency"]

    panel = (
        panel
        .stack(level="currency", future_stack=True)
        .reset_index()
        .sort_values(["currency", REQUIRED_INDEX_NAME])
        .reset_index(drop=True)
    )

    panel["rate"] = panel["rate"].astype("float64")
    panel["is_update_day"] = panel["is_update_day"].astype(bool)
    panel["source"] = source

    if (panel["source_available_at"] > panel[REQUIRED_INDEX_NAME]).any():
        raise AssertionError("Panel содержит данные из будущего")

    return panel[
        [
            REQUIRED_INDEX_NAME,
            "currency",
            "rate",
            "is_update_day",
            "source_available_at",
            "source",
        ]
    ]


def market_snapshot(
    panel: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    """Вернуть последнее известное состояние каждого коридора на ``as_of``."""
    timestamp = pd.Timestamp(as_of).normalize()
    known = panel.loc[panel[REQUIRED_INDEX_NAME].le(timestamp)].copy()

    if known.empty:
        raise ValueError("На указанный as_of ещё нет доступных данных")

    snapshot = (
        known
        .sort_values(["currency", REQUIRED_INDEX_NAME])
        .groupby("currency", sort=True, as_index=False)
        .tail(1)
        .sort_values("currency")
        .reset_index(drop=True)
    )

    if (snapshot["source_available_at"] > timestamp).any():
        raise AssertionError("Snapshot нарушает point-in-time ограничение")

    return snapshot

