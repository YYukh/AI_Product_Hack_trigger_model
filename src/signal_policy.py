"""Causal deterministic policy applied after a replaceable meta-model."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalPolicyConfig:
    cooldown_days: int = 3
    max_signals_per_7d: int = 2

    def __post_init__(self) -> None:
        if self.cooldown_days < 0:
            raise ValueError("cooldown_days не может быть отрицательным")
        if self.max_signals_per_7d <= 0:
            raise ValueError("max_signals_per_7d должен быть положительным")


def apply_signal_policy(
    events: pd.DataFrame,
    config: SignalPolicyConfig = SignalPolicyConfig(),
    *,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select events online per currency without looking at future rows.

    At most one event is sent for a currency on a market date.  The winner is
    deterministic: confidence, evidence count, shorter horizon, target name.
    Cooldown and the hard cap are then checked against already sent events.
    """
    if events.empty:
        return events.copy().reset_index(drop=True)
    required = {
        "available_at", "currency", "confidence", "evidence_count",
        "horizon", "target",
    }
    if missing := required.difference(events.columns):
        raise KeyError(f"В events нет policy-полей: {sorted(missing)}")

    ordered = events.copy()
    ordered["available_at"] = pd.to_datetime(ordered["available_at"])
    ordered["_confidence_sort"] = pd.to_numeric(
        ordered["confidence"], errors="coerce"
    ).fillna(-np.inf)
    ordered["_evidence_sort"] = pd.to_numeric(
        ordered["evidence_count"], errors="coerce"
    ).fillna(0)
    ordered = ordered.sort_values(
        ["currency", "available_at", "_confidence_sort", "_evidence_sort",
         "horizon", "target"],
        ascending=[True, True, False, False, True, True],
    )

    accepted: list[object] = []
    for _, currency_events in ordered.groupby("currency", sort=True):
        currency = str(currency_events["currency"].iloc[0])
        previous_dates = pd.Series(dtype="datetime64[ns]")
        if history is not None and not history.empty:
            if missing := {"available_at", "currency"}.difference(history.columns):
                raise KeyError(f"В policy history нет полей: {sorted(missing)}")
            previous_dates = pd.to_datetime(
                history.loc[history["currency"].astype(str).eq(currency), "available_at"]
            ).sort_values()
            first_date = currency_events["available_at"].min()
            previous_dates = previous_dates.loc[previous_dates.lt(first_date)]
        last_sent_at = previous_dates.max() if len(previous_dates) else None
        first_date = currency_events["available_at"].min()
        recent: deque[pd.Timestamp] = deque(
            previous_dates.loc[
                previous_dates.gt(first_date - pd.Timedelta(days=7))
            ].tolist()
        )
        for available_at, same_day in currency_events.groupby(
            "available_at", sort=True
        ):
            while recent and recent[0] <= available_at - pd.Timedelta(days=7):
                recent.popleft()
            if (
                last_sent_at is not None
                and available_at - last_sent_at
                < pd.Timedelta(days=config.cooldown_days)
            ):
                continue
            if len(recent) >= config.max_signals_per_7d:
                continue
            accepted.append(same_day.index[0])
            recent.append(available_at)
            last_sent_at = available_at

    return (
        ordered.loc[accepted]
        .drop(columns=["_confidence_sort", "_evidence_sort"])
        .sort_values(["available_at", "currency"])
        .reset_index(drop=True)
    )
