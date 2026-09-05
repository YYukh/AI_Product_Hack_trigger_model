"""Deterministic, stateful product constraints applied after signal selection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SignalPolicyConfig:
    cooldown_days: int = 2
    max_signals_per_7d: int = 2


def apply_signal_policy(
    ranked_events: pd.DataFrame,
    config: SignalPolicyConfig,
) -> pd.DataFrame:
    """Apply only online frequency constraints; never estimate signal quality."""
    if ranked_events.empty:
        return ranked_events.copy().reset_index(drop=True)
    ordered = ranked_events.copy()
    ordered["available_at"] = pd.to_datetime(ordered["available_at"])
    ordered = ordered.sort_values(
        ["currency", "available_at", "decision_score", "confidence",
         "expected_bps", "confidence_lift", "evidence_count", "horizon",
         "target_family"],
        ascending=[True, True, False, False, False, False, False, True, True],
    )
    accepted: list[object] = []
    for _, currency_rows in ordered.groupby("currency", sort=True):
        recent: deque[pd.Timestamp] = deque()
        last_sent: pd.Timestamp | None = None
        for available_at, same_day in currency_rows.groupby("available_at", sort=True):
            while recent and recent[0] <= available_at - pd.Timedelta(days=7):
                recent.popleft()
            if (
                last_sent is not None
                and available_at - last_sent < pd.Timedelta(days=config.cooldown_days)
            ):
                continue
            if len(recent) >= config.max_signals_per_7d:
                continue
            accepted.append(same_day.index[0])
            recent.append(available_at)
            last_sent = available_at
    return ordered.loc[accepted].sort_values(
        ["available_at", "currency"]
    ).reset_index(drop=True)
