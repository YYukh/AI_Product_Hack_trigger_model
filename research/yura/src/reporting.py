"""Like-for-like comparison with the existing production backtest."""

from __future__ import annotations

import pandas as pd


def build_action_summary(backtest_rows: pd.DataFrame) -> pd.DataFrame:
    """Economic report over unique client pushes, without mixing target labels."""
    required = {
        "available_at", "currency", "signal", "event_id", "benefit_bps",
        "_stratum_random_benefit_bps",
    }
    if missing := required.difference(backtest_rows.columns):
        raise KeyError(f"В backtest_rows нет полей: {sorted(missing)}")
    signals = (
        backtest_rows.loc[backtest_rows["signal"]]
        .sort_values(["currency", "available_at"])
        .drop_duplicates("event_id")
    )
    full_dates = pd.to_datetime(backtest_rows["available_at"])
    calendar_weeks = ((full_dates.max() - full_dates.min()).days + 1) / 7
    rows = []
    for currency in sorted(backtest_rows["currency"].unique()):
        sample = signals.loc[signals["currency"].eq(currency)]
        benefit = pd.to_numeric(sample["benefit_bps"], errors="coerce").dropna()
        random = pd.to_numeric(
            sample["_stratum_random_benefit_bps"], errors="coerce"
        ).dropna()
        mean_benefit = float(benefit.mean()) if len(benefit) else float("nan")
        random_mean = float(random.mean()) if len(random) else float("nan")
        rows.append({
            "currency": currency,
            "signal_count": len(sample),
            "calendar_weeks": calendar_weeks,
            "signals_per_week": len(sample) / calendar_weeks if calendar_weeks else 0.0,
            "mean_benefit_bps": mean_benefit,
            "random_mean_benefit_bps": random_mean,
            "benefit_uplift_bps": mean_benefit - random_mean,
            "positive_benefit_rate": float(benefit.gt(0).mean()) if len(benefit) else 0.0,
        })
    return pd.DataFrame(rows)


def compare_backtest_summaries(
    current: pd.DataFrame,
    alternative: pd.DataFrame,
) -> pd.DataFrame:
    """Join equal backtest scopes and calculate transparent deltas."""
    keys = ["scope", "currency", "horizon", "target_family"]
    metrics = [
        "signal_count", "signals_per_week", "precision", "lift",
        "mean_benefit_bps", "benefit_uplift_bps", "positive_benefit_rate",
    ]
    required = set(keys + metrics)
    for name, frame in (("current", current), ("alternative", alternative)):
        if missing := required.difference(frame.columns):
            raise KeyError(f"В {name} summary нет полей: {sorted(missing)}")
        if frame.duplicated(keys).any():
            raise ValueError(f"В {name} summary повторяются ключи")
    comparison = current.loc[:, keys + metrics].merge(
        alternative.loc[:, keys + metrics],
        on=keys, how="outer", validate="one_to_one",
        suffixes=("_current", "_yura"), indicator=True,
    )
    for metric in metrics:
        comparison[f"{metric}_delta"] = (
            comparison[f"{metric}_yura"] - comparison[f"{metric}_current"]
        )
    return comparison.sort_values(keys).reset_index(drop=True)
