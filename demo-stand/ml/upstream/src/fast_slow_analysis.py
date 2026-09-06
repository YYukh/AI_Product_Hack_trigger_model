"""Сравнение быстрых и медленных OOS-сигналов."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _prepare_signals(
    signals: pd.DataFrame,
    *,
    source: str,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    required = {"available_at", "currency", "target_family", "horizon"}
    missing = required.difference(signals.columns)
    if missing:
        raise KeyError(f"В {source} signals нет полей: {sorted(missing)}")
    result = signals.copy()
    result["available_at"] = pd.to_datetime(result["available_at"])
    result["horizon"] = pd.to_numeric(result["horizon"]).astype(int)
    result = result.loc[result["horizon"].isin(horizons)].copy()
    result["engine_type"] = source
    result["is_correct"] = result["target_value"].astype(bool) if "target_value" in result else False
    if "benefit_bps" not in result:
        result["benefit_bps"] = np.nan
    return result


def analyze_fast_slow_signals(
    indicator_signals: pd.DataFrame,
    ml_signals: pd.DataFrame,
    *,
    fast_horizons: tuple[int, ...] = (1, 3),
    slow_horizons: tuple[int, ...] = (10, 20),
    max_confirmation_gap_days: int = 20,
    evaluation_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Сравнить fast (h=1/3) и slow (h=10/20) OOS-сигналы.

    Сигналы сопоставляются внутри currency × target_family × engine_type:
    для каждого fast-сигнала берётся первое slow-срабатывание после него в
    заданном окне. Будущие значения используются только для оценки.
    """
    if not fast_horizons or not slow_horizons or max_confirmation_gap_days <= 0:
        raise ValueError("Некорректные горизонты или окно подтверждения")
    fast_frames = [
        _prepare_signals(indicator_signals, source="indicator", horizons=fast_horizons),
        _prepare_signals(ml_signals, source="ml", horizons=fast_horizons),
    ]
    slow_frames = [
        _prepare_signals(indicator_signals, source="indicator", horizons=slow_horizons),
        _prepare_signals(ml_signals, source="ml", horizons=slow_horizons),
    ]
    fast = pd.concat(fast_frames, ignore_index=True)
    slow = pd.concat(slow_frames, ignore_index=True)
    group_columns = ["currency", "target_family", "engine_type"]
    summary_rows: list[dict] = []
    pair_rows: list[dict] = []

    for keys, fast_group in fast.groupby(group_columns, sort=True):
        keys = tuple(keys) if isinstance(keys, tuple) else (keys,)
        values = dict(zip(group_columns, keys))
        slow_group = slow
        for column, value in values.items():
            slow_group = slow_group.loc[slow_group[column].eq(value)]
        fast_group = fast_group.sort_values("available_at")
        slow_group = slow_group.sort_values("available_at")
        fast_benefit = pd.to_numeric(fast_group["benefit_bps"], errors="coerce").dropna()
        slow_benefit = pd.to_numeric(slow_group["benefit_bps"], errors="coerce").dropna()
        matched = 0
        lags: list[float] = []
        waiting_costs: list[float] = []
        available_slow = list(slow_group.index)
        fast_baselines = []
        slow_baselines = []
        if evaluation_data is not None:
            for _, row in fast_group.iterrows():
                subset = evaluation_data.loc[
                    evaluation_data["currency"].eq(row["currency"])
                ]
                fast_baselines.append(
                    float(subset[row["target"]].mean())
                    if row.get("target") in subset.columns else np.nan
                )
            for _, row in slow_group.iterrows():
                subset = evaluation_data.loc[
                    evaluation_data["currency"].eq(row["currency"])
                ]
                slow_baselines.append(
                    float(subset[row["target"]].mean())
                    if row.get("target") in subset.columns else np.nan
                )
        for fast_row in fast_group.itertuples(index=False):
            candidates = slow_group.loc[available_slow].loc[
                slow_group["available_at"].gt(fast_row.available_at)
                & slow_group["available_at"].le(
                    fast_row.available_at + pd.Timedelta(days=max_confirmation_gap_days)
                )
            ]
            if candidates.empty:
                continue
            slow_row = candidates.iloc[0]
            available_slow.remove(slow_row.name)
            lag = (slow_row["available_at"] - fast_row.available_at).total_seconds() / 86400
            matched += 1
            lags.append(lag)
            if "rate" in fast_group.columns and "rate" in slow_group.columns:
                rate_fast, rate_slow = float(fast_row.rate), float(slow_row["rate"])
                waiting_costs.append(10_000 * (rate_slow - rate_fast) / rate_fast)
            pair_rows.append({
                **values,
                "fast_available_at": fast_row.available_at,
                "slow_available_at": slow_row["available_at"],
                "confirmation_lag_days": lag,
                "fast_correct": bool(fast_row.is_correct),
                "slow_correct": bool(slow_row["is_correct"]),
                "waiting_cost_bps": waiting_costs[-1] if waiting_costs else np.nan,
            })
        summary_rows.append({
            **values,
            "fast_horizons": ",".join(map(str, fast_horizons)),
            "slow_horizons": ",".join(map(str, slow_horizons)),
            "fast_signal_count": len(fast_group),
            "fast_precision": float(fast_group["is_correct"].mean()) if len(fast_group) else 0.0,
            "fast_uplift": (
                float(fast_group["is_correct"].mean()) / float(np.nanmean(fast_baselines))
                if fast_baselines and np.nanmean(fast_baselines) > 0 else np.nan
            ),
            "fast_mean_benefit_bps": float(fast_benefit.mean()) if len(fast_benefit) else np.nan,
            "fast_positive_benefit_rate": float((fast_benefit > 0).mean()) if len(fast_benefit) else 0.0,
            "slow_signal_count": len(slow_group),
            "slow_precision": float(slow_group["is_correct"].mean()) if len(slow_group) else 0.0,
            "slow_uplift": (
                float(slow_group["is_correct"].mean()) / float(np.nanmean(slow_baselines))
                if slow_baselines and np.nanmean(slow_baselines) > 0 else np.nan
            ),
            "slow_mean_benefit_bps": float(slow_benefit.mean()) if len(slow_benefit) else np.nan,
            "slow_positive_benefit_rate": float((slow_benefit > 0).mean()) if len(slow_benefit) else 0.0,
            "matched_signal_count": matched,
            "confirmation_rate": matched / len(fast_group) if len(fast_group) else 0.0,
            "median_confirmation_lag_days": float(np.median(lags)) if lags else np.nan,
            "mean_waiting_cost_bps": float(np.mean(waiting_costs)) if waiting_costs else np.nan,
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)
