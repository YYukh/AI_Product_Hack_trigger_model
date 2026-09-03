"""Независимый backtest замороженных rule-based индикаторов."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import IndicatorCandidate


def _period_metrics(
    sample: pd.DataFrame,
    *,
    target: str,
    candidate: IndicatorCandidate,
    evaluation_start: pd.Timestamp,
    evaluation_end_exclusive: pd.Timestamp,
) -> tuple[dict, pd.DataFrame]:
    prediction = candidate.predict(sample)
    target_values = sample[target].to_numpy(dtype=bool)
    observations = len(sample)
    positive_count = int(target_values.sum())
    signal_count = int(prediction.sum())
    true_positive = int(np.sum(prediction & target_values))
    false_positive = signal_count - true_positive
    signal_precision = true_positive / signal_count if signal_count else 0.0
    random_precision = positive_count / observations if observations else 0.0
    lift = signal_precision / random_precision if random_precision else 0.0
    calendar_weeks = (
        evaluation_end_exclusive - evaluation_start
    ).total_seconds() / pd.Timedelta(days=7).total_seconds()

    signals = sample.loc[
        prediction,
        ["available_at", "currency", target],
    ].rename(columns={target: "target_value"})
    return (
        {
            "observations": observations,
            "positive_count": positive_count,
            "signal_count": signal_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "signal_precision": signal_precision,
            "random_precision": random_precision,
            "lift": lift,
            "calendar_weeks": calendar_weeks,
            "signals_per_week": signal_count / calendar_weeks,
        },
        signals,
    )


def backtest_fixed_indicators(
    data: pd.DataFrame,
    *,
    selected_indicators: pd.DataFrame,
    fitted_indicators: dict[tuple[str, str, int], IndicatorCandidate],
    backtest_start: str | pd.Timestamp,
    target_families: tuple[str, ...] = ("G0", "W1"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Применить frozen candidates к untouched временной выборке.

    ``test_months`` победителя используется как частота контрольных срезов.
    Параметры candidate внутри backtest не переоптимизируются.
    """
    selected_required = {
        "currency",
        "scenario",
        "target_family",
        "target",
        "horizon",
        "indicator",
        "test_months",
        "fitted_candidate",
        "fitted_logic",
    }
    missing = selected_required.difference(selected_indicators.columns)
    if missing:
        raise KeyError(f"В selected_indicators нет полей: {sorted(missing)}")
    data_required = {"available_at", "currency"}
    missing = data_required.difference(data.columns)
    if missing:
        raise KeyError(f"В data нет полей: {sorted(missing)}")

    start = pd.Timestamp(backtest_start)
    selected = selected_indicators.loc[
        selected_indicators["target_family"].isin(target_families)
    ].copy()
    if selected.empty:
        raise ValueError("Нет выбранных индикаторов заданных target families")

    summary_rows: list[dict] = []
    fold_rows: list[dict] = []
    signal_frames: list[pd.DataFrame] = []

    for row in selected.itertuples(index=False):
        key = (row.currency, row.target, int(row.horizon))
        if key not in fitted_indicators:
            raise KeyError(f"Нет fitted candidate для {key}")
        candidate = fitted_indicators[key]
        if candidate.name != row.fitted_candidate:
            raise ValueError(f"Candidate metadata не совпадает для {key}")
        if row.target not in data.columns:
            raise KeyError(f"В data нет target {row.target!r}")

        sample = data.loc[
            data["currency"].eq(row.currency)
            & pd.to_datetime(data["available_at"]).ge(start)
            & data[row.target].notna()
        ].sort_values("available_at")
        if sample.empty:
            continue

        rebalance_months = int(row.test_months)
        period_start = start
        fold_id = 1
        totals = {
            "observations": 0,
            "positive_count": 0,
            "signal_count": 0,
            "true_positive": 0,
            "false_positive": 0,
            "calendar_weeks": 0.0,
        }

        while period_start <= sample["available_at"].max():
            period_end_exclusive = period_start + pd.DateOffset(
                months=rebalance_months
            )
            fold_sample = sample.loc[
                sample["available_at"].ge(period_start)
                & sample["available_at"].lt(period_end_exclusive)
            ]
            if not fold_sample.empty:
                evaluation_end_exclusive = min(
                    period_end_exclusive,
                    pd.Timestamp(sample["available_at"].max())
                    + pd.Timedelta(days=1),
                )
                metrics, signals = _period_metrics(
                    fold_sample,
                    target=row.target,
                    candidate=candidate,
                    evaluation_start=period_start,
                    evaluation_end_exclusive=evaluation_end_exclusive,
                )
                metadata = {
                    "currency": row.currency,
                    "scenario": row.scenario,
                    "target_family": row.target_family,
                    "target": row.target,
                    "horizon": int(row.horizon),
                    "indicator": row.indicator,
                    "fixed_candidate": candidate.name,
                    "fixed_logic": candidate.logic,
                    "rebalance_months": rebalance_months,
                }
                fold_rows.append(
                    {
                        **metadata,
                        "fold_id": fold_id,
                        "test_start": fold_sample["available_at"].min(),
                        "test_end": fold_sample["available_at"].max(),
                        **metrics,
                    }
                )
                if not signals.empty:
                    signals = signals.assign(
                        **metadata,
                        fold_id=fold_id,
                    )
                    signal_frames.append(signals)
                for name in totals:
                    totals[name] += metrics[name]
                fold_id += 1
            period_start = period_end_exclusive

        signal_count = int(totals["signal_count"])
        observations = int(totals["observations"])
        positive_count = int(totals["positive_count"])
        true_positive = int(totals["true_positive"])
        signal_precision = (
            true_positive / signal_count if signal_count else 0.0
        )
        random_precision = (
            positive_count / observations if observations else 0.0
        )
        summary_rows.append(
            {
                "currency": row.currency,
                "scenario": row.scenario,
                "target_family": row.target_family,
                "target": row.target,
                "horizon": int(row.horizon),
                "indicator": row.indicator,
                "fixed_candidate": candidate.name,
                "fixed_logic": candidate.logic,
                "rebalance_months": rebalance_months,
                "folds": fold_id - 1,
                "test_observations": observations,
                "test_positive_count": positive_count,
                "test_signal_count": signal_count,
                "test_true_positive": true_positive,
                "test_false_positive": int(totals["false_positive"]),
                "test_signal_precision": signal_precision,
                "test_random_precision": random_precision,
                "test_lift": (
                    signal_precision / random_precision
                    if random_precision
                    else 0.0
                ),
                "test_calendar_weeks": totals["calendar_weeks"],
                "test_signals_per_week": (
                    signal_count / totals["calendar_weeks"]
                    if totals["calendar_weeks"]
                    else 0.0
                ),
            }
        )

    signal_columns = [
        "available_at",
        "currency",
        "target_value",
        "scenario",
        "target_family",
        "target",
        "horizon",
        "indicator",
        "fixed_candidate",
        "fixed_logic",
        "rebalance_months",
        "fold_id",
    ]
    return (
        pd.DataFrame(summary_rows).sort_values(
            ["target_family", "currency", "horizon", "target"]
        ).reset_index(drop=True),
        pd.DataFrame(fold_rows).sort_values(
            ["target_family", "currency", "horizon", "target", "fold_id"]
        ).reset_index(drop=True),
        (
            pd.concat(signal_frames, ignore_index=True)[signal_columns]
            if signal_frames
            else pd.DataFrame(columns=signal_columns)
        ),
    )

