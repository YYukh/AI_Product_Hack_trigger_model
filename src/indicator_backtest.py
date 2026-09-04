"""Walk-forward backtest выбранных rule-архитектур."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicator_optimization import fit_indicator_space
from .indicators import IndicatorCandidate, prediction_matrix


def _period_metrics(
    sample: pd.DataFrame,
    *,
    target: str,
    horizon: int,
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
    precision = true_positive / signal_count if signal_count else 0.0
    random_precision = positive_count / observations if observations else 0.0
    calendar_weeks = (
        evaluation_end_exclusive - evaluation_start
    ).total_seconds() / pd.Timedelta(days=7).total_seconds()
    signals = sample.loc[
        prediction, ["available_at", "currency", target]
    ].rename(columns={target: "target_value"})
    benefit_column = f"local_advantage_{horizon}d_bps"
    benefit = (
        pd.to_numeric(sample.loc[prediction, benefit_column], errors="coerce")
        if benefit_column in sample.columns else pd.Series(dtype=float)
    )
    benefit = benefit.dropna()
    return {
        "observations": observations,
        "positive_count": positive_count,
        "signal_count": signal_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "signal_precision": precision,
        "random_precision": random_precision,
        "lift": precision / random_precision if random_precision else 0.0,
        "calendar_weeks": calendar_weeks,
        "signals_per_week": signal_count / calendar_weeks if calendar_weeks else 0.0,
        "benefit_bps_sum": float(benefit.sum()),
        "benefit_bps_count": int(benefit.size),
        "positive_benefit_count": int((benefit > 0).sum()),
        "mean_benefit_bps": float(benefit.mean()) if len(benefit) else np.nan,
        "median_benefit_bps": float(benefit.median()) if len(benefit) else np.nan,
        "positive_benefit_rate": float((benefit > 0).mean()) if len(benefit) else 0.0,
    }, signals


def backtest_selected_indicators(
    data: pd.DataFrame,
    *,
    selected_indicators: pd.DataFrame,
    indicator_spaces: dict[str, list[IndicatorCandidate]],
    backtest_start: str | pd.Timestamp,
    train_months: int = 24,
    min_signals_per_week: float = 2.0,
    target_families: tuple[str, ...] = ("G0", "W1"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest winners while reoptimizing thresholds before every test fold.

    Indicator architecture and test cadence come from discovery. Concrete
    candidate thresholds are selected again on the rolling train window and
    then frozen only for the immediately following OOS period.
    """
    if train_months <= 0:
        raise ValueError("train_months должен быть положительным")
    if min_signals_per_week < 0:
        raise ValueError("min_signals_per_week не может быть отрицательным")
    selected_required = {
        "currency", "scenario", "target_family", "target", "horizon",
        "indicator", "test_months",
    }
    if missing := selected_required.difference(selected_indicators.columns):
        raise KeyError(f"В selected_indicators нет полей: {sorted(missing)}")
    if missing := {"available_at", "currency"}.difference(data.columns):
        raise KeyError(f"В data нет полей: {sorted(missing)}")

    start = pd.Timestamp(backtest_start)
    selected = selected_indicators.loc[
        selected_indicators["target_family"].isin(target_families)
    ].copy()
    if selected.empty:
        raise ValueError("Нет выбранных индикаторов заданных target families")

    dates = pd.to_datetime(data["available_at"])
    summary_rows: list[dict] = []
    fold_rows: list[dict] = []
    signal_frames: list[pd.DataFrame] = []

    for row in selected.itertuples(index=False):
        horizon = int(row.horizon)
        if row.indicator not in indicator_spaces:
            raise KeyError(f"Нет пространства indicator={row.indicator!r}")
        if row.target not in data.columns:
            raise KeyError(f"В data нет target {row.target!r}")
        candidates = indicator_spaces[row.indicator]
        required_features = sorted({
            rule.feature for candidate in candidates for rule in candidate.rules
        })
        if missing := set(required_features).difference(data.columns):
            raise KeyError(f"Не хватает indicator features: {sorted(missing)}")

        currency_mask = data["currency"].eq(row.currency)
        configuration_data = data.loc[
            currency_mask & data[row.target].notna()
        ].copy()
        configuration_data["available_at"] = pd.to_datetime(
            configuration_data["available_at"]
        )
        configuration_data = configuration_data.dropna(
            subset=required_features
        ).sort_values("available_at")
        if configuration_data.empty:
            continue

        rebalance_months = int(row.test_months)
        period_start = start
        fold_id = 1
        configuration_folds: list[dict] = []
        while period_start <= configuration_data["available_at"].max():
            period_end = period_start + pd.DateOffset(months=rebalance_months)
            rolling_start = period_start - pd.DateOffset(months=train_months)
            train = configuration_data.loc[
                configuration_data["available_at"].ge(rolling_start)
                & configuration_data["available_at"].add(
                    pd.Timedelta(days=horizon)
                ).lt(period_start)
            ]
            test = configuration_data.loc[
                configuration_data["available_at"].ge(period_start)
                & configuration_data["available_at"].lt(period_end)
            ]
            if test.empty:
                period_start = period_end
                continue
            if train.empty:
                raise ValueError(
                    f"Пустой rolling train: {row.currency}, {row.target}, "
                    f"start={period_start.date()}"
                )

            train_predictions = prediction_matrix(train, candidates)
            selected_index, train_stats, train_metrics = fit_indicator_space(
                train[row.target],
                train_predictions,
                candidates,
                dates=train["available_at"],
                min_signals_per_week=min_signals_per_week,
            )
            if selected_index is None:
                raise ValueError(
                    f"Ни один candidate не прошёл frequency constraint: "
                    f"{row.currency}, {row.target}, start={period_start.date()}"
                )
            candidate = candidates[selected_index]
            evaluation_end = min(
                period_end,
                test["available_at"].max() + pd.Timedelta(days=1),
            )
            metrics, signals = _period_metrics(
                test,
                target=row.target,
                horizon=horizon,
                candidate=candidate,
                evaluation_start=period_start,
                evaluation_end_exclusive=evaluation_end,
            )
            metadata = {
                "currency": row.currency,
                "scenario": row.scenario,
                "target_family": row.target_family,
                "target": row.target,
                "horizon": horizon,
                "indicator": row.indicator,
                "fixed_candidate": candidate.name,
                "fixed_logic": candidate.logic,
                "train_months": train_months,
                "rebalance_months": rebalance_months,
                "fold_id": fold_id,
            }
            fold_row = {
                **metadata,
                "train_start": train["available_at"].min(),
                "train_end": train["available_at"].max(),
                "test_start": test["available_at"].min(),
                "test_end": test["available_at"].max(),
                "train_signal_count": int(
                    train_stats["predicted_positive_count"][selected_index]
                ),
                "train_precision": float(train_stats["precision"][selected_index]),
                "train_random_precision": float(train_metrics["random_precision"]),
                "train_lift": float(train_metrics["uplift"][selected_index]),
                **metrics,
            }
            fold_rows.append(fold_row)
            configuration_folds.append(fold_row)
            if not signals.empty:
                signal_frames.append(signals.assign(**metadata))
            fold_id += 1
            period_start = period_end

        if not configuration_folds:
            continue
        totals = {
            name: sum(fold[name] for fold in configuration_folds)
            for name in (
                "observations", "positive_count", "signal_count",
                "true_positive", "false_positive", "calendar_weeks",
                "benefit_bps_sum", "benefit_bps_count", "positive_benefit_count",
            )
        }
        signal_count = int(totals["signal_count"])
        observations = int(totals["observations"])
        true_positive = int(totals["true_positive"])
        precision = true_positive / signal_count if signal_count else 0.0
        random_precision = (
            int(totals["positive_count"]) / observations if observations else 0.0
        )
        summary_rows.append({
            "currency": row.currency,
            "scenario": row.scenario,
            "target_family": row.target_family,
            "target": row.target,
            "horizon": horizon,
            "indicator": row.indicator,
            "train_months": train_months,
            "rebalance_months": rebalance_months,
            "folds": len(configuration_folds),
            "distinct_candidates": len({
                fold["fixed_candidate"] for fold in configuration_folds
            }),
            "test_observations": observations,
            "test_positive_count": int(totals["positive_count"]),
            "test_signal_count": signal_count,
            "test_true_positive": true_positive,
            "test_false_positive": int(totals["false_positive"]),
            "test_signal_precision": precision,
            "test_random_precision": random_precision,
            "test_lift": precision / random_precision if random_precision else 0.0,
            "test_calendar_weeks": float(totals["calendar_weeks"]),
            "test_signals_per_week": (
                signal_count / float(totals["calendar_weeks"])
                if totals["calendar_weeks"] else 0.0
            ),
            "test_mean_benefit_bps": (
                totals["benefit_bps_sum"] / totals["benefit_bps_count"]
                if totals["benefit_bps_count"] else np.nan
            ),
            "test_positive_benefit_rate": (
                totals["positive_benefit_count"] / totals["benefit_bps_count"]
                if totals["benefit_bps_count"] else 0.0
            ),
        })

    signal_columns = [
        "available_at", "currency", "target_value", "scenario",
        "target_family", "target", "horizon", "indicator",
        "fixed_candidate", "fixed_logic", "train_months",
        "rebalance_months", "fold_id",
    ]
    summary = pd.DataFrame(summary_rows)
    folds = pd.DataFrame(fold_rows)
    return (
        summary.sort_values(
            ["target_family", "currency", "horizon", "target"]
        ).reset_index(drop=True),
        folds.sort_values(
            ["target_family", "currency", "horizon", "target", "fold_id"]
        ).reset_index(drop=True),
        (
            pd.concat(signal_frames, ignore_index=True)[signal_columns]
            if signal_frames else pd.DataFrame(columns=signal_columns)
        ),
    )
