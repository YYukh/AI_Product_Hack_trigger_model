"""Векторная walk-forward оптимизация библиотеки индикаторов."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import IndicatorCandidate, prediction_matrix
from .walk_forward import make_periodic_walk_forward_folds


def _precision_counts(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict[str, np.ndarray | int]:
    """Одновременно посчитать precision-компоненты всех candidates."""
    y = y_true.to_numpy(dtype=bool)
    matrix = np.asarray(predictions, dtype=bool)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if len(y) != len(matrix):
        raise ValueError("Размер target не совпадает с predictions")
    if not len(y):
        raise ValueError("Нельзя оценить пустую выборку")

    predicted_positive = matrix.sum(axis=0, dtype=np.int64)
    true_positive = np.sum(matrix & y[:, None], axis=0, dtype=np.int64)
    false_positive = predicted_positive - true_positive
    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros(matrix.shape[1], dtype=float),
        where=predicted_positive != 0,
    )
    return {
        "observations": len(y),
        "positive_count": int(y.sum()),
        "predicted_positive_count": predicted_positive,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "precision": precision,
    }


def _flatten_spaces(
    indicator_spaces: dict[str, list[IndicatorCandidate]],
) -> tuple[list[IndicatorCandidate], dict[str, np.ndarray]]:
    if not indicator_spaces:
        raise ValueError("indicator_spaces не должен быть пустым")

    all_candidates: list[IndicatorCandidate] = []
    indices_by_space: dict[str, np.ndarray] = {}
    for space_name, candidates in indicator_spaces.items():
        if not candidates:
            raise ValueError(f"Пустое пространство индикатора: {space_name}")
        start = len(all_candidates)
        all_candidates.extend(candidates)
        stop = len(all_candidates)
        # Лексикографический порядок задаёт только детерминированный tie-break.
        indices_by_space[space_name] = np.array(
            sorted(
                range(start, stop),
                key=lambda index: all_candidates[index].name,
            ),
            dtype=int,
        )

    names = [candidate.name for candidate in all_candidates]
    if len(names) != len(set(names)):
        raise ValueError("Один candidate попал в несколько indicator spaces")
    return all_candidates, indices_by_space


def _best_candidate_index(
    precision: np.ndarray,
    candidate_indices: np.ndarray,
) -> int:
    """Максимум precision; при равенстве — первое имя в sorted indices."""
    local_precision = precision[candidate_indices]
    return int(candidate_indices[int(np.argmax(local_precision))])


def _calendar_weeks(dates: pd.Series) -> float:
    """Календарная длительность набора дат в неделях."""
    timestamps = pd.to_datetime(dates)
    if timestamps.empty:
        raise ValueError("Нельзя посчитать длительность пустого периода")
    calendar_days = (timestamps.max() - timestamps.min()).days + 1
    return calendar_days / 7


def _backtest_metrics(
    stats: dict[str, np.ndarray | int],
    *,
    calendar_weeks: float,
) -> dict[str, np.ndarray | float]:
    """Precision, random baseline, lift и частота candidates."""
    observations = int(stats["observations"])
    positive_count = int(stats["positive_count"])
    random_precision = positive_count / observations
    precision = np.asarray(stats["precision"], dtype=float)
    uplift = np.divide(
        precision,
        random_precision,
        out=np.zeros_like(precision),
        where=random_precision > 0,
    )
    signals_per_week = (
        np.asarray(stats["predicted_positive_count"], dtype=float)
        / calendar_weeks
    )
    return {
        "random_precision": random_precision,
        "uplift": uplift,
        "signals_per_week": signals_per_week,
    }


def _best_backtest_candidate_index(
    stats: dict[str, np.ndarray | int],
    candidate_indices: np.ndarray,
    *,
    calendar_weeks: float,
    min_signals_per_week: float,
) -> int | None:
    """Максимальный lift среди candidates с достаточной частотой."""
    metrics = _backtest_metrics(stats, calendar_weeks=calendar_weeks)
    local_frequency = metrics["signals_per_week"][candidate_indices]
    eligible = local_frequency >= min_signals_per_week
    if not eligible.any():
        return None
    eligible_indices = candidate_indices[eligible]
    local_uplift = metrics["uplift"][eligible_indices]
    return int(eligible_indices[int(np.argmax(local_uplift))])


def fit_indicator_space(
    y_true: pd.Series,
    predictions: np.ndarray,
    candidates: list[IndicatorCandidate],
    *,
    dates: pd.Series,
    min_signals_per_week: float,
) -> tuple[int | None, dict, dict]:
    """Fit one fixed indicator architecture using the research objective.

    This is the public production-facing equivalent of the train operation
    performed inside every research WF fold.
    """
    if not candidates:
        raise ValueError("Пустое пространство candidates")
    stats = _precision_counts(y_true, predictions)
    calendar_weeks = _calendar_weeks(dates)
    metrics = _backtest_metrics(stats, calendar_weeks=calendar_weeks)
    indices = np.array(sorted(
        range(len(candidates)), key=lambda index: candidates[index].name
    ))
    selected = _best_backtest_candidate_index(
        stats,
        indices,
        calendar_weeks=calendar_weeks,
        min_signals_per_week=min_signals_per_week,
    )
    return selected, stats, {**metrics, "calendar_weeks": calendar_weeks}


def optimize_indicator_library(
    data: pd.DataFrame,
    *,
    target_registry: pd.DataFrame,
    indicator_spaces: dict[str, list[IndicatorCandidate]],
    first_test_date: str | pd.Timestamp,
    test_months_options: tuple[int, ...] | list[int],
    train_months: int | None = None,
    refit_date: str | pd.Timestamp | None = None,
    min_signals_per_week: float = 2.0,
    ensemble_lift_thresholds: tuple[float, ...] | list[float] = (1.30,),
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple, IndicatorCandidate],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Оптимизировать всю библиотеку за один проход по данным одной валюты.

    Простые rules вычисляются один раз. AND/OR строятся из готовых булевых
    массивов. Для horizon × test_months folds создаются один раз и
    переиспользуются всеми targets этого горизонта и всеми индикаторами.

    На train каждого fold выбирается максимальный uplift среди параметров,
    дающих не меньше ``min_signals_per_week``. Для каждого target × horizon
    × indicator × test_months возвращается один pooled OOS lift по всем
    test-folds вместе. Подробные folds и fitted objects возвращаются только
    для победителя target × horizon.
    Ensemble объединяет по уникальным датам сигналы конфигураций,
    прошедших заданный порог индивидуальной OOS precision.
    """
    registry_required = {"name", "horizon"}
    registry_missing = registry_required.difference(target_registry.columns)
    if registry_missing:
        raise KeyError(
            f"В target_registry нет полей: {sorted(registry_missing)}"
        )
    test_months_values = tuple(dict.fromkeys(int(x) for x in test_months_options))
    if not test_months_values or any(x <= 0 for x in test_months_values):
        raise ValueError("test_months_options должны быть положительными")
    if min_signals_per_week < 0:
        raise ValueError("min_signals_per_week не может быть отрицательным")
    if train_months is not None and train_months <= 0:
        raise ValueError("train_months должен быть положительным")
    ensemble_thresholds = tuple(
        dict.fromkeys(float(x) for x in ensemble_lift_thresholds)
    )
    if not ensemble_thresholds or any(x < 0 for x in ensemble_thresholds):
        raise ValueError("ensemble lift thresholds не могут быть отрицательными")

    candidates, indices_by_space = _flatten_spaces(indicator_spaces)
    features = sorted(
        {rule.feature for candidate in candidates for rule in candidate.rules}
    )
    target_names = target_registry["name"].tolist()
    required = {"available_at", *features, *target_names}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Не хватает полей для оптимизации: {sorted(missing)}")

    prepared = (
        data.dropna(subset=features)
        .sort_values("available_at")
        .reset_index(drop=True)
    )
    # Главный кэш: все primitive/SINGLE/AND/OR predictions одной валюты.
    predictions = prediction_matrix(prepared, candidates)
    prepared_dates = pd.to_datetime(prepared["available_at"])

    summary_rows: list[dict] = []
    fold_rows: list[dict] = []
    fitted_candidates: dict[tuple, IndicatorCandidate] = {}
    oos_signal_masks: dict[tuple, np.ndarray] = {}
    oos_evaluation_masks: dict[tuple, np.ndarray] = {}

    for horizon in sorted(target_registry["horizon"].unique()):
        definitions = target_registry.loc[
            target_registry["horizon"].eq(horizon)
        ]

        # Финальная настройка после OOS-оценки не зависит от test_months.
        final_by_target: dict[str, tuple[dict, dict, dict[str, int | None]]] = {}
        for definition in definitions.itertuples(index=False):
            target = definition.name
            # pandas может вернуть read-only view; дальше маска изменяется
            # условиями maturity и rolling train window.
            labelled = prepared[target].notna().to_numpy(dtype=bool, copy=True)
            if refit_date is not None:
                refit_at = pd.Timestamp(refit_date)
                labelled &= prepared_dates.add(
                    pd.Timedelta(days=int(horizon))
                ).lt(refit_at).to_numpy()
                if train_months is not None:
                    labelled &= prepared_dates.ge(
                        refit_at - pd.DateOffset(months=train_months)
                    ).to_numpy()
            if not labelled.any():
                raise ValueError(f"Нет доступных значений target: {target}")
            final_stats = _precision_counts(
                prepared.loc[labelled, target],
                predictions[labelled],
            )
            final_weeks = _calendar_weeks(
                prepared.loc[labelled, "available_at"]
            )
            final_metrics = _backtest_metrics(
                final_stats,
                calendar_weeks=final_weeks,
            )
            final_indices = {
                space_name: _best_backtest_candidate_index(
                    final_stats,
                    candidate_indices,
                    calendar_weeks=final_weeks,
                    min_signals_per_week=min_signals_per_week,
                )
                for space_name, candidate_indices in indices_by_space.items()
            }
            final_by_target[target] = (
                final_stats,
                final_metrics,
                final_indices,
            )

        for test_months in test_months_values:
            folds = make_periodic_walk_forward_folds(
                prepared,
                horizon=int(horizon),
                first_test_date=first_test_date,
                test_months=test_months,
                train_months=train_months,
            )

            for definition in definitions.itertuples(index=False):
                target = definition.name
                evaluation_mask = np.zeros(len(prepared), dtype=bool)
                signal_masks = {
                    space_name: np.zeros(len(prepared), dtype=bool)
                    for space_name in indicator_spaces
                }
                aggregate = {
                    space_name: {
                        "folds": 0,
                        "frequency_eligible_folds": 0,
                        "observations": 0,
                        "positive_count": 0,
                        "true_positive": 0,
                        "false_positive": 0,
                        "calendar_weeks": 0.0,
                    }
                    for space_name in indicator_spaces
                }

                for fold in folds:
                    train = prepared.loc[fold.train_index]
                    test = prepared.loc[fold.test_index]
                    train_valid = train[target].notna().to_numpy()
                    test_valid = test[target].notna().to_numpy()
                    if not train_valid.any() or not test_valid.any():
                        continue

                    train_positions = fold.train_index[train_valid]
                    test_positions = fold.test_index[test_valid]
                    evaluation_mask[test_positions] = True
                    train_stats = _precision_counts(
                        prepared.loc[train_positions, target],
                        predictions[train_positions],
                    )
                    test_stats = _precision_counts(
                        prepared.loc[test_positions, target],
                        predictions[test_positions],
                    )
                    train_weeks = _calendar_weeks(
                        prepared.loc[train_positions, "available_at"]
                    )
                    test_weeks = _calendar_weeks(
                        prepared.loc[test_positions, "available_at"]
                    )
                    train_metrics = _backtest_metrics(
                        train_stats,
                        calendar_weeks=train_weeks,
                    )
                    test_metrics = _backtest_metrics(
                        test_stats,
                        calendar_weeks=test_weeks,
                    )

                    for space_name, candidate_indices in indices_by_space.items():
                        selected_index = _best_backtest_candidate_index(
                            train_stats,
                            candidate_indices,
                            calendar_weeks=train_weeks,
                            min_signals_per_week=min_signals_per_week,
                        )
                        observations = int(test_stats["observations"])
                        positive_count = int(test_stats["positive_count"])
                        totals = aggregate[space_name]
                        totals["folds"] += 1
                        totals["observations"] += observations
                        totals["positive_count"] += positive_count
                        totals["calendar_weeks"] += test_weeks

                        if selected_index is None:
                            fold_row = {
                                "fold_id": fold.fold_id,
                                "target": target,
                                "horizon": int(horizon),
                                "indicator": space_name,
                                "test_months": test_months,
                                "train_months": train_months,
                                "train_start": fold.train_start,
                                "train_end": fold.train_end,
                                "test_start": fold.test_start,
                                "test_end": fold.test_end,
                                "selected_candidate": None,
                                "selected_logic": None,
                                "selected_rule_count": 0,
                                "train_frequency_constraint_met": False,
                                "train_random_precision": float(
                                    train_metrics["random_precision"]
                                ),
                                "train_precision": np.nan,
                                "train_lift": np.nan,
                                "train_signals_per_week": 0.0,
                                "test_observations": observations,
                                "test_true_positive": 0,
                                "test_false_positive": 0,
                                "test_predicted_positive_count": 0,
                                "test_random_precision": float(
                                    test_metrics["random_precision"]
                                ),
                                "test_precision": 0.0,
                                "test_lift": 0.0,
                                "test_signals_per_week": 0.0,
                            }
                            if hasattr(definition, "scenario"):
                                fold_row["scenario"] = definition.scenario
                            if hasattr(definition, "family"):
                                fold_row["target_family"] = definition.family
                            fold_rows.append(fold_row)
                            continue

                        selected = candidates[selected_index]
                        tp = int(test_stats["true_positive"][selected_index])
                        fp = int(test_stats["false_positive"][selected_index])
                        predicted_positive = tp + fp
                        test_precision = (
                            tp / predicted_positive if predicted_positive else 0.0
                        )
                        test_lift = float(test_metrics["uplift"][selected_index])
                        test_signals_per_week = float(
                            test_metrics["signals_per_week"][selected_index]
                        )
                        signal_masks[space_name][test_positions] |= predictions[
                            test_positions,
                            selected_index,
                        ]

                        totals["frequency_eligible_folds"] += 1
                        totals["true_positive"] += tp
                        totals["false_positive"] += fp

                        fold_row = {
                                "fold_id": fold.fold_id,
                                "target": target,
                                "horizon": int(horizon),
                                "indicator": space_name,
                                "test_months": test_months,
                                "train_months": train_months,
                                "train_start": fold.train_start,
                                "train_end": fold.train_end,
                                "test_start": fold.test_start,
                                "test_end": fold.test_end,
                                "selected_candidate": selected.name,
                                "selected_logic": selected.logic,
                                "selected_rule_count": len(selected.rules),
                                "train_frequency_constraint_met": True,
                                "train_random_precision": float(
                                    train_metrics["random_precision"]
                                ),
                                "train_precision": float(
                                    train_stats["precision"][selected_index]
                                ),
                                "train_lift": float(
                                    train_metrics["uplift"][selected_index]
                                ),
                                "train_signals_per_week": float(
                                    train_metrics["signals_per_week"][selected_index]
                                ),
                                "test_observations": observations,
                                "test_true_positive": tp,
                                "test_false_positive": fp,
                                "test_predicted_positive_count": predicted_positive,
                                "test_random_precision": float(
                                    test_metrics["random_precision"]
                                ),
                                "test_precision": test_precision,
                                "test_lift": test_lift,
                                "test_signals_per_week": test_signals_per_week,
                            }
                        if hasattr(definition, "scenario"):
                            fold_row["scenario"] = definition.scenario
                        if hasattr(definition, "family"):
                            fold_row["target_family"] = definition.family
                        fold_rows.append(fold_row)

                oos_evaluation_masks[
                    (target, int(horizon), test_months)
                ] = evaluation_mask
                final_stats, final_metrics, final_indices = final_by_target[target]
                for space_name, totals in aggregate.items():
                    final_index = final_indices[space_name]
                    predicted_positive = (
                        totals["true_positive"] + totals["false_positive"]
                    )
                    pooled_precision = (
                        totals["true_positive"] / predicted_positive
                        if predicted_positive
                        else 0.0
                    )
                    random_precision = (
                        totals["positive_count"] / totals["observations"]
                        if totals["observations"]
                        else 0.0
                    )
                    pooled_lift = (
                        pooled_precision / random_precision
                        if random_precision
                        else 0.0
                    )
                    oos_signals_per_week = (
                        predicted_positive / totals["calendar_weeks"]
                        if totals["calendar_weeks"]
                        else 0.0
                    )
                    fitted = (
                        candidates[final_index]
                        if final_index is not None
                        else None
                    )
                    row = {
                        "indicator": space_name,
                        "target": target,
                        "horizon": int(horizon),
                        "test_months": test_months,
                        "train_months": train_months,
                        "folds": totals["folds"],
                        "frequency_eligible_folds": totals[
                            "frequency_eligible_folds"
                        ],
                        "test_observations": totals["observations"],
                        "test_positive_count": totals["positive_count"],
                        "test_true_positive": totals["true_positive"],
                        "test_false_positive": totals["false_positive"],
                        "test_predicted_positive_count": predicted_positive,
                        "oos_precision": pooled_precision,
                        "oos_random_precision": random_precision,
                        "oos_lift": pooled_lift,
                        "oos_signals_per_week": oos_signals_per_week,
                        "oos_frequency_constraint_met": (
                            oos_signals_per_week >= min_signals_per_week
                        ),
                        "min_signals_per_week": min_signals_per_week,
                        "fitted_candidate": fitted.name if fitted else None,
                        "fitted_logic": fitted.logic if fitted else None,
                        "fitted_rule_count": len(fitted.rules) if fitted else 0,
                        "fitted_frequency_constraint_met": fitted is not None,
                        "fitted_train_precision": (
                            float(final_stats["precision"][final_index])
                            if final_index is not None
                            else np.nan
                        ),
                        "fitted_train_lift": (
                            float(final_metrics["uplift"][final_index])
                            if final_index is not None
                            else np.nan
                        ),
                        "fitted_train_signals_per_week": (
                            float(final_metrics["signals_per_week"][final_index])
                            if final_index is not None
                            else 0.0
                        ),
                    }
                    if hasattr(definition, "scenario"):
                        row["scenario"] = definition.scenario
                    if hasattr(definition, "family"):
                        row["target_family"] = definition.family
                    summary_rows.append(row)
                    if fitted is not None:
                        fitted_candidates[
                            (target, int(horizon), space_name, test_months)
                        ] = fitted
                    oos_signal_masks[
                        (target, int(horizon), space_name, test_months)
                    ] = signal_masks[space_name]

    if not summary_rows:
        raise ValueError("Оптимизация не сформировала результатов")

    summaries = pd.DataFrame(summary_rows)
    eligible_summaries = summaries.loc[
        summaries["fitted_frequency_constraint_met"]
        & summaries["oos_frequency_constraint_met"]
        & summaries["frequency_eligible_folds"].eq(summaries["folds"])
    ]
    winner_keys = (
        eligible_summaries.sort_values(
            ["target", "horizon", "oos_lift", "indicator", "test_months"],
            ascending=[True, True, False, True, True],
        )
        .groupby(["target", "horizon"], as_index=False)
        .head(1)
        [["target", "horizon", "indicator", "test_months"]]
    )
    winner_folds = (
        pd.DataFrame(fold_rows)
        .merge(
            winner_keys,
            on=["target", "horizon", "indicator", "test_months"],
            how="inner",
            validate="many_to_one",
        )
        .reset_index(drop=True)
    )
    winner_candidates = {
        (
            row.target,
            int(row.horizon),
            row.indicator,
            int(row.test_months),
        ): fitted_candidates[
            (
                row.target,
                int(row.horizon),
                row.indicator,
                int(row.test_months),
            )
        ]
        for row in winner_keys.itertuples(index=False)
    }

    ensemble_rows = []
    ensemble_signal_rows = []
    for (target, horizon), target_results in summaries.groupby(
        ["target", "horizon"],
        sort=True,
    ):
        target_values = prepared[target].fillna(0).to_numpy(dtype=bool)
        metadata = target_results.iloc[0]

        for threshold in ensemble_thresholds:
            selected = target_results.loc[
                target_results["oos_lift"].ge(threshold)
                & target_results["oos_frequency_constraint_met"]
                & target_results["fitted_frequency_constraint_met"]
                & target_results["frequency_eligible_folds"].eq(
                    target_results["folds"]
                )
            ]
            unique_signals = np.zeros(len(prepared), dtype=bool)
            evaluated_dates = np.zeros(len(prepared), dtype=bool)
            reliability = np.zeros(len(prepared), dtype=float)
            active_configuration_count = np.zeros(
                len(prepared),
                dtype=np.int16,
            )
            source_signal_count = 0
            for row in selected.itertuples(index=False):
                key = (
                    target,
                    int(horizon),
                    row.indicator,
                    int(row.test_months),
                )
                mask = oos_signal_masks[key]
                unique_signals |= mask
                evaluated_dates |= oos_evaluation_masks[
                    (target, int(horizon), int(row.test_months))
                ]
                reliability[mask] = np.maximum(
                    reliability[mask],
                    float(row.oos_precision),
                )
                active_configuration_count[mask] += 1
                source_signal_count += int(mask.sum())

            unique_signal_count = int(unique_signals.sum())
            unique_true_positive = int(
                np.sum(unique_signals & target_values)
            )
            unique_false_positive = (
                unique_signal_count - unique_true_positive
            )
            random_positive_count = int(
                np.sum(evaluated_dates & target_values)
            )
            random_observations = int(evaluated_dates.sum())
            random_precision = (
                random_positive_count / random_observations
                if random_observations
                else 0.0
            )
            ensemble_precision = (
                unique_true_positive / unique_signal_count
                if unique_signal_count
                else 0.0
            )
            ensemble_rows.append(
                {
                    "scenario": metadata.get("scenario"),
                    "target_family": metadata.get("target_family"),
                    "target": target,
                    "horizon": int(horizon),
                    "lift_threshold": threshold,
                    "selected_configuration_count": len(selected),
                    "selected_indicator_count": selected[
                        "indicator"
                    ].nunique(),
                    "source_signal_count_with_duplicates": source_signal_count,
                    "unique_signal_count": unique_signal_count,
                    "overlapping_signal_count": (
                        source_signal_count - unique_signal_count
                    ),
                    "unique_true_positive": unique_true_positive,
                    "unique_false_positive": unique_false_positive,
                    "post_selection_oos_precision": ensemble_precision,
                    "post_selection_random_precision": random_precision,
                    "post_selection_oos_lift": (
                        ensemble_precision / random_precision
                        if random_precision
                        else 0.0
                    ),
                }
            )
            for position in np.flatnonzero(unique_signals):
                ensemble_signal_rows.append(
                    {
                        "available_at": prepared.loc[
                            position, "available_at"
                        ],
                        "target": target,
                        "horizon": int(horizon),
                        "lift_threshold": threshold,
                        "signal": True,
                        "score": float(reliability[position]),
                        "active_configuration_count": int(
                            active_configuration_count[position]
                        ),
                        "target_value": int(target_values[position]),
                    }
                )
    signal_columns = [
        "available_at",
        "target",
        "horizon",
        "lift_threshold",
        "signal",
        "score",
        "active_configuration_count",
        "target_value",
    ]
    return (
        summaries,
        winner_folds,
        winner_candidates,
        pd.DataFrame(ensemble_rows),
        pd.DataFrame(ensemble_signal_rows, columns=signal_columns),
    )


def optimize_indicator(
    data: pd.DataFrame,
    *,
    indicator_name: str,
    target: str,
    horizon: int,
    candidates: list[IndicatorCandidate],
    first_test_date: str | pd.Timestamp,
    test_months: int = 12,
    train_months: int | None = None,
    refit_date: str | pd.Timestamp | None = None,
    min_signals_per_week: float = 2.0,
) -> tuple[dict, pd.DataFrame, IndicatorCandidate]:
    """Упрощённый интерфейс для оптимизации одного пространства."""
    summaries, folds, fitted, _, _ = optimize_indicator_library(
        data,
        target_registry=pd.DataFrame([{"name": target, "horizon": horizon}]),
        indicator_spaces={indicator_name: candidates},
        first_test_date=first_test_date,
        test_months_options=(test_months,),
        train_months=train_months,
        refit_date=refit_date,
        min_signals_per_week=min_signals_per_week,
    )
    key = (target, horizon, indicator_name, test_months)
    return summaries.iloc[0].to_dict(), folds, fitted[key]
