"""Walk-forward selection and continued OOS backtest of an ML indicator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .walk_forward import make_periodic_walk_forward_folds


SUPPORTED_MODEL_TYPES = (
    "logistic_regression",
    "hist_gradient_boosting",
    "extra_trees",
)


@dataclass(frozen=True)
class MLIndicatorConfig:
    """Fixed model settings; these values are not optimized on the targets."""

    learning_rate: float = 0.08
    max_iter: int = 60
    max_leaf_nodes: int = 7
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    random_state: int = 42


def build_ml_indicator(
    model_type: str = "logistic_regression",
    *,
    config: MLIndicatorConfig = MLIndicatorConfig(),
):
    """Построить одну из поддерживаемых моделей с фиксированными настройками."""
    if model_type == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2_000,
                class_weight="balanced",
                random_state=config.random_state,
            ),
        )
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=config.learning_rate,
            max_iter=config.max_iter,
            max_leaf_nodes=config.max_leaf_nodes,
            min_samples_leaf=config.min_samples_leaf,
            l2_regularization=config.l2_regularization,
            class_weight="balanced",
            random_state=config.random_state,
        )
    if model_type == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=60,
            max_depth=8,
            min_samples_leaf=10,
            max_features=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.random_state,
        )
    raise ValueError(
        f"Неизвестный model_type={model_type!r}; "
        f"допустимы {SUPPORTED_MODEL_TYPES}"
    )


def _calendar_weeks(
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> float:
    return (end_exclusive - start).total_seconds() / pd.Timedelta(
        days=7
    ).total_seconds()


def _choose_probability_threshold(
    probabilities: np.ndarray,
    target: np.ndarray,
    *,
    calendar_weeks: float,
    min_signals_per_week: float,
) -> tuple[float, dict] | None:
    """Maximize validation lift subject to the signal-frequency constraint."""
    probabilities = np.asarray(probabilities, dtype="float64")
    target = np.asarray(target, dtype=bool)
    if not len(target) or calendar_weeks <= 0:
        return None

    minimum_count = max(1, int(np.ceil(min_signals_per_week * calendar_weeks)))
    thresholds = np.unique(probabilities)[::-1]
    best: tuple[tuple[float, float, int], float, dict] | None = None
    baseline = float(target.mean())
    for threshold in thresholds:
        prediction = probabilities >= threshold
        signal_count = int(prediction.sum())
        if signal_count < minimum_count:
            continue
        true_positive = int(np.sum(prediction & target))
        precision = true_positive / signal_count
        lift = precision / baseline if baseline else 0.0
        # Lift first; then precision; then fewer signals for deterministic tie-break.
        score = (lift, precision, -signal_count)
        metrics = {
            "validation_observations": len(target),
            "validation_positive_count": int(target.sum()),
            "validation_signal_count": signal_count,
            "validation_true_positive": true_positive,
            "validation_precision": precision,
            "validation_random_precision": baseline,
            "validation_lift": lift,
            "validation_signals_per_week": signal_count / calendar_weeks,
        }
        if best is None or score > best[0]:
            best = (score, float(threshold), metrics)
    if best is None:
        return None
    return best[1], best[2]


def _metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    calendar_weeks: float,
    benefit_bps: np.ndarray | None = None,
) -> dict:
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    observations = len(target)
    positive_count = int(target.sum())
    signal_count = int(prediction.sum())
    true_positive = int(np.sum(target & prediction))
    false_positive = signal_count - true_positive
    precision = true_positive / signal_count if signal_count else 0.0
    random_precision = positive_count / observations if observations else 0.0
    if benefit_bps is not None:
        benefit_array = np.asarray(benefit_bps)
        if len(benefit_array) == len(prediction):
            benefit_array = benefit_array[prediction]
        benefit = pd.to_numeric(
            pd.Series(benefit_array), errors="coerce"
        ).dropna()
    else:
        benefit = pd.Series(dtype=float)
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
        "signals_per_week": (
            signal_count / calendar_weeks if calendar_weeks else 0.0
        ),
        "benefit_bps_sum": float(benefit.sum()),
        "benefit_bps_count": int(benefit.size),
        "positive_benefit_count": int((benefit > 0).sum()),
        "mean_benefit_bps": float(benefit.mean()) if len(benefit) else np.nan,
        "median_benefit_bps": float(benefit.median()) if len(benefit) else np.nan,
        "positive_benefit_rate": float((benefit > 0).mean()) if len(benefit) else 0.0,
    }


def _fit_calibrate_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str,
    feature_names: tuple[str, ...],
    horizon: int,
    test_start: pd.Timestamp,
    validation_months: int,
    min_signals_per_week: float,
    model_type: str,
    model_config: MLIndicatorConfig,
) -> tuple[np.ndarray, np.ndarray, float, dict] | None:
    """Calibrate on past OOS scores, refit on all mature train, predict test."""
    trained = fit_calibrated_ml_indicator(
        train,
        target=target,
        feature_names=feature_names,
        horizon=horizon,
        retrain_at=test_start,
        validation_months=validation_months,
        min_signals_per_week=min_signals_per_week,
        model_type=model_type,
        model_config=model_config,
    )
    if trained is None:
        return None
    model, threshold, audit = trained
    probability = model.predict_proba(test.loc[:, feature_names])[:, 1]
    prediction = probability >= threshold
    return probability, prediction, threshold, audit


def fit_calibrated_ml_indicator(
    train: pd.DataFrame,
    *,
    target: str,
    feature_names: tuple[str, ...],
    horizon: int,
    retrain_at: pd.Timestamp,
    validation_months: int,
    min_signals_per_week: float,
    model_type: str,
    model_config: MLIndicatorConfig,
) -> tuple[object, float, dict] | None:
    """Fit one ML artifact using only matured history before ``retrain_at``."""
    validation_start = retrain_at - pd.DateOffset(months=validation_months)
    train_dates = pd.to_datetime(train["available_at"])
    fit_mask = train_dates.add(pd.Timedelta(days=horizon)).lt(validation_start)
    validation_mask = train_dates.ge(validation_start)
    fit = train.loc[fit_mask]
    validation = train.loc[validation_mask]

    if fit.empty or validation.empty:
        return None
    if fit[target].nunique() < 2 or train[target].nunique() < 2:
        return None

    calibration_model = build_ml_indicator(model_type, config=model_config)
    calibration_model.fit(fit.loc[:, feature_names], fit[target].astype(int))
    validation_probability = calibration_model.predict_proba(
        validation.loc[:, feature_names]
    )[:, 1]
    validation_weeks = _calendar_weeks(
        validation["available_at"].min(),
        validation["available_at"].max() + pd.Timedelta(days=1),
    )
    selected = _choose_probability_threshold(
        validation_probability,
        validation[target].to_numpy(dtype=bool),
        calendar_weeks=validation_weeks,
        min_signals_per_week=min_signals_per_week,
    )
    if selected is None:
        return None
    threshold, validation_metrics = selected

    model = build_ml_indicator(model_type, config=model_config)
    model.fit(train.loc[:, feature_names], train[target].astype(int))
    audit = {
        **validation_metrics,
        "fit_start": fit["available_at"].min(),
        "fit_end": fit["available_at"].max(),
        "validation_start": validation["available_at"].min(),
        "validation_end": validation["available_at"].max(),
        "train_start": train["available_at"].min(),
        "train_end": train["available_at"].max(),
    }
    return model, threshold, audit


def _pooled_metrics(folds: list[dict]) -> dict:
    totals = {
        name: sum(int(row[name]) for row in folds)
        for name in (
            "observations",
            "positive_count",
            "signal_count",
            "true_positive",
            "false_positive",
            "benefit_bps_sum", "benefit_bps_count", "positive_benefit_count",
        )
    }
    calendar_weeks = sum(float(row["calendar_weeks"]) for row in folds)
    precision = (
        totals["true_positive"] / totals["signal_count"]
        if totals["signal_count"]
        else 0.0
    )
    random_precision = (
        totals["positive_count"] / totals["observations"]
        if totals["observations"]
        else 0.0
    )
    return {
        **totals,
        "signal_precision": precision,
        "random_precision": random_precision,
        "lift": precision / random_precision if random_precision else 0.0,
        "calendar_weeks": calendar_weeks,
        "signals_per_week": (
            totals["signal_count"] / calendar_weeks if calendar_weeks else 0.0
        ),
        "mean_benefit_bps": (
            totals["benefit_bps_sum"] / totals["benefit_bps_count"]
            if totals["benefit_bps_count"] else np.nan
        ),
        "positive_benefit_rate": (
            totals["positive_benefit_count"] / totals["benefit_bps_count"]
            if totals["benefit_bps_count"] else 0.0
        ),
    }


def _prepare_configuration(
    data: pd.DataFrame,
    *,
    currency: str,
    target: str,
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    required = {"available_at", "currency", target, *feature_names}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Не хватает ML-полей: {sorted(missing)}")
    return (
        data.loc[data["currency"].eq(currency)]
        .dropna(subset=[target, *feature_names])
        .sort_values("available_at")
        .reset_index(drop=True)
    )


def run_ml_indicator_backtest(
    data: pd.DataFrame,
    *,
    target_registry: pd.DataFrame,
    feature_names: list[str] | tuple[str, ...],
    first_test_date: str | pd.Timestamp,
    winner_backtest_start: str | pd.Timestamp,
    test_months_options: tuple[int, ...] = (3, 6, 12),
    train_months: int = 24,
    target_families: tuple[str, ...] = ("G0", "W1"),
    min_signals_per_week: float = 2.0,
    validation_months: int = 12,
    model_type: str = "logistic_regression",
    currencies: tuple[str, ...] | list[str] | None = None,
    model_config: MLIndicatorConfig = MLIndicatorConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select cadence on discovery and continue rolling WF with the winner.

    Returns cadence leaderboard, winner pooled summary, OOS folds and every
    emitted ML signal. There is exactly one model configuration for each
    ``currency × target × horizon`` in the requested target families.
    """
    features = tuple(dict.fromkeys(feature_names))
    if not features:
        raise ValueError("feature_names не должен быть пустым")
    if validation_months <= 0:
        raise ValueError("validation_months должен быть положительным")
    if train_months <= 0:
        raise ValueError("train_months должен быть положительным")
    if validation_months >= train_months:
        raise ValueError("validation_months должен быть меньше train_months")
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Неизвестный model_type={model_type!r}; "
            f"допустимы {SUPPORTED_MODEL_TYPES}"
        )
    indicator_name = f"ml_{model_type}"
    cadences = tuple(dict.fromkeys(int(value) for value in test_months_options))
    if not cadences or any(value <= 0 for value in cadences):
        raise ValueError("test_months_options должны быть положительными")

    winner_start = pd.Timestamp(winner_backtest_start)
    definitions = target_registry.loc[
        target_registry["family"].isin(target_families)
    ].copy()
    if definitions.empty:
        raise ValueError("В registry нет выбранных target families")

    selection_rows: list[dict] = []
    selected_cadence: dict[tuple[str, str, int], int] = {}
    prepared_by_key: dict[tuple[str, str, int], pd.DataFrame] = {}

    available_currencies = set(data["currency"].dropna().unique())
    selected_currencies = (
        tuple(dict.fromkeys(currencies))
        if currencies is not None
        else tuple(sorted(available_currencies))
    )
    if not selected_currencies:
        raise ValueError("Нужна хотя бы одна валюта для ML backtest")
    unknown_currencies = set(selected_currencies).difference(available_currencies)
    if unknown_currencies:
        raise ValueError(
            f"В data нет валют: {sorted(unknown_currencies)}"
        )

    for currency in selected_currencies:
        for definition in definitions.itertuples(index=False):
            target = definition.name
            horizon = int(definition.horizon)
            key = (currency, target, horizon)
            prepared = _prepare_configuration(
                data,
                currency=currency,
                target=target,
                feature_names=features,
            )
            prepared_by_key[key] = prepared
            discovery = prepared.loc[
                prepared["available_at"].add(pd.Timedelta(days=horizon)).lt(
                    winner_start
                )
            ].reset_index(drop=True)

            key_rows = []
            for test_months in cadences:
                folds = make_periodic_walk_forward_folds(
                    discovery,
                    horizon=horizon,
                    first_test_date=first_test_date,
                    test_months=test_months,
                    train_months=train_months,
                )
                metric_rows = []
                successful_folds = 0
                for fold in folds:
                    train = discovery.loc[fold.train_index]
                    test = discovery.loc[fold.test_index]
                    fitted = _fit_calibrate_predict(
                        train,
                        test,
                        target=target,
                        feature_names=features,
                        horizon=horizon,
                        test_start=fold.test_start,
                        validation_months=validation_months,
                        min_signals_per_week=min_signals_per_week,
                        model_type=model_type,
                        model_config=model_config,
                    )
                    if fitted is None:
                        continue
                    _, prediction, _, _ = fitted
                    metric_rows.append(
                        _metrics(
                            test[target].to_numpy(dtype=bool),
                            prediction,
                            calendar_weeks=_calendar_weeks(
                                fold.test_start,
                                fold.test_end + pd.Timedelta(days=1),
                            ),
                            benefit_bps=test.get(
                                f"local_advantage_{horizon}d_bps"
                            ).to_numpy() if f"local_advantage_{horizon}d_bps" in test else None,
                        )
                    )
                    successful_folds += 1
                if not metric_rows:
                    continue
                pooled = _pooled_metrics(metric_rows)
                row = {
                    "currency": currency,
                    "scenario": definition.scenario,
                    "target_family": definition.family,
                    "target": target,
                    "horizon": horizon,
                    "indicator": indicator_name,
                    "test_months": test_months,
                    "train_months": train_months,
                    "folds": len(folds),
                    "successful_folds": successful_folds,
                    "feature_count": len(features),
                    "feature_names": ", ".join(features),
                    "oos_observations": pooled["observations"],
                    "oos_signal_count": pooled["signal_count"],
                    "oos_true_positive": pooled["true_positive"],
                    "oos_false_positive": pooled["false_positive"],
                    "oos_precision": pooled["signal_precision"],
                    "oos_random_precision": pooled["random_precision"],
                    "oos_lift": pooled["lift"],
                    "oos_calendar_weeks": pooled["calendar_weeks"],
                    "oos_signals_per_week": pooled["signals_per_week"],
                    "all_folds_fitted": successful_folds == len(folds),
                    "frequency_constraint_met": (
                        pooled["signals_per_week"] >= min_signals_per_week
                    ),
                }
                key_rows.append(row)
                selection_rows.append(row)

            if not key_rows:
                raise ValueError(f"ML не построил discovery folds для {key}")
            candidates = [
                row
                for row in key_rows
                if row["all_folds_fitted"] and row["frequency_constraint_met"]
            ]
            if not candidates:
                candidates = key_rows
            winner = sorted(
                candidates,
                key=lambda row: (-row["oos_lift"], row["test_months"]),
            )[0]
            selected_cadence[key] = int(winner["test_months"])

    selection = pd.DataFrame(selection_rows)
    selection["selected"] = False
    for key, cadence in selected_cadence.items():
        currency, target, horizon = key
        mask = (
            selection["currency"].eq(currency)
            & selection["target"].eq(target)
            & selection["horizon"].eq(horizon)
            & selection["test_months"].eq(cadence)
        )
        selection.loc[mask, "selected"] = True

    fold_rows: list[dict] = []
    signal_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for definition in definitions.itertuples(index=False):
        target = definition.name
        horizon = int(definition.horizon)
        for currency in selected_currencies:
            key = (currency, target, horizon)
            prepared = prepared_by_key[key]
            cadence = selected_cadence[key]
            period_start = winner_start
            fold_id = 1
            configuration_folds = []
            while period_start <= prepared["available_at"].max():
                period_end = period_start + pd.DateOffset(months=cadence)
                train = prepared.loc[
                    prepared["available_at"].ge(
                        period_start - pd.DateOffset(months=train_months)
                    )
                    & prepared["available_at"].add(pd.Timedelta(days=horizon)).lt(
                        period_start
                    )
                ]
                test = prepared.loc[
                    prepared["available_at"].ge(period_start)
                    & prepared["available_at"].lt(period_end)
                ]
                if test.empty:
                    period_start = period_end
                    continue
                fitted = _fit_calibrate_predict(
                    train,
                    test,
                    target=target,
                    feature_names=features,
                    horizon=horizon,
                    test_start=period_start,
                    validation_months=validation_months,
                    min_signals_per_week=min_signals_per_week,
                    model_type=model_type,
                    model_config=model_config,
                )
                if fitted is None:
                    raise ValueError(
                        f"ML fold нельзя обучить: {key}, start={period_start.date()}"
                    )
                probability, prediction, threshold, audit = fitted
                evaluation_end = min(
                    period_end,
                    test["available_at"].max() + pd.Timedelta(days=1),
                )
                metrics = _metrics(
                    test[target].to_numpy(dtype=bool),
                    prediction,
                    calendar_weeks=_calendar_weeks(period_start, evaluation_end),
                    benefit_bps=test.get(
                        f"local_advantage_{horizon}d_bps"
                    ).to_numpy() if f"local_advantage_{horizon}d_bps" in test else None,
                )
                metadata = {
                    "currency": currency,
                    "scenario": definition.scenario,
                    "target_family": definition.family,
                    "target": target,
                    "horizon": horizon,
                    "indicator": indicator_name,
                    "rebalance_months": cadence,
                    "train_months": train_months,
                    "fold_id": fold_id,
                }
                fold_row = {
                    **metadata,
                    "test_start": test["available_at"].min(),
                    "test_end": test["available_at"].max(),
                    "probability_threshold": threshold,
                    **audit,
                    **metrics,
                }
                fold_rows.append(fold_row)
                configuration_folds.append(fold_row)

                if prediction.any():
                    emitted = test.loc[
                        prediction,
                        ["available_at", "currency", target],
                    ].rename(columns={target: "target_value"})
                    emitted["probability"] = probability[prediction]
                    emitted["probability_threshold"] = threshold
                    # Comparable confidence for the downstream meta-model.
                    # This is estimated only on the past validation window;
                    # the current final-test labels are never used here.
                    emitted["confidence"] = audit["validation_precision"]
                    emitted["confidence_method"] = (
                        "validation_precision_at_probability_threshold"
                    )
                    emitted["confidence_support"] = audit[
                        "validation_signal_count"
                    ]
                    for name, value in metadata.items():
                        if name != "currency":
                            emitted[name] = value
                    signal_frames.append(emitted)
                fold_id += 1
                period_start = period_end

            pooled = _pooled_metrics(configuration_folds)
            summary_rows.append(
                {
                    "currency": currency,
                    "scenario": definition.scenario,
                    "target_family": definition.family,
                    "target": target,
                    "horizon": horizon,
                    "indicator": indicator_name,
                    "fixed_candidate": indicator_name,
                    "fixed_logic": "MODEL",
                    "train_months": train_months,
                    "rebalance_months": cadence,
                    "folds": len(configuration_folds),
                    "feature_count": len(features),
                    "feature_names": ", ".join(features),
                    "test_observations": pooled["observations"],
                    "test_positive_count": pooled["positive_count"],
                    "test_signal_count": pooled["signal_count"],
                    "test_true_positive": pooled["true_positive"],
                    "test_false_positive": pooled["false_positive"],
                    "test_signal_precision": pooled["signal_precision"],
                    "test_random_precision": pooled["random_precision"],
                    "test_lift": pooled["lift"],
                    "test_calendar_weeks": pooled["calendar_weeks"],
                    "test_signals_per_week": pooled["signals_per_week"],
                    "test_mean_benefit_bps": pooled["mean_benefit_bps"],
                    "test_positive_benefit_rate": pooled["positive_benefit_rate"],
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["target_family", "currency", "horizon"]
    ).reset_index(drop=True)
    folds = pd.DataFrame(fold_rows).sort_values(
        ["target_family", "currency", "horizon", "fold_id"]
    ).reset_index(drop=True)
    signals = (
        pd.concat(signal_frames, ignore_index=True)
        if signal_frames
        else pd.DataFrame()
    )
    return (
        selection.sort_values(
            ["target_family", "currency", "horizon", "test_months"]
        ).reset_index(drop=True),
        summary,
        folds,
        signals,
    )
