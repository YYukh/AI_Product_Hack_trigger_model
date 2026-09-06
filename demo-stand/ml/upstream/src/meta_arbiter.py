"""Unified fitting and execution API for replaceable signal arbiters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .meta_model import (
    ConfidenceFilterConfig,
    LOGISTIC_CATEGORICAL_FEATURES,
    LOGISTIC_FEATURES,
    LOGISTIC_NUMERIC_FEATURES,
    _candidate_events,
    build_meta_candidates,
    confidence_filter_meta_model,
)
from .signal_backtest import backtest_signal_stream
from .signal_policy import SignalPolicyConfig, apply_signal_policy


SUPPORTED_META_MODEL_TYPES = (
    "confidence_filter",
    "logistic_regression",
    "hist_gradient_boosting",
    "extra_trees",
)


@dataclass
class FittedMetaArbiter:
    model_type: str
    estimator: object | None
    decision_threshold: float | None
    filter_config: ConfidenceFilterConfig | None
    policy_config: SignalPolicyConfig
    feature_names: tuple[str, ...]
    trained_at: pd.Timestamp
    trained_through: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    validation_macro_lift: float
    validation_min_currency_lift: float
    validation_mean_signals_per_week: float


def _meta_periods(
    candidates: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    validation_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    validation_start = train_end - pd.DateOffset(months=validation_months)
    dates = pd.to_datetime(candidates["available_at"])
    delta = pd.to_timedelta(candidates["horizon"], unit="D")
    fit = candidates.loc[
        dates.lt(validation_start) & dates.add(delta).lt(validation_start)
    ].copy()
    validation = candidates.loc[
        dates.ge(validation_start)
        & dates.lt(train_end)
        & dates.add(delta).lt(train_end)
    ].copy()
    training = candidates.loc[
        dates.lt(train_end) & dates.add(delta).lt(train_end)
    ].copy()
    if fit.empty or validation.empty or training.empty:
        raise ValueError("Недостаточно meta fit/validation данных")
    if fit["target_value"].nunique() < 2 or training["target_value"].nunique() < 2:
        raise ValueError("В meta train присутствует только один класс")
    return fit, validation, training, validation_start


def _build_estimator(model_type: str) -> Pipeline:
    if model_type == "logistic_regression":
        preprocessor = ColumnTransformer([
            ("numeric", Pipeline([("scale", StandardScaler())]),
             list(LOGISTIC_NUMERIC_FEATURES)),
            ("categorical", OneHotEncoder(handle_unknown="ignore"),
             list(LOGISTIC_CATEGORICAL_FEATURES)),
        ])
        model = LogisticRegression(max_iter=2000, random_state=42)
    elif model_type == "hist_gradient_boosting":
        preprocessor = ColumnTransformer([
            ("numeric", "passthrough", list(LOGISTIC_NUMERIC_FEATURES)),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             list(LOGISTIC_CATEGORICAL_FEATURES)),
        ])
        model = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=100, max_leaf_nodes=7,
            min_samples_leaf=20, l2_regularization=1.0,
            class_weight="balanced", random_state=42,
        )
    elif model_type == "extra_trees":
        preprocessor = ColumnTransformer([
            ("numeric", "passthrough", list(LOGISTIC_NUMERIC_FEATURES)),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             list(LOGISTIC_CATEGORICAL_FEATURES)),
        ])
        model = ExtraTreesClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            max_features=0.8, class_weight="balanced", n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"Неизвестный обучаемый meta model: {model_type}")
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _candidate_events_from_scores(
    candidates: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    model_type: str,
) -> pd.DataFrame:
    scored = candidates.copy()
    scored["meta_probability"] = np.asarray(scores, dtype=float)
    accepted = scored.loc[scored["meta_probability"].ge(threshold)].copy()
    return _candidate_events(
        accepted,
        confidence_column="meta_probability",
        confidence_method=f"{model_type}_meta_probability",
        meta_model_name=model_type,
        meta_model_version="validation_selected",
    )


def _candidate_evidence(candidates: pd.DataFrame) -> pd.DataFrame:
    records = [record for bundle in candidates["evidence"] for record in bundle]
    return pd.DataFrame(records)


def _validation_universe(
    universe: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    dates = pd.to_datetime(universe["available_at"])
    delta = pd.to_timedelta(universe["horizon"], unit="D")
    return universe.loc[
        dates.ge(start) & dates.lt(end) & dates.add(delta).lt(end)
    ].copy()


def _evaluate(
    events: pd.DataFrame,
    *,
    universe: pd.DataFrame,
    policy: SignalPolicyConfig,
    min_signals_per_week: float,
    max_signals_per_week: float,
) -> dict:
    filtered = apply_signal_policy(events, policy)
    summary, _ = backtest_signal_stream(filtered, evaluation_universe=universe)
    currency = summary.loc[summary["scope"].eq("currency")].copy()
    frequency_ok = bool(currency["signals_per_week"].between(
        min_signals_per_week, max_signals_per_week, inclusive="both"
    ).all())
    return {
        "events": filtered,
        "frequency_ok": frequency_ok,
        "macro_lift": float(currency["lift"].mean()),
        "min_currency_lift": float(currency["lift"].min()),
        "mean_precision": float(currency["precision"].mean()),
        "mean_signals_per_week": float(currency["signals_per_week"].mean()),
    }


def fit_meta_arbiter(
    labelled_candidates: pd.DataFrame,
    *,
    evaluation_universe: pd.DataFrame,
    model_type: str,
    train_end: str | pd.Timestamp,
    validation_months: int = 12,
    min_signals_per_week: float = 1.0,
    max_signals_per_week: float = 2.0,
    max_signals_per_7d: int = 2,
    min_support: int = 10,
    min_sources: int = 1,
    cooldown_options: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
    rule_threshold_options: tuple[float, ...] = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60),
    ml_threshold_options: tuple[float, ...] = (0.00, 0.50, 0.60, 0.70, 0.80, 0.90),
    learned_threshold_options: tuple[float, ...] = (
        0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
    ),
) -> tuple[FittedMetaArbiter, pd.DataFrame]:
    """Select meta thresholds and policy strictly on the last validation year."""
    if model_type not in SUPPORTED_META_MODEL_TYPES:
        raise ValueError(
            f"model_type={model_type!r}; доступны {SUPPORTED_META_MODEL_TYPES}"
        )
    end = pd.Timestamp(train_end)
    fit, validation, training, validation_start = _meta_periods(
        labelled_candidates, train_end=end, validation_months=validation_months
    )
    validation_universe = _validation_universe(
        evaluation_universe, start=validation_start, end=end
    )
    rows: list[dict] = []
    configurations: list[tuple[dict, dict]] = []

    if model_type == "confidence_filter":
        validation_evidence = _candidate_evidence(validation)
        for rule_threshold in rule_threshold_options:
            for ml_threshold in ml_threshold_options:
                filter_config = ConfidenceFilterConfig(
                    min_confidence=0.0,
                    min_support=min_support,
                    min_sources=min_sources,
                    rule_min_confidence=float(rule_threshold),
                    ml_min_confidence=float(ml_threshold),
                )
                events = confidence_filter_meta_model(
                    validation_evidence, filter_config
                )
                for cooldown in cooldown_options:
                    policy = SignalPolicyConfig(
                        cooldown_days=int(cooldown),
                        max_signals_per_7d=max_signals_per_7d,
                    )
                    metrics = _evaluate(
                        events, universe=validation_universe, policy=policy,
                        min_signals_per_week=min_signals_per_week,
                        max_signals_per_week=max_signals_per_week,
                    )
                    params = {
                        "rule_threshold": float(rule_threshold),
                        "ml_threshold": float(ml_threshold),
                        "decision_threshold": np.nan,
                        "cooldown_days": int(cooldown),
                    }
                    rows.append({**params, **{k: v for k, v in metrics.items() if k != "events"}})
                    configurations.append((params, {"filter": filter_config, "policy": policy, **metrics}))
        estimator = None
    else:
        estimator = _build_estimator(model_type)
        estimator.fit(fit.loc[:, LOGISTIC_FEATURES], fit["target_value"].astype(int))
        validation_scores = estimator.predict_proba(
            validation.loc[:, LOGISTIC_FEATURES]
        )[:, 1]
        for threshold in learned_threshold_options:
            events = _candidate_events_from_scores(
                validation, validation_scores, float(threshold), model_type
            )
            for cooldown in cooldown_options:
                policy = SignalPolicyConfig(
                    cooldown_days=int(cooldown),
                    max_signals_per_7d=max_signals_per_7d,
                )
                metrics = _evaluate(
                    events, universe=validation_universe, policy=policy,
                    min_signals_per_week=min_signals_per_week,
                    max_signals_per_week=max_signals_per_week,
                )
                params = {
                    "rule_threshold": np.nan,
                    "ml_threshold": np.nan,
                    "decision_threshold": float(threshold),
                    "cooldown_days": int(cooldown),
                }
                rows.append({**params, **{k: v for k, v in metrics.items() if k != "events"}})
                configurations.append((params, {"policy": policy, **metrics}))

    leaderboard = pd.DataFrame(rows)
    feasible_positions = [
        index for index, (_, item) in enumerate(configurations)
        if item["frequency_ok"]
    ]
    if not feasible_positions:
        raise ValueError(
            "На validation нет конфигурации с 1–2 сигналами в неделю "
            "для каждой валюты при жёстком лимите 2 за 7 дней"
        )
    winner_position = max(
        feasible_positions,
        key=lambda index: (
            configurations[index][1]["macro_lift"],
            configurations[index][1]["min_currency_lift"],
            configurations[index][1]["mean_precision"],
            -abs(configurations[index][1]["mean_signals_per_week"] - 1.5),
        ),
    )
    winner_params, winner = configurations[winner_position]
    leaderboard["selected"] = False
    leaderboard.loc[winner_position, "selected"] = True

    if model_type != "confidence_filter":
        estimator = _build_estimator(model_type)
        estimator.fit(
            training.loc[:, LOGISTIC_FEATURES], training["target_value"].astype(int)
        )
    fitted = FittedMetaArbiter(
        model_type=model_type,
        estimator=estimator,
        decision_threshold=(
            None if model_type == "confidence_filter"
            else float(winner_params["decision_threshold"])
        ),
        filter_config=winner.get("filter"),
        policy_config=winner["policy"],
        feature_names=tuple(LOGISTIC_FEATURES),
        trained_at=end,
        trained_through=pd.to_datetime(training["available_at"]).max(),
        validation_start=validation_start,
        validation_end=end,
        validation_macro_lift=float(winner["macro_lift"]),
        validation_min_currency_lift=float(winner["min_currency_lift"]),
        validation_mean_signals_per_week=float(winner["mean_signals_per_week"]),
    )
    return fitted, leaderboard.sort_values(
        ["selected", "macro_lift"], ascending=[False, False]
    ).reset_index(drop=True)


def run_meta_arbiter(
    evidence: pd.DataFrame,
    fitted: FittedMetaArbiter,
    *,
    policy_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run any supported meta-model and the same frozen downstream policy."""
    if fitted.model_type == "confidence_filter":
        events = confidence_filter_meta_model(evidence, fitted.filter_config)
    else:
        candidates = build_meta_candidates(evidence)
        if candidates.empty:
            events = _candidate_events(
                candidates, confidence_column="meta_probability",
                confidence_method=f"{fitted.model_type}_meta_probability",
                meta_model_name=fitted.model_type,
                meta_model_version=f"trained_{fitted.trained_at.date()}",
            )
        else:
            scores = fitted.estimator.predict_proba(
                candidates.loc[:, fitted.feature_names]
            )[:, 1]
            events = _candidate_events_from_scores(
                candidates, scores, float(fitted.decision_threshold),
                fitted.model_type,
            )
    return apply_signal_policy(
        events, fitted.policy_config, history=policy_history
    )
