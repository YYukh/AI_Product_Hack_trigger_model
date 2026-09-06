"""Temporally trained drop-in opportunity selectors.

The selector estimates P(target=1 | currently available evidence).  Model
fitting, probability calibration and policy-threshold validation use three
strictly ordered pre-holdout intervals.  Neither labels nor metrics from the
final holdout can affect the fitted selector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .signal_backtest import backtest_signal_stream

from .arbiter import (
    _collapse_evidence,
    _coverage_metrics,
    _quarterly_stability,
    _remove_horizon_dominance,
    _validation_metrics,
    _validation_universe,
)
from .config import YuraPipelineConfig
from .engines import KEYS
from .policy import SignalPolicyConfig, apply_signal_policy


SUPPORTED_LEARNED_SELECTORS = ("logistic_regression", "extra_trees")

NUMERIC_FEATURES = (
    "confidence",
    "baseline_probability",
    "confidence_lift",
    "expected_bps",
    "ml_confidence",
    "ml_lift",
    "rule_confidence",
    "rule_lift",
    "rule_count",
    "evidence_count",
    "statistical_evidence",
    "economic_evidence",
    "statistical_rank",
    "economic_rank",
    "decision_score",
)
CATEGORICAL_FEATURES = ("currency", "target_family", "horizon", "engine_type")


@dataclass(frozen=True)
class LearnedSelectorPeriods:
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


@dataclass
class FittedLearnedSelector:
    model_type: str
    estimator: Pipeline
    calibrator: Pipeline
    threshold: float
    policy: SignalPolicyConfig
    periods: LearnedSelectorPeriods
    trained_through: pd.Timestamp
    validation_geometric_lift: float
    validation_macro_lift: float
    validation_min_currency_lift: float
    validation_macro_benefit_uplift_bps: float
    validation_mean_signals_per_week: float
    validation_quarterly_geometric_lift: float
    validation_min_currency_quarterly_geometric_lift: float
    validation_quarterly_lift_p10: float
    validation_stability_ok: bool
    validation_scenario_coverage: float
    validation_horizon_coverage: float


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _feature_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {*NUMERIC_FEATURES, *CATEGORICAL_FEATURES}
    if missing := required.difference(candidates.columns):
        raise KeyError(f"В opportunities нет признаков learned selector: {sorted(missing)}")
    result = candidates.loc[:, [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]].copy()
    for column in NUMERIC_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].astype("string").fillna("UNKNOWN")
    return result


def _build_estimator(model_type: str, random_state: int) -> Pipeline:
    if model_type not in SUPPORTED_LEARNED_SELECTORS:
        raise ValueError(
            f"Неизвестный learned selector {model_type!r}; "
            f"доступны {SUPPORTED_LEARNED_SELECTORS}"
        )
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if model_type == "logistic_regression":
        numeric_steps.append(("scaler", StandardScaler()))
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", _one_hot_encoder()),
        ]), list(CATEGORICAL_FEATURES)),
    ])
    if model_type == "logistic_regression":
        model = LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=1_000,
            solver="lbfgs", random_state=random_state,
        )
    else:
        model = ExtraTreesClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            max_features="sqrt", class_weight="balanced",
            n_jobs=-1, random_state=random_state,
        )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _build_calibrator(random_state: int) -> Pipeline:
    # The raw model is class-balanced, so its raw probability is deliberately
    # not treated as confidence.  This temporal calibrator restores an actual
    # target probability and permits only broad, predeclared stratum offsets.
    categorical = ["currency", "target_family", "horizon"]
    preprocessor = ColumnTransformer([
        ("raw_score", StandardScaler(), ["raw_logit"]),
        ("categorical", _one_hot_encoder(), categorical),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            C=1.0, max_iter=1_000, solver="lbfgs", random_state=random_state,
        )),
    ])


def _selector_periods(config: YuraPipelineConfig) -> LearnedSelectorPeriods:
    fit_start = pd.Timestamp(config.arbiter_validation_start)
    validation_end = pd.Timestamp(config.holdout_start)
    validation_start = validation_end - pd.DateOffset(
        months=config.learned_selector_validation_months
    )
    calibration_start = validation_start - pd.DateOffset(
        months=config.learned_selector_calibration_months
    )
    if not fit_start < calibration_start < validation_start < validation_end:
        raise ValueError(
            "Learned selector требует последовательные непустые fit, calibration "
            "и validation интервалы до holdout"
        )
    return LearnedSelectorPeriods(
        fit_start=fit_start,
        fit_end=calibration_start,
        calibration_start=calibration_start,
        calibration_end=validation_start,
        validation_start=validation_start,
        validation_end=validation_end,
    )


def _mature_slice(
    candidates: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    dates = pd.to_datetime(candidates["available_at"], errors="raise")
    maturity = dates + pd.to_timedelta(
        pd.to_numeric(candidates["horizon"], errors="raise"), unit="D"
    )
    # maturity < end is the purge: labels crossing the next stage boundary are
    # never available to the stage that is fitted at that boundary.
    return candidates.loc[dates.ge(start) & dates.lt(end) & maturity.lt(end)].copy()


def _check_training_sample(sample: pd.DataFrame, stage: str) -> None:
    if sample.empty:
        raise ValueError(f"Пустая выборка learned selector: {stage}")
    target = pd.to_numeric(sample["target_value"], errors="raise")
    if target.nunique() < 2:
        raise ValueError(f"В выборке {stage} learned selector только один класс")


def _raw_probability(estimator: Pipeline, candidates: pd.DataFrame) -> np.ndarray:
    return estimator.predict_proba(_feature_frame(candidates))[:, 1]


def _calibration_frame(candidates: pd.DataFrame, raw_probability: np.ndarray) -> pd.DataFrame:
    probability = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return pd.DataFrame({
        "raw_logit": np.log(probability / (1.0 - probability)),
        "currency": candidates["currency"].astype("string").to_numpy(),
        "target_family": candidates["target_family"].astype("string").to_numpy(),
        "horizon": candidates["horizon"].astype("string").to_numpy(),
    }, index=candidates.index)


def _score_candidates(
    candidates: pd.DataFrame,
    estimator: Pipeline,
    calibrator: Pipeline,
) -> pd.DataFrame:
    result = candidates.copy()
    raw = _raw_probability(estimator, result)
    calibrated = calibrator.predict_proba(_calibration_frame(result, raw))[:, 1]
    result["base_confidence"] = result["confidence"]
    result["base_confidence_lift"] = result["confidence_lift"]
    result["base_decision_score"] = result["decision_score"]
    result["meta_raw_probability"] = raw
    result["meta_probability"] = calibrated
    result["confidence"] = calibrated
    baseline = pd.to_numeric(result["baseline_probability"], errors="coerce")
    result["confidence_lift"] = calibrated / baseline.clip(lower=1e-6)
    # Policy ranks same-day alternatives by the learned probability; the
    # original evidence score remains auditable in base_decision_score.
    result["decision_score"] = calibrated
    return result


def _select_scored(
    scored: pd.DataFrame,
    *,
    threshold: float,
    model_type: str,
) -> pd.DataFrame:
    expected_bps = pd.to_numeric(scored["expected_bps"], errors="coerce")
    eligible = scored.loc[
        pd.to_numeric(scored["meta_probability"], errors="coerce").ge(threshold)
        & expected_bps.ge(0.0)
    ].copy()
    eligible = _remove_horizon_dominance(eligible)
    events = _collapse_evidence(eligible)
    if events.empty:
        return events
    meta = eligible.sort_values(
        [*KEYS, "meta_probability", "expected_bps"],
        ascending=[True] * len(KEYS) + [False, False],
    ).drop_duplicates(list(KEYS))
    meta = meta.loc[:, [*KEYS, "meta_raw_probability", "meta_probability",
                        "base_confidence", "base_confidence_lift",
                        "base_decision_score"]]
    events = events.merge(meta, on=list(KEYS), how="left", validate="one_to_one")
    events["selector_model_type"] = model_type
    events["confidence_method"] = "temporally_calibrated_probability"
    events["selector_threshold"] = float(threshold)
    return events


def _frequency_is_valid(summary: pd.DataFrame, config: YuraPipelineConfig) -> bool:
    currency = summary.loc[summary["scope"].eq("currency")].copy()
    if len(currency) != len(config.currencies):
        return False
    rounding = 1.0 / currency["calendar_weeks"]
    return bool(
        currency["signals_per_week"].add(rounding).ge(
            config.min_average_signals_per_week
        ).all()
        and currency["signals_per_week"].le(
            config.max_average_signals_per_week
        ).all()
    )


class LearnedOpportunitySelector:
    """Logistic or Extra-Trees selector with frozen temporal calibration."""

    def __init__(self, model_type: str = "logistic_regression") -> None:
        if model_type not in SUPPORTED_LEARNED_SELECTORS:
            raise ValueError(
                f"model_type должен быть одним из {SUPPORTED_LEARNED_SELECTORS}"
            )
        self.model_type = model_type
        self.selector_name = model_type

    def fit(
        self,
        candidates: pd.DataFrame,
        *,
        evaluation_universe: pd.DataFrame,
        config: YuraPipelineConfig,
    ) -> tuple[FittedLearnedSelector, pd.DataFrame]:
        periods = _selector_periods(config)
        fit_rows = _mature_slice(
            candidates, start=periods.fit_start, end=periods.fit_end
        )
        calibration_rows = _mature_slice(
            candidates,
            start=periods.calibration_start,
            end=periods.calibration_end,
        )
        validation_rows = _mature_slice(
            candidates,
            start=periods.validation_start,
            end=periods.validation_end,
        )
        for sample, stage in (
            (fit_rows, "fit"),
            (calibration_rows, "calibration"),
            (validation_rows, "validation"),
        ):
            _check_training_sample(sample, stage)

        estimator = _build_estimator(self.model_type, config.random_state)
        estimator.fit(
            _feature_frame(fit_rows), fit_rows["target_value"].astype(int)
        )
        calibration_raw = _raw_probability(estimator, calibration_rows)
        calibrator = _build_calibrator(config.random_state)
        calibrator.fit(
            _calibration_frame(calibration_rows, calibration_raw),
            calibration_rows["target_value"].astype(int),
        )
        scored_validation = _score_candidates(
            validation_rows, estimator, calibrator
        )
        universe = _validation_universe(
            evaluation_universe,
            start=periods.validation_start,
            end=periods.validation_end,
        )
        if universe.empty:
            raise ValueError("Пустой evaluation universe learned selector validation")

        rows: list[dict] = []
        policies: list[SignalPolicyConfig] = []
        for threshold in config.learned_selector_thresholds:
            for cooldown in config.cooldown_options:
                policy = SignalPolicyConfig(
                    cooldown_days=int(cooldown),
                    max_signals_per_7d=config.max_signals_per_7d,
                )
                opportunities = _select_scored(
                    scored_validation,
                    threshold=float(threshold),
                    model_type=self.model_type,
                )
                events = apply_signal_policy(opportunities, policy)
                summary, scored_rows = backtest_signal_stream(
                    events, evaluation_universe=universe
                )
                metrics = _validation_metrics(summary)
                stability = _quarterly_stability(scored_rows)
                coverage = _coverage_metrics(
                    opportunities, tuple(config.currencies), tuple(config.horizons)
                )
                rows.append({
                    "model_type": self.model_type,
                    "probability_threshold": float(threshold),
                    "cooldown_days": int(cooldown),
                    **metrics,
                    **stability,
                    **coverage,
                    "stability_ok": bool(
                        stability["quarterly_lift_p10"]
                        >= config.validation_quarterly_lift_floor
                    ),
                    "frequency_ok": _frequency_is_valid(summary, config),
                    "fit_start": periods.fit_start,
                    "fit_end": periods.fit_end,
                    "calibration_start": periods.calibration_start,
                    "calibration_end": periods.calibration_end,
                    "validation_start": periods.validation_start,
                    "validation_end": periods.validation_end,
                })
                policies.append(policy)

        leaderboard = pd.DataFrame(rows)
        structurally_eligible = leaderboard["frequency_ok"] & leaderboard["coverage_ok"]
        stable = structurally_eligible & leaderboard["stability_ok"]
        selection_mask = stable if stable.any() else structurally_eligible
        eligible_indices = leaderboard.index[selection_mask].tolist()
        if not eligible_indices:
            diagnostic = leaderboard.sort_values(
                ["min_signals_per_week", "geometric_lift"], ascending=False
            ).head(5)
            raise ValueError(
                "На learned-selector validation нет конфигурации с 1–2 "
                "сигналами в неделю для каждой валюты и обоими сценариями.\n"
                f"{diagnostic.to_string(index=False)}"
            )
        winner_index = max(
            eligible_indices,
            key=lambda index: (
                leaderboard.at[index, "min_currency_quarterly_geometric_lift"],
                leaderboard.at[index, "quarterly_geometric_lift"],
                leaderboard.at[index, "quarterly_lift_p10"],
                leaderboard.at[index, "geometric_lift"],
                leaderboard.at[index, "min_currency_lift"],
                leaderboard.at[index, "macro_lift"],
                leaderboard.at[index, "macro_benefit_uplift_bps"],
                -abs(leaderboard.at[index, "mean_signals_per_week"] - 1.5),
            ),
        )
        leaderboard["selected"] = False
        leaderboard.at[winner_index, "selected"] = True
        winner = leaderboard.loc[winner_index]
        fitted = FittedLearnedSelector(
            model_type=self.model_type,
            estimator=estimator,
            calibrator=calibrator,
            threshold=float(winner["probability_threshold"]),
            policy=policies[winner_index],
            periods=periods,
            trained_through=periods.validation_end,
            validation_geometric_lift=float(winner["geometric_lift"]),
            validation_macro_lift=float(winner["macro_lift"]),
            validation_min_currency_lift=float(winner["min_currency_lift"]),
            validation_macro_benefit_uplift_bps=float(
                winner["macro_benefit_uplift_bps"]
            ),
            validation_mean_signals_per_week=float(winner["mean_signals_per_week"]),
            validation_quarterly_geometric_lift=float(
                winner["quarterly_geometric_lift"]
            ),
            validation_min_currency_quarterly_geometric_lift=float(
                winner["min_currency_quarterly_geometric_lift"]
            ),
            validation_quarterly_lift_p10=float(winner["quarterly_lift_p10"]),
            validation_stability_ok=bool(winner["stability_ok"]),
            validation_scenario_coverage=float(winner["scenario_coverage"]),
            validation_horizon_coverage=float(winner["horizon_coverage"]),
        )
        return fitted, leaderboard.sort_values(
            ["selected", "min_currency_quarterly_geometric_lift", "geometric_lift"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    def select(
        self,
        candidates: pd.DataFrame,
        fitted: FittedLearnedSelector,
    ) -> pd.DataFrame:
        scored = _score_candidates(candidates, fitted.estimator, fitted.calibrator)
        return _select_scored(
            scored, threshold=fitted.threshold, model_type=fitted.model_type
        )

    def policy_config(self, fitted: FittedLearnedSelector) -> SignalPolicyConfig:
        return fitted.policy
