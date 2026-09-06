"""Fixed ML estimators and causal probability calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .advanced_indicators import PRODUCTION_ADVANCED_FEATURES
from .config import YuraPipelineConfig
from .rules import RULE_FEATURES


EXTRA_FEATURES = (
    "zscore_30d", "zscore_60d", "zscore_90d", "zscore_180d",
    "distance_from_high_30d_bps", "distance_from_high_90d_bps",
    "consecutive_down", "consecutive_up",
    "momentum_change_1d_5d_bps", "acceleration_1d_bps",
    "days_since_local_min_30d", "absolute_return_1d_bps",
    "absolute_return_5d_bps", "rolling_std_5d_bps",
    "rolling_std_7d_bps", "rolling_std_20d_bps", "rolling_std_30d_bps",
    "month_sin", "month_cos", "month_start", "month_end",
    "recipient_days_to_holiday_30", "recipient_days_since_holiday_30",
    "russia_days_to_holiday_30", "russia_days_since_holiday_30",
    "usd_return_1d_bps", "usd_return_5d_bps", "usd_return_20d_bps",
    "eur_return_1d_bps", "eur_return_5d_bps", "eur_return_20d_bps",
    "cny_return_1d_bps", "cny_return_5d_bps", "cny_return_20d_bps",
)
ML_FEATURES = tuple(dict.fromkeys((
    *RULE_FEATURES, *EXTRA_FEATURES, *PRODUCTION_ADVANCED_FEATURES,
)))


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", "passthrough", list(ML_FEATURES)),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["currency", "horizon"],
        ),
    ])


def build_classifier(config: YuraPipelineConfig) -> Pipeline:
    model = HistGradientBoostingClassifier(
        learning_rate=config.ml_learning_rate,
        max_iter=config.ml_max_iter,
        max_leaf_nodes=config.ml_max_leaf_nodes,
        min_samples_leaf=config.ml_min_samples_leaf,
        l2_regularization=config.ml_l2_regularization,
        class_weight="balanced",
        random_state=config.random_state,
    )
    return Pipeline([("features", _preprocessor()), ("model", model)])


def build_benefit_regressor(config: YuraPipelineConfig) -> Pipeline:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=config.ml_learning_rate,
        max_iter=config.ml_max_iter,
        max_leaf_nodes=config.ml_max_leaf_nodes,
        min_samples_leaf=config.ml_min_samples_leaf,
        l2_regularization=config.ml_l2_regularization,
        random_state=config.random_state,
    )
    return Pipeline([("features", _preprocessor()), ("model", model)])


def calibration_frame(
    raw_probability: np.ndarray,
    context: pd.DataFrame,
    *,
    include_currency: bool,
    include_horizon: bool,
) -> pd.DataFrame:
    """Build a calibration frame from information known at prediction time."""
    bounded = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    result = pd.DataFrame({
        "raw_logit": np.log(bounded / (1.0 - bounded)),
    }, index=context.index)
    if include_currency:
        result["currency"] = context["currency"].astype("string")
    if include_horizon:
        result["horizon"] = context["horizon"].astype("string")
    return result


def build_probability_calibrator(
    config: YuraPipelineConfig,
    *,
    include_currency: bool,
    include_horizon: bool,
) -> Pipeline:
    """Regularized temporal Platt calibration with optional group offsets."""
    transformers: list[tuple[str, object, list[str]]] = [
        ("raw_score", StandardScaler(), ["raw_logit"]),
    ]
    categories = []
    if include_currency:
        categories.append("currency")
    if include_horizon:
        categories.append("horizon")
    if categories:
        transformers.append((
            "context",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categories,
        ))
    return Pipeline([
        ("features", ColumnTransformer(transformers)),
        ("model", LogisticRegression(
            C=1.0, max_iter=1_000, solver="lbfgs",
            random_state=config.random_state,
        )),
    ])


def model_columns() -> tuple[str, ...]:
    return (*ML_FEATURES, "currency", "horizon")


def validate_feature_schema(columns: set[str]) -> None:
    missing = set(ML_FEATURES).difference(columns)
    if missing:
        raise KeyError(f"Не хватает ML features: {sorted(missing)}")
    forbidden = [name for name in ML_FEATURES if name.startswith("target_")]
    if forbidden:
        raise ValueError(f"Target leakage in ML features: {forbidden}")
