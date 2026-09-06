"""Nested temporal tests of compact supervised models as signal indicators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .advanced_indicators import ADVANCED_FEATURES
from .indicator_discovery import (
    IndicatorDiscoveryConfig, IndicatorDiscoveryResult,
    _fold_record, _weeks, build_discovery_temporal_plan,
    summarize_discovery_folds,
)
from .models import ML_FEATURES


MODEL_INDICATOR_SPECS = (
    "logistic_all",
    "hgb_core",
    "hgb_all",
    "extra_trees_all",
)


def _features_for(specification: str) -> tuple[str, ...]:
    if specification == "hgb_core":
        return tuple(ML_FEATURES)
    if specification in {"logistic_all", "hgb_all", "extra_trees_all"}:
        return tuple(dict.fromkeys((*ML_FEATURES, *ADVANCED_FEATURES)))
    raise ValueError(f"Неизвестная model-indicator specification: {specification}")


def _model(specification: str, features: tuple[str, ...], random_state: int) -> Pipeline:
    categorical = ["currency"]
    if specification == "logistic_all":
        numeric = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        estimator = LogisticRegression(
            max_iter=2_000, class_weight="balanced", random_state=random_state
        )
    elif specification in {"hgb_core", "hgb_all"}:
        numeric = SimpleImputer(strategy="median")
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=80, max_leaf_nodes=7,
            min_samples_leaf=30, l2_regularization=2.0,
            class_weight="balanced", random_state=random_state,
        )
    elif specification == "extra_trees_all":
        numeric = SimpleImputer(strategy="median")
        estimator = ExtraTreesClassifier(
            n_estimators=150, max_depth=8, min_samples_leaf=10,
            max_features=0.8, class_weight="balanced", n_jobs=-1,
            random_state=random_state,
        )
    else:
        raise ValueError(specification)
    preprocessor = ColumnTransformer([
        ("numeric", numeric, list(features)),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical,
        ),
    ])
    return Pipeline([("features", preprocessor), ("model", estimator)])


def _calibrator(raw_probability: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
    bounded = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return pd.DataFrame({
        "raw_logit": np.log(bounded / (1.0 - bounded)),
        "currency": frame["currency"].astype("string").to_numpy(),
    }, index=frame.index)


def _fit_calibrator(
    raw_probability: np.ndarray,
    frame: pd.DataFrame,
    target: pd.Series,
    random_state: int,
) -> Pipeline | None:
    if frame.empty or target.nunique() < 2:
        return None
    result = Pipeline([
        ("features", ColumnTransformer([
            ("score", StandardScaler(), ["raw_logit"]),
            (
                "currency", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["currency"],
            ),
        ])),
        ("model", LogisticRegression(
            C=1.0, max_iter=1_000, random_state=random_state
        )),
    ])
    result.fit(_calibrator(raw_probability, frame), target.astype(int))
    return result


def _predict_probability(
    estimator: Pipeline,
    calibrator: Pipeline | None,
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> np.ndarray:
    columns = [*features, "currency"]
    raw = estimator.predict_proba(frame.loc[:, columns])[:, 1]
    if calibrator is None:
        return raw
    return calibrator.predict_proba(_calibrator(raw, frame))[:, 1]


def _choose_threshold(
    validation: pd.DataFrame,
    probability: np.ndarray,
    *,
    target: str,
    benefit_column: str,
    config: IndicatorDiscoveryConfig,
) -> tuple[float, dict] | None:
    quantiles = np.linspace(0.05, 0.95, 37)
    thresholds = np.unique(np.r_[
        0.05, 0.95, np.quantile(probability, quantiles)
    ])
    candidates = []
    for threshold in thresholds:
        selected = probability >= threshold
        currency_metrics = []
        feasible = True
        for currency, group in validation.groupby("currency", sort=True):
            positions = validation.index.get_indexer(group.index)
            prediction = selected[positions]
            count = int(prediction.sum())
            frequency = count / max(_weeks(group["available_at"]), 1e-9)
            if not config.min_train_signals_per_week <= frequency <= config.max_train_signals_per_week:
                feasible = False
                break
            y = group[target].astype(bool).to_numpy()
            baseline = float(y.mean())
            true_positive = int(np.sum(prediction & y))
            smoothed = (
                true_positive + config.precision_prior_strength * baseline
            ) / (count + config.precision_prior_strength)
            lift = smoothed / baseline if baseline > 0 else 0.0
            benefit = pd.to_numeric(
                group.loc[prediction, benefit_column], errors="coerce"
            ).dropna()
            currency_metrics.append((
                lift, float(benefit.mean()) if len(benefit) else -np.inf,
                frequency,
            ))
        if not feasible or not currency_metrics:
            continue
        mean_frequency = float(np.mean([item[2] for item in currency_metrics]))
        candidates.append({
            "threshold": float(threshold),
            "min_lift": min(item[0] for item in currency_metrics),
            "mean_lift": float(np.mean([item[0] for item in currency_metrics])),
            "mean_benefit": float(np.mean([item[1] for item in currency_metrics])),
            "mean_frequency": mean_frequency,
        })
    if not candidates:
        return None
    winner = max(candidates, key=lambda item: (
        item["min_lift"], item["mean_lift"], item["mean_benefit"],
        -abs(item["mean_frequency"] - 1.0), item["threshold"],
    ))
    return winner["threshold"], {
        "smoothed_lift": winner["mean_lift"],
        "signals_per_week": winner["mean_frequency"],
        "min_currency_smoothed_lift": winner["min_lift"],
        "mean_benefit_bps": winner["mean_benefit"],
    }


def run_model_indicator_discovery(
    data: pd.DataFrame,
    *,
    target_registry: pd.DataFrame,
    currencies: tuple[str, ...],
    model_specs: tuple[str, ...] = MODEL_INDICATOR_SPECS,
    config: IndicatorDiscoveryConfig = IndicatorDiscoveryConfig(),
    random_state: int = 42,
) -> IndicatorDiscoveryResult:
    """Nested WF: estimator fit -> calibration -> threshold -> OOS test."""
    all_features = sorted({feature for spec in model_specs for feature in _features_for(spec)})
    required = {"available_at", "currency", *all_features, *target_registry["name"]}
    if missing := required.difference(data.columns):
        raise KeyError(f"Model discovery data не содержит: {sorted(missing)}")
    prepared = data.loc[data["currency"].isin(currencies)].copy()
    prepared["available_at"] = pd.to_datetime(prepared["available_at"])
    prepared = prepared.sort_values(["available_at", "currency"]).reset_index(drop=True)
    plan = build_discovery_temporal_plan(prepared, config)
    dates = prepared["available_at"]
    rows: list[dict] = []

    for definition in target_registry.itertuples(index=False):
        target = str(definition.name)
        horizon = int(definition.horizon)
        benefit_column = f"local_advantage_{horizon}d_bps"
        maturity = dates.add(pd.Timedelta(days=horizon))
        for fold in plan.itertuples(index=False):
            calibration_start = fold.test_start - pd.DateOffset(months=12)
            threshold_start = fold.test_start - pd.DateOffset(months=6)
            train_floor = fold.test_start - pd.DateOffset(months=config.train_months)
            known = prepared[target].notna()
            fit_mask = (
                dates.ge(train_floor) & maturity.lt(calibration_start) & known
            )
            calibration_mask = (
                dates.ge(calibration_start) & dates.lt(threshold_start)
                & maturity.lt(threshold_start) & known
            )
            validation_mask = (
                dates.ge(threshold_start) & dates.lt(fold.test_start)
                & maturity.lt(fold.test_start) & known
            )
            test_mask = (
                dates.ge(fold.test_start) & dates.lt(fold.test_end)
                & maturity.lt(fold.discovery_end) & known
            )
            fit = prepared.loc[fit_mask].copy()
            calibration = prepared.loc[calibration_mask].copy()
            validation = prepared.loc[validation_mask].copy()
            test = prepared.loc[test_mask].copy()
            if (
                min(len(fit), len(calibration), len(validation), len(test)) == 0
                or fit[target].nunique() < 2
            ):
                continue
            for specification in model_specs:
                features = _features_for(specification)
                columns = [*features, "currency"]
                estimator = _model(specification, features, random_state)
                estimator.fit(fit.loc[:, columns], fit[target].astype(int))
                calibration_raw = estimator.predict_proba(
                    calibration.loc[:, columns]
                )[:, 1]
                calibrator = _fit_calibrator(
                    calibration_raw, calibration, calibration[target], random_state
                )
                validation_probability = _predict_probability(
                    estimator, calibrator, validation, features
                )
                selected = _choose_threshold(
                    validation, validation_probability, target=target,
                    benefit_column=benefit_column, config=config,
                )
                if selected is None:
                    continue
                threshold, train_metrics = selected
                test_probability = _predict_probability(
                    estimator, calibrator, test, features
                )
                test_prediction = test_probability >= threshold
                for currency in currencies:
                    local = test["currency"].eq(currency).to_numpy()
                    if not local.any():
                        continue
                    local_frame = test.loc[local]
                    base = {
                        "fold_id": int(fold.fold_id),
                        "train_start": fold.train_start,
                        "test_start": fold.test_start,
                        "test_end": fold.test_end,
                        "currency": currency,
                        "target_family": str(definition.family),
                        "target": target,
                        "horizon": horizon,
                    }
                    rows.append(_fold_record(
                        base=base, strategy_kind="model",
                        strategy_name=specification, logic="MODEL",
                        selected_spec=f"{specification}:p>={threshold:.6f}",
                        train_metrics=pd.Series(train_metrics),
                        test_prediction=test_prediction[local],
                        test_target=local_frame[target].astype(bool).to_numpy(),
                        test_benefit=pd.to_numeric(
                            local_frame[benefit_column], errors="coerce"
                        ).to_numpy(float),
                        test_weeks=_weeks(local_frame["available_at"]),
                    ))

    folds = pd.DataFrame(rows)
    if folds.empty:
        raise ValueError("Model discovery не создал ни одного OOS результата")
    leaderboard, best, family_summary = summarize_discovery_folds(
        folds, min_oos_signals=config.min_oos_signals
    )
    catalogue = pd.DataFrame({
        "model_indicator": list(model_specs),
        "feature_set": [
            "core" if spec == "hgb_core" else "core+advanced"
            for spec in model_specs
        ],
        "feature_count": [len(_features_for(spec)) for spec in model_specs],
        "confidence_method": "temporal_contextual_platt_probability",
    })
    return IndicatorDiscoveryResult(
        temporal_plan=plan, catalogue=catalogue, fold_results=folds,
        leaderboard=leaderboard, best_by_configuration=best,
        family_summary=family_summary,
    )
