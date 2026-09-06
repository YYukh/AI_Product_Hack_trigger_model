"""Causal rolling walk-forward for simple rules and configurable ML scope."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
from .config import YuraPipelineConfig
from .engine_registry import EngineRegistry, default_engine_registry
from .models import (
    ML_FEATURES,
    build_probability_calibrator,
    calibration_frame,
    model_columns,
    validate_feature_schema,
)
from .advanced_indicators import (
    PRODUCTION_ADVANCED_FEATURES, add_production_indicator_features,
)
from .rules import RuleCandidate


KEYS = ("available_at", "currency", "scenario", "target_family", "target", "horizon")


@dataclass
class BaseReplayResult:
    candidates: pd.DataFrame
    audit: pd.DataFrame
    rule_oos_metrics: pd.DataFrame


@dataclass
class _ProbabilityEngine:
    classifier: object
    calibrator: object | None
    include_currency_in_calibration: bool
    include_horizon_in_calibration: bool
    architecture: str


def _target_definitions(
    target_registry: pd.DataFrame,
    config: YuraPipelineConfig,
) -> pd.DataFrame:
    definitions = target_registry.loc[
        target_registry["family"].isin(config.target_families)
        & target_registry["horizon"].isin(config.horizons)
    ].copy()
    duplicates = definitions.duplicated(["family", "horizon"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Альтернативный pipeline ожидает один target на family × horizon"
        )
    expected = len(config.target_families) * len(config.horizons)
    if len(definitions) != expected:
        raise ValueError(f"Ожидалось {expected} targets, получено {len(definitions)}")
    return definitions.sort_values(["family", "horizon"]).reset_index(drop=True)


def _period_starts(start: pd.Timestamp, end: pd.Timestamp, months: int) -> list[pd.Timestamp]:
    values: list[pd.Timestamp] = []
    current = start
    while current <= end:
        values.append(current)
        current += pd.DateOffset(months=months)
    return values


def _calendar_weeks(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    dates = pd.to_datetime(frame["available_at"])
    return ((dates.max() - dates.min()).days + 1) / 7


def _fit_rule_variant(
    train: pd.DataFrame,
    variants: tuple[RuleCandidate, ...],
    *,
    target: str,
    horizon: int,
    config: YuraPipelineConfig,
) -> tuple[RuleCandidate, dict] | None:
    weeks = _calendar_weeks(train)
    currencies = tuple(sorted(train["currency"].unique()))
    if not weeks or not currencies:
        return None
    baseline_by_currency = train.groupby("currency")[target].mean().to_dict()
    benefit_column = f"local_advantage_{horizon}d_bps"
    best: tuple[tuple[float, float, float, int], RuleCandidate, dict] | None = None

    for variant in variants:
        prediction = variant.predict(train)
        per_currency = []
        frequency_ok = True
        for currency in currencies:
            mask = train["currency"].eq(currency).to_numpy()
            selected = prediction & mask
            count = int(selected.sum())
            rate = count / weeks
            if not config.rule_min_signals_per_week <= rate <= config.rule_max_signals_per_week:
                frequency_ok = False
                break
            base = float(baseline_by_currency[currency])
            true_positive = int(
                train.loc[selected, target].astype(bool).sum()
            )
            smoothed_precision = (
                true_positive + config.rule_confidence_prior_strength * base
            ) / (count + config.rule_confidence_prior_strength)
            lift = smoothed_precision / base if base > 0 else 0.0
            per_currency.append(lift)
        if not frequency_ok:
            continue
        signal_count = int(prediction.sum())
        target_values = train[target].to_numpy(dtype=bool)
        true_positive = int(np.sum(prediction & target_values))
        pooled_precision = true_positive / signal_count if signal_count else 0.0
        pooled_base = float(train[target].mean())
        pooled_lift = pooled_precision / pooled_base if pooled_base else 0.0
        benefit = pd.to_numeric(
            train.loc[prediction, benefit_column], errors="coerce"
        ).dropna()
        mean_benefit = float(benefit.mean()) if len(benefit) else -np.inf
        # Robust across currencies first; pooled lift and BPS are tie-breakers.
        score = (float(min(per_currency)), float(np.mean(per_currency)), mean_benefit, -signal_count)
        metrics = {
            "train_signal_count": signal_count,
            "train_true_positive": true_positive,
            "train_precision": pooled_precision,
            "train_random_precision": pooled_base,
            "train_lift": pooled_lift,
            "train_min_currency_smoothed_lift": float(min(per_currency)),
            "train_macro_currency_smoothed_lift": float(np.mean(per_currency)),
            "train_mean_benefit_bps": mean_benefit,
        }
        if best is None or score > best[0]:
            best = (score, variant, metrics)
    if best is None:
        return None
    return best[1], best[2]


def _train_slice(
    data: pd.DataFrame,
    *,
    target: str,
    horizon: int,
    retrain_at: pd.Timestamp,
    train_months: int,
) -> pd.DataFrame:
    dates = pd.to_datetime(data["available_at"])
    # Strictly mature before retrain_at; equality is excluded deliberately.
    mask = (
        dates.ge(retrain_at - pd.DateOffset(months=train_months))
        & dates.add(pd.Timedelta(days=horizon)).lt(retrain_at)
        & data[target].notna()
    )
    return data.loc[mask].sort_values(["available_at", "currency"]).copy()


def _score_slice(
    data: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    target: str,
) -> pd.DataFrame:
    dates = pd.to_datetime(data["available_at"])
    return data.loc[
        dates.ge(start) & dates.lt(end) & data[target].notna()
    ].sort_values(["available_at", "currency"]).copy()


def _version(*parts: object) -> str:
    raw = "|".join(map(str, parts))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _base_candidate(
    frame: pd.DataFrame,
    *,
    definition: object,
    engine_type: str,
    engine_name: str,
    engine_version: str,
    confidence: np.ndarray,
    baseline: float | np.ndarray,
    expected_bps: np.ndarray,
    trained_at: pd.Timestamp,
    trained_through: pd.Timestamp,
) -> pd.DataFrame:
    result = frame.loc[:, ["available_at", "currency"]].copy()
    result["scenario"] = definition.scenario
    result["target_family"] = definition.family
    result["target"] = definition.name
    result["horizon"] = int(definition.horizon)
    result["engine_type"] = engine_type
    result["engine_name"] = engine_name
    result["engine_version"] = engine_version
    result["confidence"] = np.asarray(confidence, dtype=float)
    baseline_values = np.broadcast_to(
        np.asarray(baseline, dtype=float), len(result)
    ).copy()
    result["baseline_probability"] = baseline_values
    result["confidence_lift"] = np.divide(
        result["confidence"].to_numpy(dtype=float), baseline_values,
        out=np.zeros(len(result), dtype=float), where=baseline_values > 0,
    )
    result["expected_bps"] = np.asarray(expected_bps, dtype=float)
    result["trained_at"] = trained_at
    result["trained_through"] = trained_through
    result["target_value"] = frame[definition.name].astype(bool).to_numpy()
    result["benefit_bps"] = pd.to_numeric(
        frame[f"local_advantage_{int(definition.horizon)}d_bps"], errors="coerce"
    ).to_numpy()
    return result


def _causal_rule_confidence(
    candidates: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    config: YuraPipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = candidates.copy()
    rule_mask = result["engine_type"].eq("rule")
    metric_rows: list[dict] = []

    def trailing_counts(
        query_dates: pd.Series,
        event_dates: pd.Series,
        values: pd.Series,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        query = pd.to_datetime(query_dates)
        events = pd.DataFrame({
            "event": pd.to_datetime(event_dates),
            "value": values.astype(int).to_numpy(),
        }).sort_values("event")
        maturity = events["event"] + pd.Timedelta(days=horizon)
        event_ns = events["event"].astype("int64").to_numpy()
        maturity_ns = maturity.astype("int64").to_numpy()
        lower_ns = (
            query - pd.DateOffset(months=config.rule_confidence_months)
        ).astype("int64").to_numpy()
        query_ns = query.astype("int64").to_numpy()
        lower = np.searchsorted(event_ns, lower_ns, side="left")
        # side=left implements strictly matured before the scoring timestamp.
        upper = np.searchsorted(maturity_ns, query_ns, side="left")
        prefix = np.r_[0, events["value"].cumsum().to_numpy(dtype=int)]
        return (upper - lower).astype(int), (prefix[upper] - prefix[lower]).astype(int)

    for engine_name, rows in result.loc[rule_mask].groupby("engine_name", sort=False):
        indices = rows.sort_values("available_at").index
        ordered = result.loc[indices]
        family = str(ordered["target_family"].iloc[0])
        horizon = int(ordered["horizon"].iloc[0])
        target = str(ordered["target"].iloc[0])
        history = ordered.loc[:, ["available_at", "target_value"]].copy()
        history["maturity"] = pd.to_datetime(history["available_at"]) + pd.Timedelta(days=horizon)
        population = universe.loc[
            universe["target"].eq(target), ["available_at", "target_value"]
        ].copy()
        support, true_positive = trailing_counts(
            ordered["available_at"], history["available_at"],
            history["target_value"], horizon,
        )
        observations, positives = trailing_counts(
            ordered["available_at"], population["available_at"],
            population["target_value"], horizon,
        )
        pooled_baseline = np.divide(
            positives, observations,
            out=np.zeros(len(observations), dtype=float), where=observations > 0,
        )
        pooled_confidence = np.divide(
            true_positive + config.rule_confidence_prior_strength * pooled_baseline,
            support + config.rule_confidence_prior_strength,
        )
        pooled_prior = pd.Series(pooled_confidence, index=indices)

        # Currency evidence is more relevant for the final client corridor,
        # while pooled evidence prevents unstable 2/2 estimates. Both layers
        # are causal and use the same trailing matured OOS window.
        for currency, currency_rows in ordered.groupby("currency", sort=False):
            local_indices = currency_rows.sort_values("available_at").index
            local_history = history.loc[ordered["currency"].eq(currency)]
            local_population = universe.loc[
                universe["target"].eq(target) & universe["currency"].eq(currency),
                ["available_at", "target_value"],
            ]
            local_support, local_true_positive = trailing_counts(
                result.loc[local_indices, "available_at"],
                local_history["available_at"], local_history["target_value"], horizon,
            )
            local_observations, local_positives = trailing_counts(
                result.loc[local_indices, "available_at"],
                local_population["available_at"], local_population["target_value"], horizon,
            )
            local_baseline = np.divide(
                local_positives, local_observations,
                out=np.zeros(len(local_observations), dtype=float),
                where=local_observations > 0,
            )
            prior = pooled_prior.loc[local_indices].to_numpy(dtype=float)
            local_confidence = np.divide(
                local_true_positive + config.rule_confidence_prior_strength * prior,
                local_support + config.rule_confidence_prior_strength,
            )
            local_lift = np.divide(
                local_confidence, local_baseline,
                out=np.zeros(len(local_confidence), dtype=float),
                where=local_baseline > 0,
            )
            result.loc[local_indices, "confidence"] = local_confidence
            result.loc[local_indices, "baseline_probability"] = local_baseline
            result.loc[local_indices, "confidence_lift"] = local_lift
            result.loc[local_indices, "confidence_support"] = local_support

        metric_rows.append({
            "engine_name": engine_name,
            "target_family": family,
            "horizon": horizon,
            "oos_signal_count": len(ordered),
            "oos_true_positive": int(ordered["target_value"].sum()),
            "oos_precision": float(ordered["target_value"].mean()),
            "oos_mean_benefit_bps": float(pd.to_numeric(ordered["benefit_bps"], errors="coerce").mean()),
        })
    result["confidence_support"] = result.get("confidence_support", 0).fillna(0).astype(int)
    return result, pd.DataFrame(metric_rows)


def _group_balanced_weights(
    frame: pd.DataFrame,
    groups: tuple[str, ...],
) -> np.ndarray:
    """Give every declared stratum equal total loss mass."""
    counts = frame.groupby(list(groups), sort=False)[groups[0]].transform("size")
    weights = 1.0 / counts.to_numpy(dtype=float)
    return weights / weights.mean()


def _fit_probability_engine(
    training: pd.DataFrame,
    *,
    model_spec: object,
    config: YuraPipelineConfig,
    period_start: pd.Timestamp,
    mode: str,
) -> _ProbabilityEngine | None:
    """Fit one model and a strictly later temporal probability calibrator."""
    if training.empty or training["_target_value"].nunique() < 2:
        return None
    calibration_start = period_start - pd.DateOffset(months=config.retrain_months)
    dates = pd.to_datetime(training["available_at"])
    maturity = dates + pd.to_timedelta(training["horizon"].astype(int), unit="D")
    fit_part = training.loc[maturity.lt(calibration_start)].copy()
    calibration_part = training.loc[
        dates.ge(calibration_start) & maturity.lt(period_start)
    ].copy()
    can_calibrate = (
        not fit_part.empty
        and not calibration_part.empty
        and fit_part["_target_value"].nunique() == 2
        and calibration_part["_target_value"].nunique() == 2
    )
    fit_rows = fit_part if can_calibrate else training
    classifier = model_spec.builder(config)
    fit_kwargs = {}
    if mode != "pooled":
        balance_groups = (
            ("currency", "horizon") if mode == "hybrid" else ("horizon",)
        )
        fit_kwargs["model__sample_weight"] = _group_balanced_weights(
            fit_rows, balance_groups
        )
    classifier.fit(
        fit_rows.loc[:, list(model_columns())],
        fit_rows["_target_value"],
        **fit_kwargs,
    )

    include_currency = mode == "hybrid"
    include_horizon = mode in {"hybrid", "per_currency"}
    calibrator = None
    if can_calibrate:
        raw = classifier.predict_proba(
            calibration_part.loc[:, list(model_columns())]
        )[:, 1]
        calibrator = build_probability_calibrator(
            config,
            include_currency=include_currency,
            include_horizon=include_horizon,
        )
        calibrator.fit(
            calibration_frame(
                raw,
                calibration_part,
                include_currency=include_currency,
                include_horizon=include_horizon,
            ),
            calibration_part["_target_value"],
        )
    architecture = {
        "pooled": "pooled_currency_and_horizon",
        "hybrid": "pooled_balanced_with_contextual_calibration",
        "per_currency": "per_currency_pooled_horizons",
    }[mode]
    return _ProbabilityEngine(
        classifier=classifier,
        calibrator=calibrator,
        include_currency_in_calibration=include_currency,
        include_horizon_in_calibration=include_horizon,
        architecture=architecture,
    )


def _predict_probability(
    fitted: _ProbabilityEngine,
    frame: pd.DataFrame,
) -> np.ndarray:
    raw = fitted.classifier.predict_proba(
        frame.loc[:, list(model_columns())]
    )[:, 1]
    if fitted.calibrator is None:
        return raw
    calibration = calibration_frame(
        raw,
        frame,
        include_currency=fitted.include_currency_in_calibration,
        include_horizon=fitted.include_horizon_in_calibration,
    )
    return fitted.calibrator.predict_proba(calibration)[:, 1]


def _local_training_baseline(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    target: str,
    prior_strength: float,
) -> np.ndarray:
    """Smoothed train prevalence for the score row's own currency stratum."""
    pooled = float(train[target].mean())
    statistics = train.groupby("currency", sort=False)[target].agg(["sum", "count"])
    numerator = statistics["sum"] + prior_strength * pooled
    denominator = statistics["count"] + prior_strength
    local = (numerator / denominator).to_dict()
    return score["currency"].map(local).fillna(pooled).to_numpy(dtype=float)


def replay_base_engines(
    data: pd.DataFrame,
    *,
    target_registry: pd.DataFrame,
    config: YuraPipelineConfig = YuraPipelineConfig(),
    engine_registry: EngineRegistry | None = None,
) -> BaseReplayResult:
    """Generate causal OOS rule and ML candidates through rolling WF."""
    registry = engine_registry or default_engine_registry()
    prepared_input = data
    if set(PRODUCTION_ADVANCED_FEATURES).difference(prepared_input.columns):
        prepared_input = add_production_indicator_features(prepared_input)
    required = {"available_at", "currency", *ML_FEATURES}
    if missing := required.difference(prepared_input.columns):
        raise KeyError(f"В data нет полей: {sorted(missing)}")
    validate_feature_schema(set(prepared_input.columns))
    prepared = prepared_input.loc[
        prepared_input["currency"].isin(config.currencies)
    ].copy()
    prepared["available_at"] = pd.to_datetime(prepared["available_at"])
    definitions = _target_definitions(target_registry, config)
    last_date = prepared["available_at"].max()
    periods = _period_starts(
        pd.Timestamp(config.base_oos_start), last_date, config.retrain_months
    )
    candidate_frames: list[pd.DataFrame] = []
    audit_rows: list[dict] = []

    for period_start in periods:
        period_end = period_start + pd.DateOffset(months=config.retrain_months)
        contexts: list[tuple[object, pd.DataFrame, pd.DataFrame]] = []
        for definition in definitions.itertuples(index=False):
            horizon = int(definition.horizon)
            train = _train_slice(
                prepared, target=definition.name, horizon=horizon,
                retrain_at=period_start, train_months=config.train_months,
            ).dropna(subset=list(ML_FEATURES))
            score = _score_slice(
                prepared, start=period_start, end=period_end, target=definition.name
            ).dropna(subset=list(ML_FEATURES))
            if train.empty or score.empty or train[definition.name].nunique() < 2:
                audit_rows.append({
                    "retrain_at": period_start, "target_family": definition.family,
                    "horizon": horizon, "engine_type": "all", "fitted": False,
                    "reason": "insufficient_data",
                })
                continue
            contexts.append((definition, train, score))

        if not contexts:
            continue

        columns = list(model_columns())
        # One benefit model shares statistical strength across all horizons.
        # Each stacked row still uses its own fully matured horizon label.
        benefit_frames = []
        seen_horizons: set[int] = set()
        for definition, train, _ in contexts:
            horizon = int(definition.horizon)
            if horizon in seen_horizons:
                continue
            seen_horizons.add(horizon)
            benefit_column = f"local_advantage_{horizon}d_bps"
            frame = train.dropna(subset=[benefit_column]).copy()
            frame["horizon"] = str(horizon)
            frame["_benefit_target"] = frame[benefit_column].astype(float)
            benefit_frames.append(frame)
        benefit_training = pd.concat(benefit_frames, ignore_index=True)
        # Predict incremental value relative to a random entry from the same
        # currency × horizon on this train window. This matches final BPS uplift.
        benefit_training["_random_benefit_baseline"] = (
            benefit_training.groupby(["currency", "horizon"], sort=False)["_benefit_target"]
            .transform("mean")
        )
        benefit_target = (
            benefit_training["_benefit_target"]
            - benefit_training["_random_benefit_baseline"]
        )
        lower, upper = benefit_target.quantile([0.01, 0.99])
        benefit_regressor = registry.benefit_model_builder(config)
        benefit_fit_kwargs = {}
        if config.ml_scope != "pooled":
            benefit_fit_kwargs["model__sample_weight"] = _group_balanced_weights(
                benefit_training, ("currency", "horizon")
            )
        benefit_regressor.fit(
            benefit_training.loc[:, columns], benefit_target.clip(lower, upper),
            **benefit_fit_kwargs,
        )

        # Every registered probability engine follows the configured scope.
        # New models do not change candidate, selector or policy contracts.
        family_models: dict[tuple[str, str, str | None], _ProbabilityEngine] = {}
        family_versions: dict[tuple[str, str, str | None], str] = {}
        for model_spec in registry.ml_models:
            for family in config.target_families:
                family_frames = []
                for definition, train, _ in contexts:
                    if definition.family != family:
                        continue
                    frame = train.copy()
                    frame["horizon"] = str(int(definition.horizon))
                    frame["_target_value"] = frame[definition.name].astype(int)
                    family_frames.append(frame)
                family_training = pd.concat(family_frames, ignore_index=True)
                if family_training["_target_value"].nunique() < 2:
                    continue
                if config.ml_scope == "per_currency":
                    # The hybrid model is a deterministic fallback when one
                    # currency has insufficient mature observations/classes.
                    scopes = [(None, family_training, "hybrid")]
                    scopes.extend(
                        (str(currency), rows.copy(), "per_currency")
                        for currency, rows in family_training.groupby(
                            "currency", sort=True
                        )
                    )
                else:
                    scopes = [(None, family_training, config.ml_scope)]
                for currency, training_rows, mode in scopes:
                    fitted_model = _fit_probability_engine(
                        training_rows,
                        model_spec=model_spec,
                        config=config,
                        period_start=period_start,
                        mode=mode,
                    )
                    if fitted_model is None:
                        continue
                    key = (model_spec.name, family, currency)
                    family_models[key] = fitted_model
                    family_versions[key] = _version(
                        model_spec.name, family, currency or "all_currencies",
                        "all_horizons", config.ml_scope, period_start,
                    )

        for definition, train, score in contexts:
            horizon = int(definition.horizon)
            score_for_model = score.copy()
            score_for_model["horizon"] = str(horizon)
            expected_bps = benefit_regressor.predict(
                score_for_model.loc[:, columns]
            )
            expected_bps_by_index = pd.Series(expected_bps, index=score.index)
            pooled_baseline = float(train[definition.name].mean())
            for model_spec in registry.ml_models:
                score_groups = (
                    list(score_for_model.groupby("currency", sort=True))
                    if config.ml_scope == "per_currency"
                    else [(None, score_for_model)]
                )
                for currency, score_group in score_groups:
                    currency_key = str(currency) if currency is not None else None
                    model_key = (model_spec.name, definition.family, currency_key)
                    if model_key not in family_models:
                        model_key = (model_spec.name, definition.family, None)
                    if model_key not in family_models:
                        continue
                    fitted_model = family_models[model_key]
                    ml_probability = _predict_probability(fitted_model, score_group)
                    baseline = _local_training_baseline(
                        train,
                        score_group,
                        target=definition.name,
                        prior_strength=config.rule_confidence_prior_strength,
                    )
                    ml_version = family_versions[model_key]
                    engine_suffix = (
                        f"{currency_key}_all_horizons"
                        if currency_key is not None else "all_horizons"
                    )
                    engine_name = (
                        f"{model_spec.name}_{definition.family}_{engine_suffix}"
                    )
                    local_train = (
                        train.loc[train["currency"].eq(currency_key)]
                        if currency_key is not None else train
                    )
                    trained_through = pd.Timestamp(local_train["available_at"].max())
                    candidate_frames.append(_base_candidate(
                        score_group, definition=definition, engine_type="ml",
                        engine_name=engine_name, engine_version=ml_version,
                        confidence=ml_probability, baseline=baseline,
                        expected_bps=expected_bps_by_index.loc[
                            score_group.index
                        ].to_numpy(),
                        trained_at=period_start,
                        trained_through=trained_through,
                    ))
                    audit_rows.append({
                        "retrain_at": period_start,
                        "target_family": definition.family,
                        "horizon": horizon, "engine_type": "ml",
                        "engine_name": model_spec.name, "currency": currency_key,
                        "fitted": True,
                        "architecture": fitted_model.architecture,
                        "probability_calibration": (
                            "temporal_contextual_platt_last_refit_block"
                            if fitted_model.calibrator is not None else "none"
                        ),
                        "baseline_scope": "currency_target_horizon_shrunk",
                        "train_start": local_train["available_at"].min(),
                        "trained_through": trained_through,
                        "train_observations": len(local_train),
                        "train_positive_rate": float(local_train[definition.name].mean()),
                        "score_observations": len(score_group),
                        "engine_version": ml_version,
                    })

            rule_trained_through = pd.Timestamp(train["available_at"].max())
            for rule_name, variants in registry.rule_library.items():
                fitted = _fit_rule_variant(
                    train, variants, target=definition.name, horizon=horizon, config=config
                )
                if fitted is None:
                    audit_rows.append({
                        "retrain_at": period_start, "target_family": definition.family,
                        "horizon": horizon, "engine_type": "rule",
                        "engine_name": rule_name, "fitted": False,
                        "reason": "frequency_constraint",
                    })
                    continue
                variant, metrics = fitted
                prediction = variant.predict(score)
                fired = score.loc[prediction].copy()
                rule_version = _version("rule", definition.family, horizon, variant.variant_id, period_start)
                if not fired.empty:
                    candidate_frames.append(_base_candidate(
                        fired, definition=definition, engine_type="rule",
                        engine_name=f"{rule_name}:{definition.family}:h{horizon}",
                        engine_version=rule_version,
                        confidence=np.full(len(fired), np.nan), baseline=pooled_baseline,
                        expected_bps=expected_bps[prediction],
                        trained_at=period_start, trained_through=rule_trained_through,
                    ))
                audit_rows.append({
                    "retrain_at": period_start, "target_family": definition.family,
                    "horizon": horizon, "engine_type": "rule",
                    "engine_name": rule_name, "fitted": True,
                    "selected_variant": variant.variant_id,
                    "train_start": train["available_at"].min(),
                    "trained_through": rule_trained_through,
                    "score_observations": len(score), "score_signal_count": int(prediction.sum()),
                    "engine_version": rule_version, **metrics,
                })

    if not candidate_frames:
        raise ValueError("Base replay не создал ни одного candidate")
    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates["available_at"] = pd.to_datetime(candidates["available_at"])

    universe_frames = []
    for definition in definitions.itertuples(index=False):
        frame = prepared.loc[prepared[definition.name].notna(), ["available_at", "currency", definition.name]].rename(
            columns={definition.name: "target_value"}
        )
        frame["target"] = definition.name
        universe_frames.append(frame)
    confidence_universe = pd.concat(universe_frames, ignore_index=True)
    candidates, rule_metrics = _causal_rule_confidence(
        candidates, confidence_universe, config=config
    )
    candidates = candidates.sort_values(
        ["available_at", "currency", "target_family", "horizon", "engine_type", "engine_name"]
    ).reset_index(drop=True)
    return BaseReplayResult(
        candidates=candidates,
        audit=pd.DataFrame(audit_rows),
        rule_oos_metrics=rule_metrics,
    )
