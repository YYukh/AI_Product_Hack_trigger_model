"""Stateful daily scoring pipeline shared by production and historical replay."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .indicator_optimization import fit_indicator_space
from .indicators import IndicatorCandidate, prediction_matrix
from .meta_model import (
    ConfidenceFilterConfig,
    MetaModel,
    confidence_filter_meta_model,
    market_event_records,
    run_meta_model,
)
from .ml_backtest import MLIndicatorConfig, fit_calibrated_ml_indicator
from .production_config import FixedIndicatorConfig


@dataclass
class EngineState:
    """Latest deployable artifact and its retraining schedule."""

    engine_id: str
    engine_type: str
    currency: str
    scenario: str
    target_family: str
    target: str
    horizon: int
    architecture: str
    retrain_months: int
    train_months: int
    next_retrain_at: pd.Timestamp
    feature_names: tuple[str, ...] = ()
    trained_at: pd.Timestamp | None = None
    trained_through: pd.Timestamp | None = None
    model_version: str | None = None
    configuration_version: str | None = None
    last_retrain_attempt_at: pd.Timestamp | None = None
    last_retrain_error: str | None = None
    payload: Any = None
    decision_threshold: float = 1.0
    confidence: float = 0.0
    confidence_support: int = 0
    baseline_probability: float = 0.0
    confidence_lift: float = 0.0

    @property
    def ready(self) -> bool:
        return self.payload is not None and self.trained_at is not None


@dataclass
class ReplayResult:
    states: dict[str, EngineState]
    raw_signals: pd.DataFrame
    final_events: pd.DataFrame
    training_audit: pd.DataFrame


@dataclass
class DailyPipelineResult:
    """Result of the single production operation for one market date."""

    raw_signals: list[dict]
    final_events: pd.DataFrame
    training_audit: list[dict]

    def final_event_records(self) -> list[dict]:
        return market_event_records(self.final_events)


@dataclass
class EngineReplayResult:
    """Historical output of base engines before any meta-model filtering."""

    states: dict[str, EngineState]
    raw_signals: pd.DataFrame
    training_audit: pd.DataFrame


def engine_state_registry(states: dict[str, EngineState]) -> pd.DataFrame:
    """Human-readable snapshot of deployed artifacts and their schedules."""
    columns = (
        "engine_id", "engine_type", "currency", "target_family", "target",
        "horizon", "architecture", "train_months", "model_version",
        "configuration_version", "trained_at",
        "trained_through", "next_retrain_at", "last_retrain_attempt_at",
        "last_retrain_error", "confidence",
        "confidence_support", "ready",
    )
    rows = [
        {
            **{name: getattr(state, name) for name in columns if name != "ready"},
            "ready": state.ready,
        }
        for state in states.values()
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["engine_type", "currency", "target_family", "horizon"]
    ).reset_index(drop=True)


def initialize_engine_states(
    *,
    rule_configurations: tuple[FixedIndicatorConfig, ...],
    target_registry: pd.DataFrame,
    currencies: tuple[str, ...],
    target_families: tuple[str, ...],
    first_score_date: str | pd.Timestamp,
    train_months: int,
    ml_feature_names: tuple[str, ...],
    ml_model_type: str,
    ml_retrain_months: int,
) -> dict[str, EngineState]:
    """Create empty states; first daily update will train every artifact."""
    if train_months <= 0:
        raise ValueError("train_months должен быть положительным")
    first_due = pd.Timestamp(first_score_date)
    states: dict[str, EngineState] = {}
    for config in rule_configurations:
        engine_id = (
            f"rule:{config.currency}:{config.target_family}:h{config.horizon}"
        )
        if engine_id in states:
            raise ValueError(f"Duplicate engine_id={engine_id}")
        states[engine_id] = EngineState(
            engine_id=engine_id,
            engine_type="rule",
            currency=config.currency,
            scenario=config.scenario,
            target_family=config.target_family,
            target=config.target,
            horizon=config.horizon,
            architecture=config.indicator,
            retrain_months=config.retrain_months,
            train_months=train_months,
            next_retrain_at=first_due,
        )

    definitions = target_registry.loc[
        target_registry["family"].isin(target_families)
    ]
    for currency in currencies:
        for definition in definitions.itertuples(index=False):
            horizon = int(definition.horizon)
            engine_id = f"ml:{currency}:{definition.family}:h{horizon}"
            if engine_id in states:
                raise ValueError(f"Duplicate engine_id={engine_id}")
            states[engine_id] = EngineState(
                engine_id=engine_id,
                engine_type="ml",
                currency=currency,
                scenario=definition.scenario,
                target_family=definition.family,
                target=definition.name,
                horizon=horizon,
                architecture=ml_model_type,
                retrain_months=ml_retrain_months,
                train_months=train_months,
                next_retrain_at=first_due,
                feature_names=ml_feature_names,
            )
    return states


def _next_due(previous_due: pd.Timestamp, as_of: pd.Timestamp, months: int) -> pd.Timestamp:
    next_due = previous_due
    while next_due <= as_of:
        next_due += pd.DateOffset(months=months)
    return next_due


def _version(state: EngineState, trained_at: pd.Timestamp, payload_name: str) -> str:
    value = (
        f"{state.engine_id}|{trained_at.isoformat()}|{payload_name}|"
        f"{state.configuration_version}"
    )
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _configuration_version(
    state: EngineState,
    *,
    indicator_spaces: dict[str, list[IndicatorCandidate]],
    ml_model_type: str,
    ml_model_config: MLIndicatorConfig,
) -> str:
    if state.engine_type == "rule":
        payload = "|".join(
            candidate.name for candidate in indicator_spaces[state.architecture]
        )
    else:
        payload = f"{ml_model_type}|{ml_model_config!r}|{state.feature_names}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _mature_train(data: pd.DataFrame, state: EngineState, as_of: pd.Timestamp) -> pd.DataFrame:
    if state.target not in data.columns:
        raise KeyError(f"В data нет target {state.target!r} для {state.engine_id}")
    dates = pd.to_datetime(data["available_at"])
    return data.loc[
        data["currency"].eq(state.currency)
        & dates.ge(as_of - pd.DateOffset(months=state.train_months))
        & dates.add(pd.Timedelta(days=state.horizon)).lt(as_of)
        & data[state.target].notna()
    ].sort_values("available_at")


def _fit_rule_state(
    state: EngineState,
    *,
    as_of: pd.Timestamp,
    data: pd.DataFrame,
    indicator_spaces: dict[str, list[IndicatorCandidate]],
    min_signals_per_week: float,
) -> dict:
    candidates = indicator_spaces[state.architecture]
    required_features = sorted({
        rule.feature for candidate in candidates for rule in candidate.rules
    })
    train = _mature_train(data, state, as_of).dropna(
        subset=required_features
    )
    if train.empty:
        return {"fitted": False, "reason": "empty_train"}
    predictions = prediction_matrix(train, candidates)
    selected, stats, metrics = fit_indicator_space(
        train[state.target],
        predictions,
        candidates,
        dates=train["available_at"],
        min_signals_per_week=min_signals_per_week,
    )
    if selected is None:
        return {"fitted": False, "reason": "frequency_constraint"}

    candidate = candidates[selected]
    support = int(stats["predicted_positive_count"][selected])
    state.trained_at = as_of
    state.configuration_version = _configuration_version(
        state, indicator_spaces=indicator_spaces, ml_model_type="", ml_model_config=MLIndicatorConfig()
    )
    state.trained_through = pd.Timestamp(train["available_at"].max())
    state.payload = candidate
    state.decision_threshold = 1.0
    state.confidence = float(stats["precision"][selected])
    state.confidence_support = support
    state.baseline_probability = float(metrics["random_precision"])
    state.confidence_lift = float(metrics["uplift"][selected])
    state.model_version = _version(state, as_of, candidate.name)
    state.next_retrain_at = _next_due(
        state.next_retrain_at, as_of, state.retrain_months
    )
    return {
        "fitted": True,
        "selected_candidate": candidate.name,
        "selected_logic": candidate.logic,
        "train_observations": len(train),
        "train_signals": support,
        "train_precision": state.confidence,
        "train_random_precision": state.baseline_probability,
        "train_lift": state.confidence_lift,
    }


def _fit_ml_state(
    state: EngineState,
    *,
    as_of: pd.Timestamp,
    data: pd.DataFrame,
    validation_months: int,
    min_signals_per_week: float,
    model_type: str,
    model_config: MLIndicatorConfig,
) -> dict:
    train = _mature_train(data, state, as_of).dropna(
        subset=list(state.feature_names)
    )
    fitted = fit_calibrated_ml_indicator(
        train,
        target=state.target,
        feature_names=state.feature_names,
        horizon=state.horizon,
        retrain_at=as_of,
        validation_months=validation_months,
        min_signals_per_week=min_signals_per_week,
        model_type=model_type,
        model_config=model_config,
    )
    if fitted is None:
        return {"fitted": False, "reason": "insufficient_train_or_validation"}
    model, threshold, audit = fitted
    state.trained_at = as_of
    state.configuration_version = _configuration_version(
        state, indicator_spaces={}, ml_model_type=model_type, ml_model_config=model_config
    )
    state.trained_through = pd.Timestamp(train["available_at"].max())
    state.payload = model
    state.decision_threshold = float(threshold)
    state.confidence = float(audit["validation_precision"])
    state.confidence_support = int(audit["validation_signal_count"])
    state.baseline_probability = float(audit["validation_random_precision"])
    state.confidence_lift = float(audit["validation_lift"])
    state.model_version = _version(state, as_of, model_type)
    state.next_retrain_at = _next_due(
        state.next_retrain_at, as_of, state.retrain_months
    )
    return {"fitted": True, **audit}


def update_models_if_due(
    *,
    as_of: str | pd.Timestamp,
    data: pd.DataFrame,
    states: dict[str, EngineState],
    indicator_spaces: dict[str, list[IndicatorCandidate]],
    rule_min_signals_per_week: float,
    ml_validation_months: int,
    ml_min_signals_per_week: float,
    ml_model_type: str,
    ml_model_config: MLIndicatorConfig,
) -> list[dict]:
    """Retrain only due artifacts using labels mature strictly before as_of."""
    current = pd.Timestamp(as_of)
    audit_rows = []
    for state in states.values():
        configured_version = _configuration_version(
            state,
            indicator_spaces=indicator_spaces,
            ml_model_type=ml_model_type,
            ml_model_config=ml_model_config,
        )
        if state.configuration_version != configured_version:
            state.next_retrain_at = current
        if current < state.next_retrain_at:
            continue
        previous_due = state.next_retrain_at
        state.last_retrain_attempt_at = current
        if state.engine_type == "rule":
            result = _fit_rule_state(
                state,
                as_of=current,
                data=data,
                indicator_spaces=indicator_spaces,
                min_signals_per_week=rule_min_signals_per_week,
            )
        elif state.engine_type == "ml":
            result = _fit_ml_state(
                state,
                as_of=current,
                data=data,
                validation_months=ml_validation_months,
                min_signals_per_week=ml_min_signals_per_week,
                model_type=ml_model_type,
                model_config=ml_model_config,
            )
        else:
            raise ValueError(f"Unknown engine_type={state.engine_type!r}")
        state.last_retrain_error = None if result["fitted"] else result["reason"]
        audit_rows.append({
            "as_of": current,
            "scheduled_retrain_at": previous_due,
            "engine_id": state.engine_id,
            "engine_type": state.engine_type,
            "currency": state.currency,
            "target_family": state.target_family,
            "target": state.target,
            "horizon": state.horizon,
            "architecture": state.architecture,
            "train_months": state.train_months,
            "model_version": state.model_version,
            "configuration_version": state.configuration_version,
            "trained_through": state.trained_through,
            "next_retrain_at": state.next_retrain_at,
            **result,
        })
    return audit_rows


def _local_scoring_time(as_of: pd.Timestamp) -> pd.Timestamp:
    value = as_of.normalize()
    if value.tzinfo is None:
        value = value.tz_localize("Europe/Moscow")
    else:
        value = value.tz_convert("Europe/Moscow")
    return value + pd.Timedelta(hours=9)


def _json_time(value: pd.Timestamp | None) -> str | None:
    return None if value is None else pd.Timestamp(value).isoformat()


def get_signal(
    *,
    as_of: str | pd.Timestamp,
    feature_snapshot: pd.DataFrame,
    states: dict[str, EngineState],
    validity_hours: int = 24,
) -> list[dict]:
    """Return one JSON-serializable score for every rule and ML artifact."""
    current = pd.Timestamp(as_of)
    event_time = _local_scoring_time(current)
    snapshot = feature_snapshot.loc[
        pd.to_datetime(feature_snapshot["available_at"]).eq(current)
    ]
    by_currency = {
        currency: group.iloc[-1]
        for currency, group in snapshot.groupby("currency", sort=False)
    }
    results = []
    for state in states.values():
        attempted_today = (
            state.last_retrain_attempt_at is not None
            and pd.Timestamp(state.last_retrain_attempt_at) == current
        )
        if state.ready and current >= state.next_retrain_at and not attempted_today:
            raise RuntimeError(
                f"Artifact {state.engine_id} просрочен; сначала вызовите "
                "update_models_if_due"
            )
        row = by_currency.get(state.currency)
        score: float | None = None
        fired = False
        status = "READY"
        if not state.ready:
            status = (
                "RETRAIN_FAILED_NOT_TRAINED"
                if attempted_today and state.last_retrain_error
                else "NOT_TRAINED"
            )
        elif row is None:
            status = "NO_MARKET_DATA"
        elif state.engine_type == "rule":
            candidate: IndicatorCandidate = state.payload
            if row[list({rule.feature for rule in candidate.rules})].isna().any():
                status = "FEATURES_MISSING"
            else:
                fired = bool(candidate.predict(row.to_frame().T)[0])
                score = float(fired)
        else:
            if row[list(state.feature_names)].isna().any():
                status = "FEATURES_MISSING"
            else:
                model_input = row.to_frame().T.loc[
                    :, list(state.feature_names)
                ].astype(float)
                score = float(state.payload.predict_proba(model_input)[0, 1])
                fired = score >= state.decision_threshold

        if state.ready and current >= state.next_retrain_at and attempted_today:
            status = "STALE_AFTER_FAILED_RETRAIN"

        identity = (
            f"{event_time.isoformat()}|{state.engine_id}|{state.model_version}"
        )
        results.append({
            "schema_version": "1.0",
            "signal_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
            "available_at": current.isoformat(),
            "as_of": event_time.isoformat(),
            "currency": state.currency,
            "corridor": f"RUB_{state.currency}",
            "scenario": state.scenario,
            "target_family": state.target_family,
            "target": state.target,
            "horizon": state.horizon,
            "engine_id": state.engine_id,
            "engine_type": state.engine_type,
            "engine_name": state.architecture,
            "engine_version": state.model_version,
            "model_version": state.model_version,
            "configuration_version": state.configuration_version,
            "status": status,
            "signal": fired,
            "raw_score": score,
            "decision_threshold": state.decision_threshold,
            "confidence": state.confidence,
            "confidence_method": (
                "train_precision_of_refitted_rule"
                if state.engine_type == "rule"
                else "past_validation_precision_at_threshold"
            ),
            "confidence_support": state.confidence_support,
            "baseline_probability": state.baseline_probability,
            "confidence_lift": state.confidence_lift,
            "trained_at": _json_time(state.trained_at),
            "trained_through": _json_time(state.trained_through),
            "train_window_months": state.train_months,
            "next_retrain_at": _json_time(state.next_retrain_at),
            "last_retrain_error": state.last_retrain_error,
            "expires_at": (event_time + pd.Timedelta(hours=validity_hours)).isoformat(),
        })
    return results


get_signals = get_signal


def apply_meta_model(
    signals: list[dict] | pd.DataFrame,
    *,
    meta_model: MetaModel = confidence_filter_meta_model,
    meta_config: object = ConfidenceFilterConfig(),
) -> pd.DataFrame:
    """Apply a replaceable meta-model to fired daily engine scores."""
    frame = signals.copy() if isinstance(signals, pd.DataFrame) else pd.DataFrame(signals)
    fired = frame.loc[frame["signal"].astype(bool)].copy()
    for column in ("available_at", "as_of", "expires_at"):
        fired[column] = pd.to_datetime(fired[column])
    return run_meta_model(fired, meta_model=meta_model, config=meta_config)


def filter_signal(
    signals: list[dict] | pd.DataFrame,
    *,
    meta_model: MetaModel = confidence_filter_meta_model,
    meta_config: object = ConfidenceFilterConfig(),
    output: str = "records",
) -> list[dict] | pd.DataFrame:
    """Public JSON interface for the downstream filtering stage."""
    events = apply_meta_model(
        signals, meta_model=meta_model, meta_config=meta_config
    )
    if output == "records":
        return market_event_records(events)
    if output == "frame":
        return events
    raise ValueError("output должен быть 'records' или 'frame'")


def run_signal_day(
    *,
    as_of: str | pd.Timestamp,
    data: pd.DataFrame,
    states: dict[str, EngineState],
    indicator_spaces: dict[str, list[IndicatorCandidate]],
    rule_min_signals_per_week: float,
    ml_validation_months: int,
    ml_min_signals_per_week: float,
    ml_model_type: str,
    ml_model_config: MLIndicatorConfig,
    meta_model: MetaModel = confidence_filter_meta_model,
    meta_config: object = ConfidenceFilterConfig(),
) -> DailyPipelineResult:
    """Run the production sequence once: refresh, score, then filter."""
    audit = update_models_if_due(
        as_of=as_of,
        data=data,
        states=states,
        indicator_spaces=indicator_spaces,
        rule_min_signals_per_week=rule_min_signals_per_week,
        ml_validation_months=ml_validation_months,
        ml_min_signals_per_week=ml_min_signals_per_week,
        ml_model_type=ml_model_type,
        ml_model_config=ml_model_config,
    )
    raw_signals = get_signal(
        as_of=as_of,
        feature_snapshot=data,
        states=states,
    )
    final_events = filter_signal(
        raw_signals,
        meta_model=meta_model,
        meta_config=meta_config,
        output="frame",
    )
    return DailyPipelineResult(raw_signals, final_events, audit)


def replay_daily_pipeline(
    data: pd.DataFrame,
    *,
    states: dict[str, EngineState],
    first_score_date: str | pd.Timestamp,
    indicator_spaces: dict[str, list[IndicatorCandidate]],
    rule_min_signals_per_week: float,
    ml_validation_months: int,
    ml_min_signals_per_week: float,
    ml_model_type: str,
    ml_model_config: MLIndicatorConfig,
    meta_model: MetaModel = confidence_filter_meta_model,
    meta_config: object = ConfidenceFilterConfig(),
) -> ReplayResult:
    """Replay the exact daily production sequence over historical data."""
    dates = pd.to_datetime(data["available_at"])
    scoring_dates = sorted(dates.loc[dates.ge(pd.Timestamp(first_score_date))].unique())
    raw_records: list[dict] = []
    event_frames: list[pd.DataFrame] = []
    audit_records: list[dict] = []
    for as_of in scoring_dates:
        daily = run_signal_day(
            as_of=as_of,
            data=data,
            states=states,
            indicator_spaces=indicator_spaces,
            rule_min_signals_per_week=rule_min_signals_per_week,
            ml_validation_months=ml_validation_months,
            ml_min_signals_per_week=ml_min_signals_per_week,
            ml_model_type=ml_model_type,
            ml_model_config=ml_model_config,
            meta_model=meta_model,
            meta_config=meta_config,
        )
        audit_records.extend(daily.training_audit)
        raw_records.extend(daily.raw_signals)
        event_frames.append(daily.final_events)
    return ReplayResult(
        states=states,
        raw_signals=pd.DataFrame(raw_records),
        final_events=(
            pd.concat(event_frames, ignore_index=True)
            if event_frames else pd.DataFrame()
        ),
        training_audit=pd.DataFrame(audit_records),
    )


def replay_engine_signals(
    data: pd.DataFrame,
    *,
    states: dict[str, EngineState],
    first_score_date: str | pd.Timestamp,
    indicator_spaces: dict[str, list[IndicatorCandidate]],
    rule_min_signals_per_week: float,
    ml_validation_months: int,
    ml_min_signals_per_week: float,
    ml_model_type: str,
    ml_model_config: MLIndicatorConfig,
) -> EngineReplayResult:
    """Replay only base engines; meta-model is deliberately not called."""
    dates = pd.to_datetime(data["available_at"])
    scoring_dates = sorted(
        dates.loc[dates.ge(pd.Timestamp(first_score_date))].unique()
    )
    raw_records: list[dict] = []
    audit_records: list[dict] = []
    for as_of in scoring_dates:
        audit_records.extend(update_models_if_due(
            as_of=as_of,
            data=data,
            states=states,
            indicator_spaces=indicator_spaces,
            rule_min_signals_per_week=rule_min_signals_per_week,
            ml_validation_months=ml_validation_months,
            ml_min_signals_per_week=ml_min_signals_per_week,
            ml_model_type=ml_model_type,
            ml_model_config=ml_model_config,
        ))
        raw_records.extend(get_signal(
            as_of=as_of,
            feature_snapshot=data,
            states=states,
        ))
    return EngineReplayResult(
        states=states,
        raw_signals=pd.DataFrame(raw_records),
        training_audit=pd.DataFrame(audit_records),
    )


def save_engine_states(states: dict[str, EngineState], path: str | Path) -> None:
    """Persist latest rule parameters, ML weights and schedule metadata."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(states, destination)


def load_engine_states(path: str | Path) -> dict[str, EngineState]:
    return joblib.load(Path(path))
