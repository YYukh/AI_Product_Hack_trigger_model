"""Сменяемый meta-model интерфейс и минимальный confidence filter."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


META_GROUP_COLUMNS = (
    "as_of", "available_at", "currency", "corridor", "scenario",
    "target_family", "target", "horizon",
)


@dataclass(frozen=True)
class ConfidenceFilterConfig:
    min_confidence: float = 0.70
    min_support: int = 10
    min_sources: int = 1
    rule_min_confidence: float | None = None
    ml_min_confidence: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence должен лежать в [0, 1]")
        for name, value in (
            ("rule_min_confidence", self.rule_min_confidence),
            ("ml_min_confidence", self.ml_min_confidence),
        ):
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} должен лежать в [0, 1]")
        if self.min_support < 0:
            raise ValueError("min_support не может быть отрицательным")
        if self.min_sources <= 0:
            raise ValueError("min_sources должен быть положительным")


MetaModel = Callable[[pd.DataFrame, object], pd.DataFrame]


LOGISTIC_NUMERIC_FEATURES = (
    "rule_signal", "rule_raw_score", "rule_confidence", "rule_support",
    "rule_lift", "ml_signal", "ml_probability", "ml_margin",
    "ml_confidence", "ml_support", "ml_lift", "source_count", "agreement",
)
LOGISTIC_CATEGORICAL_FEATURES = ("currency", "target_family", "horizon")
LOGISTIC_FEATURES = (*LOGISTIC_NUMERIC_FEATURES, *LOGISTIC_CATEGORICAL_FEATURES)


@dataclass
class FittedLogisticMetaModel:
    estimator: Pipeline
    threshold: float
    feature_names: tuple[str, ...]
    trained_at: pd.Timestamp
    trained_through: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    validation_precision: float
    validation_lift: float
    validation_signals_per_week: float


def build_meta_candidates(evidence: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fired rule/ML evidence into one candidate per target/horizon."""
    required = {
        *META_GROUP_COLUMNS, "signal_id", "engine_type", "engine_name",
        "engine_version", "signal", "raw_score", "decision_threshold",
        "confidence", "confidence_support", "confidence_lift", "expires_at",
    }
    missing = required.difference(evidence.columns)
    if missing:
        raise KeyError(f"В evidence нет полей: {sorted(missing)}")
    fired = evidence.loc[evidence["signal"].astype(bool)].copy()
    if fired.empty:
        return pd.DataFrame(columns=[
            *META_GROUP_COLUMNS, *LOGISTIC_FEATURES, "evidence",
        ])
    for column in ("available_at", "as_of", "expires_at"):
        fired[column] = pd.to_datetime(fired[column])

    rows: list[dict] = []
    group_columns = list(META_GROUP_COLUMNS)
    for keys, group in fired.groupby(group_columns, sort=True):
        base = dict(zip(group_columns, keys))
        values: dict[str, float] = {}
        for engine_type, prefix in (("rule", "rule"), ("ml", "ml")):
            source = group.loc[group["engine_type"].eq(engine_type)]
            if source.empty:
                values.update({
                    f"{prefix}_signal": 0.0,
                    f"{prefix}_raw_score": 0.0,
                    f"{prefix}_confidence": 0.0,
                    f"{prefix}_support": 0.0,
                    f"{prefix}_lift": 0.0,
                })
                if prefix == "ml":
                    values["ml_probability"] = 0.0
                    values["ml_margin"] = 0.0
                continue
            source = source.sort_values("confidence", ascending=False).iloc[0]
            raw_score = 0.0 if pd.isna(source.raw_score) else float(source.raw_score)
            confidence = 0.0 if pd.isna(source.confidence) else float(source.confidence)
            support = 0.0 if pd.isna(source.confidence_support) else float(source.confidence_support)
            lift = 0.0 if pd.isna(source.confidence_lift) else float(source.confidence_lift)
            values.update({
                f"{prefix}_signal": 1.0,
                f"{prefix}_raw_score": raw_score,
                f"{prefix}_confidence": confidence,
                f"{prefix}_support": support,
                f"{prefix}_lift": lift,
            })
            if prefix == "ml":
                threshold = float(source.decision_threshold)
                values["ml_probability"] = raw_score
                values["ml_margin"] = raw_score - threshold

        values["source_count"] = float(len(group))
        values["agreement"] = float(
            values["rule_signal"] > 0 and values["ml_signal"] > 0
        )
        base.update(values)
        base["expires_at"] = group["expires_at"].min()
        base["evidence"] = group.to_dict(orient="records")
        rows.append(base)
    return pd.DataFrame(rows)


def _empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "event_id", *META_GROUP_COLUMNS, "action", "confidence",
        "confidence_method", "meta_model", "meta_model_version",
        "evidence_count", "engine_types", "expires_at", "evidence",
    ])


def _json_value(value):
    if isinstance(value, (pd.Timestamp, pd.NaT.__class__)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _candidate_events(
    accepted: pd.DataFrame,
    *,
    confidence_column: str,
    confidence_method: str,
    meta_model_name: str,
    meta_model_version: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for candidate in accepted.itertuples(index=False):
        evidence = [
            {key: _json_value(value) for key, value in record.items()}
            for record in candidate.evidence
        ]
        row = {
            **{name: getattr(candidate, name) for name in META_GROUP_COLUMNS},
            "action": "TRIGGER",
            "confidence": float(getattr(candidate, confidence_column)),
            "confidence_method": confidence_method,
            "meta_model": meta_model_name,
            "meta_model_version": meta_model_version,
            "evidence_count": len(evidence),
            "engine_types": ",".join(sorted({
                str(item.get("engine_type")) for item in evidence
            })),
            "expires_at": candidate.expires_at,
            "evidence": evidence,
        }
        row["event_id"] = _event_id(row)
        rows.append(row)
    if not rows:
        return _empty_events_frame()
    return pd.DataFrame(rows).sort_values(
        ["as_of", "currency", "target_family", "horizon"]
    ).reset_index(drop=True)


def _logistic_metrics(
    target: pd.Series,
    probability: np.ndarray,
    threshold: float,
    dates: pd.Series,
) -> dict:
    y = target.to_numpy(dtype=bool)
    prediction = probability >= threshold
    signal_count = int(prediction.sum())
    true_positive = int(np.sum(prediction & y))
    observations = len(y)
    positive_count = int(y.sum())
    precision = true_positive / signal_count if signal_count else 0.0
    baseline = positive_count / observations if observations else 0.0
    weeks = max(
        (pd.to_datetime(dates).max() - pd.to_datetime(dates).min()).days + 1,
        1,
    ) / 7
    return {
        "precision": precision,
        "baseline": baseline,
        "lift": precision / baseline if baseline else 0.0,
        "signal_count": signal_count,
        "true_positive": true_positive,
        "signals_per_week": signal_count / weeks,
    }


def _signals_per_week_by_currency(
    data: pd.DataFrame,
    prediction: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for currency, group in data.groupby("currency", sort=True):
        # ``data`` may retain original row labels after date filtering;
        # prediction is positional, therefore map group labels to positions.
        positions = data.index.get_indexer(group.index)
        span_days = max(
            (pd.to_datetime(group["available_at"]).max()
             - pd.to_datetime(group["available_at"]).min()).days + 1,
            1,
        )
        result[str(currency)] = float(
            prediction[positions].sum() / (span_days / 7)
        )
    return result


def fit_logistic_meta_model(
    labelled_candidates: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
    validation_months: int = 12,
    min_signals_per_week: float = 1.0,
    max_signals_per_week: float = 2.0,
) -> FittedLogisticMetaModel:
    """Fit on historical OOS engine predictions.

    The decision threshold is selected on the validation period.  Its primary
    objective remains uplift, while the admissible signal-rate range
    ``[min_signals_per_week, max_signals_per_week]`` is checked separately for
    every currency (rather than on a pooled stream).
    """
    required = {*LOGISTIC_FEATURES, "available_at", "horizon", "target_value"}
    missing = required.difference(labelled_candidates.columns)
    if missing:
        raise KeyError(f"В meta training data нет полей: {sorted(missing)}")
    if validation_months <= 0:
        raise ValueError("validation_months должен быть положительным")
    if min_signals_per_week < 0 or max_signals_per_week < min_signals_per_week:
        raise ValueError("Некорректный диапазон signals_per_week")
    end = pd.Timestamp(train_end)
    validation_start = end - pd.DateOffset(months=validation_months)
    data = labelled_candidates.copy()
    dates = pd.to_datetime(data["available_at"])
    horizon_delta = pd.to_timedelta(data["horizon"], unit="D")
    fit_mature = dates.add(horizon_delta).lt(validation_start)
    validation_mature = dates.add(horizon_delta).lt(end)
    validation = data.loc[
        dates.ge(validation_start) & dates.lt(end) & validation_mature
    ].copy()
    fit = data.loc[dates.lt(validation_start) & fit_mature].copy()
    if fit.empty or validation.empty or fit["target_value"].nunique() < 2:
        raise ValueError("Недостаточно OOS-данных для обучения meta-model")

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("scale", StandardScaler())]), list(LOGISTIC_NUMERIC_FEATURES)),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), list(LOGISTIC_CATEGORICAL_FEATURES)),
    ])
    estimator = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    estimator.fit(fit.loc[:, LOGISTIC_FEATURES], fit["target_value"].astype(int))
    validation_probability = estimator.predict_proba(
        validation.loc[:, LOGISTIC_FEATURES]
    )[:, 1]

    candidate_thresholds = np.unique(
        np.r_[0.5, validation_probability, 1.0]
    )
    evaluated = []
    for threshold in candidate_thresholds:
        prediction = validation_probability >= float(threshold)
        metrics = _logistic_metrics(
            validation["target_value"], validation_probability,
            float(threshold), validation["available_at"],
        )
        frequency = _signals_per_week_by_currency(validation, prediction)
        in_range = all(
            min_signals_per_week <= value <= max_signals_per_week
            for value in frequency.values()
        )
        distance = sum(
            max(min_signals_per_week - value, 0.0)
            + max(value - max_signals_per_week, 0.0)
            for value in frequency.values()
        )
        if in_range:
            evaluated.append((metrics["lift"], metrics["precision"], threshold, metrics))
        else:
            evaluated.append((
                metrics["lift"] - distance,
                metrics["precision"],
                threshold,
                metrics,
            ))
    if not evaluated:
        raise ValueError("Не удалось подобрать threshold meta-model")
    _, _, threshold, validation_metrics = max(
        evaluated, key=lambda row: (row[0], row[1], row[2])
    )

    training = data.loc[dates.lt(end) & validation_mature].copy()
    if training["target_value"].nunique() < 2:
        raise ValueError("В полном meta train только один класс")
    estimator.fit(training.loc[:, LOGISTIC_FEATURES], training["target_value"].astype(int))
    return FittedLogisticMetaModel(
        estimator=estimator,
        threshold=float(threshold),
        feature_names=tuple(LOGISTIC_FEATURES),
        trained_at=end,
        trained_through=pd.to_datetime(training["available_at"]).max(),
        validation_start=validation_start,
        validation_end=end,
        validation_precision=float(validation_metrics["precision"]),
        validation_lift=float(validation_metrics["lift"]),
        validation_signals_per_week=float(validation_metrics["signals_per_week"]),
    )


def logistic_meta_model(
    evidence: pd.DataFrame,
    config: FittedLogisticMetaModel,
) -> pd.DataFrame:
    """Apply a fitted logistic meta-model without retraining base engines."""
    candidates = build_meta_candidates(evidence)
    if candidates.empty:
        return _empty_events_frame()
    candidates["meta_probability"] = config.estimator.predict_proba(
        candidates.loc[:, config.feature_names]
    )[:, 1]
    accepted = candidates.loc[
        candidates["meta_probability"].ge(config.threshold)
    ].copy()
    if accepted.empty:
        return _empty_events_frame()
    return _candidate_events(
        accepted,
        confidence_column="meta_probability",
        confidence_method="logistic_meta_probability",
        meta_model_name="logistic_regression",
        meta_model_version=f"trained_{config.trained_at.date()}",
    )


def _event_id(row: dict) -> str:
    identity = "|".join(str(row[name]) for name in META_GROUP_COLUMNS)
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


def confidence_filter_meta_model(
    evidence: pd.DataFrame,
    config: ConfidenceFilterConfig = ConfidenceFilterConfig(),
) -> pd.DataFrame:
    """Filter evidence, then emit one event per target and horizon.

    Evidence from different horizons is deliberately not collapsed. Confidence
    is the maximum accepted source confidence; no probability independence is
    assumed.
    """
    required = {
        *META_GROUP_COLUMNS, "signal_id", "engine_type", "engine_name",
        "engine_version", "signal", "raw_score", "decision_threshold",
        "confidence", "confidence_method", "confidence_support", "expires_at",
    }
    missing = required.difference(evidence.columns)
    if missing:
        raise KeyError(f"В evidence нет полей: {sorted(missing)}")

    thresholds = evidence["engine_type"].map({
        "rule": (
            config.min_confidence
            if config.rule_min_confidence is None
            else config.rule_min_confidence
        ),
        "ml": (
            config.min_confidence
            if config.ml_min_confidence is None
            else config.ml_min_confidence
        ),
    }).fillna(config.min_confidence)
    accepted = evidence.loc[
        evidence["signal"].astype(bool)
        & evidence["confidence"].ge(thresholds)
        & evidence["confidence_support"].ge(config.min_support)
    ].copy()
    if accepted.empty:
        return pd.DataFrame(columns=[
            "event_id", *META_GROUP_COLUMNS, "action", "confidence",
            "confidence_method", "meta_model", "meta_model_version",
            "evidence_count", "engine_types", "expires_at", "evidence",
        ])

    rows: list[dict] = []
    for keys, group in accepted.groupby(list(META_GROUP_COLUMNS), sort=True):
        if len(group) < config.min_sources:
            continue
        base = dict(zip(META_GROUP_COLUMNS, keys))
        evidence_bundle = [
            {
                "signal_id": row.signal_id,
                "engine_type": row.engine_type,
                "engine_name": row.engine_name,
                "engine_version": row.engine_version,
                "raw_score": float(row.raw_score),
                "decision_threshold": float(row.decision_threshold),
                "confidence": float(row.confidence),
                "confidence_method": row.confidence_method,
                "confidence_support": int(row.confidence_support),
                "baseline_probability": (
                    None if pd.isna(row.baseline_probability)
                    else float(row.baseline_probability)
                ),
                "confidence_lift": (
                    None if pd.isna(row.confidence_lift)
                    else float(row.confidence_lift)
                ),
                "model_version": getattr(row, "model_version", None),
                "trained_at": getattr(row, "trained_at", None),
                "trained_through": getattr(row, "trained_through", None),
                "next_retrain_at": getattr(row, "next_retrain_at", None),
            }
            for row in group.sort_values(
                ["confidence", "engine_type"], ascending=[False, True]
            ).itertuples(index=False)
        ]
        row = {
            **base,
            "action": "TRIGGER",
            "confidence": float(group["confidence"].max()),
            "confidence_method": "max_accepted_evidence_confidence",
            "meta_model": "confidence_filter",
            "meta_model_version": "1.0",
            "evidence_count": len(group),
            "engine_types": ",".join(sorted(group["engine_type"].unique())),
            "expires_at": group["expires_at"].min(),
            "evidence": evidence_bundle,
        }
        row["event_id"] = _event_id(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["as_of", "currency", "target_family", "horizon"]
    ).reset_index(drop=True)


def run_meta_model(
    evidence: pd.DataFrame,
    *,
    meta_model: MetaModel = confidence_filter_meta_model,
    config: object = ConfidenceFilterConfig(),
) -> pd.DataFrame:
    """Stable integration point for a deterministic or learned meta-model."""
    return meta_model(evidence.copy(), config)


def market_event_records(events: pd.DataFrame) -> list[dict]:
    """Return JSON-serializable production records (without future truth)."""
    records = []
    for row in events.itertuples(index=False):
        records.append({
            "schema_version": "1.0",
            "event_id": row.event_id,
            "as_of": row.as_of.isoformat(),
            "corridor": row.corridor,
            "currency": row.currency,
            "scenario": row.scenario,
            "target_family": row.target_family,
            "target": row.target,
            "horizon_days": int(row.horizon),
            "action": row.action,
            "confidence": float(row.confidence),
            "confidence_method": row.confidence_method,
            "validity": {"expires_at": row.expires_at.isoformat()},
            "policy": {
                "name": row.meta_model,
                "version": row.meta_model_version,
            },
            "evidence": row.evidence,
        })
    return records
