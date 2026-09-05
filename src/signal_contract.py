"""Единый контракт evidence-сигналов rule-based и ML движков."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


SIGNAL_SCHEMA_VERSION = "1.0"
SIGNAL_COLUMNS = (
    "schema_version",
    "signal_id",
    "available_at",
    "as_of",
    "currency",
    "corridor",
    "scenario",
    "target_family",
    "target",
    "horizon",
    "engine_id",
    "engine_type",
    "engine_name",
    "engine_version",
    "model_version",
    "status",
    "signal",
    "raw_score",
    "decision_threshold",
    "confidence",
    "confidence_method",
    "confidence_support",
    "baseline_probability",
    "confidence_lift",
    "trained_at",
    "trained_through",
    "next_retrain_at",
    "last_retrain_error",
    "expires_at",
)


OOS_CONFIDENCE_KEYS = (
    "available_at", "currency", "scenario", "target_family", "target", "horizon",
)


def _require_columns(data: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"В {name} нет полей: {sorted(missing)}")


def _rolling_past_counts(
    query_dates: pd.Series,
    event_dates: pd.Series,
    maturity_dates: pd.Series,
    values: np.ndarray,
    *,
    window_months: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return count/value in a trailing window, matured before each query."""
    if len(maturity_dates) == 0:
        zeros = np.zeros(len(query_dates), dtype=int)
        return zeros, zeros.copy()
    history = pd.DataFrame({
        "event": pd.to_datetime(event_dates),
        "maturity": pd.to_datetime(maturity_dates),
        "value": np.asarray(values, dtype=int),
    }).sort_values(["event", "maturity"])
    if not history["maturity"].is_monotonic_increasing:
        raise ValueError("В одной confidence-группе horizons должны быть одинаковыми")
    event_ns = history["event"].astype("int64").to_numpy()
    maturity_ns = history["maturity"].astype("int64").to_numpy()
    queries = pd.to_datetime(query_dates)
    lower_dates = queries - pd.DateOffset(months=window_months)
    lower = np.searchsorted(
        event_ns, lower_dates.astype("int64").to_numpy(), side="left"
    )
    upper = np.searchsorted(
        maturity_ns,
        queries.astype("int64").to_numpy(),
        side="left",
    )
    prefix = np.r_[0, history["value"].cumsum().to_numpy(dtype=int)]
    return (upper - lower).astype(int), (prefix[upper] - prefix[lower]).astype(int)


def calibrate_rule_confidence_from_oos(
    signals: pd.DataFrame,
    *,
    evaluation_universe: pd.DataFrame,
    window_months: int = 24,
    freeze_at: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Replace rule train precision with causal rolling OOS precision.

    Every row uses only signals from the trailing ``window_months`` whose
    outcomes had strictly matured before that row.  ``freeze_at`` remains an
    optional audit mode; normal production replay leaves it unset and updates
    confidence causally as new OOS outcomes mature.  ML rows are unchanged.
    """
    signal_required = {
        *OOS_CONFIDENCE_KEYS, "engine_id", "engine_type", "signal",
        "confidence", "confidence_method", "confidence_support",
        "baseline_probability", "confidence_lift",
    }
    universe_required = {*OOS_CONFIDENCE_KEYS, "target_value"}
    _require_columns(signals, signal_required, "signals")
    _require_columns(evaluation_universe, universe_required, "evaluation_universe")
    if window_months <= 0:
        raise ValueError("window_months должен быть положительным")

    result = signals.copy()
    universe = evaluation_universe.copy()
    result["available_at"] = pd.to_datetime(result["available_at"])
    universe["available_at"] = pd.to_datetime(universe["available_at"])
    if universe.duplicated(list(OOS_CONFIDENCE_KEYS)).any():
        raise ValueError("evaluation_universe содержит повторяющиеся target rows")

    rule_mask = result["engine_type"].eq("rule")
    if not rule_mask.any():
        return result

    fired = result.loc[rule_mask & result["signal"].astype(bool), [
        *OOS_CONFIDENCE_KEYS, "engine_id",
    ]].merge(
        universe.loc[:, [*OOS_CONFIDENCE_KEYS, "target_value"]],
        on=list(OOS_CONFIDENCE_KEYS), how="left", validate="many_to_one",
    ).dropna(subset=["target_value"])
    fired["maturity"] = fired["available_at"].add(
        pd.to_timedelta(fired["horizon"], unit="D")
    )
    universe["maturity"] = universe["available_at"].add(
        pd.to_timedelta(universe["horizon"], unit="D")
    )

    freeze = pd.Timestamp(freeze_at) if freeze_at is not None else None
    for engine_id, engine_rows in result.loc[rule_mask].groupby("engine_id", sort=False):
        row_index = engine_rows.index
        query_dates = engine_rows["available_at"]
        if freeze is not None:
            query_dates = query_dates.where(query_dates.lt(freeze), freeze)
        signal_history = fired.loc[fired["engine_id"].eq(engine_id)]
        support, true_positive = _rolling_past_counts(
            query_dates,
            signal_history["available_at"],
            signal_history["maturity"],
            signal_history["target_value"].astype(int).to_numpy(),
            window_months=window_months,
        )

        first = engine_rows.iloc[0]
        observation_history = universe.loc[
            universe["currency"].eq(first["currency"])
            & universe["scenario"].eq(first["scenario"])
            & universe["target_family"].eq(first["target_family"])
            & universe["target"].eq(first["target"])
            & universe["horizon"].eq(int(first["horizon"]))
        ]
        observations, positives = _rolling_past_counts(
            query_dates,
            observation_history["available_at"],
            observation_history["maturity"],
            observation_history["target_value"].astype(int).to_numpy(),
            window_months=window_months,
        )
        confidence = np.divide(
            true_positive, support,
            out=np.zeros(len(support), dtype=float), where=support > 0,
        )
        baseline = np.divide(
            positives, observations,
            out=np.zeros(len(observations), dtype=float), where=observations > 0,
        )
        lift = np.divide(
            confidence, baseline,
            out=np.zeros(len(confidence), dtype=float), where=baseline > 0,
        )
        result.loc[row_index, "confidence"] = confidence
        result.loc[row_index, "confidence_support"] = support
        result.loc[row_index, "baseline_probability"] = baseline
        result.loc[row_index, "confidence_lift"] = lift
        result.loc[row_index, "confidence_method"] = (
            f"causal_rolling_{window_months}m_oos_precision"
        )
    result["confidence_support"] = result["confidence_support"].astype(int)
    return result


def _moscow_scoring_time(values: pd.Series, hour: int = 9) -> pd.Series:
    dates = pd.to_datetime(values).dt.normalize()
    if dates.dt.tz is None:
        dates = dates.dt.tz_localize("Europe/Moscow")
    else:
        dates = dates.dt.tz_convert("Europe/Moscow")
    return dates + pd.Timedelta(hours=hour)


def _signal_ids(data: pd.DataFrame) -> pd.Series:
    identity = data[
        ["as_of", "currency", "target", "horizon", "engine_type", "engine_version"]
    ].astype(str).agg("|".join, axis=1)
    return identity.map(lambda value: hashlib.sha256(value.encode()).hexdigest()[:24])


def _finish_contract(data: pd.DataFrame, *, validity_hours: int) -> pd.DataFrame:
    result = data.copy()
    result["schema_version"] = SIGNAL_SCHEMA_VERSION
    result["available_at"] = pd.to_datetime(result["available_at"])
    result["as_of"] = _moscow_scoring_time(result["available_at"])
    result["corridor"] = "RUB_" + result["currency"].astype(str)
    result["signal"] = True
    result["expires_at"] = result["as_of"] + pd.Timedelta(hours=validity_hours)
    result["signal_id"] = _signal_ids(result)
    result["confidence"] = pd.to_numeric(result["confidence"], errors="coerce")
    result["confidence_support"] = pd.to_numeric(
        result["confidence_support"], errors="coerce"
    ).fillna(0).astype(int)
    for column in ("baseline_probability", "confidence_lift"):
        if column not in result:
            result[column] = float("nan")
    if "engine_id" not in result:
        result["engine_id"] = (
            result["engine_type"].astype(str) + ":"
            + result["currency"].astype(str) + ":"
            + result["target_family"].astype(str) + ":h"
            + result["horizon"].astype(str)
        )
    if "model_version" not in result:
        result["model_version"] = result["engine_version"]
    if "status" not in result:
        result["status"] = "READY"
    for column in (
        "trained_at", "trained_through", "next_retrain_at",
        "last_retrain_error",
    ):
        if column not in result:
            result[column] = None
    return result.loc[:, SIGNAL_COLUMNS].sort_values(
        ["as_of", "currency", "target", "horizon", "engine_type"]
    ).reset_index(drop=True)


def standardize_rule_signals(
    signals: pd.DataFrame,
    *,
    selected_indicators: pd.DataFrame,
    validity_hours: int = 24,
) -> pd.DataFrame:
    """Convert rule backtest emissions to the shared evidence contract.

    Confidence is discovery pooled OOS precision, never final holdout precision.
    """
    signal_required = {
        "available_at", "currency", "target_value", "scenario",
        "target_family", "target", "horizon", "indicator",
        "fixed_candidate", "rebalance_months", "fold_id",
    }
    selection_required = {
        "currency", "target", "horizon", "oos_precision",
        "test_predicted_positive_count",
    }
    _require_columns(signals, signal_required, "rule signals")
    _require_columns(selected_indicators, selection_required, "selected_indicators")

    confidence = selected_indicators.loc[:, [
        "currency", "target", "horizon", "oos_precision",
        "test_predicted_positive_count",
    ]].drop_duplicates(["currency", "target", "horizon"])
    result = signals.merge(
        confidence,
        on=["currency", "target", "horizon"],
        how="left",
        validate="many_to_one",
    )
    result["engine_type"] = "rule"
    result["engine_name"] = result["indicator"]
    result["engine_version"] = (
        result["fixed_candidate"].astype(str)
        + ":rebalance_"
        + result["rebalance_months"].astype(str)
        + "m"
    )
    result["raw_score"] = 1.0
    result["decision_threshold"] = 1.0
    result["confidence"] = result["oos_precision"]
    result["confidence_method"] = "pooled_discovery_oos_precision"
    result["confidence_support"] = result["test_predicted_positive_count"]
    result["baseline_probability"] = float("nan")
    result["confidence_lift"] = float("nan")
    return _finish_contract(result, validity_hours=validity_hours)


def standardize_ml_signals(
    signals: pd.DataFrame,
    *,
    validity_hours: int = 24,
) -> pd.DataFrame:
    """Convert ML backtest emissions to the shared evidence contract."""
    required = {
        "available_at", "currency", "target_value", "scenario",
        "target_family", "target", "horizon", "indicator",
        "rebalance_months", "fold_id", "probability",
        "probability_threshold", "confidence", "confidence_method",
        "confidence_support",
    }
    _require_columns(signals, required, "ML signals")
    result = signals.copy()
    result["engine_type"] = "ml"
    result["engine_name"] = result["indicator"]
    result["engine_version"] = (
        result["indicator"].astype(str)
        + ":rebalance_"
        + result["rebalance_months"].astype(str)
        + "m"
    )
    result["raw_score"] = result["probability"]
    result["decision_threshold"] = result["probability_threshold"]
    result["confidence"] = result["probability"]
    result["confidence_method"] = "model_predict_proba"
    return _finish_contract(result, validity_hours=validity_hours)


def finalize_signal_evidence(
    signals: pd.DataFrame,
    *,
    validity_hours: int = 24,
) -> pd.DataFrame:
    """Finalize engine output that already contains contract attributes."""
    required = {
        "available_at", "currency", "scenario", "target_family", "target",
        "horizon", "engine_type", "engine_name", "engine_version",
        "raw_score", "decision_threshold", "confidence",
        "confidence_method", "confidence_support",
    }
    _require_columns(signals, required, "engine signals")
    return _finish_contract(signals, validity_hours=validity_hours)


def combine_evidence_streams(*streams: pd.DataFrame) -> pd.DataFrame:
    """Concatenate already standardized streams and reject duplicate IDs."""
    non_empty = [stream for stream in streams if not stream.empty]
    if not non_empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    for stream in non_empty:
        _require_columns(stream, set(SIGNAL_COLUMNS), "evidence stream")
    result = pd.concat(non_empty, ignore_index=True).loc[:, SIGNAL_COLUMNS]
    if result["signal_id"].duplicated().any():
        duplicates = result.loc[result["signal_id"].duplicated(), "signal_id"].tolist()
        raise ValueError(f"Повторяющиеся signal_id: {duplicates[:5]}")
    return result.sort_values(
        ["as_of", "currency", "target", "horizon", "engine_type"]
    ).reset_index(drop=True)
