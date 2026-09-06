"""Сервис 1 — ML-модель выявления выгодного момента.

Обёртка над ml/upstream/src/production_pipeline.py. Держит обученные состояния
движков (rule + ML, GOOD_NOW/WINDOW_CLOSING × горизонты 1/3/5/10/20 × 5 валют)
и отдаёт их сигналы на дату среза через тот же get_signal, что в production.
"""
from __future__ import annotations

import sys
import threading

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/upstream")

from common import artifacts as A       # noqa: E402

app = FastAPI(title="Favorable-moment ML model", version="1.0")
_S: dict = {"ready": None, "states": None, "scoring": None, "raw": None,
            "registry": None, "dates": None}
_lock = threading.Lock()


def _boot() -> None:
    with _lock:
        if _S["states"] is not None:
            return
        _S["ready"] = A.wait_ready(timeout_s=7200)
        from src.production_pipeline import load_engine_states
        _S["states"] = load_engine_states(str(A.p("engine_states.joblib")))
        scoring = A.load_parquet("scoring_data.parquet")
        scoring["available_at"] = pd.to_datetime(scoring["available_at"])
        _S["scoring"] = scoring
        _S["dates"] = sorted(scoring["available_at"].unique())
        _S["raw"] = A.load_parquet("raw_signals.parquet")
        _S["registry"] = A.load_parquet("engine_registry.parquet")
        print(f"[moment] загружено: {len(_S['states'])} движков, "
              f"{len(scoring)} строк scoring, срез {_S['dates'][0]}…{_S['dates'][-1]}", flush=True)


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_boot, daemon=True).start()


def _snap_date(as_of: str) -> pd.Timestamp:
    try:
        want = pd.Timestamp(as_of).normalize()
    except (ValueError, TypeError):
        raise HTTPException(400, f"неверный as_of: {as_of!r}")
    if want is pd.NaT or pd.isna(want):
        raise HTTPException(400, f"неверный as_of: {as_of!r}")
    dates = _S["dates"]
    prior = [d for d in dates if d <= want]
    if not prior:
        raise HTTPException(400, f"as_of {as_of} раньше начала данных {dates[0]}")
    return pd.Timestamp(prior[-1])


@app.get("/health")
def health() -> dict:
    if _S["states"] is None:
        return {"status": "warming", "service": "moment-model"}
    reg = _S["registry"]
    return {
        "status": "ok",
        "service": "moment-model",
        "ready": _S["ready"],
        "engines": int(len(_S["states"])),
        "engine_types": reg["engine_type"].value_counts().to_dict() if "engine_type" in reg else {},
        "model_versions": int(reg["model_version"].dropna().nunique()) if "model_version" in reg else 0,
        "trained_through": str(reg["trained_through"].dropna().max()) if "trained_through" in reg else None,
        "scoring_from": str(pd.Timestamp(_S["dates"][0]).date()),
        "scoring_to": str(pd.Timestamp(_S["dates"][-1]).date()),
    }


@app.get("/registry")
def registry() -> list[dict]:
    if _S["registry"] is None:
        raise HTTPException(503, "warming")
    return _json(_S["registry"])


@app.get("/engine-signals")
def engine_signals(as_of: str = Query(...), corridor: str | None = Query(None)):
    """Live-скоринг движков на дату среза (production get_signal)."""
    if _S["states"] is None:
        raise HTTPException(503, "warming")
    from src.production_pipeline import get_signal

    snap = _snap_date(as_of)
    rows = get_signal(as_of=snap, feature_snapshot=_S["scoring"], states=_S["states"])
    if corridor:
        rows = [r for r in rows if r.get("corridor") == corridor]
    fired = [r for r in rows if r.get("signal")]
    return {
        "as_of_requested": as_of,
        "as_of_scored": str(snap.date()),
        "n_engines": len(rows),
        "n_fired": len(fired),
        "signals": rows,
    }


@app.get("/engine-signals/replay")
def engine_signals_replay(corridor: str | None = Query(None),
                          from_: str | None = Query(None, alias="from"),
                          to: str | None = Query(None),
                          fired_only: bool = Query(False)):
    if _S["raw"] is None:
        raise HTTPException(503, "warming")
    df = _S["raw"].copy()
    df["available_at"] = pd.to_datetime(df["available_at"])
    if corridor:
        df = df[df["corridor"] == corridor]
    if from_:
        df = df[df["available_at"] >= pd.Timestamp(from_)]
    if to:
        df = df[df["available_at"] <= pd.Timestamp(to)]
    if fired_only:
        df = df[df["signal"].astype(bool)]
    return {"rows": len(df), "signals": _json(df.tail(3000))}


def _json(df: pd.DataFrame) -> list[dict]:
    return json_ready(df).to_dict(orient="records")


def json_ready(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].astype(str)
    return out.where(pd.notnull(out), None)
