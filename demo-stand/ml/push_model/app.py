"""Сервис 2 — ML-модель «какой сигнал пойдёт в пуш».

Обёртка над ml/upstream/src/meta_model.py + signal_policy.py. Берёт сырой поток
движков «выгодного момента», прогоняет через сменяемую метамодель и частотную
политику (cooldown 3д, ≤2 сигнала за 7д) и отдаёт финальный поток в контракте
демо-стенда (GET /health, GET /signals?as_of=&corridors=).
"""
from __future__ import annotations

import sys
import threading

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/upstream")

from common import artifacts as A       # noqa: E402
from common import contract as C        # noqa: E402

app = FastAPI(title="Signal-to-push meta model", version="1.0")
_S: dict = {"ready": None, "push": None, "raw": None, "scoring": None,
            "rates": None, "records": None}
_lock = threading.Lock()


def _boot() -> None:
    with _lock:
        if _S["push"] is not None:
            return
        _S["ready"] = A.wait_ready(timeout_s=7200)
        push = A.load_parquet("push_events.parquet") if A.exists("push_events.parquet") else pd.DataFrame()
        if len(push):
            push["available_at"] = pd.to_datetime(push["available_at"])
        _S["push"] = push
        raw_name = "raw_signals_calibrated.parquet" if A.exists("raw_signals_calibrated.parquet") else "raw_signals.parquet"
        raw = A.load_parquet(raw_name)
        raw["available_at"] = pd.to_datetime(raw["available_at"])
        _S["raw"] = raw
        scoring = A.load_parquet("scoring_data.parquet")
        scoring["available_at"] = pd.to_datetime(scoring["available_at"])
        _S["scoring"] = scoring
        rw = A.load_parquet("rates_wide.parquet").set_index("available_at")
        rw.index = pd.to_datetime(rw.index)
        _S["rates"] = rw
        import json
        try:
            _S["records"] = json.loads(A.p("push_events_records.json").read_text())
        except Exception:  # noqa: BLE001
            _S["records"] = None
        try:
            _S["seeds"] = json.loads(A.p("signal_seeds.json").read_text())
        except Exception:  # noqa: BLE001
            _S["seeds"] = []
        print(f"[push] загружено: push_events={len(push)}, raw={len(raw)}, "
              f"meta={_S['ready'].get('meta_model')}", flush=True)


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_boot, daemon=True).start()


def _rate_on(corridor: str, d: pd.Timestamp) -> float | None:
    cur = corridor.replace("RUB_", "")
    rw = _S["rates"]
    if rw is None or cur not in rw.columns:
        return None
    s = rw[cur].loc[rw.index <= d]
    return float(s.iloc[-1]) if len(s) else None


def _feature_row(corridor: str, d: pd.Timestamp):
    cur = corridor.replace("RUB_", "")
    sub = _S["scoring"].loc[
        (_S["scoring"]["currency"] == cur) & (_S["scoring"]["available_at"] <= d)
    ]
    return sub.iloc[-1] if len(sub) else None


@app.get("/health")
def health() -> dict:
    if _S["push"] is None:
        return {"status": "warming", "service": "push-model"}
    r = _S["ready"] or {}
    return {
        "status": "ok",
        "service": "push-model",
        "model_version": r.get("model_version"),
        "meta_model": r.get("meta_model"),
        "cbr_source": r.get("cbr_source"),
        "policy": {"cooldown_days": 3, "max_signals_per_7d": 2},
        "n_push_events": int(len(_S["push"])),
        "coverage": {"from": r.get("scoring_from"), "to": r.get("scoring_to")},
    }


@app.get("/signals")
def signals(as_of: str = Query(...), corridors: str | None = Query(None)):
    """Контракт стенда: финальный поток сигналов, уже прошедших метамодель и
    частотную политику, переведённый в схему стенда."""
    if _S["push"] is None:
        raise HTTPException(503, "warming")
    want = pd.Timestamp(as_of).normalize()
    wanted = {c.strip() for c in corridors.split(",")} if corridors else None
    lo = want - pd.Timedelta(days=400)
    df = _S["push"]
    if not len(df):
        return {"as_of": as_of, "model_version": (_S["ready"] or {}).get("model_version"), "signals": []}
    sub = df.loc[(df["available_at"] <= want) & (df["available_at"] >= lo)].copy()
    if wanted:
        sub = sub.loc[sub["corridor"].isin(wanted)]
    out = []
    for row in sub.itertuples(index=False):
        d = pd.Timestamp(row.available_at)
        corr = row.corridor
        ev_raw = getattr(row, "evidence", None)
        evidence = list(ev_raw) if ev_raw is not None else []
        eng = ""
        for e in evidence:
            if not isinstance(e, dict):
                continue
            eng = e.get("engine_name") or e.get("engine_id") or eng
            if e.get("engine_type") == "ml":
                break
        rate = _rate_on(corr, d)
        out.append(C.to_stand_signal(
            date=d.date().isoformat(), corridor=corr, scenario=str(row.scenario),
            engine_name=eng, horizon=int(row.horizon), strength=float(row.confidence),
            feature_row=_feature_row(corr, d), rate=rate,
        ))
    # сид-сигналы (реконструкция из ряда для покрытия сценариев, seed=true)
    for s in (_S.get("seeds") or []):
        sd = pd.Timestamp(s["date"]).normalize()
        if sd > want or sd < lo:
            continue
        if wanted and s["corridor"] not in wanted:
            continue
        out.append(dict(s))
    out.sort(key=lambda s: s["date"])
    return {
        "as_of": as_of,
        "model_version": (_S["ready"] or {}).get("model_version"),
        "meta_model": (_S["ready"] or {}).get("meta_model"),
        "signals": out,
    }


@app.get("/push-events")
def push_events(as_of: str | None = Query(None), corridor: str | None = Query(None)):
    """Богатый вид финальных событий (сырой ML event contract)."""
    if _S["push"] is None:
        raise HTTPException(503, "warming")
    df = _S["push"].copy()
    if corridor:
        df = df[df["corridor"] == corridor]
    if as_of:
        df = df[df["available_at"] <= pd.Timestamp(as_of)]
    df = df.drop(columns=[c for c in ("evidence",) if c in df.columns])
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype(str)
    return {"rows": len(df), "events": df.where(pd.notnull(df), None).to_dict(orient="records")}


@app.get("/decisions")
def decisions(as_of: str = Query(...), corridor: str | None = Query(None)):
    """Что сработало у движков в этот день и что из этого прошло в пуш."""
    if _S["raw"] is None:
        raise HTTPException(503, "warming")
    d = pd.Timestamp(as_of).normalize()
    raw = _S["raw"]
    day = raw[raw["available_at"] == d].copy()
    if corridor:
        day = day[day["corridor"] == corridor]
    fired = day[day["signal"].astype(bool)]
    push = _S["push"]
    pushed = push[push["available_at"] == d] if len(push) else push
    if corridor and len(pushed):
        pushed = pushed[pushed["corridor"] == corridor]
    cols = ["corridor", "scenario", "target_family", "horizon", "engine_type",
            "engine_name", "raw_score", "confidence", "status"]
    return {
        "as_of": as_of,
        "engines_evaluated": int(len(day)),
        "engines_fired": _rows(fired, cols),
        "went_to_push": _rows(pushed, ["corridor", "scenario", "horizon", "confidence"]) if len(pushed) else [],
        "note": "движок сработал → метамодель по confidence/uplift → частотная политика (cooldown 3д, ≤2/7д)",
    }


def _rows(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    have = [c for c in cols if c in df.columns]
    out = df[have].copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].astype(str)
    return out.where(pd.notnull(out), None).to_dict(orient="records")
