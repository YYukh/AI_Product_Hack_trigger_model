"""Сервис 3 — парсер котировок валютных пар (ЦБ РФ).

Обёртка над ml/upstream/src/cbr_loader.py. Отдаёт дневной нормированный ряд
RUB за единицу валюты в формате, который понимает демо-стенд.
"""
from __future__ import annotations

import os
import sys
import threading
from datetime import date

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/upstream")

from common import artifacts as A       # noqa: E402
from common import pipeline as P        # noqa: E402

CBR_PREFER = os.getenv("CBR_PREFER", "auto")
START = os.getenv("PARSER_START", "2020-01-01")

PAIR_META = {
    "RUB_AMD": {"country": "Армения", "currency": "AMD", "cbr_id": "R01060"},
    "RUB_KZT": {"country": "Казахстан", "currency": "KZT", "cbr_id": "R01335"},
    "RUB_KGS": {"country": "Кыргызстан", "currency": "KGS", "cbr_id": "R01370"},
    "RUB_TJS": {"country": "Таджикистан", "currency": "TJS", "cbr_id": "R01670"},
    "RUB_UZS": {"country": "Узбекистан", "currency": "UZS", "cbr_id": "R01717"},
}

app = FastAPI(title="FX quote parser (CBR)", version="1.0")
_state: dict = {"rates": None, "source": None, "loaded_at": None, "error": None}
_lock = threading.Lock()


def _load(prefer: str = CBR_PREFER) -> None:
    with _lock:
        # приоритет — снапшот из общего тома (его пишет ml-warmup), иначе live/fallback
        if prefer == "auto" and A.exists("rates_wide.parquet"):
            df = A.load_parquet("rates_wide.parquet").set_index("available_at")
            df.index = pd.to_datetime(df.index)
            _state.update(rates=df, source="mldata_snapshot", loaded_at=pd.Timestamp.utcnow().isoformat(), error=None)
            return
        try:
            rates, src = P.load_wide_rates(start=START, prefer=prefer, raw_dir=str(A.p("raw")))
            _state.update(rates=rates, source=src, loaded_at=pd.Timestamp.utcnow().isoformat(), error=None)
        except Exception as exc:  # noqa: BLE001
            _state["error"] = f"{type(exc).__name__}: {exc}"
            raise


@app.on_event("startup")
def _startup() -> None:
    try:
        _load()
    except Exception as exc:  # noqa: BLE001
        print(f"[parser] стартовая загрузка не удалась: {exc}", flush=True)


def _rates() -> pd.DataFrame:
    if _state["rates"] is None:
        _load()
    return _state["rates"]


@app.get("/health")
def health() -> dict:
    r = _state["rates"]
    return {
        "status": "ok" if r is not None else "loading",
        "service": "parser",
        "source": _state["source"],
        "loaded_at": _state["loaded_at"],
        "error": _state["error"],
        "currencies": list(r.columns) if r is not None else [],
        "date_from": str(r.index.min().date()) if r is not None else None,
        "date_to": str(r.index.max().date()) if r is not None else None,
        "rows": int(len(r)) if r is not None else 0,
    }


@app.get("/pairs")
def pairs() -> list[dict]:
    r = _rates()
    out = []
    for corr, meta in PAIR_META.items():
        cur = meta["currency"]
        last = None
        if cur in r.columns and len(r):
            last = {"date": str(r.index.max().date()), "rate": round(float(r[cur].iloc[-1]), 6)}
        out.append({"corridor": corr, **meta, "last": last})
    return out


@app.get("/rates")
def rates(corridor: str = Query(...), from_: str | None = Query(None, alias="from"),
          to: str | None = Query(None)):
    """Формат стенда: {corridor, points:[{date,rate,is_stale}]}."""
    if corridor not in PAIR_META:
        raise HTTPException(404, f"unknown corridor {corridor}")
    cur = PAIR_META[corridor]["currency"]
    r = _rates()
    if cur not in r.columns:
        raise HTTPException(404, f"no data for {cur}")
    s = r[cur].dropna()
    if from_:
        s = s.loc[s.index >= pd.Timestamp(from_)]
    if to:
        s = s.loc[s.index <= pd.Timestamp(to)]
    points = [{"date": str(pd.Timestamp(ts).date()), "rate": round(float(v), 6), "is_stale": False}
              for ts, v in s.items()]
    return {"corridor": corridor, "currency": cur, "source": _state["source"], "points": points}


@app.get("/rates/wide")
def rates_wide(from_: str | None = Query(None, alias="from"), to: str | None = Query(None)):
    r = _rates()
    if from_:
        r = r.loc[r.index >= pd.Timestamp(from_)]
    if to:
        r = r.loc[r.index <= pd.Timestamp(to)]
    return {
        "source": _state["source"],
        "columns": list(r.columns),
        "rows": [{"date": str(pd.Timestamp(ts).date()),
                  **{c: round(float(r.loc[ts, c]), 6) for c in r.columns}}
                 for ts in r.index],
    }


@app.post("/refresh")
def refresh(prefer: str = Query("live")):
    _load(prefer=prefer)
    return health()
