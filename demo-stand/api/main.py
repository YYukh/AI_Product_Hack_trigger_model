"""Роуты стенда. Префикс /api. Все ответы JSON."""
from __future__ import annotations

from datetime import date

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import httpx

from . import config, evaluate as ev, policy, reserve as rsv, signals as sig
from . import texts as texts_mod
from .data_access import (CORRIDORS, corridor_series, date_range, personas,
                          rates_source, scenarios, rate_on)
from .events import log_event, recent


def _ml_probe(url: str) -> dict:
    """Быстрый health любого ML-сервиса. Никогда не бросает."""
    if not url:
        return {"configured": False}
    try:
        with httpx.Client(timeout=config.ML_TIMEOUT_S) as c:
            r = c.get(url.rstrip("/") + "/health")
            r.raise_for_status()
            return {"configured": True, "reachable": True, "health": r.json()}
    except Exception as e:  # noqa: BLE001
        return {"configured": True, "reachable": False, "error": f"{type(e).__name__}: {e}"}


def _ml_get(url: str, path: str, params: dict):
    if not url:
        raise HTTPException(503, "ML-сервис не сконфигурирован")
    try:
        with httpx.Client(timeout=config.ML_TIMEOUT_S * 4) as c:
            r = c.get(url.rstrip("/") + path, params={k: v for k, v in params.items() if v is not None})
            r.raise_for_status()
            return r.json()
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"ML-сервис недоступен: {type(e).__name__}: {e}")

app = FastAPI(title="FX-trigger demo stand", version=config.VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.exception_handler(KeyError)
async def _key_error(_: Request, exc: KeyError):
    return JSONResponse(status_code=404, content={"error": "not_found", "detail": str(exc)})


@app.exception_handler(ValueError)
async def _value_error(_: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"error": "bad_request", "detail": str(exc)})


# --- справочные ручки ---------------------------------------------------
@app.get("/api/health")
def health():
    lo, hi = date_range()
    src = sig.health()
    return {
        "status": "ok",
        "version": config.VERSION,
        "signals_source": src,
        "rates_source": rates_source(),
        "ml_services": {
            "parser": _ml_probe(config.RATES_URL),
            "moment_model": _ml_probe(config.MOMENT_URL),
            "push_model": _ml_probe(config.ML_URL),
        },
        "dates_available": {"from": lo, "to": hi},
        "features": {
            "reserve": config.FEATURE_RESERVE,
            "recipient_limit": config.FEATURE_RECIPIENT_LIMIT,
        },
        "params": {
            "drift_threshold_bp": config.DRIFT_THRESHOLD_BP,
            "push_ttl_minutes": config.PUSH_TTL_MINUTES,
            "push_budget_month": config.PUSH_BUDGET_MONTH,
            "push_cooldown_days": config.PUSH_COOLDOWN_DAYS,
            "reserve_ttl_days": config.RESERVE_TTL_DAYS,
            "reserve_percentile": config.RESERVE_PERCENTILE,
            "reserve_window_days": config.RESERVE_WINDOW_DAYS,
        },
    }


@app.get("/api/corridors")
def corridors():
    return [{"corridor": k, **v} for k, v in CORRIDORS.items()]


@app.get("/api/rates")
def rates(corridor: str = Query(...), from_: str | None = Query(None, alias="from"),
          to: str | None = Query(None)):
    if corridor not in CORRIDORS:
        raise HTTPException(404, f"unknown corridor {corridor}")
    df = corridor_series(corridor)
    if from_:
        df = df[df["date"] >= date.fromisoformat(from_)]
    if to:
        df = df[df["date"] <= date.fromisoformat(to)]
    return {
        "corridor": corridor,
        "points": [{"date": str(r.date), "rate": float(r.rate),
                    "is_stale": bool(r.is_stale)} for r in df.itertuples()],
    }


@app.get("/api/signals")
def signals(as_of: str = Query(...), corridor: str | None = Query(None)):
    corr = [corridor] if corridor else None
    return sig.get_signals(as_of, corr)


# --- проксирование к ML-сервисам (для экрана «Данные» / режима разбора) ---
@app.get("/api/ml/engine-signals")
def ml_engine_signals(as_of: str = Query(...), corridor: str | None = Query(None)):
    """Сырой поток движков «выгодного момента» (moment-model)."""
    return _ml_get(config.MOMENT_URL, "/engine-signals",
                   {"as_of": as_of, "corridor": corridor})


@app.get("/api/ml/registry")
def ml_registry():
    return _ml_get(config.MOMENT_URL, "/registry", {})


@app.get("/api/ml/push-events")
def ml_push_events(as_of: str | None = Query(None), corridor: str | None = Query(None)):
    return _ml_get(config.ML_URL, "/push-events", {"as_of": as_of, "corridor": corridor})


@app.get("/api/ml/decisions")
def ml_decisions(as_of: str = Query(...), corridor: str | None = Query(None)):
    """Что сработало у движков и что прошло метамодель + частотную политику."""
    return _ml_get(config.ML_URL, "/decisions", {"as_of": as_of, "corridor": corridor})


@app.get("/api/personas")
def api_personas():
    return personas()


@app.get("/api/scenarios")
def api_scenarios():
    return scenarios()


@app.get("/api/texts")
def api_texts():
    return texts_mod.library()


# --- главная ручка -----------------------------------------------------
@app.post("/api/evaluate")
async def api_evaluate(req: Request):
    body = await req.json()
    for f in ("corridor", "sim_date", "sim_minutes"):
        if f not in body:
            raise HTTPException(400, f"missing field: {f}")
    if body["corridor"] not in CORRIDORS:
        raise HTTPException(404, f"unknown corridor {body['corridor']}")
    result = ev.evaluate(body)
    if body.get("log", True):
        log_event("screen_shown", {"state": result["state"],
                  "scenario_code": result["plaque"]["scenario_code"],
                  "corridor": body["corridor"]},
                  body["sim_date"], body.get("session_id"))
    return result


@app.get("/api/scenario/{sid}/run")
def scenario_run(sid: str, drift_mechanic: str = Query("C"),
                 drift_threshold_bp: int | None = Query(None),
                 amount_rub: int | None = Query(None),
                 sim_date: str | None = Query(None),
                 open_delay_min: int | None = Query(None)):
    """Собирает параметры evaluate из сценария и портрета и прогоняет их.
    Удобно для фронта и для проверки expected_state."""
    sc = next((s for s in scenarios() if s["id"] == sid), None)
    if not sc:
        raise HTTPException(404, f"no scenario {sid}")
    p = next((x for x in personas() if x["id"] == sc["persona"]), None)
    amount = amount_rub or sc.get("amount_rub_override") or (p or {}).get("typical_amount_rub", 20000)
    delay = sc.get("open_delay_min", 0) if open_delay_min is None else int(open_delay_min)
    push_min = None
    sim_min = 12 * 60
    if sc.get("push_sent_at"):
        h, m = sc["push_sent_at"].split(":")
        push_min = int(h) * 60 + int(m)
        sim_min = push_min + int(delay)
    body = {
        "corridor": sc["corridor"],
        "sim_date": sim_date or sc["as_of_date"],
        "sim_minutes": sim_min,
        "entry": sc["entry"],
        "push_sent_at_minutes": push_min,
        "push_rate": sc.get("push_rate"),
        "amount_rub": amount,
        "drift_mechanic": drift_mechanic,
        "drift_threshold_bp": drift_threshold_bp,
    }
    result = ev.evaluate(body)
    result["scenario"] = sc
    result["persona"] = p
    result["expected_state"] = sc.get("expected_state")
    result["state_matches_expected"] = result["state"] == sc.get("expected_state")
    return result


# --- резерв ----------------------------------------------------------
@app.post("/api/reserve")
async def api_reserve_create(req: Request):
    body = await req.json()
    try:
        return rsv.create(body, body.get("session_id"))
    except PermissionError as e:
        raise HTTPException(409, str(e))


@app.get("/api/reserve/{rid}")
def api_reserve_view(rid: str, as_of: str = Query(...)):
    return rsv.view(rid, date.fromisoformat(as_of))


@app.post("/api/reserve/{rid}/cancel")
async def api_reserve_cancel(rid: str, req: Request):
    body = await req.json()
    return rsv.cancel(rid, date.fromisoformat(body["as_of"]), body.get("session_id"))


@app.post("/api/reserve/{rid}/transfer-now")
async def api_reserve_supersede(rid: str, req: Request):
    body = await req.json()
    return rsv.supersede(rid, date.fromisoformat(body["as_of"]), body.get("session_id"))


@app.get("/api/reserves")
def api_reserves(as_of: str | None = Query(None)):
    return rsv.list_all(date.fromisoformat(as_of) if as_of else None)


# --- события и политика ---------------------------------------------
@app.post("/api/events")
async def api_events_post(req: Request):
    body = await req.json()
    return log_event(body.get("type", "unknown"), body.get("payload"),
                     body.get("sim_date"), body.get("session_id"))


@app.get("/api/events")
def api_events_get(session_id: str | None = Query(None), limit: int = Query(200)):
    return recent(limit, session_id)


@app.post("/api/policy/preview")
async def api_policy_preview(req: Request):
    body = await req.json()
    return policy.plan(body.get("pushes", []), budget=body.get("budget"),
                       cooldown_days=body.get("cooldown_days"),
                       timezone_offset_h=body.get("tz_offset_h", 3))


@app.get("/api/recipient-limit/check")
def recipient_limit_check(persona: str, amount_rub: float, corridor: str,
                          as_of: str):
    """O1 / S7: исчерпает ли перевод месячный лимит получателя."""
    p = next((x for x in personas() if x["id"] == persona), None)
    if not p or not p.get("recipient_limit"):
        return {"applies": False}
    lim = p["recipient_limit"]
    rate, _ = rate_on(corridor, date.fromisoformat(as_of))
    recipient_units = amount_rub / rate
    return {
        "applies": True,
        "reason": lim.get("reason"),
        "recipient_units": round(recipient_units),
        "per_operation": lim.get("per_operation_kgs"),
        "per_month": lim.get("per_month_kgs"),
        "exceeds_operation": recipient_units > lim.get("per_operation_kgs", 1e18),
        "exceeds_month": recipient_units > lim.get("per_month_kgs", 1e18),
        "currency_short": CORRIDORS[corridor]["currency_short"],
    }
