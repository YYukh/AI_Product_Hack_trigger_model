"""Резервирование средств под условие по курсу (расширенный уровень).

Хранилище — в памяти процесса. Перезапуск контейнера обнуляет резервы, это
осознанно (ТЗ §3). Проверка условия — раз в сутки при сдвиге sim_date;
реализована лениво: состояние пересчитывается при чтении GET /reserve/{id}?as_of=.

Машина состояний: ACTIVE → EXECUTED | CANCELLED | EXPIRED | SUPERSEDED.
При истечении — разблокировка, НЕ автоисполнение (асимметрия ошибки λ=3),
кроме явного fallback_send_on_expiry=true.
"""
from __future__ import annotations

import uuid
from datetime import date, timedelta

from . import config
from .data_access import (rate_on, window_values, percentile_value,
                          next_trading_day)
from .events import log_event

_STORE: dict[str, dict] = {}


def _threshold(corridor: str, as_of: date, percentile: int, window_days: int) -> float:
    vals = window_values(corridor, as_of, window_days)
    if not vals:
        vals = [rate_on(corridor, as_of)[0]]
    return percentile_value(vals, percentile)


def create(req: dict, session_id: str | None = None) -> dict:
    if not config.FEATURE_RESERVE:
        raise PermissionError("резервирование выключено (FEATURE_RESERVE=false)")
    rid = uuid.uuid4().hex[:12]
    created_on = date.fromisoformat(req["created_on"])
    corridor = req["corridor"]
    pct = int(req.get("percentile", config.RESERVE_PERCENTILE))
    win = int(req.get("window_days", config.RESERVE_WINDOW_DAYS))
    ttl = int(req.get("ttl_days", config.RESERVE_TTL_DAYS))
    rate_at_creation, _ = rate_on(corridor, created_on)
    r = {
        "id": rid,
        "corridor": corridor,
        "amount_rub": int(req["amount_rub"]),
        "created_on": created_on.isoformat(),
        "percentile": pct,
        "window_days": win,
        "ttl_days": ttl,
        "fallback_send_on_expiry": bool(req.get("fallback_send_on_expiry", False)),
        "state": "ACTIVE",
        "rate_at_creation": round(rate_at_creation, 6),
        "executed_on": None,
        "exec_rate": None,
        "waited_days": None,
        "gain_bp": None,
    }
    _STORE[rid] = r
    log_event("reserve_created", {"id": rid, "corridor": corridor,
              "condition": f"нижние {pct}% за {win} дней", "ttl_days": ttl,
              "amount_rub": r["amount_rub"]}, created_on.isoformat(), session_id)
    return view(rid, created_on)


def _advance_state(r: dict, as_of: date) -> None:
    if r["state"] != "ACTIVE":
        return
    created = date.fromisoformat(r["created_on"])
    corridor = r["corridor"]
    cur, _ = rate_on(corridor, as_of)
    thr = _threshold(corridor, as_of, r["percentile"], r["window_days"])

    # условие проверяется по курсу на новую дату — «в день, когда курс войдёт
    # в нижние N%», внутридневного срабатывания не обещаем
    if cur <= thr:
        r["state"] = "EXECUTED"
        r["executed_on"] = as_of.isoformat()
        r["exec_rate"] = round(cur, 6)
        r["waited_days"] = (as_of - created).days
        r["gain_bp"] = round((r["rate_at_creation"] - cur) / r["rate_at_creation"] * 10000, 1)
        log_event("reserve_executed", {"id": r["id"], "waited_days": r["waited_days"],
                  "gain_bp": r["gain_bp"], "exec_rate": r["exec_rate"]},
                  as_of.isoformat())
        return

    if (as_of - created).days >= r["ttl_days"]:
        r["state"] = "EXPIRED"
        log_event("reserve_expired", {"id": r["id"],
                  "condition": f"нижние {r['percentile']}% за {r['window_days']} дней",
                  "ttl_days": r["ttl_days"]}, as_of.isoformat())


def view(rid: str, as_of: date) -> dict:
    r = _STORE.get(rid)
    if not r:
        raise KeyError(rid)
    _advance_state(r, as_of)
    created = date.fromisoformat(r["created_on"])
    corridor = r["corridor"]
    cur, stale = rate_on(corridor, as_of)
    thr = _threshold(corridor, as_of, r["percentile"], r["window_days"])
    days_left = max(0, r["ttl_days"] - (as_of - created).days)
    out = dict(r)
    out.update({
        "as_of": as_of.isoformat(),
        "current_rate": round(cur, 6),
        "current_rate_is_stale": stale,
        "threshold_rate": round(thr, 6),
        "distance_bp": round((cur - thr) / thr * 10000, 1),  # >0 — ещё не дотянул
        "days_left": days_left,
        "condition_text": f"курс в нижних {r['percentile']}% за {r['window_days']} дней",
        "recipient_gets_now": round(r["amount_rub"] / cur),
        "recipient_gets_at_creation": round(r["amount_rub"] / r["rate_at_creation"]),
    })
    return out


def cancel(rid: str, as_of: date, session_id: str | None = None) -> dict:
    r = _STORE.get(rid)
    if not r:
        raise KeyError(rid)
    if r["state"] == "ACTIVE":
        r["state"] = "CANCELLED"
        log_event("reserve_cancelled", {"id": rid}, as_of.isoformat(), session_id)
    return view(rid, as_of)


def supersede(rid: str, as_of: date, session_id: str | None = None) -> dict:
    """Клиент выбрал «Перевести сейчас» при активном резерве."""
    r = _STORE.get(rid)
    if not r:
        raise KeyError(rid)
    if r["state"] == "ACTIVE":
        r["state"] = "SUPERSEDED"
        log_event("reserve_cancelled", {"id": rid, "reason": "superseded_by_transfer"},
                  as_of.isoformat(), session_id)
    return view(rid, as_of)


def list_all(as_of: date | None = None) -> list[dict]:
    out = []
    for rid in list(_STORE):
        d = as_of or date.fromisoformat(_STORE[rid]["created_on"])
        out.append(view(rid, d))
    return out


def reset() -> None:
    _STORE.clear()
