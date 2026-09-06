"""Главная ручка: по параметрам открытия вычисляет состояние экрана перевода.

Правила состояний — из user-path-v2 §2 и ТЗ §8.2. Вердикта «выгодно/невыгодно»
на выходе нет: только состояние, факты, числа и период. Вывод делает клиент.
"""
from __future__ import annotations

from datetime import date

from . import config
from .data_access import CORRIDORS, rate_on, window_values, percentile_rank
from . import signals as signals_src
from . import texts as texts_mod

DisclaimerText = "Показан официальный курс ЦБ РФ. Курс перевода может отличаться"


def _delta_bp(current: float, push: float) -> float:
    return (current - push) / push * 10000.0


def _state(entry: str, elapsed_min: int | None, delta_bp: float,
           has_signal: bool, thr: int | None = None) -> str:
    thr = config.DRIFT_THRESHOLD_BP if thr is None else int(thr)
    if entry == "PUSH":
        if elapsed_min is not None and elapsed_min > config.PUSH_TTL_MINUTES:
            return "NEUTRAL"
        if delta_bp > thr:
            return "DRIFT"
        if delta_bp < -thr:
            return "BETTER"
        return "OK"
    # entry == SELF
    return "OK" if has_signal else "NEUTRAL"


def evaluate(req: dict) -> dict:
    corridor = req["corridor"]
    sim_date = date.fromisoformat(req["sim_date"])
    sim_minutes = int(req["sim_minutes"])
    entry = req.get("entry", "SELF")
    amount_rub = int(req.get("amount_rub") or 0)
    mechanic = (req.get("drift_mechanic") or "C").upper()

    current_rate, is_stale = rate_on(corridor, sim_date)
    push_rate = req.get("push_rate")
    push_sent_at_minutes = req.get("push_sent_at_minutes")
    elapsed = None
    if entry == "PUSH" and push_sent_at_minutes is not None:
        elapsed = sim_minutes - int(push_sent_at_minutes)

    delta_bp = _delta_bp(current_rate, push_rate) if (entry == "PUSH" and push_rate) else 0.0

    # Для входа SELF и для плашки берём сигнал ИМЕННО этого дня — обычный день
    # не выдаётся за сигнальный (S5, визуальный эквивалент lift ≈ 1,0).
    sig = signals_src.signal_on(req["sim_date"], corridor)
    has_signal = sig is not None

    thr_override = req.get("drift_threshold_bp")
    state = _state(entry, elapsed, delta_bp, has_signal, thr_override)

    # --- суммы у получателя
    recipient_gets = round(amount_rub / current_rate) if amount_rub else None
    recipient_gets_at_push = (round(amount_rub / push_rate)
                              if (amount_rub and push_rate) else None)
    recipient_delta = (abs(recipient_gets_at_push - recipient_gets)
                       if (recipient_gets is not None and recipient_gets_at_push is not None)
                       else None)

    # --- перцентиль текущего курса за 90 дней (для контекста)
    w90 = window_values(corridor, sim_date, 90)
    pct_now = percentile_rank(w90, current_rate)

    plaque = _plaque(state, mechanic, corridor, sig, {
        "push_rate": round(push_rate, 5) if push_rate else None,
        "current_rate": round(current_rate, 5),
        "delta_pct": round(abs(delta_bp) / 100, 2),
        "amount_rub": amount_rub,
        "recipient_delta": recipient_delta,
        "percentile": pct_now,
        "percentile_inv": 100 - pct_now,
        "window_days": 90,
    })

    secondary = None
    if state == "DRIFT":
        secondary = "RESERVE" if config.FEATURE_RESERVE else "RETURN"
    elif state == "NEUTRAL":
        secondary = "RESERVE" if config.FEATURE_RESERVE else None

    prefill = entry == "PUSH" and state in ("OK", "BETTER")

    # Текст пуша берётся от сигнала, который его вызвал: сначала пробуем сигнал
    # ровно на дату среза, иначе — последний сигнал по коридору до неё (пуш мог
    # уйти на несколько дней раньше момента открытия, как в сценарии «момент
    # изменился»).
    push_sig = sig or signals_src.signal_for(req["sim_date"], corridor)
    push_text = None
    if entry == "PUSH" and push_sig:
        push_text = texts_mod.render(
            texts_mod.texts().get(push_sig["scenario_code"], {}).get("push"),
            push_sig.get("facts", {}), corridor)

    return {
        "state": state,
        "entry": entry,
        "push_text": push_text,
        "delta_bp": round(delta_bp, 1),
        "current_rate": round(current_rate, 6),
        "current_rate_is_stale": is_stale,
        "push_rate": push_rate,
        "recipient_gets": recipient_gets,
        "recipient_gets_at_push": recipient_gets_at_push,
        "recipient_delta": recipient_delta,
        "recipient_currency": CORRIDORS[corridor]["currency_name"],
        "recipient_currency_short": CORRIDORS[corridor]["currency_short"],
        "percentile_now": pct_now,
        "prefill": prefill,
        "plaque": plaque,
        "actions": {"primary": "TRANSFER", "secondary": secondary},
        "signal": _signal_echo(sig),
        "disclaimer": DisclaimerText,
        "params_are_stand": {  # что из этого — параметр стенда, а не данные
            "drift_threshold_bp": config.DRIFT_THRESHOLD_BP,
            "push_ttl_minutes": config.PUSH_TTL_MINUTES,
        },
    }


def _plaque(state: str, mechanic: str, corridor: str, sig: dict | None,
            drift_facts: dict) -> dict:
    if state == "DRIFT":
        code = {"A": "DRIFT_A", "B": "DRIFT_B", "C": "DRIFT_C"}.get(mechanic, "DRIFT_C")
        b = texts_mod.bundle(code, drift_facts, corridor)
        context = None
        if mechanic == "C":
            context = texts_mod.render(texts_mod.texts()["DRIFT_B"]["plaque"],
                                       drift_facts, corridor)
        return {"scenario_code": code, "text": b["plaque"], "context": context,
                "forbidden": b["forbidden"], "why_forbidden": b["why_forbidden"]}

    if state == "BETTER":
        b = texts_mod.bundle("BETTER", drift_facts, corridor)
        return {"scenario_code": "BETTER", "text": b["plaque"], "context": None,
                "forbidden": b["forbidden"], "why_forbidden": b["why_forbidden"]}

    if state == "NEUTRAL":
        b = texts_mod.bundle("NEUTRAL", {
            "window_days": drift_facts["window_days"],
            "percentile": drift_facts["percentile"],
        }, corridor)
        return {"scenario_code": "NEUTRAL", "text": b["plaque"], "context": None,
                "forbidden": b["forbidden"], "why_forbidden": b["why_forbidden"]}

    # state == OK — берём код и факты из сигнала
    if sig:
        b = texts_mod.bundle(sig["scenario_code"], sig.get("facts", {}), corridor)
        return {"scenario_code": sig["scenario_code"], "text": b["plaque"],
                "context": None, "forbidden": b["forbidden"],
                "why_forbidden": b["why_forbidden"]}
    b = texts_mod.bundle("NEUTRAL", {"window_days": drift_facts["window_days"]}, corridor)
    return {"scenario_code": "NEUTRAL", "text": b["plaque"], "context": None,
            "forbidden": b["forbidden"], "why_forbidden": b["why_forbidden"]}


def _signal_echo(sig: dict | None) -> dict | None:
    if not sig:
        return None
    return {k: sig.get(k) for k in
            ("date", "corridor", "indicator", "direction", "speed", "strength",
             "scenario_code", "facts")}
