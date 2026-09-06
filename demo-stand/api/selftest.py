"""Быстрый прогон инвариантов стенда без поднятия HTTP.

    python -m api.selftest      (внутри контейнера api или venv с зависимостями)

Проверяет: expected_state всех сценариев, рендер текстов только из фактов,
запрет заглядывания вперёд, жизненный цикл резерва. Ненулевой код выхода —
сборка сломана.
"""
from __future__ import annotations

import sys
from datetime import date

from .data_access import scenarios, personas, corridor_series
from .evaluate import evaluate
from . import reserve as rsv
from . import config

FAIL = []


def check(cond: bool, msg: str):
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        FAIL.append(msg)


def hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def run_scenarios():
    print("Сценарии: expected_state == факт")
    peo = {p["id"]: p for p in personas()}
    for sc in scenarios():
        p = peo.get(sc["persona"], {})
        amount = sc.get("amount_rub_override") or p.get("typical_amount_rub", 20000)
        push_min = hhmm(sc["push_sent_at"]) if sc.get("push_sent_at") else None
        sim_min = (push_min + sc.get("open_delay_min", 0)) if push_min is not None else 720
        res = evaluate({
            "corridor": sc["corridor"],
            "sim_date": sc["as_of_date"],
            "sim_minutes": sim_min,
            "entry": sc["entry"],
            "push_sent_at_minutes": push_min,
            "push_rate": sc.get("push_rate"),
            "amount_rub": amount,
            "drift_mechanic": "C",
        })
        check(res["state"] == sc["expected_state"],
              f'{sc["id"]}: ожидали {sc["expected_state"]}, получили {res["state"]}')
        # плашка не пустая и не содержит незаполненных плейсхолдеров
        txt = res["plaque"]["text"] or ""
        check("{" not in txt and "}" not in txt and txt != "",
              f'{sc["id"]}: плашка отрендерена без дырок ("{txt[:60]}...")')


def run_drift_math():
    print("DRIFT: обе величины и дельта в валюте получателя")
    res = evaluate({
        "corridor": "RUB_TJS", "sim_date": "2026-06-15", "sim_minutes": 910,
        "entry": "PUSH", "push_sent_at_minutes": 550, "push_rate": 0.108830,
        "amount_rub": 20000, "drift_mechanic": "C",
    })
    check(res["state"] == "DRIFT", "state == DRIFT")
    check(res["push_rate"] == 0.108830 and res["current_rate"] > res["push_rate"],
          "показаны курс из пуша и текущий")
    check(res["recipient_delta"] and res["recipient_delta"] > 0,
          f'дельта у получателя = {res["recipient_delta"]} смн')
    check(abs(res["delta_bp"] - 45.0) < 1.0, f'delta_bp ≈ 45 (факт {res["delta_bp"]})')


def run_no_lookahead():
    print("Запрет заглядывания вперёд")
    s1 = corridor_series("RUB_TJS", upto=date(2026, 6, 15))
    s2 = corridor_series("RUB_TJS", upto=date(2026, 7, 15))
    m1 = {r.date: r.rate for r in s1.itertuples()}
    m2 = {r.date: r.rate for r in s2.itertuples()}
    common = set(m1) & set(m2)
    check(all(m1[d] == m2[d] for d in common) and len(common) > 100,
          "срезы на разные as_of совпадают на общих датах")


def run_reserve():
    print("Резерв: создание → +1 день → исполнение/истечение")
    if not config.FEATURE_RESERVE:
        print("  skip (FEATURE_RESERVE=false)")
        return
    rsv.reset()
    rv = rsv.create({
        "corridor": "RUB_TJS", "amount_rub": 20000, "created_on": "2026-06-15",
        "percentile": 40, "window_days": 30, "ttl_days": 7,
    })
    check(rv["state"] == "ACTIVE", "создан в ACTIVE")
    d = date(2026, 6, 15)
    terminal = None
    for _ in range(20):
        nd = None
        from .data_access import next_trading_day
        nd = next_trading_day("RUB_TJS", d)
        if not nd:
            break
        d = nd
        v = rsv.view(rv["id"], d)
        if v["state"] != "ACTIVE":
            terminal = v["state"]
            break
    check(terminal in ("EXECUTED", "EXPIRED"),
          f"за 20 торговых дней резерв дошёл до терминального состояния ({terminal})")


def main():
    run_scenarios()
    run_drift_math()
    run_no_lookahead()
    run_reserve()
    print()
    if FAIL:
        print(f"{len(FAIL)} проверок упало")
        sys.exit(1)
    print("всё зелёное")


if __name__ == "__main__":
    main()
