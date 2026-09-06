"""Коммуникационная политика: месячный бюджет, cooldown, тихие часы.

Влияет на панель разбора и на сценарий перебора коммуникаций (S6).
Молчание должно быть объяснимым — каждый подавленный сигнал получает причину.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from . import config


def plan(candidate_pushes: list[dict], *, budget: int | None = None,
         cooldown_days: int | None = None, timezone_offset_h: int = 3) -> list[dict]:
    """candidate_pushes: [{"date": "YYYY-MM-DD", "minutes": int, "corridor": str,
    "scenario_code": str}]. Возвращает те же элементы с полями sent / reason."""
    budget = config.PUSH_BUDGET_MONTH if budget is None else budget
    cooldown_days = config.PUSH_COOLDOWN_DAYS if cooldown_days is None else cooldown_days
    q_start, q_end = config.QUIET_HOURS

    out = []
    sent_in_month: dict[tuple[int, int, str], int] = {}
    last_sent: dict[str, date] = {}

    for p in sorted(candidate_pushes, key=lambda x: (x["date"], x.get("minutes", 0))):
        d = date.fromisoformat(p["date"])
        corridor = p.get("corridor", "?")
        minutes = int(p.get("minutes", 12 * 60))
        local_hour = ((minutes // 60) + 0) % 24  # minutes уже в локальном времени портрета
        reason = None

        if q_start <= local_hour or local_hour < q_end:
            reason = "тихие часы по местному времени портрета"
        else:
            key = (d.year, d.month, corridor)
            if sent_in_month.get(key, 0) >= budget:
                reason = f"исчерпан месячный бюджет ({budget}) по коридору"
            elif corridor in last_sent and (d - last_sent[corridor]).days < cooldown_days:
                reason = f"cooldown {cooldown_days} дн. после предыдущего пуша"

        sent = reason is None
        if sent:
            key = (d.year, d.month, corridor)
            sent_in_month[key] = sent_in_month.get(key, 0) + 1
            last_sent[corridor] = d

        item = dict(p)
        item["sent"] = sent
        item["reason"] = reason
        out.append(item)
    return out
