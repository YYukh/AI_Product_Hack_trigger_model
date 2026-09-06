"""Подстановка фактов в шаблоны из texts.json.

Правило (ТЗ §5.5): в шаблон подставляются только значения из блока facts
сигнала плюс падежные формы валюты из метаданных коридора. Если хотя бы
одного плейсхолдера нет в контексте — текст НЕ рендерится (возвращается None),
а не рендерится с дыркой. Генерация текста в рантайме запрещена: только
подстановка в заранее утверждённые формулировки.
"""
from __future__ import annotations

import re
from typing import Any

from .data_access import CORRIDORS, texts

_TOKEN = re.compile(r"\{([a-z_]+)\}")


def _fmt_num(v: Any) -> Any:
    if isinstance(v, float):
        # курсы — до 5 знаков, проценты/бп — как есть
        if abs(v) < 1:
            return f"{v:.5f}".rstrip("0").rstrip(".").replace(".", ",")
        return f"{v:,.2f}".replace(",", " ").replace(".", ",")
    if isinstance(v, int):
        return f"{v:,}".replace(",", " ")
    return v


def render(template: str | None, facts: dict, corridor: str) -> str | None:
    if not template:
        return None
    ctx: dict[str, Any] = {}
    meta = CORRIDORS.get(corridor, {})
    for k in ("currency_name", "currency_gen", "currency_dat", "currency_acc", "currency_short"):
        if k in meta:
            ctx[k] = meta[k]
    for k, v in (facts or {}).items():
        ctx[k] = v
    # производные удобные поля
    if "percentile" in facts and "percentile_inv" not in ctx:
        ctx["percentile_inv"] = 100 - int(facts["percentile"])
    if "change_bp" in facts and "change_pct" not in ctx:
        ctx["change_pct"] = round(abs(facts["change_bp"]) / 100, 1)

    needed = set(_TOKEN.findall(template))
    missing = needed - ctx.keys()
    if missing:
        return None
    return _TOKEN.sub(lambda m: str(_fmt_num(ctx[m.group(1)])), template)


def bundle(scenario_code: str, facts: dict, corridor: str) -> dict:
    """Полный набор для одного кода: push, plaque, запрещённый двойник и причина."""
    lib = texts().get(scenario_code, {})
    return {
        "scenario_code": scenario_code,
        "push": render(lib.get("push"), facts, corridor),
        "plaque": render(lib.get("plaque"), facts, corridor),
        "forbidden": lib.get("forbidden"),
        "why_forbidden": lib.get("why_forbidden"),
    }


def library() -> list[dict]:
    """Вся библиотека для панели разбора: разрешённое ↔ запрещённое ↔ почему."""
    out = []
    for code, lib in texts().items():
        out.append({
            "scenario_code": code,
            "push_template": lib.get("push"),
            "plaque_template": lib.get("plaque"),
            "forbidden": lib.get("forbidden"),
            "why_forbidden": lib.get("why_forbidden"),
        })
    return out
