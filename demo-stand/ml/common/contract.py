"""Перевод выхода ML-пайплайна в схему сигналов демо-стенда.

Стенд ждёт (ТЗ §7 / api/signals.py):
    {as_of, model_version, signals: [{
        date, corridor, indicator, direction, speed, strength,
        scenario_code, facts: {rate, percentile, window_days, streak_days, change_bp}
    }]}
где scenario_code ∈ {MOMENTUM_DOWN, LEVEL_LOW, REVERSAL_UP, SEASONAL, NEUTRAL}.

ML отдаёт scenario ∈ {GOOD_NOW, WINDOW_CLOSING}, target_family ∈ {G0, W1},
архитектуру движка (level_low / momentum_down / down_streak / trend_down /
near_low / up_streak / momentum_up / high_volatility) и горизонт 1..20.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

STAND_CODES = ("MOMENTUM_DOWN", "LEVEL_LOW", "REVERSAL_UP", "SEASONAL",
               "NEUTRAL", "ML_MOMENT")


def scenario_code(scenario: str, engine_name: str, month: int | None = None,
                  feature_row: "pd.Series | dict | None" = None) -> str:
    """ML-сценарий → код текста стенда, но только если факт-строка это
    подтверждает. Иначе — нейтральный ML_MOMENT (факт + число + период, без
    прогноза), потому что G0-таргет модели (локальный минимум в ±h) на
    трендовом рынке не равен «курс исторически низкий»."""
    name = (engine_name or "").lower()
    if scenario == "WINDOW_CLOSING":
        return "REVERSAL_UP"

    g = (lambda k, d=0.0: _num(feature_row.get(k) if feature_row is not None else None, d))
    pctl = g("percentile_20d", g("percentile_30d", 0.9))
    down_streak = g("consecutive_down", 0.0)

    if down_streak >= 2 and ("momentum_down" in name or "down_streak" in name
                             or "trend_down" in name):
        code = "MOMENTUM_DOWN"
    elif pctl <= 0.35:
        code = "LEVEL_LOW"
    else:
        code = "ML_MOMENT"

    if month in (11, 12, 1) and code in ("LEVEL_LOW", "ML_MOMENT"):
        return "SEASONAL"
    return code


def direction(scenario: str) -> str:
    return "favorable" if scenario == "GOOD_NOW" else "closing"


def speed(horizon: int) -> str:
    return "fast" if int(horizon) <= 3 else "slow"


def _num(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


def facts_from_feature_row(row: pd.Series | dict | None, rate: float | None,
                           *, window_days: int = 20) -> dict:
    """facts для подстановки в тексты стенда — из causal-признаков (features.py).
    percentile берём по короткому окну (локальное положение курса), а не по 90д —
    иначе на трендовом рынке всё «в 99 перцентиле»."""
    g = (lambda k, d=0.0: _num(row.get(k) if row is not None else None, d))
    pkey = f"percentile_{window_days}d"
    percentile_frac = g(pkey, g("percentile_30d", g("percentile_90d", 0.5)))
    percentile = int(round(max(0.0, min(1.0, percentile_frac)) * 100))
    down_streak = int(round(g("consecutive_down", 0.0)))
    up_streak = int(round(g("consecutive_up", 0.0)))
    change_bp = int(round(g("return_5d_bps", g("return_3d_bps", 0.0))))
    facts = {
        "rate": round(_num(rate, 0.0), 6) if rate is not None else None,
        "percentile": percentile,
        "window_days": window_days,
        "streak_days": down_streak or up_streak or 1,
        "change_bp": change_bp,
    }
    return {k: v for k, v in facts.items() if v is not None}


def to_stand_signal(
    *,
    date: str,
    corridor: str,
    scenario: str,
    engine_name: str,
    horizon: int,
    strength: float,
    feature_row: pd.Series | dict | None,
    rate: float | None,
) -> dict:
    ts = pd.Timestamp(date)
    code = scenario_code(scenario, engine_name, month=ts.month, feature_row=feature_row)
    return {
        "date": ts.date().isoformat(),
        "corridor": corridor,
        "indicator": engine_name,
        "direction": direction(scenario),
        "speed": speed(horizon),
        "strength": round(_num(strength, 0.0), 3),
        "scenario_code": code,
        "facts": facts_from_feature_row(feature_row, rate),
    }
