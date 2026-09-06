"""Чтение файлов данных и расчётные помощники по ряду курсов.

Базы данных нет. rates.csv грузится в pandas один раз при старте, JSON-файлы
читаются с диска (маленькие). Реальное время нигде не используется — стенд
работает на исторических датах (sim_date).
"""
from __future__ import annotations

import json
import os
from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

from . import config

# --- метаданные коридоров -------------------------------------------------
# Падежные формы валют нужны текстам из texts.json. Числа сюда не попадают.
CORRIDORS: dict[str, dict] = {
    "RUB_TJS": dict(country="Таджикистан", flag="🇹🇯", currency_code="TJS",
                    currency_name="сомони", currency_gen="сомони", currency_dat="сомони",
                    currency_acc="сомони", currency_short="смн"),
    "RUB_UZS": dict(country="Узбекистан", flag="🇺🇿", currency_code="UZS",
                    currency_name="сум", currency_gen="сума", currency_dat="суму",
                    currency_acc="сум", currency_short="сум"),
    "RUB_KGS": dict(country="Кыргызстан", flag="🇰🇬", currency_code="KGS",
                    currency_name="сом", currency_gen="сома", currency_dat="сому",
                    currency_acc="сом", currency_short="сом"),
    "RUB_AMD": dict(country="Армения", flag="🇦🇲", currency_code="AMD",
                    currency_name="драм", currency_gen="драма", currency_dat="драму",
                    currency_acc="драм", currency_short="драм"),
    "RUB_KZT": dict(country="Казахстан", flag="🇰🇿", currency_code="KZT",
                    currency_name="тенге", currency_gen="тенге", currency_dat="тенге",
                    currency_acc="тенге", currency_short="тг"),
}


def _path(name: str) -> str:
    return os.path.abspath(os.path.join(config.DATA_DIR, name))


_rates_source = {"active": "file", "error": None}


def rates_source() -> dict:
    return dict(_rates_source)


def _rates_from_parser() -> pd.DataFrame:
    """Тянет широкий ряд у парсера котировок (RATES_URL) и приводит к формату
    data/rates.csv: columns date, corridor, rate, is_stale."""
    import httpx

    with httpx.Client(timeout=config.ML_TIMEOUT_S * 3) as c:
        r = c.get(config.RATES_URL.rstrip("/") + "/rates/wide")
        r.raise_for_status()
        payload = r.json()
    cols = [x for x in payload.get("columns", []) if x in {"AMD", "KZT", "KGS", "TJS", "UZS"}]
    recs = []
    for row in payload.get("rows", []):
        for cur in cols:
            if row.get(cur) is None:
                continue
            recs.append({"date": row["date"], "corridor": f"RUB_{cur}",
                         "rate": float(row[cur]), "is_stale": False})
    df = pd.DataFrame(recs)
    if df.empty:
        raise ValueError("парсер вернул пустой ряд")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values(["corridor", "date"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def rates() -> pd.DataFrame:
    if config.RATES_URL:
        try:
            df = _rates_from_parser()
            _rates_source.update(active="parser", error=None)
            return df
        except Exception as e:  # noqa: BLE001 — откат на файл
            _rates_source.update(active="file", error=f"{type(e).__name__}: {e}")
    df = pd.read_csv(_path("rates.csv"), dtype={"corridor": str})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["rate"] = df["rate"].astype(float)
    df["is_stale"] = df["is_stale"].astype(str).str.lower().eq("true")
    return df.sort_values(["corridor", "date"]).reset_index(drop=True)


def load_json(name: str):
    with open(_path(name), encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def personas() -> list[dict]:
    return load_json("personas.json")


@lru_cache(maxsize=1)
def scenarios() -> list[dict]:
    return load_json("scenarios.json")


@lru_cache(maxsize=1)
def texts() -> dict:
    return load_json("texts.json")


def date_range() -> tuple[str, str]:
    df = rates()
    return str(df["date"].min()), str(df["date"].max())


# --- операции над рядом --------------------------------------------------
def corridor_series(corridor: str, upto: date | None = None) -> pd.DataFrame:
    """Ряд по коридору, только данные <= upto (запрет заглядывания вперёд)."""
    df = rates()
    df = df[df["corridor"] == corridor]
    if upto is not None:
        df = df[df["date"] <= upto]
    return df.reset_index(drop=True)


def rate_on(corridor: str, d: date) -> tuple[float, bool]:
    """Курс на дату d (или ближайший предыдущий торговый день). Возвращает
    (rate, is_stale)."""
    df = corridor_series(corridor, upto=d)
    if df.empty:
        raise KeyError(f"нет курса для {corridor} на {d}")
    last = df.iloc[-1]
    stale = bool(last["is_stale"]) or last["date"] != d
    return float(last["rate"]), stale


def next_trading_day(corridor: str, d: date) -> date | None:
    df = rates()
    fut = df[(df["corridor"] == corridor) & (df["date"] > d) & (~df["is_stale"])]
    if fut.empty:
        return None
    return fut.iloc[0]["date"]


def window_values(corridor: str, as_of: date, days: int) -> list[float]:
    lo = as_of - timedelta(days=days)
    df = corridor_series(corridor, upto=as_of)
    df = df[df["date"] >= lo]
    return df["rate"].tolist()


def percentile_rank(values: list[float], x: float) -> int:
    """Доля значений <= x, в процентах 0..100."""
    if not values:
        return 50
    below = sum(1 for v in values if v <= x)
    return round(100 * below / len(values))


def percentile_value(values: list[float], p: int) -> float:
    """P-й перцентиль ряда (линейная интерполяция), p в 0..100."""
    if not values:
        raise ValueError("пустой ряд")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
