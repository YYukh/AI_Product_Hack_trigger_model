"""Данные → признаки → targets → scoring_data.

Тонкая обёртка над ml/upstream/src. Повторяет цепочку из
notebooks/prod_pipline.ipynb, но с оффлайн-фолбэком на снапшот курсов ЦБ.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd

# Все 8 валют загрузчика ЦБ (5 продакшн + контекстные USD/EUR/CNY для признаков).
CBR_COLUMNS = ("AMD", "KZT", "KGS", "TJS", "UZS", "USD", "EUR", "CNY")
PRODUCTION_CURRENCIES = ("AMD", "KZT", "KGS", "TJS", "UZS")
CORRIDOR_BY_CURRENCY = {c: f"RUB_{c}" for c in PRODUCTION_CURRENCIES}

FALLBACK_CSV = Path(os.getenv("CBR_FALLBACK_CSV", "/app/fallback/cbr_rates.csv"))


def load_wide_rates(
    start: str = "2020-01-01",
    end: str | None = None,
    *,
    prefer: str = "auto",
    raw_dir: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Широкий ряд курсов (индекс available_at × валюта, RUB за единицу).

    prefer: 'live' — только ЦБ; 'fallback' — только снапшот; 'auto' — ЦБ,
    при ошибке снапшот. Возвращает (rates, source) где source ∈ {'live','fallback'}.
    """
    end = end or date.today().isoformat()

    if prefer != "fallback":
        try:
            from src.cbr_loader import CURRENCIES, load_cbr_rates

            rates = load_cbr_rates(
                pd.Timestamp(start).date(),
                pd.Timestamp(end).date(),
                currencies=CURRENCIES,
                raw_dir=Path(raw_dir) if raw_dir else None,
            )
            rates = _clean_wide(rates)
            return rates, "live"
        except Exception as exc:  # noqa: BLE001
            if prefer == "live":
                raise
            print(f"[pipeline] ЦБ недоступен ({type(exc).__name__}: {exc}); "
                  f"использую снапшот {FALLBACK_CSV}", flush=True)

    df = pd.read_csv(FALLBACK_CSV, parse_dates=["available_at"]).set_index("available_at")
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.Timedelta(days=1)
    df = df.loc[(df.index >= lo) & (df.index <= hi)]
    return _clean_wide(df), "fallback"


def _clean_wide(rates: pd.DataFrame) -> pd.DataFrame:
    rates = rates.copy()
    rates.index = pd.DatetimeIndex(pd.to_datetime(rates.index)).normalize()
    rates.index.name = "available_at"
    keep = [c for c in CBR_COLUMNS if c in rates.columns]
    rates = rates[keep]
    # market_data.validate_wide_rates запрещает пропуски → берём только полные дни
    rates = rates.dropna(how="any")
    rates = rates[~rates.index.duplicated(keep="last")].sort_index()
    return rates


def build_scoring_data(rates: pd.DataFrame):
    """rates(wide) → (scoring_data, target_registry, features, panel)."""
    from src.features import build_features
    from src.market_data import build_daily_market_panel
    from src.outcomes import add_future_outcomes
    from src.production_config import PRODUCTION_CURRENCIES as PROD
    from src.production_config import PRODUCTION_HORIZONS
    from src.targets import build_targets

    panel = build_daily_market_panel(rates)
    features = build_features(panel)
    outcomes = add_future_outcomes(features, horizons=PRODUCTION_HORIZONS)
    dataset, target_registry = build_targets(outcomes, horizons=PRODUCTION_HORIZONS)
    scoring = dataset.loc[
        dataset["is_update_day"] & dataset["currency"].isin(PROD)
    ].copy()
    scoring["available_at"] = pd.to_datetime(scoring["available_at"])
    return scoring, target_registry, features, panel


def rates_long(rates: pd.DataFrame) -> pd.DataFrame:
    """Широкий ряд → длинный {corridor,date,rate,is_stale} для стенда."""
    rows = []
    for cur in PRODUCTION_CURRENCIES:
        if cur not in rates.columns:
            continue
        s = rates[cur].dropna()
        for ts, val in s.items():
            rows.append({
                "corridor": CORRIDOR_BY_CURRENCY[cur],
                "date": pd.Timestamp(ts).date().isoformat(),
                "rate": round(float(val), 6),
                "is_stale": False,
            })
    return pd.DataFrame(rows).sort_values(["corridor", "date"]).reset_index(drop=True)
