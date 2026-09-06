"""Общие фикстуры. Тесты гоняются в образе ml (см. tests/README.md)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "ml", ROOT / "ml" / "upstream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture
def wide_rates() -> pd.DataFrame:
    """Мини широкий ряд курсов: индекс available_at × 8 валют ЦБ, без пропусков."""
    idx = pd.to_datetime(["2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13"])
    cols = ["AMD", "KZT", "KGS", "TJS", "UZS", "USD", "EUR", "CNY"]
    data = {c: [1.0 + i * 0.01 + j * 0.1 for j in range(len(idx))]
            for i, c in enumerate(cols)}
    df = pd.DataFrame(data, index=idx)
    df.index.name = "available_at"
    return df


@pytest.fixture
def feature_row_local_low() -> dict:
    """Строка признаков: локальный минимум (низкий короткий перцентиль)."""
    return {"percentile_20d": 0.08, "percentile_30d": 0.12, "percentile_90d": 0.9,
            "consecutive_down": 3.0, "consecutive_up": 0.0, "return_5d_bps": -140.0}


@pytest.fixture
def feature_row_local_high() -> dict:
    """Строка признаков: курс у локального максимума (трендовый рынок)."""
    return {"percentile_20d": 0.95, "percentile_30d": 0.96, "percentile_90d": 0.99,
            "consecutive_down": 1.0, "consecutive_up": 0.0, "return_5d_bps": 112.0}
