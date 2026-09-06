"""Все сценарии стенда резолвятся в ожидаемое состояние на текущих data/*
(после ml-warmup это реальные курсы ЦБ и реальные срабатывания модели)."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import config
from api import data_access as da
from api import signals as sig
from api.main import app

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = json.loads((ROOT / "data" / "scenarios.json").read_text(encoding="utf-8"))


@pytest.fixture
def client(monkeypatch):
    # чистый file-режим: без ML-сервисов
    monkeypatch.setattr(config, "ML_URL", "")
    monkeypatch.setattr(config, "RATES_URL", "")
    monkeypatch.setattr(config, "MOMENT_URL", "")
    da.rates.cache_clear()
    sig._last_source.update(active="file", model_version=None, fell_back=False, error=None)
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_scenario_resolves_to_expected_state(client, sc):
    r = client.get(f"/api/scenario/{sc['id']}/run")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["state"] == d["expected_state"], (
        f"{sc['id']}: ждали {d['expected_state']}, получили {d['state']} "
        f"(delta_bp={d['delta_bp']})"
    )
    # плашка отрендерена без незаполненных плейсхолдеров
    txt = d["plaque"]["text"] or ""
    assert "{" not in txt and "}" not in txt and txt != ""


def test_signals_file_has_only_allowed_codes():
    payload = json.loads((ROOT / "data" / "signals.json").read_text(encoding="utf-8"))
    allowed = {"MOMENTUM_DOWN", "LEVEL_LOW", "REVERSAL_UP", "SEASONAL", "NEUTRAL", "ML_MOMENT"}
    codes = {s["scenario_code"] for s in payload["signals"]}
    assert codes <= allowed, f"неизвестные коды: {codes - allowed}"
    for s in payload["signals"]:
        assert {"date", "corridor", "direction", "speed", "strength",
                "scenario_code", "facts"} <= set(s)


def test_texts_has_ml_moment_code():
    texts = json.loads((ROOT / "data" / "texts.json").read_text(encoding="utf-8"))
    assert "ML_MOMENT" in texts and texts["ML_MOMENT"]["plaque"]
