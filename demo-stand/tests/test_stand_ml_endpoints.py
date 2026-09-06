"""api/main.py — /api/health c ml_services и проксирование /api/ml/*."""
import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from api import config
from api import data_access as da
from api import signals as sig
from api.main import app

PARSER, MOMENT, PUSH = "http://parser.test", "http://moment.test", "http://push.test"


@pytest.fixture
def client(monkeypatch):
    da.rates.cache_clear()
    monkeypatch.setattr(config, "RATES_URL", PARSER)
    monkeypatch.setattr(config, "MOMENT_URL", MOMENT)
    monkeypatch.setattr(config, "ML_URL", PUSH)
    sig._last_source.update(active="file", model_version=None, fell_back=False, error=None)
    with TestClient(app) as c:
        yield c
    da.rates.cache_clear()


@respx.mock
def test_health_reports_all_three_ml_services(client):
    respx.get(f"{PARSER}/health").mock(return_value=httpx.Response(200, json={"status": "ok", "source": "snap"}))
    respx.get(f"{MOMENT}/health").mock(return_value=httpx.Response(200, json={"status": "ok", "engines": 100}))
    respx.get(f"{PUSH}/health").mock(return_value=httpx.Response(200, json={"status": "ok", "model_version": "logi_v1"}))
    respx.get(f"{PUSH}/signals").mock(return_value=httpx.Response(200, json={"model_version": "logi_v1", "signals": []}))
    respx.get(f"{PARSER}/rates/wide").mock(return_value=httpx.Response(500))

    h = client.get("/api/health").json()
    assert h["version"] == "0.2.0"
    ml = h["ml_services"]
    assert ml["parser"]["reachable"] and ml["moment_model"]["reachable"] and ml["push_model"]["reachable"]
    assert ml["push_model"]["health"]["model_version"] == "logi_v1"
    assert "rates_source" in h


@respx.mock
def test_ml_engine_signals_proxied(client):
    payload = {"as_of_scored": "2026-08-13", "n_engines": 20, "n_fired": 2, "signals": []}
    respx.get(f"{MOMENT}/engine-signals").mock(return_value=httpx.Response(200, json=payload))
    r = client.get("/api/ml/engine-signals", params={"as_of": "2026-08-13", "corridor": "RUB_TJS"})
    assert r.status_code == 200 and r.json()["n_engines"] == 20


@respx.mock
def test_ml_passthrough_502_when_service_down(client):
    respx.get(f"{MOMENT}/engine-signals").mock(return_value=httpx.Response(500))
    r = client.get("/api/ml/engine-signals", params={"as_of": "2026-08-13"})
    assert r.status_code == 502


def test_ml_probe_unconfigured(monkeypatch):
    from api.main import _ml_probe
    assert _ml_probe("") == {"configured": False}
