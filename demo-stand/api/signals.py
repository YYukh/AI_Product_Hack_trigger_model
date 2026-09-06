"""Источник сигналов: файл data/signals.json или HTTP-модель по ML_URL.

Контракт модели (ТЗ §7): GET /health и GET /signals?as_of=&corridors=.
Модель отдаёт факты и scenario_code, НЕ текст. При ошибке/таймауте HTTP —
молчаливый откат на файл, факт отката виден в /api/health.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from . import config
from .data_access import load_json

_last_source = {"active": "file", "model_version": None, "fell_back": False,
                "checked_at": None, "error": None}


def status() -> dict:
    return dict(_last_source)


def _from_file(as_of: str, corridors: list[str]) -> dict:
    payload = load_json("signals.json")
    sig = [s for s in payload.get("signals", [])
           if s["date"] <= as_of and (not corridors or s["corridor"] in corridors)]
    _last_source.update(active="file", model_version=payload.get("model_version"),
                        checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return {"as_of": as_of, "model_version": payload.get("model_version"), "signals": sig}


def _from_http(as_of: str, corridors: list[str]) -> dict:
    url = config.ML_URL.rstrip("/") + "/signals"
    params = {"as_of": as_of}
    if corridors:
        params["corridors"] = ",".join(corridors)
    with httpx.Client(timeout=config.ML_TIMEOUT_S) as c:
        r = c.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    sig = [s for s in data.get("signals", [])
           if s["date"] <= as_of and (not corridors or s["corridor"] in corridors)]
    _last_source.update(active="http", model_version=data.get("model_version"),
                        fell_back=False, error=None,
                        checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return {"as_of": as_of, "model_version": data.get("model_version"), "signals": sig}


def get_signals(as_of: str, corridors: list[str] | None = None) -> dict:
    corridors = corridors or []
    if not config.ML_URL:
        _last_source["fell_back"] = False
        return _from_file(as_of, corridors)
    try:
        return _from_http(as_of, corridors)
    except Exception as e:  # noqa: BLE001 — любой сбой модели = откат на файл
        _last_source.update(fell_back=True, error=f"{type(e).__name__}: {e}")
        out = _from_file(as_of, corridors)
        _last_source["active"] = "file"
        return out


def health() -> dict:
    """Проверка модели для /api/health. Никогда не бросает."""
    if not config.ML_URL:
        return {"active": "file", "model_version": load_json("signals.json").get("model_version")}
    try:
        with httpx.Client(timeout=config.ML_TIMEOUT_S) as c:
            r = c.get(config.ML_URL.rstrip("/") + "/health")
            r.raise_for_status()
            return {"active": "http", "model_version": r.json().get("model_version"),
                    "fell_back": False}
    except Exception as e:  # noqa: BLE001
        return {"active": "file", "model_version": load_json("signals.json").get("model_version"),
                "fell_back": True, "error": f"{type(e).__name__}: {e}"}


def signal_for(as_of: str, corridor: str) -> dict[str, Any] | None:
    """Последний сигнал по коридору на дату <= as_of (для истории/ленты)."""
    sig = get_signals(as_of, [corridor]).get("signals", [])
    sig = [s for s in sig if s["corridor"] == corridor]
    if not sig:
        return None
    return sorted(sig, key=lambda s: s["date"])[-1]


def signal_on(as_of: str, corridor: str) -> dict[str, Any] | None:
    """Сигнал, сработавший ИМЕННО на дату среза. Это то, что делает день
    сигнальным для входа SELF и что подставляется в плашку."""
    sig = get_signals(as_of, [corridor]).get("signals", [])
    exact = [s for s in sig if s["corridor"] == corridor and s["date"] == as_of]
    return exact[-1] if exact else None
