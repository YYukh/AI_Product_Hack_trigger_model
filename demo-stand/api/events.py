"""Единый лог событий. Append в logs/events.jsonl, отдаётся в панель разбора.
Персональных данных в payload быть не должно — портрет анонимен, сессия
идентифицируется случайным session_id с фронта.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque

from . import config

_LOCK = threading.Lock()
_RECENT: deque[dict] = deque(maxlen=500)

ALLOWED_TYPES = {
    "signal_emitted", "push_suppressed", "push_sent", "push_opened", "push_expired",
    "screen_shown", "transfer_confirmed", "abandoned", "unsubscribed",
    "reserve_created", "reserve_executed", "reserve_cancelled", "reserve_expired",
    "postponed",
}


def _path() -> str:
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    return os.path.join(config.LOGS_DIR, "events.jsonl")


def log_event(etype: str, payload: dict | None = None, sim_date: str | None = None,
              session_id: str | None = None) -> dict:
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "type": etype,
        "sim_date": sim_date,
        "session_id": session_id,
        "payload": payload or {},
        "known_type": etype in ALLOWED_TYPES,
    }
    with _LOCK:
        _RECENT.append(rec)
        try:
            with open(_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass  # лог не должен ронять запрос
    return rec


def recent(limit: int = 200, session_id: str | None = None) -> list[dict]:
    items = list(_RECENT)
    if session_id:
        items = [e for e in items if e.get("session_id") == session_id]
    return items[-limit:]
