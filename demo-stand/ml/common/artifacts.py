"""Работа с общим томом артефактов /mldata.

ml-warmup пишет сюда тяжёлые артефакты (parquet/joblib) один раз, три сервиса
читают. Файл ready.json — барьер: сервисы ждут его перед стартом.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

MLDATA = Path(os.getenv("MLDATA_DIR", "/mldata"))
READY = MLDATA / "ready.json"


def p(name: str) -> Path:
    return MLDATA / name


def read_ready() -> dict | None:
    try:
        return json.loads(READY.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_ready(payload: dict) -> None:
    MLDATA.mkdir(parents=True, exist_ok=True)
    tmp = READY.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(READY)


def wait_ready(timeout_s: int = 3600, interval_s: float = 3.0) -> dict:
    """Блокируется до появления ready.json. Бросает TimeoutError по истечении."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = read_ready()
        if r:
            return r
        time.sleep(interval_s)
    raise TimeoutError(f"ready.json не появился за {timeout_s} c ({READY})")


# --- parquet / joblib -----------------------------------------------------
def save_parquet(df, name: str) -> None:
    import pandas as pd  # noqa: F401  (ensure engine imported)

    MLDATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p(name), index=False)


def load_parquet(name: str):
    import pandas as pd

    return pd.read_parquet(p(name))


def save_joblib(obj: Any, name: str) -> None:
    import joblib

    MLDATA.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p(name))


def load_joblib(name: str) -> Any:
    import joblib

    return joblib.load(p(name))


def exists(name: str) -> bool:
    return p(name).exists()
