"""ml/common/artifacts.py — общий том /mldata и барьер ready.json."""
import time

import pandas as pd
import pytest


@pytest.fixture
def art(tmp_path, monkeypatch):
    monkeypatch.setenv("MLDATA_DIR", str(tmp_path))
    import importlib

    from common import artifacts as A
    importlib.reload(A)
    return A


def test_ready_roundtrip(art):
    assert art.read_ready() is None
    art.write_ready({"model_version": "v1", "n_push_events": 3})
    got = art.read_ready()
    assert got["model_version"] == "v1" and got["n_push_events"] == 3


def test_wait_ready_returns_when_present(art):
    art.write_ready({"ok": True})
    t0 = time.monotonic()
    assert art.wait_ready(timeout_s=5, interval_s=0.1)["ok"] is True
    assert time.monotonic() - t0 < 1


def test_wait_ready_times_out(art):
    with pytest.raises(TimeoutError):
        art.wait_ready(timeout_s=1, interval_s=0.2)


def test_parquet_roundtrip(art):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    art.save_parquet(df, "t.parquet")
    assert art.exists("t.parquet")
    back = art.load_parquet("t.parquet")
    pd.testing.assert_frame_equal(df, back)


def test_joblib_roundtrip(art):
    art.save_joblib({"threshold": 0.61}, "m.joblib")
    assert art.load_joblib("m.joblib")["threshold"] == 0.61
