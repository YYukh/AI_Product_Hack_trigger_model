import numpy as np
import pandas as pd

from research.yura.src.client_simulation import RUSSIAN_TIME_ZONES
from research.yura.src.timezone_map import (
    SUBJECT_TIMEZONE, build_subject_lift_table, build_subject_savings_table,
)


def test_subject_mapping_covers_all_active_timezones_and_excludes_kaliningrad():
    active = {row[0] for row in RUSSIAN_TIME_ZONES}
    mapped = {zone for zone in SUBJECT_TIMEZONE.values() if zone is not None}
    assert len(SUBJECT_TIMEZONE) == 85
    assert SUBJECT_TIMEZONE["Kaliningrad"] is None
    assert mapped == active


def test_subject_lift_table_propagates_zone_metrics_without_inventing_kalt():
    zones = [row[0] for row in RUSSIAN_TIME_ZONES]
    summary = pd.DataFrame({
        "timezone": zones,
        "currency": "KZT",
        "lift": np.arange(len(zones), dtype=float) + 1.0,
    })
    table = build_subject_lift_table(summary, currency="KZT")
    assert len(table) == 85
    assert table.loc[table["subject_id"].eq("Kaliningrad"), "lift"].isna().all()
    moscow_lift = summary.loc[summary["timezone"].eq("MSK"), "lift"].iloc[0]
    assert table.loc[table["subject_id"].eq("Moscow City"), "lift"].iloc[0] == moscow_lift


def test_savings_table_converts_bps_to_percent_and_pools_currencies():
    summary = pd.DataFrame({
        "timezone": ["MSK", "MSK"],
        "currency": ["KZT", "AMD"],
        "mean_realized_benefit_bps": [100.0, 300.0],
        "expected_client_transactions": [3.0, 1.0],
    })
    table = build_subject_savings_table(summary, currency="ALL")
    value = table.loc[
        table["subject_id"].eq("Moscow City"), "mean_client_savings_pct"
    ].iloc[0]
    assert value == 1.5
