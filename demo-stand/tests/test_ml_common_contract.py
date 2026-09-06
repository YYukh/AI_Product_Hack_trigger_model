"""ml/common/contract.py — перевод выхода ML в схему сигналов стенда."""
from common import contract as C


def test_scenario_code_window_closing_is_reversal_up():
    assert C.scenario_code("WINDOW_CLOSING", "anything", month=6) == "REVERSAL_UP"


def test_scenario_code_local_low_is_level_low(feature_row_local_low):
    code = C.scenario_code("GOOD_NOW", "hist_gradient_boosting", month=6,
                           feature_row=feature_row_local_low)
    assert code == "LEVEL_LOW"


def test_scenario_code_trending_high_falls_back_to_ml_moment(feature_row_local_high):
    # G0 на трендовом рынке (курс у максимума) не должен называться LEVEL_LOW
    code = C.scenario_code("GOOD_NOW", "hist_gradient_boosting", month=6,
                           feature_row=feature_row_local_high)
    assert code == "ML_MOMENT"


def test_scenario_code_momentum_down_needs_streak_and_rule_name(feature_row_local_low):
    code = C.scenario_code("GOOD_NOW", "momentum_down__OR__down_streak", month=6,
                           feature_row=feature_row_local_low)
    assert code == "MOMENTUM_DOWN"
    # тот же движок, но без серии снижения → не MOMENTUM_DOWN
    weak = dict(feature_row_local_low, consecutive_down=1.0)
    assert C.scenario_code("GOOD_NOW", "momentum_down", month=6, feature_row=weak) != "MOMENTUM_DOWN"


def test_scenario_code_seasonal_slot_in_winter(feature_row_local_low):
    assert C.scenario_code("GOOD_NOW", "hist_gradient_boosting", month=12,
                           feature_row=feature_row_local_low) == "SEASONAL"


def test_direction_and_speed():
    assert C.direction("GOOD_NOW") == "favorable"
    assert C.direction("WINDOW_CLOSING") == "closing"
    assert C.speed(1) == "fast" and C.speed(3) == "fast"
    assert C.speed(5) == "slow" and C.speed(20) == "slow"


def test_facts_use_short_window_percentile(feature_row_local_high):
    f = C.facts_from_feature_row(feature_row_local_high, rate=8.9, window_days=20)
    assert f["window_days"] == 20
    assert f["percentile"] == 95            # из percentile_20d, не 99 из 90д
    assert f["rate"] == 8.9
    assert f["streak_days"] >= 1
    assert isinstance(f["change_bp"], int)


def test_to_stand_signal_shape(feature_row_local_low):
    s = C.to_stand_signal(
        date="2026-08-13", corridor="RUB_TJS", scenario="GOOD_NOW",
        engine_name="hist_gradient_boosting", horizon=3, strength=0.71,
        feature_row=feature_row_local_low, rate=8.8,
    )
    assert set(s) == {"date", "corridor", "indicator", "direction", "speed",
                      "strength", "scenario_code", "facts"}
    assert s["date"] == "2026-08-13"
    assert s["corridor"] == "RUB_TJS"
    assert s["scenario_code"] in C.STAND_CODES
    assert s["direction"] == "favorable"
    assert s["speed"] == "fast"
    assert 0.0 <= s["strength"] <= 1.0
    assert {"rate", "percentile", "window_days"} <= set(s["facts"])
