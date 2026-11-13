# -*- coding: utf-8 -*-
"""Tests for report level grounding and ATR position sizing."""

from src.core.grounding_verifier import (
    compute_atr_position_sizing,
    verify_and_normalize_levels,
)


def test_verify_and_normalize_levels_valid_long_setup() -> None:
    levels = verify_and_normalize_levels(
        {
            "ideal_buy": "100.00元",
            "secondary_buy": "95.00元",
            "stop_loss": "92.00元",
            "take_profit": "120.00元",
        },
        current_price=102.0,
        action="buy",
    )

    assert levels["ideal_buy"] == 100.0
    assert levels["reference_price"] == 102.0
    assert levels["status"] == "active"
    assert levels["validation"] == "ok"
    assert levels["risk_reward_ratio"] == 2.5


def test_verify_and_normalize_levels_detects_conflict_and_anchor_drift() -> None:
    conflict = verify_and_normalize_levels(
        {"ideal_buy": "100", "stop_loss": "105", "take_profit": "120"},
        current_price=101.0,
        action="watch",
    )
    assert conflict["status"] == "conditional_watch"
    assert conflict["validation"] == "direction_conflict"

    drift = verify_and_normalize_levels(
        {"ideal_buy": "60", "stop_loss": "55", "take_profit": "80"},
        current_price=100.0,
        action="buy",
    )
    assert drift["validation"] == "out_of_range"


def test_compute_atr_position_sizing_is_bounded_and_safe() -> None:
    levels = {"reference_price": 100.0, "stop_loss": 96.0, "validation": "ok"}

    buy_position = compute_atr_position_sizing("buy", 80, levels, data_quality_level="good")
    assert 5 <= buy_position["min_pct"] <= buy_position["max_pct"] <= 35
    assert buy_position["stop_distance_pct"] == 4.0

    limited_position = compute_atr_position_sizing("buy", 85, levels, data_quality_level="limited")
    assert limited_position["max_pct"] <= 20
    assert limited_position["cap_reason"] == "limited_data_quality"

    blocked_position = compute_atr_position_sizing(
        "buy", 80, {**levels, "validation": "direction_conflict"}, data_quality_level="good"
    )
    assert blocked_position["min_pct"] == blocked_position["max_pct"] == 0

