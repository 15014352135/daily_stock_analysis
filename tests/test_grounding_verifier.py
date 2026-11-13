# -*- coding: utf-8 -*-
"""Tests for numeric grounding verifier, ATR position sizing, and auto-populated structured_report."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("litellm", MagicMock())

from api.v1.schemas.history import AnalysisReport, ReportMeta, ReportStrategy, ReportSummary
from src.core.grounding_verifier import (
    compute_atr_position_sizing,
    verify_and_normalize_levels,
)


def test_verify_and_normalize_levels_valid_long_setup() -> None:
    strategy = {
        "ideal_buy": "100.00元",
        "secondary_buy": "95.00元",
        "stop_loss": "92.00元",
        "take_profit": "120.00元",
    }
    levels = verify_and_normalize_levels(strategy, current_price=102.0, action="buy")

    assert levels["ideal_buy"] == 100.0
    assert levels["secondary_buy"] == 95.0
    assert levels["stop_loss"] == 92.0
    assert levels["take_profit"] == 120.0
    assert levels["reference_price"] == 102.0
    assert levels["status"] == "active"
    assert levels["validation"] == "ok"
    assert levels["risk_reward_ratio"] == 2.5


def test_verify_and_normalize_levels_detects_conflict_and_watch_status() -> None:
    strategy = {
        "ideal_buy": "100.0",
        "stop_loss": "105.0",
        "take_profit": "120.0",
    }
    levels = verify_and_normalize_levels(strategy, current_price=101.0, action="watch")

    assert levels["status"] == "conditional_watch"
    assert levels["validation"] == "direction_conflict"
    assert levels["risk_reward_ratio"] is None


def test_verify_and_normalize_levels_preserves_entry_semantics_and_checks_both_entries() -> None:
    levels = verify_and_normalize_levels(
        {
            "ideal_buy": "100",
            "secondary_buy": "110",
            "stop_loss": "105",
            "take_profit": "120",
        },
        current_price=100.0,
        action="buy",
    )

    assert levels["ideal_buy"] == 100.0
    assert levels["secondary_buy"] == 110.0
    assert levels["validation"] == "direction_conflict"
    position = compute_atr_position_sizing("buy", 80, levels, data_quality_level="good")
    assert position["max_pct"] == 0
    assert position["cap_reason"] == "invalid_level_validation"


def test_verify_and_normalize_levels_detects_out_of_range_anchor() -> None:
    strategy = {
        "ideal_buy": "60.0",
        "stop_loss": "55.0",
        "take_profit": "80.0",
    }
    levels = verify_and_normalize_levels(strategy, current_price=100.0, action="buy")

    assert levels["validation"] == "out_of_range"


def test_compute_atr_position_sizing_caps_and_non_offensive_zero() -> None:
    levels = {"reference_price": 100.0, "stop_loss": 96.0}

    buy_pos = compute_atr_position_sizing("buy", 80, levels, data_quality_level="good")
    assert buy_pos["min_pct"] >= 5
    assert buy_pos["max_pct"] <= 35
    assert buy_pos["min_pct"] <= buy_pos["max_pct"]
    assert buy_pos["stop_distance_pct"] == 4.0

    limited_pos = compute_atr_position_sizing("buy", 85, levels, data_quality_level="limited")
    assert limited_pos["max_pct"] <= 20
    assert limited_pos["cap_reason"] == "limited_data_quality"

    hold_pos = compute_atr_position_sizing("hold", 80, levels, data_quality_level="good")
    assert hold_pos["min_pct"] == 0
    assert hold_pos["max_pct"] == 0
    assert hold_pos["cap_reason"] == "non_offensive_action"


def test_compute_atr_position_sizing_never_exceeds_risk_budget_after_rounding() -> None:
    position = compute_atr_position_sizing(
        "buy",
        80,
        {"reference_price": 100.0, "stop_loss": 40.0},
        data_quality_level="good",
    )

    assert position["stop_distance_pct"] == 60.0
    assert position["max_pct"] == 2
    assert position["stop_distance_pct"] * position["max_pct"] / 100 <= position["risk_budget_pct"]


def test_compute_atr_position_sizing_high_conviction_respects_risk_budget() -> None:
    position = compute_atr_position_sizing(
        "buy", 100, {"reference_price": 100.0, "stop_loss": 96.0}, data_quality_level="good"
    )

    assert position["stop_distance_pct"] == 4.0
    assert position["max_pct"] == 30
    assert position["risk_budget_pct"] == 1.2


@pytest.mark.parametrize("action", ["buy", "add"])
@pytest.mark.parametrize("score", [60, 80, 81, 88, 100])
@pytest.mark.parametrize("quality", ["good", "limited", None])
@pytest.mark.parametrize("stop_loss", [99.0, 96.0, 94.0, 40.0, 95.8619])
def test_compute_atr_position_sizing_final_risk_bound(
    action: str, score: int, quality: str | None, stop_loss: float
) -> None:
    position = compute_atr_position_sizing(
        action, score, {"reference_price": 100.0, "stop_loss": stop_loss}, data_quality_level=quality
    )

    assert 0 <= position["min_pct"] <= position["max_pct"] <= (20 if quality == "limited" else 35)
    assert position["max_pct"] * position["stop_distance_pct"] / 100 <= position["risk_budget_pct"]
    assert position["max_pct"] * (100.0 - stop_loss) / 100 <= position["risk_budget_pct"]


@pytest.mark.parametrize("score", [76, 100])
def test_analysis_report_auto_populates_structured_report(score: int) -> None:
    report = AnalysisReport(
        meta=ReportMeta(
            query_id="q-100",
            stock_code="600519",
            stock_name="贵州茅台",
            report_type="detailed",
            current_price=1680.0,
        ),
        summary=ReportSummary(
            analysis_summary="均线多头排列，回踩支撑位。",
            operation_advice="买入",
            action="buy",
            trend_prediction="看多",
            sentiment_score=score,
        ),
        strategy=ReportStrategy(
            ideal_buy="1660.0",
            secondary_buy="1620.0",
            stop_loss="1580.0",
            take_profit="1840.0",
        ),
    )

    assert report.structured_report is not None
    assert report.structured_report.schema_version == "research-artifact-v1"
    meta = report.structured_report.metadata
    assert meta["levels"]["ideal_buy"] == 1660.0
    assert meta["levels"]["status"] == "active"
    assert meta["levels"]["validation"] == "ok"
    assert meta["position"]["max_pct"] >= 5
    position = meta["position"]
    assert position["max_pct"] * position["stop_distance_pct"] / 100 <= position["risk_budget_pct"]
