# -*- coding: utf-8 -*-
"""Tests for numeric grounding verifier, ATR position sizing, and auto-populated structured_report."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

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


def test_analysis_report_auto_populates_structured_report() -> None:
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
            sentiment_score=76,
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
