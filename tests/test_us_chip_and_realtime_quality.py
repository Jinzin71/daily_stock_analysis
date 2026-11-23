# -*- coding: utf-8 -*-
"""US chip-distribution estimate, Yahoo realtime source labelling and final-bar guard."""
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from data_provider.chip_estimator import compute_cyq_metrics, estimate_chip_distribution
from data_provider.realtime_types import RealtimeSource
from data_provider.yfinance_fetcher import YfinanceFetcher
from src.core.pipeline import StockAnalysisPipeline


def _bars(n=250, start=10.0, end=20.0, volume=1_000_000):
    close = np.linspace(start, end, n)
    idx = pd.date_range("2025-11-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": close, "High": close * 1.02, "Low": close * 0.98, "Close": close, "Volume": volume},
        index=idx,
    )


def test_estimate_chip_distribution_rising_trend_is_mostly_in_profit():
    bars = _bars().rename(columns=str.lower)
    dist = estimate_chip_distribution(bars)
    assert dist is not None and abs(dist["percent"].sum() - 100) < 1e-6
    metrics = compute_cyq_metrics(dist, current_price=float(bars["close"].iloc[-1]))
    assert metrics["获利比例"] > 0.8
    assert 10 < metrics["平均成本"] < 20
    assert metrics["90成本-低"] < metrics["70成本-低"] <= metrics["70成本-高"] < metrics["90成本-高"]
    assert 0 < metrics["70集中度"] < metrics["90集中度"] < 1


def test_estimate_chip_distribution_rejects_short_history():
    assert estimate_chip_distribution(_bars(n=10).rename(columns=str.lower)) is None


def test_tushare_compute_cyq_metrics_delegates_to_shared_impl():
    from data_provider.tushare_fetcher import TushareFetcher

    df = pd.DataFrame({"price": [9.0, 10.0, 11.0], "percent": [20.0, 50.0, 30.0]})
    fetcher = object.__new__(TushareFetcher)
    assert fetcher.compute_cyq_metrics(df, 10.0) == compute_cyq_metrics(df, 10.0)


@patch("yfinance.Ticker")
def test_yfinance_chip_distribution_for_us_symbol(ticker_factory: MagicMock):
    ticker = MagicMock()
    ticker.history.return_value = _bars()
    ticker.fast_info = SimpleNamespace(shares=None)
    ticker_factory.return_value = ticker

    chip = YfinanceFetcher().get_chip_distribution("SOXL")
    assert chip is not None
    assert chip.source == "yfinance_estimated"
    assert chip.avg_cost > 0 and chip.concentration_90 > 0
    assert chip.date == _bars().index[-1].strftime("%Y-%m-%d")


def test_yfinance_chip_distribution_skips_a_share_and_index():
    fetcher = YfinanceFetcher()
    with patch("yfinance.Ticker") as ticker_factory:
        assert fetcher.get_chip_distribution("600519") is None
        assert fetcher.get_chip_distribution("SPX") is None
        ticker_factory.assert_not_called()


@patch("yfinance.Ticker")
def test_yfinance_realtime_quote_is_labelled_yfinance_with_volume_ratio(ticker_factory: MagicMock):
    ticker = MagicMock()
    ticker.fast_info = SimpleNamespace(
        lastPrice=30.0, previousClose=29.0, open=29.5, dayHigh=30.5, dayLow=29.0,
        lastVolume=150_000_000, marketCap=None, tenDayAverageVolume=100_000_000, shares=None,
    )
    ticker.info = {"shortName": "Direxion Daily Semiconductor Bull 3X Shares", "currency": "USD",
                   "sharesOutstanding": 1_000_000_000}
    ticker_factory.return_value = ticker

    quote = YfinanceFetcher().get_realtime_quote("SOXL")
    assert quote.source == RealtimeSource.YFINANCE
    assert quote.volume_ratio == 1.5
    assert quote.turnover_rate == 15.0
    assert quote.amount == 4_500_000_000.0


def test_daily_bar_is_final_only_after_session_with_covering_bar():
    final = StockAnalysisPipeline._daily_bar_is_final
    phase_closed = {"is_partial_bar": False, "effective_daily_bar_date": "2026-11-20"}
    assert final({"date": "2026-11-20"}, phase_closed) is True
    assert final({"date": date(2026, 11, 20)}, phase_closed) is True
    assert final({"date": "2026-11-19"}, phase_closed) is False
    assert final({"date": "2026-11-20"}, {"is_partial_bar": True, "effective_daily_bar_date": "2026-11-20"}) is False
    assert final({"date": "2026-11-20"}, None) is False


def test_daily_fetch_days_backfills_when_history_short():
    pipeline = object.__new__(StockAnalysisPipeline)
    pipeline.db = MagicMock()
    pipeline.db.get_data_range.return_value = [object()] * 44
    assert pipeline._daily_fetch_days("SOXL", date(2026, 11, 20)) == 120
    pipeline.db.get_data_range.return_value = [object()] * 61
    assert pipeline._daily_fetch_days("SOXL", date(2026, 11, 20)) == 30
    pipeline.db.get_data_range.side_effect = RuntimeError("db down")
    assert pipeline._daily_fetch_days("SOXL", date(2026, 11, 20)) == 120


@patch("yfinance.Ticker")
def test_yfinance_etf_turnover_uses_total_assets_over_nav(ticker_factory: MagicMock):
    ticker = MagicMock()
    ticker.fast_info = SimpleNamespace(
        lastPrice=30.0, previousClose=29.0, open=29.5, dayHigh=30.5, dayLow=29.0,
        lastVolume=100_000_000, marketCap=None, tenDayAverageVolume=50_000_000, shares=None,
    )
    ticker.info = {"shortName": "SOXL", "currency": "USD", "totalAssets": 12_000_000_000, "navPrice": 30.0}
    ticker_factory.return_value = ticker
    quote = YfinanceFetcher().get_realtime_quote("SOXL")
    assert quote.turnover_rate == 25.0


def test_chip_and_turnover_display_helpers():
    from src.notification import NotificationService

    assert NotificationService._format_ratio_display(0.002) == "0.2%"
    assert NotificationService._format_ratio_display("0.6438") == "64.4%"
    assert NotificationService._format_ratio_display("64.38") == "64.4%"
    assert NotificationService._format_ratio_display("12%") == "12%"
    assert NotificationService._format_ratio_display(None) == "N/A"
    assert NotificationService._format_turnover_display(None) == "N/A"
    assert NotificationService._format_turnover_display("N/A") == "N/A"
    assert NotificationService._format_turnover_display(3.2) == "3.2%"


def test_bare_number_chip_ratios_are_replaced_with_computed_values():
    from data_provider.realtime_types import ChipDistribution
    from src.analyzer import AnalysisResult, fill_chip_structure_if_needed

    result = AnalysisResult(
        code="SOXL", name="SOXL", sentiment_score=20, trend_prediction="看空",
        operation_advice="观望", report_language="ko",
        dashboard={"data_perspective": {"chip_structure": {
            "profit_ratio": "0.2", "avg_cost": 85.34, "concentration": 0.6438, "chip_health": "위험"}}},
    )
    chip = ChipDistribution(code="SOXL", source="yfinance_estimated", profit_ratio=0.002,
                            avg_cost=85.34, concentration_90=0.6438, concentration_70=0.4)
    fill_chip_structure_if_needed(result, chip)
    cs = result.dashboard["data_perspective"]["chip_structure"]
    assert cs["profit_ratio"] == "0.2%"
    assert cs["concentration"] == "64.38%"
    assert cs["avg_cost"] == 85.34
    assert cs["chip_health"] == "위험"
