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
