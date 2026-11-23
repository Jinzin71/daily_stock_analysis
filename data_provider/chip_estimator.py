# -*- coding: utf-8 -*-
"""
筹码分布（CYQ）估算与指标计算。

A 股可直接从东方财富 / Tushare 拿到筹码分布；美股等市场没有现成接口，
这里用日线 OHLCV 按经典 CYQ 换手衰减算法估算持仓成本分布：

    dist_t = dist_{t-1} * (1 - turnover_t) + day_dist_t * turnover_t

- day_dist_t：当日成交在 [low, high] 区间内的三角分布，峰值在典型价 (H+L+C)/3
- turnover_t：volume / 流通股本；拿不到股本时用 volume / 近 N 日成交量之和 近似

结果是估算值（source 带 ``_estimated`` 后缀），供报告参考，不等同于交易所真实持仓数据。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_PRICE_BINS = 240
DEFAULT_PROXY_WINDOW = 60  # 无股本时：近 60 日成交量之和视为一轮完整换手


def compute_cyq_metrics(df: pd.DataFrame, current_price: float) -> dict:
    """
    基于筹码分布明细（price / percent）计算常用筹码指标。

    :param df: 包含 'price' 和 'percent' 列的 DataFrame
    :param current_price: 当前价/收盘价（用于计算获利比例）
    :return: 与 akshare 口径一致的指标字典（获利比例、集中度为 0-1 小数）
    """
    df_sorted = df.sort_values(by='price', ascending=True).reset_index(drop=True)
    total_percent = df_sorted['percent'].sum()
    df_sorted['norm_percent'] = df_sorted['percent'] / total_percent * 100
    df_sorted['cumsum'] = df_sorted['norm_percent'].cumsum()

    winner_rate = df_sorted[df_sorted['price'] <= current_price]['norm_percent'].sum()
    avg_cost = np.average(df_sorted['price'], weights=df_sorted['norm_percent'])

    def get_percentile_price(target_pct):
        idx = df_sorted['cumsum'].searchsorted(target_pct)
        idx = min(idx, len(df_sorted) - 1)
        return df_sorted.loc[idx, 'price']

    cost_90_low = get_percentile_price(5)
    cost_90_high = get_percentile_price(95)
    if (cost_90_high + cost_90_low) != 0:
        concentration_90 = (cost_90_high - cost_90_low) / (cost_90_high + cost_90_low) * 100
    else:
        concentration_90 = 0.0

    cost_70_low = get_percentile_price(15)
    cost_70_high = get_percentile_price(85)
    if (cost_70_high + cost_70_low) != 0:
        concentration_70 = (cost_70_high - cost_70_low) / (cost_70_high + cost_70_low) * 100
    else:
        concentration_70 = 0.0

    return {
        "获利比例": round(winner_rate / 100, 4),
        "平均成本": round(avg_cost, 4),
        "90成本-低": round(cost_90_low, 4),
        "90成本-高": round(cost_90_high, 4),
        "90集中度": round(concentration_90 / 100, 4),
        "70成本-低": round(cost_70_low, 4),
        "70成本-高": round(cost_70_high, 4),
        "70集中度": round(concentration_70 / 100, 4),
    }


def estimate_chip_distribution(
    bars: pd.DataFrame,
    shares_outstanding: Optional[float] = None,
    *,
    price_bins: int = DEFAULT_PRICE_BINS,
    proxy_window: int = DEFAULT_PROXY_WINDOW,
) -> Optional[pd.DataFrame]:
    """
    用日线 OHLCV 估算筹码分布。

    :param bars: 含 high / low / close / volume 列、按日期升序的日线
    :param shares_outstanding: 流通股本（可选）
    :return: 含 'price' / 'percent' 列的 DataFrame；数据不足返回 None
    """
    required = {"high", "low", "close", "volume"}
    if bars is None or bars.empty or not required.issubset(bars.columns):
        return None
    data = bars[["high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce").dropna()
    data = data[(data["high"] > 0) & (data["low"] > 0) & (data["volume"] >= 0)]
    if len(data) < 20:
        return None

    lo = float(data["low"].min())
    hi = float(data["high"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    grid = np.linspace(lo, hi, price_bins)

    volume = data["volume"].to_numpy(dtype=float)
    if shares_outstanding and shares_outstanding > 0:
        turnover = volume / float(shares_outstanding)
    else:
        rolling = pd.Series(volume).rolling(proxy_window, min_periods=1).sum().to_numpy()
        turnover = np.divide(volume, rolling, out=np.zeros_like(volume), where=rolling > 0)
    turnover = np.clip(turnover, 0.0, 1.0)

    dist = np.zeros(price_bins)
    for (high, low, close, _vol), rate in zip(data.itertuples(index=False, name=None), turnover):
        if rate <= 0:
            continue
        typical = (high + low + close) / 3.0
        if high - low <= 1e-12:
            day = np.zeros(price_bins)
            day[int(np.argmin(np.abs(grid - close)))] = 1.0
        else:
            left = np.clip((grid - low) / max(typical - low, 1e-12), 0.0, 1.0)
            right = np.clip((high - grid) / max(high - typical, 1e-12), 0.0, 1.0)
            day = np.where(grid <= typical, left, right)
            day[(grid < low) | (grid > high)] = 0.0
            total = day.sum()
            if total <= 0:
                day = np.zeros(price_bins)
                day[int(np.argmin(np.abs(grid - close)))] = 1.0
            else:
                day = day / total
        dist = dist * (1.0 - rate) + day * rate

    if dist.sum() <= 0:
        return None
    return pd.DataFrame({"price": grid, "percent": dist / dist.sum() * 100})
