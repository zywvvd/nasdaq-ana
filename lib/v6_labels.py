"""V6 多尺度方向标签构建。

标签 = 前向收益的滚动百分位排名 → 平滑 → 映射到 [-1, 1]

百分位排名自动适应市场趋势：
- 50百分位 = 中性（跟近期平均水平一样）
- 95百分位 = 大涨（近期极少见的大涨）→ 接近 +1
- 5百分位 = 大跌（近期极少见的大跌）→ 接近 -1
"""
import numpy as np
import pandas as pd
from .v6_config import HORIZONS, LABEL_CONFIG


def build_direction_labels(close, horizon, rank_window=None, smooth_window=None):
    """构建单个 horizon 的方向标签（百分位排名法）。

    Parameters
    ----------
    close : pd.Series  — 日收盘价
    horizon : int       — 前向天数
    rank_window : int   — 百分位排名窗口（默认 252 ≈ 1年）
    smooth_window : int — 平滑窗口

    Returns
    -------
    pd.Series — 标签 [-1, 1]
    """
    cfg = LABEL_CONFIG
    if rank_window is None:
        rank_window = cfg.get('rank_window', 252)
    if smooth_window is None:
        smooth_window = cfg['smooth_window'](horizon)

    # 1. 原始前向收益
    raw = close.shift(-horizon) / close - 1

    # 2. 滚动百分位排名：当前前向收益在过去 rank_window 天中排第几
    #    使用 rolling + rank 实现
    ranked = raw.rolling(rank_window, min_periods=60).rank(pct=True)

    # 3. 映射到 [-1, 1]：P0→-1, P50→0, P100→+1
    normalized = ranked * 2 - 1

    # 4. 平滑
    if smooth_window >= 3:
        if cfg['smooth_type'] == 'median':
            smoothed = normalized.rolling(smooth_window, center=True, min_periods=1).median()
        else:
            smoothed = normalized.rolling(smooth_window, center=True, min_periods=1).mean()
    else:
        smoothed = normalized

    # 5. clip
    lo, hi = cfg['clip_range']
    labels = smoothed.clip(lo, hi)

    labels.name = f'label_{horizon}d'
    return labels


def build_zscore_labels(close, horizon, window=252, smooth_window=None):
    """构建 z-score 方向标签（跨股票可比）。

    label = (fwd_ret - rolling_mean) / rolling_std
    不同股票的标签在同一尺度上：+2 表示比近期均值高2个标准差。

    Returns pd.Series in [-3, 3], clipped to [-1, 1]
    """
    if smooth_window is None:
        smooth_window = max(5, horizon // 3)

    fwd_ret = close.shift(-horizon) / close - 1
    rolling_mean = fwd_ret.rolling(window, min_periods=60).mean()
    rolling_std = fwd_ret.rolling(window, min_periods=60).std()

    zscore = (fwd_ret - rolling_mean) / (rolling_std + 1e-10)

    if smooth_window >= 3:
        zscore = zscore.rolling(smooth_window, center=True, min_periods=1).mean()

    labels = zscore.clip(-1, 1)
    labels.name = f'label_{horizon}d'
    return labels


def build_all_labels(df):
    """为所有 horizon 构建方向标签。

    Parameters
    ----------
    df : DataFrame — 必须包含 'Close' 列

    Returns
    -------
    dict : {horizon: pd.Series}
    """
    close = df['Close']
    result = {}
    for h in HORIZONS:
        labels = build_direction_labels(close, h)
        valid = labels.notna().sum()
        pos = (labels > 0).sum()
        neg = (labels < 0).sum()
        zero = (labels == 0).sum()
        print(f"  {h:2d}d标签: {valid}有效, "
              f"正={pos}({pos/max(valid,1):.1%}), "
              f"负={neg}({neg/max(valid,1):.1%}), "
              f"零={zero}, "
              f"均值={labels.mean():.3f}, std={labels.std():.3f}")
        result[h] = labels
    return result
