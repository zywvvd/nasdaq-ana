"""标签构建模块 — 事件距离标签（V3）+ 前向回撤标签（V4）。"""
import pandas as pd
import numpy as np


# ══════════════════════════════════════
# V3: 事件距离标签（保留）
# ══════════════════════════════════════

def soft_score(distance_pct, max_dist=10.0):
    """距顶/底 0%=100分，线性衰减到 max_dist%=0分。"""
    if pd.isna(distance_pct) or distance_pct < 0:
        return 0.0
    if distance_pct > max_dist:
        return 0.0
    return max(0.0, 100 * (1 - distance_pct / max_dist))


def _trend_direction(ma20_series):
    """基于 MA20 的 20 日斜率判断趋势方向：1=上涨，-1=下跌。"""
    trend = pd.Series(0.0, index=ma20_series.index)
    for i in range(20, len(ma20_series)):
        now = ma20_series.iloc[i]
        ago = ma20_series.iloc[i - 20]
        if pd.notna(now) and pd.notna(ago) and ago > 0:
            trend.iloc[i] = 1.0 if now > ago else -1.0
    return trend


def build_soft_labels(df, dd_defs_csv):
    """V3 事件距离标签。返回 top_scores, bot_scores。"""
    dd_defs = pd.read_csv(dd_defs_csv, parse_dates=['peak_date', 'trough_date'])
    trend = _trend_direction(df['MA20'])

    top_scores = pd.Series(0.0, index=df.index)
    bot_scores = pd.Series(0.0, index=df.index)

    for _, ev in dd_defs.iterrows():
        peak_date = ev['peak_date']
        trough_date = ev['trough_date']
        peak_price = ev['peak_price']
        trough_price = ev['trough_price']

        pre_peak = df.loc[:peak_date].tail(31)
        for dt, row in pre_peak.iterrows():
            dist = abs(row['Close'] - peak_price) / peak_price * 100
            score = soft_score(dist)
            if trend.get(dt, 0) < 0:
                score *= 0.5
            if score > top_scores.get(dt, 0):
                top_scores.loc[dt] = score

        pre_trough = df.loc[:trough_date].tail(31)
        for dt, row in pre_trough.iterrows():
            dist = abs(row['Close'] - trough_price) / trough_price * 100
            score = soft_score(dist)
            if trend.get(dt, 0) > 0:
                score *= 0.5
            if score > bot_scores.get(dt, 0):
                bot_scores.loc[dt] = score

    return top_scores, bot_scores


def compute_sample_weights(labels):
    """计算样本权重：正样本基础5 + score/20，中性样本1。"""
    return np.where(labels > 0, 5 + labels / 20, 1.0)


# ══════════════════════════════════════
# V4: 前向回撤标签
# ══════════════════════════════════════

def _forward_max_drawdown(close, window):
    """计算每一天未来 window 天内的最大回撤（负数）。"""
    n = len(close)
    prices = close.values.astype(float)
    result = np.full(n, np.nan)
    for t in range(n - window):
        future = prices[t + 1:t + 1 + window]
        cummax = np.maximum.accumulate(future)
        dd = (future - cummax) / cummax
        result[t] = dd.min()
    return pd.Series(result, index=close.index)


def _forward_max_recovery(close, window):
    """计算每一天未来 window 天内的最大反弹（正数）。"""
    n = len(close)
    prices = close.values.astype(float)
    result = np.full(n, np.nan)
    for t in range(n - window):
        future = prices[t + 1:t + 1 + window]
        cummin = np.minimum.accumulate(future)
        rec = (future - cummin) / cummin
        result[t] = rec.max()
    return pd.Series(result, index=close.index)


def _apply_gaussian_peaks(raw_scores, sigma):
    """在原始分数峰值周围应用高斯衰减。"""
    result = np.zeros(len(raw_scores))
    vals = raw_scores.values

    # 找局部峰值
    peaks = []
    for i in range(1, len(vals) - 1):
        if vals[i] > 0 and vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]:
            peaks.append(i)
    # 边界
    if len(vals) > 0 and vals[0] > 0:
        peaks.append(0)
    if len(vals) > 1 and vals[-1] > 0 and vals[-1] >= vals[-2]:
        peaks.append(len(vals) - 1)

    radius = int(3 * sigma)
    for pi in peaks:
        peak_val = vals[pi]
        lo = max(0, pi - radius)
        hi = min(len(vals), pi + radius + 1)
        for j in range(lo, hi):
            dist = abs(j - pi)
            decayed = peak_val * np.exp(-dist ** 2 / (2 * sigma ** 2))
            result[j] = max(result[j], decayed)

    return pd.Series(result, index=raw_scores.index)


def build_forward_labels(df, config=None):
    """V4 增强标签 — 基于事件定义 + 高斯衰减 + 回撤幅度加权。

    使用 V3 的13个回撤事件定义作为锚点，但改进：
    1. 用高斯衰减(sigma=10)代替线性衰减
    2. 回撤幅度越大，标签峰值越高
    3. 覆盖范围扩大到峰值/谷值前后45天
    """
    from .config import data_path

    dd_csv = data_path('dd_definitions')
    dd_defs = pd.read_csv(dd_csv, parse_dates=['peak_date', 'trough_date'])
    sigma = 10
    close = df['Close']
    top_arr = np.zeros(len(df))
    bot_arr = np.zeros(len(df))

    for _, ev in dd_defs.iterrows():
        peak_date = ev['peak_date']
        trough_date = ev['trough_date']
        dd_pct = abs(ev['drawdown_pct'])
        magnitude = min(dd_pct / 10.0 * 100, 100)

        # 顶部标签：峰值前后45天，高斯衰减
        window = pd.Timedelta(days=45)
        mask_p = (df.index >= peak_date - window) & (df.index <= peak_date)
        for idx in np.where(mask_p)[0]:
            dt = df.index[idx]
            days_dist = abs((dt - peak_date).days)
            score = magnitude * np.exp(-days_dist ** 2 / (2 * sigma ** 2))
            top_arr[idx] = max(top_arr[idx], score)

        # 底部标签：谷值前后45天，高斯衰减
        mask_b = (df.index >= trough_date - window) & (df.index <= trough_date)
        for idx in np.where(mask_b)[0]:
            dt = df.index[idx]
            days_dist = abs((dt - trough_date).days)
            score = magnitude * np.exp(-days_dist ** 2 / (2 * sigma ** 2))
            bot_arr[idx] = max(bot_arr[idx], score)

    top_scores = pd.Series(top_arr, index=df.index)
    bot_scores = pd.Series(bot_arr, index=df.index)

    print(f"  增强标签: 顶部>0={int((top_scores > 0).sum())}, >=50={int((top_scores >= 50).sum())}, "
          f"底部>0={int((bot_scores > 0).sum())}, >=50={int((bot_scores >= 50).sum())}")

    return top_scores.clip(0, 100), bot_scores.clip(0, 100)


def build_labels(df, scheme=None, dd_defs_csv=None, config=None):
    """标签调度器。

    Parameters
    ----------
    scheme : str — 'event_distance'(V3) 或 'forward_drawdown'(V4)
    dd_defs_csv : str — 回撤定义CSV路径（V3需要）
    config : dict — 标签配置
    """
    if scheme is None:
        if config is None:
            from .config import LABEL_CONFIG
            config = LABEL_CONFIG
        scheme = config['scheme']

    if scheme == 'event_distance':
        if dd_defs_csv is None:
            from .config import data_path
            dd_defs_csv = data_path('dd_definitions')
        return build_soft_labels(df, dd_defs_csv)
    elif scheme == 'forward_drawdown':
        return build_forward_labels(df, config)
    else:
        raise ValueError(f"Unknown label scheme: {scheme}")
