"""特征计算模块 — 训练和预测共用。

输入 DataFrame 需包含以下列：
  Open, High, Low, Close, Volume, VIX, VVIX, VIX9D, VIX3M, SKEW,
  Spread_10Y_2Y, HYG, LQD, TLT, SP500_PE, Shiller_CAPE

输出 DataFrame 新增 153 个特征列（加上中间计算列）。
"""
import pandas as pd
import numpy as np


def compute_features(df):
    """计算全部 153 个特征，返回添加了特征列的 DataFrame。"""
    c = df['Close']
    h = df['High']
    lo = df['Low']
    v = df['Volume']

    # ══════════════════════════════════════
    # 第一部分：基础技术指标
    # ══════════════════════════════════════

    # ── 均线 ──
    df['MA5'] = c.rolling(5).mean()
    df['MA10'] = c.rolling(10).mean()
    df['MA20'] = c.rolling(20).mean()
    df['MA60'] = c.rolling(60).mean()
    df['MA200'] = c.rolling(200).mean()

    # ── 偏离度 ──
    df['Dist_MA5'] = (c / df['MA5'] - 1) * 100
    df['Dist_MA10'] = (c / df['MA10'] - 1) * 100
    df['Dist_MA20'] = (c / df['MA20'] - 1) * 100
    df['Dist_MA60'] = (c / df['MA60'] - 1) * 100
    df['Dist_MA200'] = (c / df['MA200'] - 1) * 100

    # ── RSI(14) ──
    delta = c.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    df['RSI'] = 100 - 100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean())

    # ── MACD ──
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    df['MACD_Hist'] = macd - macd_signal

    # ── ATR% ──
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    df['ATR_Pct'] = tr.rolling(14).mean() / c * 100

    # ── 日收益率 ──
    ret = c.pct_change()
    df['Return_1d'] = ret

    # ── 波动率与量比 ──
    df['Volatility_20d'] = ret.rolling(20).std() * np.sqrt(252) * 100
    df['Vol_Ratio'] = v / v.rolling(20).mean()

    # ── 动量 ──
    df['Mom_5d'] = c.pct_change(5) * 100
    df['Mom_10d'] = c.pct_change(10) * 100
    df['Mom_20d'] = c.pct_change(20) * 100

    # ── 回撤深度 ──
    df['DD_depth'] = (c - c.cummax()) / c.cummax() * 100

    # ── 布林带 %B + 宽度 ──
    bb_ma = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df['BB_pctB'] = (c - (bb_ma - 2 * bb_std)) / (4 * bb_std)
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    df['BB_width'] = (bb_upper - bb_lower) / bb_ma * 100

    # ── Williams %R ──
    df['Williams_R'] = (h.rolling(14).max() - c) / (h.rolling(14).max() - lo.rolling(14).min()) * -100

    # ── OBV ──
    obv = (v * np.sign(c.diff())).cumsum()
    df['OBV_change_5d'] = obv.pct_change(5) * 100
    df['OBV_change_10d'] = obv.pct_change(10) * 100

    # ── 连涨/跌天数 ──
    streak = np.sign(c.diff())
    groups = (streak != streak.shift(1)).cumsum()
    df['Price_streak'] = streak.groupby(groups).cumcount() + 1
    df.loc[streak < 0, 'Price_streak'] = -df.loc[streak < 0, 'Price_streak']

    # ── 交互特征 ──
    df['Vol_w_MACD'] = df['MACD_Hist'] * df['Vol_Ratio']

    # ── 统计/K线形态 ──
    df['Std_5d'] = ret.rolling(5).std()
    df['Intraday_range'] = (h - lo) / c * 100
    df['Upper_shadow_pct'] = (h - pd.concat([c, df['Open']], axis=1).max(axis=1)) / (h - lo + 1e-10) * 100
    df['Lower_shadow_pct'] = (pd.concat([c, df['Open']], axis=1).min(axis=1) - lo) / (h - lo + 1e-10) * 100

    # ── 位置 ──
    df['Dist_from_60d_high'] = (c - h.rolling(60).max()) / h.rolling(60).max() * 100
    df['Dist_from_60d_low'] = (c - lo.rolling(60).min()) / lo.rolling(60).min() * 100

    # ── VIX 相关性 + 趋势 ──
    df['VIX_corr_20d'] = ret.rolling(20).corr(df['VIX'].pct_change())
    df['Trend_20d'] = c.pct_change(20) * 100 / (df['Volatility_20d'] / np.sqrt(252) * np.sqrt(20) + 1e-10)

    # ── 原有衍生指标 ──
    df['VIX_change_5d'] = df['VIX'].pct_change(5) * 100
    df['RSI_change_5d'] = df['RSI'].diff(5)
    df['Vol_surge'] = df['Volatility_20d'].pct_change(5) * 100
    df['MA20_cross_MA60'] = df['Dist_MA20'] - df['Dist_MA60']
    df['VIX_rank_60d'] = df['VIX'].rolling(60).rank(pct=True) * 100
    df['RSI_rank_60d'] = df['RSI'].rolling(60).rank(pct=True) * 100
    df['Dist_MA60_change_10d'] = df['Dist_MA60'].diff(10)
    df['ATR_vs_VIX'] = df['ATR_Pct'] / (df['VIX'] / 100)

    # ══════════════════════════════════════
    # 第二部分：V2 宏观/估值指标
    # ══════════════════════════════════════

    # VIX 期限结构
    df['VIX9D_VIX_Ratio'] = df['VIX9D'] / df['VIX']
    df['VIX3M_VIX_Ratio'] = df['VIX3M'] / df['VIX']
    df['VIX3M_VIX9D_Spread'] = df['VIX3M'] - df['VIX9D']
    df['VIX9D_change_5d'] = df['VIX9D'].pct_change(5) * 100
    df['VIX3M_change_5d'] = df['VIX3M'].pct_change(5) * 100

    # SKEW
    df['SKEW_Dist_MA20'] = df['SKEW'] - df['SKEW'].rolling(20).mean()
    df['SKEW_rank_60d'] = df['SKEW'].rolling(60).rank(pct=True) * 100

    # 国债利差
    df['Spread_10Y2Y_change_20d'] = df['Spread_10Y_2Y'].diff(20)
    df['Spread_Inverted'] = (df['Spread_10Y_2Y'] < 0).astype(float)

    # 信用利差
    df['HYG_TLT_Ratio'] = df['HYG'] / df['TLT']
    df['HYG_TLT_change_10d'] = df['HYG_TLT_Ratio'].pct_change(10) * 100
    df['LQD_TLT_Ratio'] = df['LQD'] / df['TLT']
    df['LQD_TLT_change_10d'] = df['LQD_TLT_Ratio'].pct_change(10) * 100

    # 估值
    df['PE_rank_252'] = df['SP500_PE'].rolling(252).rank(pct=True) * 100
    df['CAPE_rank_252'] = df['Shiller_CAPE'].rolling(252).rank(pct=True) * 100
    df['PE_change_60d'] = df['SP500_PE'].pct_change(60) * 100
    df['CAPE_change_60d'] = df['Shiller_CAPE'].pct_change(60) * 100

    # ══════════════════════════════════════
    # 第三部分：V3 窗口变化特征（96 个）
    # ══════════════════════════════════════

    # ── 3A. 均线斜率（7） ──
    df['MA5_slope_5d'] = df['MA5'].pct_change(5) * 100
    df['MA5_slope_10d'] = df['MA5'].pct_change(10) * 100
    df['MA10_slope_5d'] = df['MA10'].pct_change(5) * 100
    df['MA10_slope_10d'] = df['MA10'].pct_change(10) * 100
    df['MA20_slope_5d'] = df['MA20'].pct_change(5) * 100
    df['MA20_slope_10d'] = df['MA20'].pct_change(10) * 100
    df['MA60_slope_10d'] = df['MA60'].pct_change(10) * 100
    df['MA60_slope_20d'] = df['MA60'].pct_change(20) * 100
    df['MA200_slope_20d'] = df['MA200'].pct_change(20) * 100
    df['MA200_slope_60d'] = df['MA200'].pct_change(60) * 100

    # ── 3B. 距离变化/加速（5） ──
    df['Dist_MA20_change_5d'] = df['Dist_MA20'].diff(5)
    df['Dist_MA200_change_5d'] = df['Dist_MA200'].diff(5)
    df['Dist_MA200_change_20d'] = df['Dist_MA200'].diff(20)
    df['MA20_cross_MA200'] = df['Dist_MA20'] - df['Dist_MA200']
    df['MA20_cross_MA200_change_10d'] = df['MA20_cross_MA200'].diff(10)

    # ── 3C. 二元位置（3） ──
    df['Close_above_MA20'] = (c > df['MA20']).astype(float)
    df['Close_above_MA60'] = (c > df['MA60']).astype(float)
    df['Close_above_MA200'] = (c > df['MA200']).astype(float)

    # ── 3D. 动量加速度（6） ──
    df['Mom_accel_5d'] = df['Mom_5d'].diff(5)
    df['Mom_accel_10d'] = df['Mom_10d'].diff(5)
    df['Mom_accel_20d_5d'] = df['Mom_20d'].diff(5)
    df['Mom_accel_20d_10d'] = df['Mom_20d'].diff(10)
    df['Mom_5d_vs_20d'] = df['Mom_5d'] - df['Mom_20d']
    mom20_m = df['Mom_20d'].rolling(60).mean()
    mom20_s = df['Mom_20d'].rolling(60).std()
    df['Mom_20d_zscore'] = (df['Mom_20d'] - mom20_m) / (mom20_s + 1e-10)

    # ── 3E. RSI 变化/极端（5） ──
    df['RSI_change_3d'] = df['RSI'].diff(3)
    df['RSI_change_10d'] = df['RSI'].diff(10)
    df['RSI_oversold'] = (df['RSI'] < 30).astype(float)
    df['RSI_overbought'] = (df['RSI'] > 70).astype(float)
    df['RSI_extreme'] = (df['RSI'] - 50).abs()

    # ── 3F. MACD 变化（3） ──
    df['MACD_Hist_change_3d'] = df['MACD_Hist'].diff(3)
    df['MACD_Hist_change_5d'] = df['MACD_Hist'].diff(5)
    df['MACD_signal_cross'] = np.sign(df['MACD_Hist']) * np.sign(df['MACD_Hist'].shift(1))

    # ── 3G. VIX/VVIX 多窗口变化（11） ──
    df['VIX_change_1d'] = df['VIX'].pct_change(1) * 100
    df['VIX_change_3d'] = df['VIX'].pct_change(3) * 100
    df['VIX_change_10d'] = df['VIX'].pct_change(10) * 100
    df['VIX_change_20d'] = df['VIX'].pct_change(20) * 100
    df['VVIX_change_1d'] = df['VVIX'].pct_change(1) * 100
    df['VVIX_change_5d'] = df['VVIX'].pct_change(5) * 100
    df['VVIX_change_10d'] = df['VVIX'].pct_change(10) * 100
    df['VIX_VVIX_Ratio_change_5d'] = df['VIX_VVIX_Ratio'].pct_change(5) * 100
    df['VIX_VVIX_Ratio_change_10d'] = df['VIX_VVIX_Ratio'].pct_change(10) * 100
    df['VIX_spike'] = df['VIX'] / df['VIX'].rolling(20).mean()
    vix_m = df['VIX'].rolling(60).mean()
    vix_s = df['VIX'].rolling(60).std()
    df['VIX_zscore_60d'] = (df['VIX'] - vix_m) / (vix_s + 1e-10)

    # ── 3H. VIX 期限结构变化（5） ──
    df['VIX9D_VIX_Ratio_change_5d'] = df['VIX9D_VIX_Ratio'].pct_change(5) * 100
    df['VIX3M_VIX_Ratio_change_5d'] = df['VIX3M_VIX_Ratio'].pct_change(5) * 100
    df['VIX3M_VIX9D_Spread_change_5d'] = df['VIX3M_VIX9D_Spread'].diff(5)
    df['VIX9D_change_10d'] = df['VIX9D'].pct_change(10) * 100
    df['VIX3M_change_10d'] = df['VIX3M'].pct_change(10) * 100

    # ── 3I. 波动率变化（6） ──
    df['Volatility_20d_change_5d'] = df['Volatility_20d'].diff(5)
    df['Volatility_20d_change_10d'] = df['Volatility_20d'].diff(10)
    df['ATR_Pct_change_5d'] = df['ATR_Pct'].diff(5)
    df['ATR_Pct_change_10d'] = df['ATR_Pct'].diff(10)
    df['Volatility_regime'] = df['Volatility_20d'] / df['Volatility_20d'].rolling(60).mean()
    df['Intraday_range_change_5d'] = df['Intraday_range'].diff(5)

    # ── 3J. SKEW 变化（4） ──
    df['SKEW_change_5d'] = df['SKEW'].diff(5)
    df['SKEW_change_10d'] = df['SKEW'].diff(10)
    df['SKEW_change_20d'] = df['SKEW'].diff(20)
    df['SKEW_extreme'] = (df['SKEW'] > 150).astype(float)

    # ── 3K. 信用利差变化（6） ──
    df['HYG_TLT_change_5d'] = df['HYG_TLT_Ratio'].pct_change(5) * 100
    df['HYG_TLT_change_20d'] = df['HYG_TLT_Ratio'].pct_change(20) * 100
    df['LQD_TLT_change_5d'] = df['LQD_TLT_Ratio'].pct_change(5) * 100
    df['LQD_TLT_change_20d'] = df['LQD_TLT_Ratio'].pct_change(20) * 100
    ht_m = df['HYG_TLT_Ratio'].rolling(60).mean()
    ht_s = df['HYG_TLT_Ratio'].rolling(60).std()
    df['HYG_TLT_zscore'] = (df['HYG_TLT_Ratio'] - ht_m) / (ht_s + 1e-10)
    lt_m = df['LQD_TLT_Ratio'].rolling(60).mean()
    lt_s = df['LQD_TLT_Ratio'].rolling(60).std()
    df['LQD_TLT_zscore'] = (df['LQD_TLT_Ratio'] - lt_m) / (lt_s + 1e-10)

    # ── 3L. 国债利差变化（3） ──
    df['Spread_10Y2Y_change_5d'] = df['Spread_10Y_2Y'].diff(5)
    df['Spread_10Y2Y_change_10d'] = df['Spread_10Y_2Y'].diff(10)
    df['Spread_deeply_inverted'] = (df['Spread_10Y_2Y'] < -0.5).astype(float)

    # ── 3M. 估值变化（6） ──
    df['PE_change_20d'] = df['SP500_PE'].pct_change(20) * 100
    df['PE_change_120d'] = df['SP500_PE'].pct_change(120) * 100
    df['CAPE_change_20d'] = df['Shiller_CAPE'].pct_change(20) * 100
    df['CAPE_change_120d'] = df['Shiller_CAPE'].pct_change(120) * 100
    pe_m = df['SP500_PE'].rolling(252).mean()
    pe_s = df['SP500_PE'].rolling(252).std()
    df['PE_zscore_252'] = (df['SP500_PE'] - pe_m) / (pe_s + 1e-10)
    cape_m = df['Shiller_CAPE'].rolling(252).mean()
    cape_s = df['Shiller_CAPE'].rolling(252).std()
    df['CAPE_zscore_252'] = (df['Shiller_CAPE'] - cape_m) / (cape_s + 1e-10)

    # ── 3N. 量变化（2） ──
    df['Vol_Ratio_change_5d'] = df['Vol_Ratio'].diff(5)
    df['Volume_surge'] = (df['Vol_Ratio'] > 2).astype(float)

    # ── 3O. 布林/威廉变化（3） ──
    df['BB_pctB_change_5d'] = df['BB_pctB'].diff(5)
    df['Williams_R_change_5d'] = df['Williams_R'].diff(5)
    df['BB_width_change_10d'] = df['BB_width'].diff(10)

    # ── 3P. 回撤/位置变化（3） ──
    df['DD_depth_change_5d'] = df['DD_depth'].diff(5)
    df['DD_depth_change_10d'] = df['DD_depth'].diff(10)
    df['Days_since_60d_high'] = h.rolling(60).apply(lambda x: 59 - np.argmax(x), raw=True)

    # ── 3Q. 统计特征（5） ──
    df['Return_skew_20d'] = ret.rolling(20).skew()
    df['Return_kurtosis_20d'] = ret.rolling(20).kurt()
    df['Max_dd_20d'] = ret.rolling(20).apply(
        lambda x: np.min(np.cumsum(x) - np.maximum.accumulate(np.cumsum(x))), raw=True
    ) * 100
    df['High_low_range_5d'] = (h.rolling(5).max() - lo.rolling(5).min()) / c * 100
    df['Std_5d_change'] = df['Std_5d'].diff(5)

    # ── 3R. 跳空缺口（2） ──
    df['Gap_pct'] = (df['Open'] - c.shift(1)) / c.shift(1) * 100
    df['Gap_abs_5d'] = df['Gap_pct'].abs().rolling(5).mean()

    # ── 3S. 相关性/趋势变化（2） ──
    df['VIX_corr_20d_change_5d'] = df['VIX_corr_20d'].diff(5)
    df['Trend_20d_change_5d'] = df['Trend_20d'].diff(5)

    # ── 3T. 跨指标比率（3） ──
    df['VIX_Volatility_ratio'] = df['VIX'] / (df['Volatility_20d'] + 1e-10)
    df['VIX9D_VIX3M_Ratio'] = df['VIX9D'] / (df['VIX3M'] + 1e-10)
    df['ATR_vs_VIX_change_5d'] = df['ATR_vs_VIX'].diff(5)

    # ══════════════════════════════════════
    # V7: 新增特征（Tier 1 研究验证）
    # ══════════════════════════════════════

    o = df['Open']

    # ── 7A. 隔夜/日内分解（4）──
    overnight_ret = (o - c.shift(1)) / c.shift(1)
    intraday_ret = (c - o) / o
    df['Overnight_ret'] = overnight_ret
    df['Intraday_ret'] = intraday_ret
    df['Overnight_intraday_spread'] = overnight_ret - intraday_ret
    df['Overnight_zscore_20'] = (overnight_ret - overnight_ret.rolling(20).mean()) / (overnight_ret.rolling(20).std() + 1e-10)

    # ── 7B. VIX期限结构斜率（5）──
    df['VIX_slope_9d_3m'] = df['VIX9D'] - df['VIX3M']
    df['VIX_slope_spot_3m'] = df['VIX'] - df['VIX3M']
    df['VIX_curvature'] = df['VIX9D'] - 2 * df['VIX'] + df['VIX3M']
    df['VIX_slope_chg_3d'] = df['VIX_slope_9d_3m'].diff(3)
    df['VIX_pctile_252'] = df['VIX'].rolling(252, min_periods=60).rank(pct=True)

    # ── 7C. 波动率风险溢价（2）──
    realized_vol_20 = c.pct_change().rolling(20).std() * np.sqrt(252)
    df['Vol_risk_premium'] = df['VIX'] / 100 - realized_vol_20
    df['Vol_ratio_10d_20d'] = (c.pct_change().rolling(10).std() * np.sqrt(252)) / (realized_vol_20 + 1e-10)

    # ── 7D. Kaufman效率系数（3）──
    def _er(close, period):
        direction = (close - close.shift(period)).abs()
        volatility = close.diff().abs().rolling(period).sum()
        return direction / (volatility + 1e-10)
    df['ER_5'] = _er(c, 5)
    df['ER_10'] = _er(c, 10)
    df['ER_20'] = _er(c, 20)

    # ── 7E. 跨资产分歧（4）──
    df['Credit_equity_div_5d'] = df['HYG'].pct_change(5) - c.pct_change(5)
    df['Credit_equity_div_10d'] = df['HYG'].pct_change(10) - c.pct_change(10)
    df['Bond_equity_div_5d'] = df['TLT'].pct_change(5) - c.pct_change(5)
    df['Eq_TLT_corr_20d'] = c.pct_change().rolling(20).corr(df['TLT'].pct_change())

    # ── 7F. 量价背离（3）──
    body_range = (c - o).abs() / (h - lo + 1e-10)
    df['Conviction'] = v * np.sign(c - o) * body_range
    df['Vol_price_corr_10d'] = c.pct_change().rolling(10).corr(v)
    up_vol = v.where(c > o, 0).rolling(10).mean()
    dn_vol = v.where(c < o, 0).rolling(10).mean()
    df['Up_down_vol_ratio'] = up_vol / (dn_vol + 1)

    # ── 7G. 回撤状态（3）──
    roll_high_63 = c.rolling(63).max()
    df['Current_drawdown'] = (c - roll_high_63) / roll_high_63
    dd_sq = df['Current_drawdown'] ** 2
    df['Ulcer_index_14'] = np.sqrt(dd_sq.rolling(14).mean())
    df['Days_since_63d_high'] = c.rolling(63).apply(lambda x: 63 - np.argmax(x), raw=True)

    # ── 7H. 日历效应（2）──
    if hasattr(df.index, 'dayofweek'):
        dow = df.index.dayofweek
        df['Dow_sin'] = np.sin(2 * np.pi * dow / 5)
        df['Dow_cos'] = np.cos(2 * np.pi * dow / 5)

    return df
