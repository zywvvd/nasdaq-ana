"""全局配置：路径、特征列表、模型超参数、数据源定义"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = BASE_DIR

# ══════════════════════════════════════
# 153 维特征列表（唯一来源）
# ══════════════════════════════════════

FEATURE_COLS = [
    # ── 原始 36 特征 ──
    'VIX', 'VVIX', 'VIX_VVIX_Ratio', 'RSI', 'MACD_Hist', 'ATR_Pct',
    'Volatility_20d', 'Vol_Ratio',
    'Dist_MA20', 'Dist_MA60', 'Dist_MA200',
    'Mom_5d', 'Mom_10d', 'Mom_20d',
    'VIX_change_5d', 'RSI_change_5d', 'Vol_surge', 'MA20_cross_MA60',
    'VIX_rank_60d', 'RSI_rank_60d', 'DD_depth', 'Dist_MA60_change_10d', 'ATR_vs_VIX',
    'BB_pctB', 'Williams_R', 'OBV_change_10d', 'Price_streak',
    'Vol_w_MACD',
    'Std_5d', 'Intraday_range', 'Upper_shadow_pct', 'Lower_shadow_pct',
    'Dist_from_60d_high', 'Dist_from_60d_low', 'VIX_corr_20d', 'Trend_20d',

    # ── V2 新增 21 宏观/估值特征 ──
    'VIX9D_VIX_Ratio', 'VIX3M_VIX_Ratio', 'VIX3M_VIX9D_Spread',
    'VIX9D_change_5d', 'VIX3M_change_5d',
    'SKEW', 'SKEW_Dist_MA20', 'SKEW_rank_60d',
    'Spread_10Y_2Y', 'Spread_10Y2Y_change_20d', 'Spread_Inverted',
    'HYG_TLT_Ratio', 'HYG_TLT_change_10d', 'LQD_TLT_Ratio', 'LQD_TLT_change_10d',
    'SP500_PE', 'Shiller_CAPE', 'PE_rank_252', 'CAPE_rank_252',
    'PE_change_60d', 'CAPE_change_60d',

    # ── V3 窗口变化特征 96 个 ──

    # 3A. 均线斜率（10）
    'MA5_slope_5d', 'MA5_slope_10d', 'MA10_slope_5d', 'MA10_slope_10d',
    'MA20_slope_5d', 'MA20_slope_10d',
    'MA60_slope_10d', 'MA60_slope_20d',
    'MA200_slope_20d', 'MA200_slope_60d',

    # 3B. 距离变化/加速（5）
    'Dist_MA20_change_5d', 'Dist_MA200_change_5d', 'Dist_MA200_change_20d',
    'MA20_cross_MA200', 'MA20_cross_MA200_change_10d',

    # 3C. 二元位置（3）
    'Close_above_MA20', 'Close_above_MA60', 'Close_above_MA200',

    # 3D. 动量加速度（6）
    'Mom_accel_5d', 'Mom_accel_10d', 'Mom_accel_20d_5d', 'Mom_accel_20d_10d',
    'Mom_5d_vs_20d', 'Mom_20d_zscore',

    # 3E. RSI 变化/极端（5）
    'RSI_change_3d', 'RSI_change_10d',
    'RSI_oversold', 'RSI_overbought', 'RSI_extreme',

    # 3F. MACD 变化（3）
    'MACD_Hist_change_3d', 'MACD_Hist_change_5d', 'MACD_signal_cross',

    # 3G. VIX/VVIX 多窗口变化（11）
    'VIX_change_1d', 'VIX_change_3d', 'VIX_change_10d', 'VIX_change_20d',
    'VVIX_change_1d', 'VVIX_change_5d', 'VVIX_change_10d',
    'VIX_VVIX_Ratio_change_5d', 'VIX_VVIX_Ratio_change_10d',
    'VIX_spike', 'VIX_zscore_60d',

    # 3H. VIX 期限结构变化（5）
    'VIX9D_VIX_Ratio_change_5d', 'VIX3M_VIX_Ratio_change_5d',
    'VIX3M_VIX9D_Spread_change_5d',
    'VIX9D_change_10d', 'VIX3M_change_10d',

    # 3I. 波动率变化（6）
    'Volatility_20d_change_5d', 'Volatility_20d_change_10d',
    'ATR_Pct_change_5d', 'ATR_Pct_change_10d',
    'Volatility_regime', 'Intraday_range_change_5d',

    # 3J. SKEW 变化（4）
    'SKEW_change_5d', 'SKEW_change_10d', 'SKEW_change_20d', 'SKEW_extreme',

    # 3K. 信用利差变化（6）
    'HYG_TLT_change_5d', 'HYG_TLT_change_20d',
    'LQD_TLT_change_5d', 'LQD_TLT_change_20d',
    'HYG_TLT_zscore', 'LQD_TLT_zscore',

    # 3L. 国债利差变化（3）
    'Spread_10Y2Y_change_5d', 'Spread_10Y2Y_change_10d', 'Spread_deeply_inverted',

    # 3M. 估值变化（6）
    'PE_change_20d', 'PE_change_120d',
    'CAPE_change_20d', 'CAPE_change_120d',
    'PE_zscore_252', 'CAPE_zscore_252',

    # 3N. 量变化（2）
    'Vol_Ratio_change_5d', 'Volume_surge',

    # 3O. 布林/威廉变化（3）+ 带宽
    'BB_pctB_change_5d', 'Williams_R_change_5d', 'BB_width_change_10d', 'BB_width',

    # 3P. 回撤/位置变化（3）
    'DD_depth_change_5d', 'DD_depth_change_10d', 'Days_since_60d_high',

    # 3Q. 统计特征（5）
    'Return_skew_20d', 'Return_kurtosis_20d', 'Max_dd_20d',
    'High_low_range_5d', 'Std_5d_change',

    # 3R. 跳空缺口（2）
    'Gap_pct', 'Gap_abs_5d',

    # 3S. 相关性/趋势变化（2）
    'VIX_corr_20d_change_5d', 'Trend_20d_change_5d',

    # 3T. 跨指标比率（3）+ 补充
    'VIX_Volatility_ratio', 'VIX9D_VIX3M_Ratio', 'ATR_vs_VIX_change_5d',
    'Return_1d', 'OBV_change_5d',
]

N_FEATURES = len(FEATURE_COLS)

# ── 模型超参数 ──
MODEL_PARAMS = {
    'n_estimators': 1500,
    'max_depth': 10,
    'min_samples_leaf': 8,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,
}

# ── yfinance 数据源定义 ──
YF_TICKERS = {
    '^IXIC': {'period': '4y', 'alias': 'nasdaq'},
    '^VIX':  {'period': '4y', 'alias': 'vix'},
    '^VVIX': {'period': '4y', 'alias': 'vvix'},
    '^VIX9D': {'period': '4y', 'alias': 'vix9d'},
    '^VIX3M': {'period': '4y', 'alias': 'vix3m'},
    '^SKEW': {'period': '4y', 'alias': 'skew'},
    '^TNX':  {'period': '4y', 'alias': 'tnx'},
    '^IRX':  {'period': '4y', 'alias': 'irx'},
    'HYG':   {'period': '4y', 'alias': 'hyg'},
    'LQD':   {'period': '4y', 'alias': 'lqd'},
    'TLT':   {'period': '4y', 'alias': 'tlt'},
}

# ── 本地 CSV 文件路径 ──
DATA_FILES = {
    'main':           '数据_全指标合并主表.csv',
    'vix9d':          '数据_VIX9D_15y.csv',
    'vix3m':          '数据_VIX3M_15y.csv',
    'skew':           '数据_SKEW.csv',
    'treasury':       '数据_国债利差10Y2Y.csv',
    'credit':         '数据_信用利差.csv',
    'pe_monthly':     '数据_SP500估值月度_填充.csv',
    'dd_definitions': '分析_所有回撤区间定义.csv',
}


def data_path(key):
    """根据 key 返回 CSV 的绝对路径"""
    return os.path.join(DATA_DIR, DATA_FILES[key])
