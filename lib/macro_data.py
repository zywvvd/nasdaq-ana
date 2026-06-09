"""宏观指标加载模块 — 从现有CSV提取12个宏观列，供多股票训练复用。"""
import pandas as pd
import numpy as np
from .config import data_path

MACRO_COLS = [
    'VIX', 'VVIX', 'VIX_VVIX_Ratio', 'VIX9D', 'VIX3M', 'SKEW',
    'Spread_10Y_2Y', 'HYG', 'LQD', 'TLT', 'SP500_PE', 'Shiller_CAPE',
]


def fetch_macro_indicators():
    """加载全部宏观指标，返回 Date-indexed DataFrame (12列)。

    列: VIX, VVIX, VIX_VVIX_Ratio, VIX9D, VIX3M, SKEW,
        Spread_10Y_2Y, HYG, LQD, TLT, SP500_PE, Shiller_CAPE
    """
    # 主表提供 VIX, VVIX, VIX_VVIX_Ratio
    main = pd.read_csv(data_path('main'), parse_dates=['Date'], index_col='Date')
    df = main[['VIX', 'VVIX', 'VIX_VVIX_Ratio']].copy()

    # VIX9D, VIX3M, SKEW
    for key in ['vix9d', 'vix3m', 'skew']:
        tmp = pd.read_csv(data_path(key), parse_dates=['Date'], index_col='Date')
        df = df.join(tmp[[tmp.columns[0]]], how='left')

    # 国债利差
    treasury = pd.read_csv(data_path('treasury'), parse_dates=['Date'], index_col='Date')
    df = df.join(treasury[['Spread_10Y_2Y']], how='left')

    # 信用利差
    credit = pd.read_csv(data_path('credit'), parse_dates=['Date'], index_col='Date')
    df = df.join(credit[['HYG', 'LQD', 'TLT']], how='left')

    # PE/CAPE 月频→日频插值
    pe = pd.read_csv(data_path('pe_monthly'), parse_dates=['date'], index_col='date')
    pe_daily = pe.reindex(df.index.union(pe.index)).interpolate(method='time').reindex(df.index)
    df['SP500_PE'] = pe_daily['SP500_PE']
    df['Shiller_CAPE'] = pe_daily['Shiller_CAPE']

    # 前向填充
    df = df.ffill()

    return df[MACRO_COLS]
