"""数据拉取模块：训练用本地 CSV，预测用 yfinance 实时数据。"""
import pandas as pd
import numpy as np
import os
from .config import data_path, DATA_DIR, SPY_DATA_PATH


def fetch_training_data():
    """从本地 CSV 加载并合并全部训练数据，返回待计算特征的 DataFrame。"""
    print("加载本地训练数据...")

    # 主表
    df = pd.read_csv(data_path('main'), parse_dates=['Date'], index_col='Date')
    print(f"  主表: {len(df)} 行")

    # 合并日频新数据
    for key in ['vix9d', 'vix3m', 'skew']:
        tmp = pd.read_csv(data_path(key), parse_dates=['Date'], index_col='Date')
        col = tmp.columns[0]
        df = df.join(tmp[[col]], how='left')
        print(f"  {key}: {col}")

    # 国债利差
    treasury = pd.read_csv(data_path('treasury'), parse_dates=['Date'], index_col='Date')
    df = df.join(treasury[['Spread_10Y_2Y']], how='left')

    # 信用利差 — 只取价格，比率在 features.py 统一计算
    credit = pd.read_csv(data_path('credit'), parse_dates=['Date'], index_col='Date')
    df = df.join(credit[['HYG', 'LQD', 'TLT']], how='left')
    print(f"  国债利差 + 信用利差")

    # PE/CAPE（月频→日频插值）
    pe = pd.read_csv(data_path('pe_monthly'), parse_dates=['date'], index_col='date')
    pe_daily = pe.reindex(df.index.union(pe.index)).interpolate(method='time').reindex(df.index)
    df['SP500_PE'] = pe_daily['SP500_PE']
    df['Shiller_CAPE'] = pe_daily['Shiller_CAPE']

    # 前向填充
    ffill_cols = ['VIX9D', 'VIX3M', 'SKEW', 'Spread_10Y_2Y', 'HYG', 'LQD', 'TLT',
                  'SP500_PE', 'Shiller_CAPE']
    df[ffill_cols] = df[ffill_cols].ffill()

    print(f"  合并完成: {df.columns.size} 列")
    return df


def fetch_live_data():
    """从 yfinance 拉取实时数据 + 本地 PE，返回待计算特征的 DataFrame。"""
    import yfinance as yf
    from .config import YF_TICKERS, data_path

    print("从 yfinance 拉取实时数据...")
    raw = {}
    for ticker, cfg in YF_TICKERS.items():
        raw[cfg['alias']] = yf.download(ticker, period=cfg['period'])

    # 扁平化列名
    for k in raw:
        raw[k].columns = [c[0] if isinstance(c, tuple) else c for c in raw[k].columns]

    # 以纳斯达克为基准
    df = raw['nasdaq'][['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # 合并各数据源
    df['VIX'] = raw['vix']['Close']
    df['VVIX'] = raw['vvix']['Close']
    df['VIX_VVIX_Ratio'] = df['VIX'] / df['VVIX']
    df['VIX9D'] = raw['vix9d']['Close']
    df['VIX3M'] = raw['vix3m']['Close']
    df['SKEW'] = raw['skew']['Close']
    df['Spread_10Y_2Y'] = raw['tnx']['Close'] - raw['irx']['Close']
    df['HYG'] = raw['hyg']['Close']
    df['LQD'] = raw['lqd']['Close']
    df['TLT'] = raw['tlt']['Close']

    # PE/CAPE 从本地
    pe_path = data_path('pe_monthly')
    if os.path.exists(pe_path):
        pe = pd.read_csv(pe_path, parse_dates=['date'], index_col='date')
        pe_daily = pe.reindex(df.index.union(pe.index)).interpolate(method='time').reindex(df.index).ffill()
        df['SP500_PE'] = pe_daily['SP500_PE']
        df['Shiller_CAPE'] = pe_daily['Shiller_CAPE']
    else:
        df['SP500_PE'] = np.nan
        df['Shiller_CAPE'] = np.nan

    print(f"  拉取完成: {len(df)} 行")
    return df


# ══════════════════════════════════════
# V4: SPY 数据 + 合并训练数据
# ══════════════════════════════════════

def fetch_spy_data():
    """加载 SPY OHLCV + 合并市场指标（VIX、利差等同 Nasdaq 共享的指标）。"""
    spy = pd.read_csv(SPY_DATA_PATH, parse_dates=['Date'], index_col='Date')
    spy = spy[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # 从 Nasdaq 主表合并市场指标
    main = pd.read_csv(data_path('main'), parse_dates=['Date'], index_col='Date')
    for col in ['VIX', 'VVIX', 'VIX_VVIX_Ratio']:
        if col in main.columns:
            spy[col] = main[col]

    # VIX 期限结构
    for key, col_name in [('vix9d', 'VIX9D'), ('vix3m', 'VIX3M'), ('skew', 'SKEW')]:
        tmp = pd.read_csv(data_path(key), parse_dates=['Date'], index_col='Date')
        spy[col_name] = tmp[tmp.columns[0]]

    # 国债利差
    treasury = pd.read_csv(data_path('treasury'), parse_dates=['Date'], index_col='Date')
    spy['Spread_10Y_2Y'] = treasury['Spread_10Y_2Y']

    # 信用利差
    credit = pd.read_csv(data_path('credit'), parse_dates=['Date'], index_col='Date')
    for col in ['HYG', 'LQD', 'TLT']:
        spy[col] = credit[col]

    # PE/CAPE
    pe_path = data_path('pe_monthly')
    if os.path.exists(pe_path):
        pe = pd.read_csv(pe_path, parse_dates=['date'], index_col='date')
        pe_daily = pe.reindex(spy.index.union(pe.index)).interpolate(method='time').reindex(spy.index).ffill()
        spy['SP500_PE'] = pe_daily['SP500_PE']
        spy['Shiller_CAPE'] = pe_daily['Shiller_CAPE']
    else:
        spy['SP500_PE'] = np.nan
        spy['Shiller_CAPE'] = np.nan

    # 前向填充
    ffill_cols = ['VIX', 'VVIX', 'VIX_VVIX_Ratio', 'VIX9D', 'VIX3M', 'SKEW',
                  'Spread_10Y_2Y', 'HYG', 'LQD', 'TLT', 'SP500_PE', 'Shiller_CAPE']
    spy[ffill_cols] = spy[ffill_cols].ffill()

    return spy


def fetch_merged_training_data():
    """V4: 合并 Nasdaq + SPY 训练数据。返回带 'source' 列的 DataFrame。"""
    df_nasdaq = fetch_training_data()
    df_nasdaq['source'] = 'nasdaq'

    df_spy = fetch_spy_data()
    df_spy['source'] = 'spy'

    df = pd.concat([df_nasdaq, df_spy], axis=0).sort_index()
    print(f"  Nasdaq: {len(df_nasdaq)} 行, SPY: {len(df_spy)} 行, 合并: {len(df)} 行")
    return df
