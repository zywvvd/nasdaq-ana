#!/usr/bin/env python3
"""数据下载与更新脚本 — 从 yfinance 拉取行情并保存到 data/ 目录。

用法:
  python download_data.py             # 下载全部 15 年数据
  python download_data.py --period 5y # 下载最近 5 年
  python download_data.py --period max # 下载最长可用数据
"""
import os
import sys
import argparse
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.config import DATA_DIR, DATA_FILES


def data_path(key):
    return os.path.join(DATA_DIR, DATA_FILES[key])


def download_yfinance(ticker, period, name):
    """下载单个 ticker 数据，返回 DataFrame。"""
    import yfinance as yf
    print(f"  下载 {name} ({ticker})...", end=" ", flush=True)
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            print("无数据!")
            return None
        # 扁平化多层列名
        data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
        print(f"{len(data)} 行")
        return data
    except Exception as e:
        print(f"失败: {e}")
        return None


def download_all(period='15y'):
    """下载全部数据并保存到 CSV。"""
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print(f"数据下载 — period={period}")
    print("=" * 60)

    # ── 下载定义 ──
    sources = {
        'nasdaq':  ('^IXIC', '纳斯达克综合指数'),
        'vix':     ('^VIX',  'VIX 恐慌指数'),
        'vvix':    ('^VVIX', 'VVIX 波动率的波动率'),
        'vix9d':   ('^VIX9D', 'VIX9D 短期波动率'),
        'vix3m':   ('^VIX3M', 'VIX3M 3月波动率'),
        'skew':    ('^SKEW',  'SKEW 偏度指数'),
        'tnx':     ('^TNX',  '10年期国债收益率'),
        'irx':     ('^IRX',  '13周期国库券收益率'),
        'hyg':     ('HYG',   '高收益公司债ETF'),
        'lqd':     ('LQD',   '投资级公司债ETF'),
        'tlt':     ('TLT',   '20+年国债ETF'),
    }

    raw = {}
    for name, (ticker, desc) in sources.items():
        data = download_yfinance(ticker, period, f"{desc}")
        if data is not None:
            raw[name] = data
        time.sleep(0.3)

    if not raw:
        print("未下载到任何数据!")
        return

    # ── 1. 主表 ──
    if 'nasdaq' in raw:
        main = raw['nasdaq'][['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        if 'vix' in raw:
            main['VIX'] = raw['vix']['Close']
        if 'vvix' in raw:
            main['VVIX'] = raw['vvix']['Close']
        if 'VIX' in main.columns and 'VVIX' in main.columns:
            main['VIX_VVIX_Ratio'] = main['VIX'] / main['VVIX']
        path = data_path('main')
        main.to_csv(path)
        print(f"\n  主表: {len(main)} 行 -> {os.path.basename(path)}")

    # ── 2. VIX9D ──
    if 'vix9d' in raw:
        vix9d_df = pd.DataFrame({'VIX9D': raw['vix9d']['Close']})
        vix9d_df.index.name = 'Date'
        vix9d_df.to_csv(data_path('vix9d'))
        print(f"  VIX9D: {len(vix9d_df)} 行")

    # ── 3. VIX3M ──
    if 'vix3m' in raw:
        vix3m_df = pd.DataFrame({'VIX3M': raw['vix3m']['Close']})
        vix3m_df.index.name = 'Date'
        vix3m_df.to_csv(data_path('vix3m'))
        print(f"  VIX3M: {len(vix3m_df)} 行")

    # ── 4. SKEW ──
    if 'skew' in raw:
        skew_df = pd.DataFrame({'SKEW': raw['skew']['Close']})
        skew_df.index.name = 'Date'
        skew_df.to_csv(data_path('skew'))
        print(f"  SKEW: {len(skew_df)} 行")

    # ── 5. 国债利差 ──
    if 'tnx' in raw and 'irx' in raw:
        spread_df = pd.DataFrame({
            'Yield_10Y': raw['tnx']['Close'],
            'Yield_Short': raw['irx']['Close'],
            'Spread_10Y_2Y': raw['tnx']['Close'] - raw['irx']['Close'],
        })
        spread_df.index.name = 'Date'
        spread_df.to_csv(data_path('treasury'))
        print(f"  国债利差: {len(spread_df)} 行")

    # ── 6. 信用利差 ──
    if all(k in raw for k in ['hyg', 'lqd', 'tlt']):
        credit_df = pd.DataFrame({
            'HYG': raw['hyg']['Close'],
            'LQD': raw['lqd']['Close'],
            'TLT': raw['tlt']['Close'],
        })
        credit_df['HYG_TLT_Ratio'] = credit_df['HYG'] / credit_df['TLT']
        credit_df.index.name = 'Date'
        credit_df.to_csv(data_path('credit'))
        print(f"  信用利差: {len(credit_df)} 行")

    # ── 7. PE/CAPE 检查 ──
    pe_path = data_path('pe_monthly')
    if os.path.exists(pe_path):
        pe = pd.read_csv(pe_path)
        latest_pe = pd.to_datetime(pe['date']).max()
        age_days = (pd.Timestamp.now() - latest_pe).days
        if age_days > 90:
            print(f"\n  ⚠ PE/CAPE 数据已 {age_days} 天未更新 (最后: {latest_pe.strftime('%Y-%m-%d')})")
            print(f"    请手动更新: {pe_path}")
        else:
            print(f"\n  PE/CAPE 数据正常 (最后: {latest_pe.strftime('%Y-%m-%d')})")
    else:
        print(f"\n  ⚠ PE/CAPE 数据不存在: {pe_path}")
        print(f"    请准备 SP500_PE + Shiller_CAPE 月度 CSV")

    print(f"\n{'=' * 60}")
    print("数据下载完成!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='下载训练数据')
    parser.add_argument('--period', default='15y', help='下载周期 (默认 15y)')
    args = parser.parse_args()
    download_all(period=args.period)
