#!/usr/bin/env python3
"""纳斯达克顶部/底部评分预测 — 每日运行 V2"""
import pandas as pd
import numpy as np
import joblib, json, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))


def fetch_latest():
    """拉取最新数据并计算所有57个特征"""
    import yfinance as yf

    print("拉取最新数据...")
    # 拉取足够长的历史数据（200日均线 + 3年排名需要）
    nasdaq = yf.download("^IXIC", period="4y")
    vix = yf.download("^VIX", period="4y")
    vvix = yf.download("^VVIX", period="4y")
    vix9d = yf.download("^VIX9D", period="4y")
    vix3m = yf.download("^VIX3M", period="4y")
    skew = yf.download("^SKEW", period="4y")
    tnx = yf.download("^TNX", period="4y")  # 10Y Treasury
    irx = yf.download("^IRX", period="4y")  # 13W T-bill (proxy for 2Y)
    hyg = yf.download("HYG", period="4y")
    lqd = yf.download("LQD", period="4y")
    tlt = yf.download("TLT", period="4y")

    def flatten(df):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df

    for d in [nasdaq, vix, vvix, vix9d, vix3m, skew, tnx, irx, hyg, lqd, tlt]:
        flatten(d)

    df = nasdaq[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['VIX'] = vix['Close']
    df['VVIX'] = vvix['Close']
    df['VIX_VVIX_Ratio'] = df['VIX'] / df['VVIX']
    df['VIX9D'] = vix9d['Close']
    df['VIX3M'] = vix3m['Close']
    df['SKEW'] = skew['Close']
    df['Spread_10Y_2Y'] = tnx['Close'] - irx['Close']
    df['HYG'] = hyg['Close']
    df['LQD'] = lqd['Close']
    df['TLT'] = tlt['Close']
    df['HYG_TLT_Ratio'] = df['HYG'] / df['TLT']
    df['LQD_TLT_Ratio'] = df['LQD'] / df['TLT']

    # PE/CAPE (从本地月度文件前向填充)
    pe_file = os.path.join(BASE, "数据_SP500估值月度_填充.csv")
    if os.path.exists(pe_file):
        pe = pd.read_csv(pe_file, parse_dates=['date'], index_col='date')
        pe_reindexed = pe.reindex(df.index.union(pe.index)).interpolate(method='time').reindex(df.index).ffill()
        df['SP500_PE'] = pe_reindexed['SP500_PE']
        df['Shiller_CAPE'] = pe_reindexed['Shiller_CAPE']
    else:
        df['SP500_PE'] = np.nan
        df['Shiller_CAPE'] = np.nan

    # ── 技术指标 ──
    c = df['Close']
    h = df['High']
    lo = df['Low']
    v = df['Volume']

    df['MA20'] = c.rolling(20).mean()
    df['MA60'] = c.rolling(60).mean()
    df['MA200'] = c.rolling(200).mean()
    df['Dist_MA20'] = (c / df['MA20'] - 1) * 100
    df['Dist_MA60'] = (c / df['MA60'] - 1) * 100
    df['Dist_MA200'] = (c / df['MA200'] - 1) * 100

    delta = c.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    df['RSI'] = 100 - 100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean())

    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    df['ATR_Pct'] = tr.rolling(14).mean() / c * 100

    df['Return'] = c.pct_change()
    df['Volatility_20d'] = df['Return'].rolling(20).std() * np.sqrt(252) * 100
    df['Vol_Ratio'] = v / v.rolling(20).mean()
    df['Mom_5d'] = c.pct_change(5) * 100
    df['Mom_10d'] = c.pct_change(10) * 100
    df['Mom_20d'] = c.pct_change(20) * 100
    df['Peak'] = c.cummax()
    df['Drawdown'] = (c - df['Peak']) / df['Peak'] * 100
    df['DD_depth'] = df['Drawdown']

    df['VIX_change_5d'] = df['VIX'].pct_change(5) * 100
    df['RSI_change_5d'] = df['RSI'].diff(5)
    df['Vol_surge'] = df['Volatility_20d'].pct_change(5) * 100
    df['MA20_cross_MA60'] = df['Dist_MA20'] - df['Dist_MA60']
    df['VIX_rank_60d'] = df['VIX'].rolling(60).rank(pct=True) * 100
    df['RSI_rank_60d'] = df['RSI'].rolling(60).rank(pct=True) * 100
    df['Dist_MA60_change_10d'] = df['Dist_MA60'].diff(10)
    df['ATR_vs_VIX'] = df['ATR_Pct'] / (df['VIX'] / 100)

    bb_ma = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df['BB_pctB'] = (c - (bb_ma - 2 * bb_std)) / (4 * bb_std)
    df['Williams_R'] = (h.rolling(14).max() - c) / (h.rolling(14).max() - lo.rolling(14).min()) * -100

    obv = (v * np.sign(c.diff())).cumsum()
    df['OBV_change_10d'] = obv.pct_change(10) * 100

    streak = np.sign(c.diff())
    groups = (streak != streak.shift(1)).cumsum()
    df['Price_streak'] = streak.groupby(groups).cumcount() + 1
    df.loc[streak < 0, 'Price_streak'] = -df.loc[streak < 0, 'Price_streak']

    df['Vol_w_MACD'] = df['MACD_Hist'] * df['Vol_Ratio']
    df['Std_5d'] = c.pct_change().rolling(5).std()
    df['Intraday_range'] = (h - lo) / c * 100
    df['Upper_shadow_pct'] = (h - pd.concat([c, df['Open']], axis=1).max(axis=1)) / (h - lo + 1e-10) * 100
    df['Lower_shadow_pct'] = (pd.concat([c, df['Open']], axis=1).min(axis=1) - lo) / (h - lo + 1e-10) * 100
    df['Dist_from_60d_high'] = (c - h.rolling(60).max()) / h.rolling(60).max() * 100
    df['Dist_from_60d_low'] = (c - lo.rolling(60).min()) / lo.rolling(60).min() * 100
    df['VIX_corr_20d'] = c.pct_change().rolling(20).corr(df['VIX'].pct_change())
    df['Trend_20d'] = c.pct_change(20) * 100 / (df['Volatility_20d'] / np.sqrt(252) * np.sqrt(20) + 1e-10)

    # ── 新增特征 ──
    df['VIX9D_VIX_Ratio'] = df['VIX9D'] / df['VIX']
    df['VIX3M_VIX_Ratio'] = df['VIX3M'] / df['VIX']
    df['VIX3M_VIX9D_Spread'] = df['VIX3M'] - df['VIX9D']
    df['VIX9D_change_5d'] = df['VIX9D'].pct_change(5) * 100
    df['VIX3M_change_5d'] = df['VIX3M'].pct_change(5) * 100

    df['SKEW_MA20'] = df['SKEW'].rolling(20).mean()
    df['SKEW_Dist_MA20'] = df['SKEW'] - df['SKEW_MA20']
    df['SKEW_rank_60d'] = df['SKEW'].rolling(60).rank(pct=True) * 100

    df['Spread_10Y2Y_change_20d'] = df['Spread_10Y_2Y'].diff(20)
    df['Spread_Inverted'] = (df['Spread_10Y_2Y'] < 0).astype(float)

    df['HYG_TLT_change_10d'] = df['HYG_TLT_Ratio'].pct_change(10) * 100
    df['LQD_TLT_change_10d'] = df['LQD_TLT_Ratio'].pct_change(10) * 100

    df['PE_rank_252'] = df['SP500_PE'].rolling(252).rank(pct=True) * 100
    df['CAPE_rank_252'] = df['Shiller_CAPE'].rolling(252).rank(pct=True) * 100
    df['PE_change_60d'] = df['SP500_PE'].pct_change(60) * 100
    df['CAPE_change_60d'] = df['Shiller_CAPE'].pct_change(60) * 100

    return df


def predict(df=None):
    """运行预测"""
    meta = json.load(open(os.path.join(BASE, "model_meta.json")))
    features = meta['features']

    model_top = joblib.load(os.path.join(BASE, meta['top_model']))
    model_bot = joblib.load(os.path.join(BASE, meta['bot_model']))

    if df is None:
        df = fetch_latest()

    latest = df[features].dropna().iloc[-1:]
    date_str = latest.index[0].strftime('%Y-%m-%d')
    close = df.loc[latest.index[0], 'Close']

    top_score = model_top.predict(latest.values)[0]
    bot_score = model_bot.predict(latest.values)[0]

    print(f"\n{'=' * 60}")
    print(f"纳斯达克顶部/底部评分 — {date_str}")
    print(f"{'=' * 60}")
    print(f"  收盘价: {close:,.0f}")
    print(f"  顶部风险评分: {top_score:.1f}/100 {'⚠️ 高风险' if top_score >= 60 else '⚡ 极高风险' if top_score >= 70 else ''}")
    print(f"  底部机会评分: {bot_score:.1f}/100 {'🟢 机会' if bot_score >= 60 else '✅ 极佳机会' if bot_score >= 70 else ''}")

    # 关键指标速览
    keys = ['VIX', 'RSI', 'DD_depth', 'Dist_MA60', 'Mom_20d',
            'SP500_PE', 'Shiller_CAPE', 'Spread_10Y_2Y', 'SKEW']
    print(f"\n  关键指标:")
    for k in keys:
        if k in latest.columns:
            v = latest[k].values[0]
            if not pd.isna(v):
                print(f"    {k:20s} = {v:.2f}")

    # 信号解读
    print(f"\n  信号解读:")
    if top_score >= 70:
        print(f"    ⚠️ 当前处于高度顶部风险区域 (评分{top_score:.0f})")
    elif top_score >= 50:
        print(f"    ⚡ 存在顶部风险信号 (评分{top_score:.0f})")
    elif top_score >= 30:
        print(f"    📊 轻微顶部信号 (评分{top_score:.0f})")
    else:
        print(f"    ✅ 暂无明显顶部风险 (评分{top_score:.0f})")

    if bot_score >= 70:
        print(f"    ✅ 当前处于极佳底部机会区域 (评分{bot_score:.0f})")
    elif bot_score >= 50:
        print(f"    🟢 存在底部机会信号 (评分{bot_score:.0f})")
    elif bot_score >= 30:
        print(f"    📊 轻微底部信号 (评分{bot_score:.0f})")
    else:
        print(f"    ⚠️ 暂无明显底部机会 (评分{bot_score:.0f})")

    return {
        'date': date_str, 'close': float(close),
        'top_score': float(top_score), 'bottom_score': float(bot_score),
    }


if __name__ == '__main__':
    predict()
