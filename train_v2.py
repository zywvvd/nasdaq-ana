#!/usr/bin/env python3
"""
纳斯达克顶部/底部评分模型 V2 — 扩展特征版
新增数据源: VIX9D, VIX3M, SKEW, 国债利差10Y-2Y, 信用利差(HYG/LQD/TLT), PE/CAPE
训练策略: 加权采样 + 样本权重处理标签稀疏问题
"""
import pandas as pd
import numpy as np
import os, json, warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
import joblib

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))

# ── 1. 加载主表 ──
print("=" * 60)
print("步骤 1: 加载主表数据")
print("=" * 60)
df = pd.read_csv(os.path.join(BASE, "数据_全指标合并主表.csv"), parse_dates=['Date'], index_col='Date')
print(f"  主表: {len(df)} 行, {df.columns.size} 列, {df.index[0].date()} ~ {df.index[-1].date()}")

# ── 2. 加载新数据源 ──
print("\n步骤 2: 加载新数据源")

new_data = {}
for fname, key in [
    ("数据_VIX9D_15y.csv", "VIX9D"),
    ("数据_VIX3M_15y.csv", "VIX3M"),
    ("数据_SKEW.csv", "SKEW"),
]:
    tmp = pd.read_csv(os.path.join(BASE, fname), parse_dates=['Date'], index_col='Date')
    new_data[key] = tmp[[key]]
    print(f"  {key}: {len(tmp)} 行")

treasury = pd.read_csv(os.path.join(BASE, "数据_国债利差10Y2Y.csv"), parse_dates=['Date'], index_col='Date')
credit = pd.read_csv(os.path.join(BASE, "数据_信用利差.csv"), parse_dates=['Date'], index_col='Date')
pe_monthly = pd.read_csv(os.path.join(BASE, "数据_SP500估值月度_填充.csv"), parse_dates=['date'], index_col='date')
print(f"  国债利差: {len(treasury)} 行, 信用利差: {len(credit)} 行, PE月度: {len(pe_monthly)} 行")

# ── 3. 合并所有数据 ──
print("\n步骤 3: 合并数据")

for key, data in new_data.items():
    df = df.join(data, how='left')

df = df.join(treasury[['Spread_10Y_2Y']], how='left')
df = df.join(credit[['HYG', 'LQD', 'TLT', 'HYG_TLT_Ratio']], how='left')

pe_daily = pe_monthly.reindex(df.index.union(pe_monthly.index)).interpolate(method='time').reindex(df.index)
df['SP500_PE'] = pe_daily['SP500_PE']
df['Shiller_CAPE'] = pe_daily['Shiller_CAPE']
df['Div_Yield'] = pe_daily['Div_Yield']

for col in ['VIX9D', 'VIX3M', 'SKEW', 'Spread_10Y_2Y', 'HYG', 'LQD', 'TLT', 'HYG_TLT_Ratio',
            'SP500_PE', 'Shiller_CAPE', 'Div_Yield']:
    df[col] = df[col].ffill()

print(f"  合并后: {df.columns.size} 列")

# ── 4. 计算所有特征 ──
print("\n步骤 4: 计算技术指标特征")

c = df['Close']
h = df['High']
lo = df['Low']
v = df['Volume']

# 均线与偏离
df['MA20'] = c.rolling(20).mean()
df['MA60'] = c.rolling(60).mean()
df['MA200'] = c.rolling(200).mean()
df['Dist_MA20'] = (c / df['MA20'] - 1) * 100
df['Dist_MA60'] = (c / df['MA60'] - 1) * 100
df['Dist_MA200'] = (c / df['MA200'] - 1) * 100

# RSI
delta = c.diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
df['RSI'] = 100 - 100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean())

# MACD
ema12 = c.ewm(span=12).mean()
ema26 = c.ewm(span=26).mean()
df['MACD'] = ema12 - ema26
df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

# ATR
tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
df['ATR_Pct'] = tr.rolling(14).mean() / c * 100

# 波动率与量比
df['Return'] = c.pct_change()
df['Volatility_20d'] = df['Return'].rolling(20).std() * np.sqrt(252) * 100
df['Vol_Ratio'] = v / v.rolling(20).mean()

# 动量
df['Mom_5d'] = c.pct_change(5) * 100
df['Mom_10d'] = c.pct_change(10) * 100
df['Mom_20d'] = c.pct_change(20) * 100

# 回撤
df['Peak'] = c.cummax()
df['Drawdown'] = (c - df['Peak']) / df['Peak'] * 100
df['DD_depth'] = df['Drawdown']

# 额外原始特征
df['VIX_change_5d'] = df['VIX'].pct_change(5) * 100
df['RSI_change_5d'] = df['RSI'].diff(5)
df['Vol_surge'] = df['Volatility_20d'].pct_change(5) * 100
df['MA20_cross_MA60'] = df['Dist_MA20'] - df['Dist_MA60']
df['VIX_rank_60d'] = df['VIX'].rolling(60).rank(pct=True) * 100
df['RSI_rank_60d'] = df['RSI'].rolling(60).rank(pct=True) * 100
df['Dist_MA60_change_10d'] = df['Dist_MA60'].diff(10)
df['ATR_vs_VIX'] = df['ATR_Pct'] / (df['VIX'] / 100)

# 布林带 %B
bb_ma = c.rolling(20).mean()
bb_std = c.rolling(20).std()
df['BB_pctB'] = (c - (bb_ma - 2 * bb_std)) / (4 * bb_std)

# Williams %R
df['Williams_R'] = (h.rolling(14).max() - c) / (h.rolling(14).max() - lo.rolling(14).min()) * -100

# OBV
obv = (v * np.sign(c.diff())).cumsum()
df['OBV_change_10d'] = obv.pct_change(10) * 100

# 连涨/连跌天数
streak = np.sign(c.diff())
groups = (streak != streak.shift(1)).cumsum()
df['Price_streak'] = streak.groupby(groups).cumcount() + 1
df.loc[streak < 0, 'Price_streak'] = -df.loc[streak < 0, 'Price_streak']

# K线形态
df['Std_5d'] = c.pct_change().rolling(5).std()
df['Intraday_range'] = (h - lo) / c * 100
df['Upper_shadow_pct'] = (h - pd.concat([c, df['Open']], axis=1).max(axis=1)) / (h - lo + 1e-10) * 100
df['Lower_shadow_pct'] = (pd.concat([c, df['Open']], axis=1).min(axis=1) - lo) / (h - lo + 1e-10) * 100

# 距离高低点
df['Dist_from_60d_high'] = (c - h.rolling(60).max()) / h.rolling(60).max() * 100
df['Dist_from_60d_low'] = (c - lo.rolling(60).min()) / lo.rolling(60).min() * 100

# VIX相关性
df['VIX_corr_20d'] = c.pct_change().rolling(20).corr(df['VIX'].pct_change())

# 趋势强度
df['Trend_20d'] = c.pct_change(20) * 100 / (df['Volatility_20d'] / np.sqrt(252) * np.sqrt(20) + 1e-10)

# 交互特征
df['Vol_w_MACD'] = df['MACD_Hist'] * df['Vol_Ratio']

# ── 5. 新增特征 ──
print("\n步骤 5: 计算新增特征")

# VIX期限结构
df['VIX9D_VIX_Ratio'] = df['VIX9D'] / df['VIX']
df['VIX3M_VIX_Ratio'] = df['VIX3M'] / df['VIX']
df['VIX3M_VIX9D_Spread'] = df['VIX3M'] - df['VIX9D']
df['VIX9D_change_5d'] = df['VIX9D'].pct_change(5) * 100
df['VIX3M_change_5d'] = df['VIX3M'].pct_change(5) * 100

# SKEW特征
df['SKEW_MA20'] = df['SKEW'].rolling(20).mean()
df['SKEW_Dist_MA20'] = df['SKEW'] - df['SKEW_MA20']
df['SKEW_rank_60d'] = df['SKEW'].rolling(60).rank(pct=True) * 100

# 国债利差特征
df['Spread_10Y2Y_MA20'] = df['Spread_10Y_2Y'].rolling(20).mean()
df['Spread_10Y2Y_change_20d'] = df['Spread_10Y_2Y'].diff(20)
df['Spread_Inverted'] = (df['Spread_10Y_2Y'] < 0).astype(float)

# 信用利差特征
df['HYG_LQD_Ratio'] = df['HYG'] / df['LQD']
df['HYG_TLT_change_10d'] = df['HYG_TLT_Ratio'].pct_change(10) * 100
df['LQD_TLT_Ratio'] = df['LQD'] / df['TLT']
df['LQD_TLT_change_10d'] = df['LQD_TLT_Ratio'].pct_change(10) * 100

# PE/CAPE特征
df['PE_MA60'] = df['SP500_PE'].rolling(60).mean()
df['CAPE_MA60'] = df['Shiller_CAPE'].rolling(60).mean()
df['PE_rank_252'] = df['SP500_PE'].rolling(252).rank(pct=True) * 100
df['CAPE_rank_252'] = df['Shiller_CAPE'].rolling(252).rank(pct=True) * 100
df['PE_change_60d'] = df['SP500_PE'].pct_change(60) * 100
df['CAPE_change_60d'] = df['Shiller_CAPE'].pct_change(60) * 100

print("  特征计算完成")

# ── 6. 定义特征列表 ──
feature_cols = [
    # 原始36特征
    'VIX', 'VVIX', 'VIX_VVIX_Ratio', 'RSI', 'MACD_Hist', 'ATR_Pct',
    'Volatility_20d', 'Vol_Ratio', 'Dist_MA20', 'Dist_MA60', 'Dist_MA200',
    'Mom_5d', 'Mom_10d', 'Mom_20d',
    'VIX_change_5d', 'RSI_change_5d', 'Vol_surge', 'MA20_cross_MA60',
    'VIX_rank_60d', 'RSI_rank_60d', 'DD_depth', 'Dist_MA60_change_10d', 'ATR_vs_VIX',
    'BB_pctB', 'Williams_R', 'OBV_change_10d', 'Price_streak',
    'Vol_w_MACD',
    'Std_5d', 'Intraday_range', 'Upper_shadow_pct', 'Lower_shadow_pct',
    'Dist_from_60d_high', 'Dist_from_60d_low', 'VIX_corr_20d', 'Trend_20d',
    # 新增特征
    'VIX9D_VIX_Ratio', 'VIX3M_VIX_Ratio', 'VIX3M_VIX9D_Spread',
    'VIX9D_change_5d', 'VIX3M_change_5d',
    'SKEW', 'SKEW_Dist_MA20', 'SKEW_rank_60d',
    'Spread_10Y_2Y', 'Spread_10Y2Y_change_20d', 'Spread_Inverted',
    'HYG_TLT_Ratio', 'HYG_TLT_change_10d', 'LQD_TLT_Ratio', 'LQD_TLT_change_10d',
    'SP500_PE', 'Shiller_CAPE', 'PE_rank_252', 'CAPE_rank_252',
    'PE_change_60d', 'CAPE_change_60d',
]

n_features = len(feature_cols)
print(f"\n总特征数: {n_features}")

# ── 7. 构建软标签 ──
print("\n步骤 6: 构建软标签")

dd_defs = pd.read_csv(os.path.join(BASE, "分析_所有回撤区间定义.csv"), parse_dates=['peak_date', 'trough_date'])
print(f"  回撤事件: {len(dd_defs)} 个")

# 趋势方向
trend_dir = pd.Series(0.0, index=df.index)
for i in range(20, len(df)):
    ma_now = df['MA20'].iloc[i]
    ma_20_ago = df['MA20'].iloc[max(0, i - 20)]
    if pd.notna(ma_now) and pd.notna(ma_20_ago) and ma_20_ago > 0:
        trend_dir.iloc[i] = 1.0 if ma_now > ma_20_ago else -1.0


def soft_score(distance_pct, max_dist=10.0):
    if pd.isna(distance_pct) or distance_pct < 0:
        return 0.0
    if distance_pct > max_dist:
        return 0.0
    return max(0, 100 * (1 - distance_pct / max_dist))


top_scores = pd.Series(0.0, index=df.index)
bottom_scores = pd.Series(0.0, index=df.index)

for _, ev in dd_defs.iterrows():
    peak_date = ev['peak_date']
    trough_date = ev['trough_date']
    peak_price = ev['peak_price']
    trough_price = ev['trough_price']

    # 顶部标签: peak_date前后各30个交易日
    pre_peak = df.loc[:peak_date].tail(31)
    for dt, row in pre_peak.iterrows():
        price_dist = abs(row['Close'] - peak_price) / peak_price * 100
        score = soft_score(price_dist)
        t = trend_dir.get(dt, 0)
        if t < 0:
            score *= 0.5
        if score > top_scores.get(dt, 0):
            top_scores.loc[dt] = score

    # 底部标签: trough_date前后各30个交易日
    pre_trough = df.loc[:trough_date].tail(31)
    for dt, row in pre_trough.iterrows():
        price_dist = abs(row['Close'] - trough_price) / trough_price * 100
        score = soft_score(price_dist)
        t = trend_dir.get(dt, 0)
        if t > 0:
            score *= 0.5
        if score > bottom_scores.get(dt, 0):
            bottom_scores.loc[dt] = score

df['top_score_label'] = top_scores
df['bottom_score_label'] = bottom_scores

print(f"  顶部标签 > 0: {(top_scores > 0).sum()} 天, >=50: {(top_scores >= 50).sum()}, >=70: {(top_scores >= 70).sum()}")
print(f"  底部标签 > 0: {(bottom_scores > 0).sum()} 天, >=50: {(bottom_scores >= 50).sum()}, >=70: {(bottom_scores >= 70).sum()}")

# ── 8. 准备训练数据（加权采样策略）──
print("\n步骤 7: 准备训练数据 (加权采样)")

keep_cols = feature_cols + ['top_score_label', 'bottom_score_label', 'Close']
valid = df[keep_cols].dropna()
print(f"  有效样本: {len(valid)} 行")

# 构建样本权重: 非零标签的样本权重更高
y_top = valid['top_score_label'].values
y_bot = valid['bottom_score_label'].values
close_vals = valid['Close'].values
dates = valid.index

# 顶部权重: 有分数的样本权重=10, 无分数=1
w_top = np.where(y_top > 0, 5 + y_top / 20, 1.0)
# 底部权重
w_bot = np.where(y_bot > 0, 5 + y_bot / 20, 1.0)

X = valid[feature_cols].values
print(f"  特征矩阵: {X.shape}")

# ── 9. 训练模型 ──
print("\n步骤 8: 训练模型")

# 使用带样本权重的RandomForest
model_top = RandomForestRegressor(
    n_estimators=1500, max_depth=10, min_samples_leaf=8,
    max_features='sqrt', random_state=42, n_jobs=-1
)
model_top.fit(X, y_top, sample_weight=w_top)

model_bot = RandomForestRegressor(
    n_estimators=1500, max_depth=10, min_samples_leaf=8,
    max_features='sqrt', random_state=42, n_jobs=-1
)
model_bot.fit(X, y_bot, sample_weight=w_bot)

# 训练集上的预测
pred_top = model_top.predict(X)
pred_bot = model_bot.predict(X)

print(f"  顶部预测: min={pred_top.min():.1f}, mean={pred_top.mean():.1f}, max={pred_top.max():.1f}")
print(f"  底部预测: min={pred_bot.min():.1f}, mean={pred_bot.mean():.1f}, max={pred_bot.max():.1f}")

# 时序交叉验证
print("\n  时序交叉验证 (5折)...")
tscv = TimeSeriesSplit(n_splits=5)
cv_scores_top = []
cv_scores_bot = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr, X_te = X[train_idx], X[test_idx]
    yt_tr, yt_te = y_top[train_idx], y_top[test_idx]
    yb_tr, yb_te = y_bot[train_idx], y_bot[test_idx]
    wt_tr = w_top[train_idx]
    wb_tr = w_bot[train_idx]

    mt = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=8,
                                max_features='sqrt', random_state=42, n_jobs=-1)
    mt.fit(X_tr, yt_tr, sample_weight=wt_tr)
    pt = mt.predict(X_te)

    mb = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=8,
                                max_features='sqrt', random_state=42, n_jobs=-1)
    mb.fit(X_tr, yb_tr, sample_weight=wb_tr)
    pb = mb.predict(X_te)

    # 用"信号命中率"代替R²: 预测>阈值时，真实标签>50的比例
    for thresh in [30, 50]:
        mask_t = pt >= thresh
        if mask_t.sum() > 0:
            hit_rate = (yt_te[mask_t] >= 30).mean()
            print(f"    Fold {fold + 1} 顶部(>={thresh}): {mask_t.sum()} 个信号, 命中率={hit_rate:.1%}")

        mask_b = pb >= thresh
        if mask_b.sum() > 0:
            hit_rate = (yb_te[mask_b] >= 30).mean()
            print(f"    Fold {fold + 1} 底部(>={thresh}): {mask_b.sum()} 个信号, 命中率={hit_rate:.1%}")

# ── 10. A% 评估 ──
print("\n" + "=" * 60)
print("步骤 9: A% 价格距离评估")
print("=" * 60)

eval_df = valid.copy()
eval_df['pred_top'] = pred_top
eval_df['pred_bot'] = pred_bot

# 对每个阈值，计算信号与真实顶部/底部的价格距离
for threshold in [40, 50, 60, 70]:
    print(f"\n--- 阈值: {threshold} ---")

    # 顶部信号
    top_sig = eval_df[eval_df['pred_top'] >= threshold]
    print(f"  顶部信号数量: {len(top_sig)}")

    if len(top_sig) > 0:
        for a in [1, 2, 3, 5, 8]:
            hits = 0
            for dt, row in top_sig.iterrows():
                for _, ev in dd_defs.iterrows():
                    price_dist = abs(row['Close'] - ev['peak_price']) / ev['peak_price'] * 100
                    if price_dist <= a:
                        hits += 1
                        break
            print(f"    A={a}%: {hits}/{len(top_sig)} = {hits / len(top_sig):.1%}")

    # 底部信号
    bot_sig = eval_df[eval_df['pred_bot'] >= threshold]
    print(f"  底部信号数量: {len(bot_sig)}")

    if len(bot_sig) > 0:
        for a in [1, 2, 3, 5, 8]:
            hits = 0
            for dt, row in bot_sig.iterrows():
                for _, ev in dd_defs.iterrows():
                    price_dist = abs(row['Close'] - ev['trough_price']) / ev['trough_price'] * 100
                    if price_dist <= a:
                        hits += 1
                        break
            print(f"    A={a}%: {hits}/{len(bot_sig)} = {hits / len(bot_sig):.1%}")

# ── 11. 逐事件验证 ──
print("\n" + "=" * 60)
print("步骤 10: 逐事件验证")
print("=" * 60)

for _, ev in dd_defs.iterrows():
    peak_date = ev['peak_date']
    trough_date = ev['trough_date']
    dd_pct = abs(ev['drawdown_pct'])

    # 找顶部附近区域的最高预测分
    mask_pre_peak = (eval_df.index <= peak_date) & (eval_df.index >= peak_date - pd.Timedelta(days=60))
    peak_region = eval_df[mask_pre_peak]
    if len(peak_region) > 0:
        max_top = peak_region['pred_top'].max()
        max_top_date = peak_region['pred_top'].idxmax()
        delta_top = (peak_date - max_top_date).days
    else:
        max_top = 0
        delta_top = 0

    # 找底部附近区域的最高预测分
    mask_pre_trough = (eval_df.index <= trough_date) & (eval_df.index >= trough_date - pd.Timedelta(days=60))
    trough_region = eval_df[mask_pre_trough]
    if len(trough_region) > 0:
        max_bot = trough_region['pred_bot'].max()
        max_bot_date = trough_region['pred_bot'].idxmax()
        delta_bot = (trough_date - max_bot_date).days
    else:
        max_bot = 0
        delta_bot = 0

    top_str = f"顶部={max_top:.0f}({'提前' if delta_top > 0 else '滞后'}{abs(delta_top)}天)"
    bot_str = f"底部={max_bot:.0f}({'提前' if delta_bot > 0 else '滞后'}{abs(delta_bot)}天)"

    print(f"  {peak_date.strftime('%Y-%m')}~{trough_date.strftime('%Y-%m')} ({dd_pct:.1f}%): {top_str}, {bot_str}")

# ── 12. 特征重要性 ──
print("\n" + "=" * 60)
print("步骤 11: 特征重要性 (Top 20)")
print("=" * 60)

imp_top = pd.Series(model_top.feature_importances_, index=feature_cols).sort_values(ascending=False)
imp_bot = pd.Series(model_bot.feature_importances_, index=feature_cols).sort_values(ascending=False)

print("\n  顶部模型 Top 20:")
for i, (feat, imp) in enumerate(imp_top.head(20).items()):
    marker = " ★" if feat not in [
        'VIX', 'VVIX', 'VIX_VVIX_Ratio', 'RSI', 'MACD_Hist', 'ATR_Pct',
        'Volatility_20d', 'Vol_Ratio', 'Dist_MA20', 'Dist_MA60', 'Dist_MA200',
        'Mom_5d', 'Mom_10d', 'Mom_20d',
        'VIX_change_5d', 'RSI_change_5d', 'Vol_surge', 'MA20_cross_MA60',
        'VIX_rank_60d', 'RSI_rank_60d', 'DD_depth', 'Dist_MA60_change_10d', 'ATR_vs_VIX',
        'BB_pctB', 'Williams_R', 'OBV_change_10d', 'Price_streak',
        'Vol_w_MACD', 'Std_5d', 'Intraday_range', 'Upper_shadow_pct', 'Lower_shadow_pct',
        'Dist_from_60d_high', 'Dist_from_60d_low', 'VIX_corr_20d', 'Trend_20d',
    ] else ""
    print(f"    {i + 1:2d}. {feat:30s} {imp:.4f}{marker}")

print("\n  底部模型 Top 20:")
for i, (feat, imp) in enumerate(imp_bot.head(20).items()):
    marker = " ★" if feat not in [
        'VIX', 'VVIX', 'VIX_VVIX_Ratio', 'RSI', 'MACD_Hist', 'ATR_Pct',
        'Volatility_20d', 'Vol_Ratio', 'Dist_MA20', 'Dist_MA60', 'Dist_MA200',
        'Mom_5d', 'Mom_10d', 'Mom_20d',
        'VIX_change_5d', 'RSI_change_5d', 'Vol_surge', 'MA20_cross_MA60',
        'VIX_rank_60d', 'RSI_rank_60d', 'DD_depth', 'Dist_MA60_change_10d', 'ATR_vs_VIX',
        'BB_pctB', 'Williams_R', 'OBV_change_10d', 'Price_streak',
        'Vol_w_MACD', 'Std_5d', 'Intraday_range', 'Upper_shadow_pct', 'Lower_shadow_pct',
        'Dist_from_60d_high', 'Dist_from_60d_low', 'VIX_corr_20d', 'Trend_20d',
    ] else ""
    print(f"    {i + 1:2d}. {feat:30s} {imp:.4f}{marker}")

# ── 13. 保存模型 ──
print("\n" + "=" * 60)
print("步骤 12: 保存模型")
print("=" * 60)

joblib.dump(model_top, os.path.join(BASE, "model_top_score.pkl"))
joblib.dump(model_bot, os.path.join(BASE, "model_bottom_score.pkl"))

meta = {
    "model": "Dual RF Regressor V2 (Weighted)",
    "features": feature_cols,
    "n_features": n_features,
    "top_model": "model_top_score.pkl",
    "bot_model": "model_bottom_score.pkl",
    "new_features_vs_v1": [
        'VIX9D_VIX_Ratio', 'VIX3M_VIX_Ratio', 'VIX3M_VIX9D_Spread',
        'VIX9D_change_5d', 'VIX3M_change_5d',
        'SKEW', 'SKEW_Dist_MA20', 'SKEW_rank_60d',
        'Spread_10Y_2Y', 'Spread_10Y2Y_change_20d', 'Spread_Inverted',
        'HYG_TLT_Ratio', 'HYG_TLT_change_10d', 'LQD_TLT_Ratio', 'LQD_TLT_change_10d',
        'SP500_PE', 'Shiller_CAPE', 'PE_rank_252', 'CAPE_rank_252',
        'PE_change_60d', 'CAPE_change_60d',
    ],
    "feature_importance_top": {k: round(float(v), 6) for k, v in imp_top.head(20).items()},
    "feature_importance_bot": {k: round(float(v), 6) for k, v in imp_bot.head(20).items()},
}

with open(os.path.join(BASE, "model_meta.json"), 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# 保存预测结果
eval_df[['Close', 'pred_top', 'pred_bot', 'top_score_label', 'bottom_score_label']].to_csv(
    os.path.join(BASE, "分析_ML预测结果_V2.csv")
)

print(f"  模型已保存")
print(f"  特征数: 36 → {n_features} (+{n_features - 36})")
print(f"\n训练完成!")
