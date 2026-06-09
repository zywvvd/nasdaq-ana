"""替代方案 — 距离评分 + 分位数评分 + 规则评分，与ML对比OOS表现。

核心思路：ML回归在13个事件上必然过拟合。改用非参数方法：
1. 距离评分：计算当天特征向量与历史事件日特征向量的马氏距离
2. 分位数评分：每个指标相对自身历史的极端程度
3. 规则评分：已知有效的规则组合
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import RobustScaler
from lib.config import MODEL_META_PATH, data_path
from lib.data_fetcher import fetch_merged_training_data
from lib.features import compute_features
from lib.labels import build_labels

N_FOLDS = 8  # 滚动窗口数


def load_data():
    df = fetch_merged_training_data()
    df = compute_features(df)
    dd_csv = data_path('dd_definitions')
    top_labels, bot_labels = build_labels(df, dd_defs_csv=dd_csv)
    df['top_score_label'] = top_labels
    df['bottom_score_label'] = bot_labels

    meta = json.load(open(MODEL_META_PATH, encoding='utf-8'))
    all_features = meta.get('all_features', [])

    keep = all_features + ['top_score_label', 'bottom_score_label', 'Close']
    valid = df[keep].dropna(subset=all_features + ['top_score_label', 'bottom_score_label'])
    return valid, all_features


def get_event_info():
    dd_csv = data_path('dd_definitions')
    dd_defs = pd.read_csv(dd_csv, parse_dates=['peak_date', 'trough_date'])
    return dd_defs


def walk_forward_eval(valid, all_features, score_func, label_col, event_dates, side_name):
    """通用滚动窗口评估框架。"""
    n = len(valid)
    train_window = 504
    test_window = 126
    gap = 60
    max_start = n - train_window - gap - test_window
    if max_start <= 0:
        return None

    step = max(max_start // N_FOLDS, 1)
    starts = list(range(0, max_start + 1, step))[:N_FOLDS]

    all_scores = []
    all_labels = []
    all_dates = []

    for start in starts:
        tr_e = start + train_window
        te_s = tr_e + gap
        te_e = min(te_s + test_window, n)
        if te_e <= te_s:
            continue

        train_df = valid.iloc[start:tr_e]
        test_df = valid.iloc[te_s:te_e]

        scores = score_func(train_df, test_df, all_features)

        all_scores.extend(scores)
        all_labels.extend(test_df[label_col].values)
        all_dates.extend(test_df.index.tolist())

    if not all_scores:
        return None

    scores = np.array(all_scores)
    labels = np.array(all_labels)
    dates = all_dates

    # 精确率/召回率
    results = {}
    for thresh in [30, 50, 70]:
        sig = scores >= thresh
        n_sig = sig.sum()
        if n_sig > 0:
            sig_dates = np.array(dates)[sig]
            prec = sum(1 for d in sig_dates
                      if any(abs((d - e).days) <= 30 for e in event_dates)) / n_sig
            relevant = [e for e in event_dates if min(dates) <= e <= max(dates)]
            rec = sum(1 for e in relevant
                     if any(abs((d - e).days) <= 30 for d in sig_dates)) / len(relevant) if relevant else 0
        else:
            prec, rec = 0, 0
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        results[f'p{thresh}'] = prec
        results[f'r{thresh}'] = rec
        results[f'f1_{thresh}'] = f1
        results[f'sig{thresh}'] = int(n_sig)

    results['score_max'] = scores.max()
    results['score_mean'] = scores.mean()
    results['n_events'] = len([e for e in event_dates if min(dates) <= e <= max(dates)])

    return results


# ══════════════════════════════════════
# 方法1: 马氏距离评分
# ══════════════════════════════════════
def mahalanobis_score(train_df, test_df, features):
    """计算测试集每天到训练集中事件日特征均值的马氏距离。"""
    # 只用前15个最重要的特征
    top_feats = features[:15]
    X_tr = train_df[top_feats].values
    X_te = test_df[top_feats].values

    # 训练集中的"事件日"（标签>=30的天）
    event_mask_tr = train_df['top_score_label'].values >= 30
    if event_mask_tr.sum() < 5:
        # 样本太少，用标签>0的
        event_mask_tr = train_df['top_score_label'].values > 0
    if event_mask_tr.sum() < 3:
        return np.zeros(len(X_te))

    X_event = X_tr[event_mask_tr]

    try:
        # 计算事件日特征的中心和协方差
        center = X_event.mean(axis=0)
        cov = np.cov(X_event.T) + np.eye(len(top_feats)) * 1e-6  # 正则化
        cov_inv = np.linalg.inv(cov)

        # 计算距离
        diff = X_te - center
        dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

        # 转换为0-100分数：距离越近分数越高
        # 用训练集非事件日的距离分布作为参考
        non_event_mask = train_df['top_score_label'].values == 0
        if non_event_mask.sum() > 10:
            diff_ne = X_tr[non_event_mask] - center
            dist_ne = np.sqrt(np.sum(diff_ne @ cov_inv * diff_ne, axis=1))
            # 分数 = 1 - (距离 / P95非事件距离)，截断到0-100
            ref_dist = np.percentile(dist_ne, 95)
            scores = np.clip((1 - dist / ref_dist) * 100, 0, 100)
        else:
            scores = np.clip(100 / (1 + dist), 0, 100)
    except Exception:
        scores = np.zeros(len(X_te))

    return scores


def mahalanobis_score_bot(train_df, test_df, features):
    """底部马氏距离评分。"""
    top_feats = features[:15]
    X_tr = train_df[top_feats].values
    X_te = test_df[top_feats].values

    event_mask_tr = train_df['bottom_score_label'].values >= 30
    if event_mask_tr.sum() < 5:
        event_mask_tr = train_df['bottom_score_label'].values > 0
    if event_mask_tr.sum() < 3:
        return np.zeros(len(X_te))

    X_event = X_tr[event_mask_tr]

    try:
        center = X_event.mean(axis=0)
        cov = np.cov(X_event.T) + np.eye(len(top_feats)) * 1e-6
        cov_inv = np.linalg.inv(cov)

        diff = X_te - center
        dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

        non_event_mask = train_df['bottom_score_label'].values == 0
        if non_event_mask.sum() > 10:
            diff_ne = X_tr[non_event_mask] - center
            dist_ne = np.sqrt(np.sum(diff_ne @ cov_inv * diff_ne, axis=1))
            ref_dist = np.percentile(dist_ne, 95)
            scores = np.clip((1 - dist / ref_dist) * 100, 0, 100)
        else:
            scores = np.clip(100 / (1 + dist), 0, 100)
    except Exception:
        scores = np.zeros(len(X_te))

    return scores


# ══════════════════════════════════════
# 方法2: 分位数极端评分
# ══════════════════════════════════════
def quantile_score_top(train_df, test_df, features):
    """顶部分位数评分：多个指标同时处于极端高位时得分高。"""
    # 使用已知有效的指标
    top_indicators = {
        'RSI': 'high',              # RSI高=顶部信号
        'Dist_MA60': 'high',        # 偏离MA60高=顶部信号
        'Dist_MA200': 'high',       # 偏离MA200高=顶部信号
        'VIX': 'low',               # VIX低=顶部信号（反向）
        'Volatility_20d': 'low',    # 波动率低=顶部信号（反向）
        'Mom_20d': 'high',          # 动量高=顶部信号
    }

    scores = np.zeros(len(test_df))
    n_indicators = 0

    for feat, direction in top_indicators.items():
        if feat not in train_df.columns or feat not in test_df.columns:
            continue

        tr_vals = train_df[feat].dropna().values
        if len(tr_vals) < 50:
            continue

        p10 = np.percentile(tr_vals, 10)
        p90 = np.percentile(tr_vals, 90)

        te_vals = test_df[feat].values

        if direction == 'high':
            # 指标越高越像顶部
            indicator_score = np.clip((te_vals - p90) / (p90 - p10 + 1e-10) * 100, 0, 100)
        else:
            # 指标越低越像顶部
            indicator_score = np.clip((p10 - te_vals) / (p90 - p10 + 1e-10) * 100, 0, 100)

        scores += np.nan_to_num(indicator_score, nan=0)
        n_indicators += 1

    if n_indicators > 0:
        scores = scores / n_indicators
    return scores


def quantile_score_bot(train_df, test_df, features):
    """底部分位数评分：多个指标同时处于极端低位时得分高。"""
    bot_indicators = {
        'RSI': 'low',               # RSI低=底部信号
        'Dist_MA60': 'low',         # 偏离MA60低=底部信号
        'Dist_MA200': 'low',        # 偏离MA200低=底部信号
        'VIX': 'high',              # VIX高=底部信号
        'Volatility_20d': 'high',   # 波动率高=底部信号
        'Mom_20d': 'low',           # 动量低=底部信号
    }

    scores = np.zeros(len(test_df))
    n_indicators = 0

    for feat, direction in bot_indicators.items():
        if feat not in train_df.columns or feat not in test_df.columns:
            continue

        tr_vals = train_df[feat].dropna().values
        if len(tr_vals) < 50:
            continue

        p10 = np.percentile(tr_vals, 10)
        p90 = np.percentile(tr_vals, 90)
        te_vals = test_df[feat].values

        if direction == 'low':
            indicator_score = np.clip((p10 - te_vals) / (p90 - p10 + 1e-10) * 100, 0, 100)
        else:
            indicator_score = np.clip((te_vals - p90) / (p90 - p10 + 1e-10) * 100, 0, 100)

        scores += np.nan_to_num(indicator_score, nan=0)
        n_indicators += 1

    if n_indicators > 0:
        scores = scores / n_indicators
    return scores


# ══════════════════════════════════════
# 方法3: 规则评分（已知有效规则）
# ══════════════════════════════════════
def rule_score_top(train_df, test_df, features):
    """顶部规则评分：基于回测中命中率最高的规则。"""
    scores = np.zeros(len(test_df))

    # 规则1: RSI > 65 (命中率100%)
    if 'RSI' in test_df.columns:
        rsi = test_df['RSI'].values
        scores += np.where(rsi > 65, 30, np.where(rsi > 60, 15, 0))

    # 规则2: 偏离MA60 > 3% (命中率77%)
    if 'Dist_MA60' in test_df.columns:
        dma60 = test_df['Dist_MA60'].values
        scores += np.where(dma60 > 5, 30, np.where(dma60 > 3, 20, 0))

    # 规则3: VIX < 16 (命中率77%)
    if 'VIX' in test_df.columns:
        vix = test_df['VIX'].values
        scores += np.where(vix < 13, 20, np.where(vix < 16, 10, 0))

    # 规则4: 偏离MA200 > 10%
    if 'Dist_MA200' in test_df.columns:
        dma200 = test_df['Dist_MA200'].values
        scores += np.where(dma200 > 15, 20, np.where(dma200 > 10, 10, 0))

    return np.clip(scores, 0, 100)


def rule_score_bot(train_df, test_df, features):
    """底部规则评分。"""
    scores = np.zeros(len(test_df))

    # 规则1: RSI < 35 (命中率100%)
    if 'RSI' in test_df.columns:
        rsi = test_df['RSI'].values
        scores += np.where(rsi < 25, 30, np.where(rsi < 35, 20, 0))

    # 规则2: 偏离MA60 < -5% (命中率85%)
    if 'Dist_MA60' in test_df.columns:
        dma60 = test_df['Dist_MA60'].values
        scores += np.where(dma60 < -10, 30, np.where(dma60 < -5, 20, 0))

    # 规则3: VIX > 25 (命中率77%)
    if 'VIX' in test_df.columns:
        vix = test_df['VIX'].values
        scores += np.where(vix > 35, 20, np.where(vix > 25, 10, 0))

    # 规则4: 偏离MA200 < -10%
    if 'Dist_MA200' in test_df.columns:
        dma200 = test_df['Dist_MA200'].values
        scores += np.where(dma200 < -15, 20, np.where(dma200 < -10, 10, 0))

    return np.clip(scores, 0, 100)


def main():
    print("=" * 80)
    print("替代方案 OOS 评估")
    print("=" * 80)

    valid, all_features = load_data()
    dd_defs = get_event_info()
    event_peaks = dd_defs['peak_date'].tolist()
    event_troughs = dd_defs['trough_date'].tolist()

    print(f"有效样本: {len(valid)} 行\n")

    # 获取特征重要性排名（用于马氏距离选择特征）
    meta = json.load(open(MODEL_META_PATH, encoding='utf-8'))
    imp_top = meta.get('feature_importance_top', {})
    imp_bot = meta.get('feature_importance_bot', {})
    all_feats = meta.get('all_features', [])
    combined = {f: max(imp_top.get(f, 0), imp_bot.get(f, 0)) for f in all_feats}
    ranked = sorted(combined.items(), key=lambda x: -x[1])
    ranked_features = [f for f, _ in ranked]

    methods = [
        ('马氏距离', mahalanobis_score, mahalanobis_score_bot),
        ('分位数评分', quantile_score_top, quantile_score_bot),
        ('规则评分', rule_score_top, rule_score_bot),
    ]

    print(f"{'方法':<12} | {'侧':<4} | {'P50':>5} {'R50':>5} {'F1_50':>6} {'Sig':>4} | "
          f"{'P30':>5} {'R30':>5} {'F1_30':>6} {'Sig':>4} | {'Max':>5} {'Mean':>5} {'Events':>7}")
    print("-" * 100)

    for method_name, score_top, score_bot in methods:
        for side, func, label_col, events in [
            ('顶部', score_top, 'top_score_label', event_peaks),
            ('底部', score_bot, 'bottom_score_label', event_troughs)
        ]:
            # 马氏距离需要特征排名
            if method_name == '马氏距离':
                result = walk_forward_eval(valid, ranked_features, func, label_col, events, side)
            else:
                result = walk_forward_eval(valid, all_features, func, label_col, events, side)

            if result:
                print(f"{method_name:<12} | {side:<4} | "
                      f"{result['p50']:5.0%} {result['r50']:5.0%} {result['f1_50']:6.3f} {result['sig50']:4d} | "
                      f"{result['p30']:5.0%} {result['r30']:5.0%} {result['f1_30']:6.3f} {result['sig30']:4d} | "
                      f"{result['score_max']:5.1f} {result['score_mean']:5.1f} {result['n_events']:7d}")
            else:
                print(f"{method_name:<12} | {side:<4} | N/A")
        print()

    # 额外：看看规则评分在训练集上 vs OOS 的差距
    print("\n" + "=" * 80)
    print("规则评分详细分析（最有前景的方法）")
    print("=" * 80)

    for side, func, label_col, events in [
        ('顶部', rule_score_top, 'top_score_label', event_peaks),
        ('底部', rule_score_bot, 'bottom_score_label', event_troughs)
    ]:
        result = walk_forward_eval(valid, all_features, func, label_col, events, side)
        if result:
            print(f"\n  {side}:")
            for thresh in [30, 50, 70]:
                print(f"    >={thresh}: P={result[f'p{thresh}']:.1%}, R={result[f'r{thresh}']:.1%}, "
                      f"F1={result[f'f1_{thresh}']:.3f}, sig={result[f'sig{thresh}']}")


if __name__ == '__main__':
    main()
