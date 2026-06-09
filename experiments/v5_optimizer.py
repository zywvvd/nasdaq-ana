"""V5 规则评分系统 — OOS优化 + 宏观指标增强。

通过walk-forward网格搜索优化规则权重和阈值。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from itertools import product
from lib.config import data_path
from lib.data_fetcher import fetch_merged_training_data
from lib.features import compute_features


def compute_indicator_scores(df, train_window=504):
    """计算各指标的极端百分位评分（滚动窗口）。"""
    n = len(df)
    scores = {}

    # ── 顶部指标 ──
    top_rules = {
        'rsi_high': ('RSI', 'high'),
        'dist_ma60_high': ('Dist_MA60', 'high'),
        'dist_ma200_high': ('Dist_MA200', 'high'),
        'vix_low': ('VIX', 'low'),
        'vol_low': ('Volatility_20d', 'low'),
        'mom_high': ('Mom_20d', 'high'),
        'cape_high': ('Shiller_CAPE', 'high'),
        'skew_high': ('SKEW', 'high'),
    }

    # ── 底部指标 ──
    bot_rules = {
        'rsi_low': ('RSI', 'low'),
        'dist_ma60_low': ('Dist_MA60', 'low'),
        'dist_ma200_low': ('Dist_MA200', 'low'),
        'vix_high': ('VIX', 'high'),
        'vol_high': ('Volatility_20d', 'high'),
        'mom_low': ('Mom_20d', 'low'),
        'dd_deep': ('DD_depth', 'low'),  # DD_depth负值越大=底部
        'hyg_tlt_low': ('HYG_TLT_Ratio', 'low'),
    }

    for rule_name, (feat, direction) in {**top_rules, **bot_rules}.items():
        if feat not in df.columns:
            scores[rule_name] = np.zeros(n)
            continue

        vals = df[feat].values
        rule_scores = np.zeros(n)

        for i in range(train_window, n):
            window = vals[i - train_window:i]
            window = window[~np.isnan(window)]
            if len(window) < 50:
                continue

            p5 = np.percentile(window, 5)
            p10 = np.percentile(window, 10)
            p25 = np.percentile(window, 25)
            p75 = np.percentile(window, 75)
            p90 = np.percentile(window, 90)
            p95 = np.percentile(window, 95)
            v = vals[i]

            if np.isnan(v):
                continue

            if direction == 'high':
                if v >= p95:
                    rule_scores[i] = 100
                elif v >= p90:
                    rule_scores[i] = 70
                elif v >= p75:
                    rule_scores[i] = 30
            else:  # low
                if v <= p5:
                    rule_scores[i] = 100
                elif v <= p10:
                    rule_scores[i] = 70
                elif v <= p25:
                    rule_scores[i] = 30

        scores[rule_name] = rule_scores

    return scores


def walk_forward_eval_rule(df, top_weights, bot_weights, top_thresh, bot_thresh,
                           train_window=504, test_window=126, gap=60, n_folds=10):
    """Walk-forward评估规则评分。"""
    n = len(df)

    top_rule_names = list(top_weights.keys())
    bot_rule_names = list(bot_weights.keys())

    max_start = n - train_window - gap - test_window
    if max_start <= 0:
        return None

    step = max(max_start // n_folds, 1)
    starts = list(range(0, max_start + 1, step))[:n_folds]

    dd_defs = pd.read_csv(data_path('dd_definitions'), parse_dates=['peak_date', 'trough_date'])
    event_peaks = dd_defs['peak_date'].tolist()
    event_troughs = dd_defs['trough_date'].tolist()

    all_top_scores = []
    all_bot_scores = []
    all_dates = []

    # 预计算指标分数
    ind_scores = compute_indicator_scores(df, train_window)

    for start in starts:
        tr_e = start + train_window
        te_s = tr_e + gap
        te_e = min(te_s + test_window, n)
        if te_e <= te_s:
            continue

        for i in range(te_s, te_e):
            # 顶部评分
            top_s = sum(ind_scores[r][i] * w for r, w in top_weights.items()) / sum(top_weights.values())
            all_top_scores.append(top_s)
            # 底部评分
            bot_s = sum(ind_scores[r][i] * w for r, w in bot_weights.items()) / sum(bot_weights.values())
            all_bot_scores.append(bot_s)
            all_dates.append(df.index[i])

    if not all_top_scores:
        return None

    top_arr = np.array(all_top_scores)
    bot_arr = np.array(all_bot_scores)
    dates_arr = all_dates

    result = {}
    for side, scores, events, thresh in [
        ('top', top_arr, event_peaks, top_thresh),
        ('bot', bot_arr, event_troughs, bot_thresh)
    ]:
        sig = scores >= thresh
        n_sig = sig.sum()
        if n_sig > 0:
            sig_dates = np.array(dates_arr)[sig]
            prec = sum(1 for d in sig_dates
                      if any(abs((d - e).days) <= 30 for e in events)) / n_sig
            relevant = [e for e in events if min(dates_arr) <= e <= max(dates_arr)]
            rec = sum(1 for e in relevant
                     if any(abs((d - e).days) <= 30 for d in sig_dates)) / len(relevant) if relevant else 0
        else:
            prec, rec = 0, 0
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        result[f'{side}_p'] = prec
        result[f'{side}_r'] = rec
        result[f'{side}_f1'] = f1
        result[f'{side}_sig'] = int(n_sig)
        result[f'{side}_max'] = scores.max()
        result[f'{side}_mean'] = scores.mean()

    result['n_events_top'] = len([e for e in event_peaks if min(dates_arr) <= e <= max(dates_arr)])
    result['n_events_bot'] = len([e for e in event_troughs if min(dates_arr) <= e <= max(dates_arr)])
    return result


def main():
    print("=" * 80)
    print("V5 规则评分优化")
    print("=" * 80)

    df = fetch_merged_training_data()
    df = compute_features(df)
    print(f"数据: {len(df)} 行\n")

    # ── 基础规则权重 ──
    # 顶部规则
    top_weights_base = {
        'rsi_high': 3.0,       # RSI高 (100%命中率)
        'dist_ma60_high': 2.5, # 偏离MA60高 (77%命中率)
        'dist_ma200_high': 1.5,
        'vix_low': 1.5,        # VIX低 (77%命中率)
        'vol_low': 1.0,
        'mom_high': 1.0,
        'cape_high': 1.5,      # 估值高
        'skew_high': 1.0,      # 尾部风险
    }

    # 底部规则
    bot_weights_base = {
        'rsi_low': 3.0,        # RSI低 (100%命中率)
        'dist_ma60_low': 2.5,  # 偏离MA60低 (85%命中率)
        'dist_ma200_low': 1.5,
        'vix_high': 1.5,       # VIX高 (77%命中率)
        'vol_high': 1.0,
        'mom_low': 1.0,
        'dd_deep': 2.0,        # 回撤深
        'hyg_tlt_low': 1.0,    # 信用利差扩大
    }

    # ── 网格搜索阈值 ──
    print("阈值网格搜索:")
    print(f"{'top_t':>6} {'bot_t':>6} | {'top_P':>6} {'top_R':>6} {'top_F1':>7} {'top_S':>5} | "
          f"{'bot_P':>6} {'bot_R':>6} {'bot_F1':>7} {'bot_S':>5}")
    print("-" * 80)

    best_top_f1 = 0
    best_bot_f1 = 0
    best_config = None

    for top_t, bot_t in product([20, 30, 40, 50, 60, 70], repeat=2):
        r = walk_forward_eval_rule(df, top_weights_base, bot_weights_base, top_t, bot_t)
        if r:
            if r['top_f1'] > best_top_f1 or r['bot_f1'] > best_bot_f1:
                if r['top_f1'] >= best_top_f1:
                    best_top_f1 = r['top_f1']
                if r['bot_f1'] >= best_bot_f1:
                    best_bot_f1 = r['bot_f1']
                best_config = (top_t, bot_t)

            if r['top_f1'] > 0.01 or r['bot_f1'] > 0.01:
                print(f"{top_t:6d} {bot_t:6d} | "
                      f"{r['top_p']:6.1%} {r['top_r']:6.1%} {r['top_f1']:7.3f} {r['top_sig']:5d} | "
                      f"{r['bot_p']:6.1%} {r['bot_r']:6.1%} {r['bot_f1']:7.3f} {r['bot_sig']:5d}")

    print(f"\n最佳配置: top_thresh={best_config[0]}, bot_thresh={best_config[1]}")
    print(f"最佳顶部F1={best_top_f1:.3f}, 最佳底部F1={best_bot_f1:.3f}")

    # ── 权重搜索（微调） ──
    print(f"\n{'=' * 80}")
    print("权重微调（基于最佳阈值）")
    print("=" * 80)

    top_t, bot_t = best_config

    # 测试不同的RSI权重
    print("\nRSI权重消融:")
    for rsi_w in [2.0, 3.0, 4.0, 5.0]:
        tw = dict(top_weights_base)
        tw['rsi_high'] = rsi_w
        bw = dict(bot_weights_base)
        bw['rsi_low'] = rsi_w
        r = walk_forward_eval_rule(df, tw, bw, top_t, bot_t)
        if r:
            print(f"  RSI_w={rsi_w:.1f}: top_F1={r['top_f1']:.3f} P={r['top_p']:.1%} R={r['top_r']:.1%} | "
                  f"bot_F1={r['bot_f1']:.3f} P={r['bot_p']:.1%} R={r['bot_r']:.1%}")

    # 测试是否加入宏观指标
    print("\n宏观指标开关:")
    for cape_w in [0, 1.5, 3.0]:
        for skew_w in [0, 1.0, 2.0]:
            tw = dict(top_weights_base)
            tw['cape_high'] = cape_w
            tw['skew_high'] = skew_w
            r = walk_forward_eval_rule(df, tw, bot_weights_base, top_t, bot_t)
            if r and r['top_f1'] > 0.01:
                print(f"  CAPE={cape_w:.1f} SKEW={skew_w:.1f}: top_F1={r['top_f1']:.3f} P={r['top_p']:.1%} R={r['top_r']:.1%}")

    # ── 最终评估 ──
    print(f"\n{'=' * 80}")
    print("最终最佳配置评估")
    print("=" * 80)

    r = walk_forward_eval_rule(df, top_weights_base, bot_weights_base, top_t, bot_t, n_folds=15)
    if r:
        print(f"  OOS顶部: P={r['top_p']:.1%} R={r['top_r']:.1%} F1={r['top_f1']:.3f} "
              f"sig={r['top_sig']} max={r['top_max']:.1f} events={r['n_events_top']}")
        print(f"  OOS底部: P={r['bot_p']:.1%} R={r['bot_r']:.1%} F1={r['bot_f1']:.3f} "
              f"sig={r['bot_sig']} max={r['bot_max']:.1f} events={r['n_events_bot']}")

    # 保存最佳配置
    config = {
        'version': 'V5',
        'method': 'rule_based',
        'top_weights': top_weights_base,
        'bot_weights': bot_weights_base,
        'top_threshold': top_t,
        'bot_threshold': bot_t,
        'oos_top_f1': r['top_f1'] if r else 0,
        'oos_bot_f1': r['bot_f1'] if r else 0,
    }
    output = os.path.join(os.path.dirname(__file__), 'v5_config.json')
    with open(output, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n配置已保存: {output}")


if __name__ == '__main__':
    main()
