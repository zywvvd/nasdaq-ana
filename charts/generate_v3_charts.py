#!/usr/bin/env python3
"""V3 模型评估结果全面可视化。"""
import os
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# 中文字体
plt.rcParams['font.family'] = ['Noto Serif CJK SC', 'Noto Serif CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHART_DIR = os.path.join(BASE, 'charts')
DATA_DIR = os.path.join(BASE, 'data')
os.makedirs(CHART_DIR, exist_ok=True)

# 加载数据
eval_df = pd.read_csv(os.path.join(DATA_DIR, '分析_评估数据_V3.csv'), index_col=0, parse_dates=True)
events_df = pd.read_csv(os.path.join(DATA_DIR, '分析_评估事件_V3.csv'), parse_dates=['peak_date', 'trough_date'])
imp_top = pd.read_csv(os.path.join(DATA_DIR, '分析_特征重要性_顶部_V3.csv'), index_col=0, header=None)
imp_top.columns = ['importance']
imp_top.index.name = 'feature'
imp_bot = pd.read_csv(os.path.join(DATA_DIR, '分析_特征重要性_底部_V3.csv'), index_col=0, header=None)
imp_bot.columns = ['importance']
imp_bot.index.name = 'feature'
cv_data = json.load(open(os.path.join(DATA_DIR, '分析_CV结果_V3.json')))

COLORS = {
    'top': '#e74c3c',
    'bot': '#2ecc71',
    'price': '#3498db',
    'gray': '#95a5a6',
    'dark': '#2c3e50',
    'warning': '#f39c12',
    'bg': '#fafafa',
}


def chart1_score_timeseries():
    """图1: 评分时间序列 (15年全景)"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 14), height_ratios=[2, 1, 1],
                                          gridspec_kw={'hspace': 0.08})
    fig.patch.set_facecolor('white')

    # 价格
    ax1.fill_between(eval_df.index, eval_df['Close'], alpha=0.15, color=COLORS['price'])
    ax1.plot(eval_df.index, eval_df['Close'], color=COLORS['price'], linewidth=1.2, label='Nasdaq Close')
    for _, ev in events_df.iterrows():
        ax1.axvline(ev['peak_date'], color=COLORS['top'], alpha=0.4, linestyle='--', linewidth=0.8)
        ax1.axvline(ev['trough_date'], color=COLORS['bot'], alpha=0.4, linestyle='--', linewidth=0.8)
    ax1.set_ylabel('Nasdaq', fontsize=12)
    ax1.set_title('V3 模型评分时间序列 (2012-2026)', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    # 顶部评分
    ax2.fill_between(eval_df.index, eval_df['pred_top'], alpha=0.3, color=COLORS['top'])
    ax2.plot(eval_df.index, eval_df['pred_top'], color=COLORS['top'], linewidth=0.8)
    ax2.axhline(60, color=COLORS['warning'], linestyle='--', alpha=0.7, label='阈值=60')
    ax2.axhline(70, color=COLORS['dark'], linestyle='--', alpha=0.5, label='阈值=70')
    ax2.set_ylabel('顶部评分', fontsize=12, color=COLORS['top'])
    ax2.set_ylim(-2, 90)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)

    # 底部评分
    ax3.fill_between(eval_df.index, eval_df['pred_bot'], alpha=0.3, color=COLORS['bot'])
    ax3.plot(eval_df.index, eval_df['pred_bot'], color=COLORS['bot'], linewidth=0.8)
    ax3.axhline(50, color=COLORS['warning'], linestyle='--', alpha=0.7, label='阈值=50')
    ax3.axhline(60, color=COLORS['dark'], linestyle='--', alpha=0.5, label='阈值=60')
    ax3.set_ylabel('底部评分', fontsize=12, color=COLORS['bot'])
    ax3.set_ylim(-2, 90)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))

    for ax in [ax1, ax2]:
        ax.set_xticklabels([])

    fig.savefig(os.path.join(CHART_DIR, '20_V3评分_15年全景.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图1: 20_V3评分_15年全景.png')


def chart2_score_recent():
    """图2: 评分时间序列 (近3年)"""
    recent = eval_df.last('3Y')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12), height_ratios=[2, 1, 1],
                                          gridspec_kw={'hspace': 0.08})
    fig.patch.set_facecolor('white')

    ax1.fill_between(recent.index, recent['Close'], alpha=0.15, color=COLORS['price'])
    ax1.plot(recent.index, recent['Close'], color=COLORS['price'], linewidth=1.5)
    for _, ev in events_df.iterrows():
        if ev['peak_date'] >= recent.index[0]:
            ax1.axvline(ev['peak_date'], color=COLORS['top'], alpha=0.5, linestyle='--', linewidth=1)
            ax1.axvline(ev['trough_date'], color=COLORS['bot'], alpha=0.5, linestyle='--', linewidth=1)
    ax1.set_title('V3 模型评分 — 近3年', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Nasdaq', fontsize=12)
    ax1.grid(alpha=0.3)

    ax2.fill_between(recent.index, recent['pred_top'], alpha=0.3, color=COLORS['top'])
    ax2.plot(recent.index, recent['pred_top'], color=COLORS['top'], linewidth=1)
    ax2.axhline(60, color=COLORS['warning'], linestyle='--', alpha=0.7)
    ax2.set_ylabel('顶部评分', fontsize=12, color=COLORS['top'])
    ax2.set_ylim(-2, 90)
    ax2.grid(alpha=0.3)

    ax3.fill_between(recent.index, recent['pred_bot'], alpha=0.3, color=COLORS['bot'])
    ax3.plot(recent.index, recent['pred_bot'], color=COLORS['bot'], linewidth=1)
    ax3.axhline(50, color=COLORS['warning'], linestyle='--', alpha=0.7)
    ax3.set_ylabel('底部评分', fontsize=12, color=COLORS['bot'])
    ax3.set_ylim(-2, 90)
    ax3.grid(alpha=0.3)

    for ax in [ax1, ax2]:
        ax.set_xticklabels([])

    fig.savefig(os.path.join(CHART_DIR, '21_V3评分_近3年.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图2: 21_V3评分_近3年.png')


def chart3_distribution():
    """图3: 评分分布直方图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('white')

    for ax, data, label, color in [
        (ax1, eval_df['pred_top'], '顶部评分分布', COLORS['top']),
        (ax2, eval_df['pred_bot'], '底部评分分布', COLORS['bot']),
    ]:
        ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(data.mean(), color=COLORS['dark'], linestyle='-', linewidth=2, label=f'均值={data.mean():.1f}')
        ax.axvline(np.percentile(data, 90), color=COLORS['warning'], linestyle='--', linewidth=2,
                    label=f'P90={np.percentile(data, 90):.1f}')
        ax.axvline(np.percentile(data, 95), color=COLORS['dark'], linestyle=':', linewidth=2,
                    label=f'P95={np.percentile(data, 95):.1f}')
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_xlabel('评分', fontsize=12)
        ax.set_ylabel('天数', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle('V3 模型评分分布', fontsize=16, fontweight='bold', y=1.02)
    fig.savefig(os.path.join(CHART_DIR, '22_V3评分分布.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图3: 22_V3评分分布.png')


def chart4_per_event():
    """图4: 逐事件评分柱状图"""
    dd_defs = events_df
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'hspace': 0.35})
    fig.patch.set_facecolor('white')

    # 计算每事件的最高评分
    event_scores = []
    for _, ev in dd_defs.iterrows():
        peak_date = ev['peak_date']
        trough_date = ev['trough_date']

        mask_p = (eval_df.index <= peak_date) & (eval_df.index >= peak_date - pd.Timedelta(days=60))
        max_top = eval_df.loc[mask_p, 'pred_top'].max() if mask_p.sum() > 0 else 0
        delta_top = (peak_date - eval_df.loc[mask_p, 'pred_top'].idxmax()).days if max_top > 0 and mask_p.sum() > 0 else 0

        mask_b = (eval_df.index <= trough_date) & (eval_df.index >= trough_date - pd.Timedelta(days=60))
        max_bot = eval_df.loc[mask_b, 'pred_bot'].max() if mask_b.sum() > 0 else 0
        delta_bot = (trough_date - eval_df.loc[mask_b, 'pred_bot'].idxmax()).days if max_bot > 0 and mask_b.sum() > 0 else 0

        event_scores.append({
            'label': peak_date.strftime('%Y-%m'),
            'dd_pct': ev['dd_pct'],
            'top_score': max_top,
            'bot_score': max_bot,
            'top_timing': delta_top,
            'bot_timing': delta_bot,
        })

    es = pd.DataFrame(event_scores)
    x = np.arange(len(es))
    w = 0.35

    # 评分
    bars1 = ax1.bar(x - w/2, es['top_score'], w, color=COLORS['top'], alpha=0.8, label='顶部评分')
    bars2 = ax1.bar(x + w/2, es['bot_score'], w, color=COLORS['bot'], alpha=0.8, label='底部评分')
    ax1.axhline(60, color=COLORS['warning'], linestyle='--', alpha=0.5, label='阈值60')
    ax1.axhline(70, color=COLORS['dark'], linestyle='--', alpha=0.3, label='阈值70')
    ax1.set_xticks(x)
    ax1.set_xticklabels(es['label'], rotation=45, ha='right')
    ax1.set_ylabel('最高评分', fontsize=12)
    ax1.set_title('逐事件最高评分', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')

    # 在柱子上标注回撤幅度
    for i, row in es.iterrows():
        ax1.text(i, max(row['top_score'], row['bot_score']) + 2, f"{row['dd_pct']:.0f}%",
                ha='center', fontsize=8, color=COLORS['dark'])

    # 提前/滞后天数
    colors_top = [COLORS['bot'] if t >= 0 else COLORS['top'] for t in es['top_timing']]
    colors_bot = [COLORS['bot'] if t >= 0 else COLORS['top'] for t in es['bot_timing']]
    ax2.bar(x - w/2, es['top_timing'], w, color=colors_top, alpha=0.8, label='顶部时机(正=提前)')
    ax2.bar(x + w/2, es['bot_timing'], w, color=colors_bot, alpha=0.8, label='底部时机(正=提前)')
    ax2.axhline(0, color=COLORS['dark'], linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(es['label'], rotation=45, ha='right')
    ax2.set_ylabel('天数 (正=提前)', fontsize=12)
    ax2.set_title('逐事件预测时机', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')

    fig.savefig(os.path.join(CHART_DIR, '23_V3逐事件评分.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图4: 23_V3逐事件评分.png')


def chart5_feature_importance():
    """图5: 特征重要性 Top 25 对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor('white')

    n = 25
    top_n = imp_top.head(n).sort_values('importance', ascending=True)
    bot_n = imp_bot.head(n).sort_values('importance', ascending=True)

    ax1.barh(range(n), top_n['importance'], color=COLORS['top'], alpha=0.8)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(top_n.index, fontsize=9)
    ax1.set_xlabel('Importance', fontsize=12)
    ax1.set_title(f'顶部模型 Top {n}', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3, axis='x')

    ax2.barh(range(n), bot_n['importance'], color=COLORS['bot'], alpha=0.8)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(bot_n.index, fontsize=9)
    ax2.set_xlabel('Importance', fontsize=12)
    ax2.set_title(f'底部模型 Top {n}', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='x')

    fig.suptitle('V3 模型特征重要性 Top 25', fontsize=16, fontweight='bold')
    fig.savefig(os.path.join(CHART_DIR, '24_V3特征重要性Top25.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图5: 24_V3特征重要性Top25.png')


def chart6_metrics():
    """图6: 精确率/召回率/F1 随阈值变化"""
    thresholds = [50, 60, 70]
    dd_defs = events_df
    dd_dates_peak = dd_defs['peak_date'].tolist()
    dd_dates_trough = dd_defs['trough_date'].tolist()

    metrics = {'top': {}, 'bot': {}}
    for thresh in thresholds:
        for side, pred, dates, label in [
            ('top', eval_df['pred_top'].values, dd_dates_peak, '顶部'),
            ('bot', eval_df['pred_bot'].values, dd_dates_trough, '底部'),
        ]:
            sig_mask = pred >= thresh
            sig_dates = eval_df.index[sig_mask]
            n_sig = int(sig_mask.sum())
            if n_sig == 0:
                continue
            n_events = len(dates)
            precision = sum(1 for d in sig_dates if any(abs((d - e).days) <= 30 for e in dates)) / n_sig
            recall = sum(1 for e in dates if any(abs((d - e).days) <= 30 for d in sig_dates)) / n_events
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            metrics[side][thresh] = {'P': precision, 'R': recall, 'F1': f1, 'n': n_sig}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('white')

    for i, metric_name in enumerate(['P', 'R', 'F1']):
        ax = axes[i]
        top_vals = [metrics['top'].get(t, {}).get(metric_name, 0) for t in thresholds]
        bot_vals = [metrics['bot'].get(t, {}).get(metric_name, 0) for t in thresholds]
        x = np.arange(len(thresholds))
        w = 0.35
        ax.bar(x - w/2, top_vals, w, color=COLORS['top'], alpha=0.8, label='顶部')
        ax.bar(x + w/2, bot_vals, w, color=COLORS['bot'], alpha=0.8, label='底部')
        for j in range(len(thresholds)):
            ax.text(x[j] - w/2, top_vals[j] + 0.01, f'{top_vals[j]:.1%}', ha='center', fontsize=9)
            ax.text(x[j] + w/2, bot_vals[j] + 0.01, f'{bot_vals[j]:.1%}', ha='center', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f'>={t}' for t in thresholds])
        ax.set_ylim(0, 1.15)
        ax.set_title({'P': '精确率 Precision', 'R': '召回率 Recall', 'F1': 'F1 Score'}[metric_name],
                      fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

    fig.suptitle('V3 模型评估指标 vs 阈值', fontsize=16, fontweight='bold')
    fig.savefig(os.path.join(CHART_DIR, '25_V3评估指标.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图6: 25_V3评估指标.png')


def chart7_cv_analysis():
    """图7: 交叉验证分析"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.patch.set_facecolor('white')

    for i, cv in enumerate(cv_data):
        ax = axes[i // 3][i % 3] if i < 6 else None
        if ax is None:
            continue

        pt = np.array(cv['pred_top'])
        pb = np.array(cv['pred_bot'])
        yt = np.array(cv['y_top'])
        yb = np.array(cv['y_bot'])

        # Scatter: predicted vs actual for signals
        ax.scatter(yt[yt > 0], pt[yt > 0], c=COLORS['top'], alpha=0.4, s=15, label='顶部')
        ax.scatter(yb[yb > 0], pb[yb > 0], c=COLORS['bot'], alpha=0.4, s=15, label='底部')
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1)
        ax.set_xlabel('实际标签', fontsize=10)
        ax.set_ylabel('预测评分', fontsize=10)
        ax.set_title(f'Fold {cv["fold"]} (训练={cv["train_size"]}, 测试={cv["test_size"]})',
                      fontsize=11, fontweight='bold')
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # 最后一个子图: 信号数量统计
    ax = axes[1][2]
    folds = [cv['fold'] for cv in cv_data]
    top_sigs = [sum(1 for p in cv['pred_top'] if p >= 50) for cv in cv_data]
    bot_sigs = [sum(1 for p in cv['pred_bot'] if p >= 50) for cv in cv_data]
    x = np.arange(len(folds))
    w = 0.35
    ax.bar(x - w/2, top_sigs, w, color=COLORS['top'], alpha=0.8, label='顶部信号(>=50)')
    ax.bar(x + w/2, bot_sigs, w, color=COLORS['bot'], alpha=0.8, label='底部信号(>=50)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.set_ylabel('信号数', fontsize=10)
    ax.set_title('各折信号数量', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle('V3 时序交叉验证 (5-Fold)', fontsize=16, fontweight='bold')
    fig.savefig(os.path.join(CHART_DIR, '26_V3交叉验证.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图7: 26_V3交叉验证.png')


def chart8_score_vs_actual():
    """图8: 评分 vs 实际标签散点图 (全量)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')

    pt = eval_df['pred_top'].values
    yt = eval_df['label_top'].values
    pb = eval_df['pred_bot'].values
    yb = eval_df['label_bot'].values

    # 只画有标签的样本
    mask_t = yt > 0
    mask_b = yb > 0

    ax1.scatter(yt[mask_t], pt[mask_t], c=COLORS['top'], alpha=0.3, s=10)
    ax1.plot([0, 100], [0, 100], 'k--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('实际顶部标签', fontsize=12)
    ax1.set_ylabel('预测顶部评分', fontsize=12)
    ax1.set_title(f'顶部模型: 预测 vs 实际 ({mask_t.sum()} 正样本)', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)

    # 训练 vs 全量的对比
    ax2.scatter(yb[mask_b], pb[mask_b], c=COLORS['bot'], alpha=0.3, s=10)
    ax2.plot([0, 100], [0, 100], 'k--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('实际底部标签', fontsize=12)
    ax2.set_ylabel('预测底部评分', fontsize=12)
    ax2.set_title(f'底部模型: 预测 vs 实际 ({mask_b.sum()} 正样本)', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)

    fig.suptitle('V3 模型预测 vs 实际标签', fontsize=16, fontweight='bold')
    fig.savefig(os.path.join(CHART_DIR, '27_V3预测vs实际.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图8: 27_V3预测vs实际.png')


def chart9_feature_groups():
    """图9: 特征类别重要性分布"""
    groups = {
        '价格/均线': ['Dist_MA', 'Close_above', 'MA', 'Mom_', 'Price_streak', 'DD_depth', 'Dist_from', 'Gap_', 'High_low_range', 'Return_', 'Max_dd', 'Days_since'],
        '波动率': ['Volatility', 'ATR', 'Std_', 'Intraday_range', 'BB_', 'Williams'],
        'RSI/MACD': ['RSI', 'MACD'],
        'VIX': ['VIX', 'VVIX', 'VIX_'],
        '信用利差': ['HYG', 'LQD'],
        '国债利差': ['Spread_'],
        '估值': ['PE_', 'CAPE_', 'SP500_PE', 'Shiller'],
        '量': ['Vol_', 'OBV', 'Volume'],
        '相关性/趋势': ['Trend_', 'VIX_corr'],
        '统计': ['Std_5d_change', 'Return_skew', 'Return_kurtosis'],
    }

    top_imp = {str(k): v for k, v in zip(imp_top.index, imp_top['importance'])}
    bot_imp = {str(k): v for k, v in zip(imp_bot.index, imp_bot['importance'])}

    group_top = {}
    group_bot = {}
    for gname, prefixes in groups.items():
        total_top = sum(v for k, v in top_imp.items() if any(k.startswith(p) or k == p for p in prefixes))
        total_bot = sum(v for k, v in bot_imp.items() if any(k.startswith(p) or k == p for p in prefixes))
        group_top[gname] = total_top
        group_bot[gname] = total_bot

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('white')

    sorted_groups = sorted(group_top.keys(), key=lambda x: group_top[x], reverse=True)
    x = np.arange(len(sorted_groups))

    ax1.bar(x, [group_top[g] for g in sorted_groups], color=COLORS['top'], alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_groups, rotation=45, ha='right', fontsize=11)
    ax1.set_ylabel('累计重要性', fontsize=12)
    ax1.set_title('顶部模型 — 特征类别重要性', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')

    sorted_groups_b = sorted(group_bot.keys(), key=lambda x: group_bot[x], reverse=True)
    x2 = np.arange(len(sorted_groups_b))
    ax2.bar(x2, [group_bot[g] for g in sorted_groups_b], color=COLORS['bot'], alpha=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(sorted_groups_b, rotation=45, ha='right', fontsize=11)
    ax2.set_ylabel('累计重要性', fontsize=12)
    ax2.set_title('底部模型 — 特征类别重要性', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')

    fig.suptitle('V3 特征类别重要性分布', fontsize=16, fontweight='bold')
    fig.savefig(os.path.join(CHART_DIR, '28_V3特征类别.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图9: 28_V3特征类别.png')


if __name__ == '__main__':
    print("生成 V3 评估图表...")
    chart1_score_timeseries()
    chart2_score_recent()
    chart3_distribution()
    chart4_per_event()
    chart5_feature_importance()
    chart6_metrics()
    chart7_cv_analysis()
    chart8_score_vs_actual()
    chart9_feature_groups()
    print("\n全部图表生成完成!")
