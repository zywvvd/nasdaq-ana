#!/usr/bin/env python3
"""V4 模型评估结果全面可视化。"""
import os, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.family'] = ['Noto Serif CJK SC', 'Noto Serif CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHART_DIR = os.path.join(BASE, 'charts')
DATA_DIR = os.path.join(BASE, 'data')
os.makedirs(CHART_DIR, exist_ok=True)

eval_nasdaq = pd.read_csv(os.path.join(DATA_DIR, '分析_评估数据_V4_nasdaq.csv'), index_col=0, parse_dates=True)
events_df = pd.read_csv(os.path.join(DATA_DIR, '分析_评估事件_V4.csv'), parse_dates=['peak_date', 'trough_date'])
imp_top = pd.read_csv(os.path.join(DATA_DIR, '分析_特征重要性_顶部_V4.csv'), index_col=0, header=None)
imp_top.columns = ['importance']
imp_bot = pd.read_csv(os.path.join(DATA_DIR, '分析_特征重要性_底部_V4.csv'), index_col=0, header=None)
imp_bot.columns = ['importance']
cv_data = json.load(open(os.path.join(DATA_DIR, '分析_CV结果_V4.json')))

C = {'top': '#e74c3c', 'bot': '#2ecc71', 'price': '#3498db', 'warn': '#f39c12', 'dark': '#2c3e50', 'cal': '#9b59b6'}


def chart1_timeseries():
    """图1: 15年全景 — 价格+顶部评分+底部评分"""
    fig, axes = plt.subplots(4, 1, figsize=(22, 16), height_ratios=[2.5, 1, 1, 1], gridspec_kw={'hspace': 0.08})
    fig.patch.set_facecolor('white')

    # 价格
    ax = axes[0]
    ax.fill_between(eval_nasdaq.index, eval_nasdaq['Close'], alpha=0.12, color=C['price'])
    ax.plot(eval_nasdaq.index, eval_nasdaq['Close'], color=C['price'], lw=1.2)
    for _, ev in events_df.iterrows():
        ax.axvline(ev['peak_date'], color=C['top'], alpha=0.35, ls='--', lw=0.8)
        ax.axvline(ev['trough_date'], color=C['bot'], alpha=0.35, ls='--', lw=0.8)
    ax.set_ylabel('Nasdaq', fontsize=12)
    ax.set_title('V4 模型评分时间序列 (2012-2026) — 63维 LightGBM', fontsize=16, fontweight='bold', pad=15)
    ax.grid(alpha=0.3)
    ax.set_xticklabels([])

    # 顶部原始评分
    ax = axes[1]
    ax.fill_between(eval_nasdaq.index, eval_nasdaq['pred_top'], alpha=0.3, color=C['top'])
    ax.plot(eval_nasdaq.index, eval_nasdaq['pred_top'], color=C['top'], lw=0.8)
    ax.axhline(60, color=C['warn'], ls='--', alpha=0.7, label='阈值=60')
    ax.set_ylabel('顶部评分', fontsize=11, color=C['top'])
    ax.set_ylim(-5, 110)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticklabels([])

    # 底部原始评分
    ax = axes[2]
    ax.fill_between(eval_nasdaq.index, eval_nasdaq['pred_bot'], alpha=0.3, color=C['bot'])
    ax.plot(eval_nasdaq.index, eval_nasdaq['pred_bot'], color=C['bot'], lw=0.8)
    ax.axhline(50, color=C['warn'], ls='--', alpha=0.7, label='阈值=50')
    ax.set_ylabel('底部评分', fontsize=11, color=C['bot'])
    ax.set_ylim(-5, 110)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticklabels([])

    # 校准概率
    ax = axes[3]
    ax.fill_between(eval_nasdaq.index, eval_nasdaq['cal_top_prob']*100, alpha=0.25, color=C['cal'], label='顶部概率')
    ax.fill_between(eval_nasdaq.index, eval_nasdaq['cal_bot_prob']*100, alpha=0.25, color=C['bot'], label='底部概率')
    ax.axhline(50, color=C['warn'], ls='--', alpha=0.5)
    ax.set_ylabel('校准概率 %', fontsize=11)
    ax.set_ylim(-2, 105)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    fig.savefig(os.path.join(CHART_DIR, '30_V4评分_15年全景.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图1: 30_V4评分_15年全景.png')


def chart2_recent():
    """图2: 近3年"""
    recent = eval_nasdaq.last('3Y')
    fig, axes = plt.subplots(3, 1, figsize=(20, 12), height_ratios=[2, 1, 1], gridspec_kw={'hspace': 0.08})
    fig.patch.set_facecolor('white')

    ax = axes[0]
    ax.fill_between(recent.index, recent['Close'], alpha=0.15, color=C['price'])
    ax.plot(recent.index, recent['Close'], color=C['price'], lw=1.5)
    for _, ev in events_df.iterrows():
        if ev['peak_date'] >= recent.index[0]:
            ax.axvline(ev['peak_date'], color=C['top'], alpha=0.5, ls='--', lw=1)
            ax.axvline(ev['trough_date'], color=C['bot'], alpha=0.5, ls='--', lw=1)
    ax.set_title('V4 模型评分 — 近3年', fontsize=16, fontweight='bold')
    ax.set_ylabel('Nasdaq', fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xticklabels([])

    ax = axes[1]
    ax.fill_between(recent.index, recent['pred_top'], alpha=0.3, color=C['top'])
    ax.plot(recent.index, recent['pred_top'], color=C['top'], lw=1)
    ax.axhline(60, color=C['warn'], ls='--', alpha=0.7)
    ax.set_ylabel('顶部评分', fontsize=11, color=C['top'])
    ax.set_ylim(-5, 110)
    ax.grid(alpha=0.3)
    ax.set_xticklabels([])

    ax = axes[2]
    ax.fill_between(recent.index, recent['pred_bot'], alpha=0.3, color=C['bot'])
    ax.plot(recent.index, recent['pred_bot'], color=C['bot'], lw=1)
    ax.axhline(50, color=C['warn'], ls='--', alpha=0.7)
    ax.set_ylabel('底部评分', fontsize=11, color=C['bot'])
    ax.set_ylim(-5, 110)
    ax.grid(alpha=0.3)

    fig.savefig(os.path.join(CHART_DIR, '31_V4评分_近3年.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图2: 31_V4评分_近3年.png')


def chart3_distribution():
    """图3: 评分分布 + 校准概率分布"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('white')

    for ax, data, label, color in [
        (axes[0][0], eval_nasdaq['pred_top'], '顶部原始评分', C['top']),
        (axes[0][1], eval_nasdaq['pred_bot'], '底部原始评分', C['bot']),
    ]:
        ax.hist(data, bins=60, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(data.mean(), color=C['dark'], ls='-', lw=2, label=f'均值={data.mean():.1f}')
        ax.axvline(np.percentile(data, 95), color=C['warn'], ls='--', lw=2, label=f'P95={np.percentile(data, 95):.1f}')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xlabel('评分'); ax.set_ylabel('天数')
        ax.legend(); ax.grid(alpha=0.3)

    for ax, data, label, color in [
        (axes[1][0], eval_nasdaq['cal_top_prob']*100, '顶部校准概率 %', C['cal']),
        (axes[1][1], eval_nasdaq['cal_bot_prob']*100, '底部校准概率 %', '#27ae60'),
    ]:
        ax.hist(data, bins=60, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(50, color=C['warn'], ls='--', lw=2, label='50%')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xlabel('概率 %'); ax.set_ylabel('天数')
        ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle('V4 评分与校准概率分布', fontsize=16, fontweight='bold', y=1.01)
    fig.savefig(os.path.join(CHART_DIR, '32_V4评分分布.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图3: 32_V4评分分布.png')


def chart4_per_event():
    """图4: 逐事件评分 + 时机"""
    event_scores = []
    for _, ev in events_df.iterrows():
        peak_date, trough_date = ev['peak_date'], ev['trough_date']
        dd_pct = abs(ev['drawdown_pct'])

        mask_p = (eval_nasdaq.index <= peak_date) & (eval_nasdaq.index >= peak_date - pd.Timedelta(days=60))
        max_top = eval_nasdaq.loc[mask_p, 'pred_top'].max() if mask_p.sum() > 0 else 0
        delta_top = (peak_date - eval_nasdaq.loc[mask_p, 'pred_top'].idxmax()).days if max_top > 0 and mask_p.sum() > 0 else 0

        mask_b = (eval_nasdaq.index <= trough_date) & (eval_nasdaq.index >= trough_date - pd.Timedelta(days=60))
        max_bot = eval_nasdaq.loc[mask_b, 'pred_bot'].max() if mask_b.sum() > 0 else 0
        delta_bot = (trough_date - eval_nasdaq.loc[mask_b, 'pred_bot'].idxmax()).days if max_bot > 0 and mask_b.sum() > 0 else 0

        event_scores.append({'label': peak_date.strftime('%Y-%m'), 'dd_pct': dd_pct,
                             'top': max_top, 'bot': max_bot, 'top_t': delta_top, 'bot_t': delta_bot})

    es = pd.DataFrame(event_scores)
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'hspace': 0.35})
    fig.patch.set_facecolor('white')
    x, w = np.arange(len(es)), 0.35

    ax = axes[0]
    ax.bar(x - w/2, es['top'], w, color=C['top'], alpha=0.8, label='顶部评分')
    ax.bar(x + w/2, es['bot'], w, color=C['bot'], alpha=0.8, label='底部评分')
    ax.axhline(60, color=C['warn'], ls='--', alpha=0.5)
    for i, row in es.iterrows():
        ax.text(i, max(row['top'], row['bot']) + 2, f"{row['dd_pct']:.0f}%", ha='center', fontsize=8, color=C['dark'])
    ax.set_xticks(x); ax.set_xticklabels(es['label'], rotation=45, ha='right')
    ax.set_ylabel('最高评分'); ax.set_title('逐事件最高评分', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ctop = [C['bot'] if t > 0 else C['top'] for t in es['top_t']]
    cbot = [C['bot'] if t > 0 else C['top'] for t in es['bot_t']]
    ax.bar(x - w/2, es['top_t'], w, color=ctop, alpha=0.8, label='顶部时机(正=提前)')
    ax.bar(x + w/2, es['bot_t'], w, color=cbot, alpha=0.8, label='底部时机(正=提前)')
    ax.axhline(0, color=C['dark'], lw=1)
    ax.set_xticks(x); ax.set_xticklabels(es['label'], rotation=45, ha='right')
    ax.set_ylabel('天数'); ax.set_title('逐事件预测时机', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    fig.savefig(os.path.join(CHART_DIR, '33_V4逐事件评分.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图4: 33_V4逐事件评分.png')


def chart5_importance():
    """图5: 特征重要性 Top 25"""
    n = 25
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor('white')

    for ax, imp, label, color in [(ax1, imp_top, '顶部模型', C['top']), (ax2, imp_bot, '底部模型', C['bot'])]:
        topn = imp.head(n).sort_values('importance', ascending=True)
        ax.barh(range(n), topn['importance'], color=color, alpha=0.8)
        ax.set_yticks(range(n)); ax.set_yticklabels(topn.index, fontsize=9)
        ax.set_xlabel('Importance'); ax.set_title(f'{label} Top {n}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')

    fig.suptitle(f'V4 特征重要性 Top {n} (63维中)', fontsize=16, fontweight='bold')
    fig.savefig(os.path.join(CHART_DIR, '34_V4特征重要性Top25.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图5: 34_V4特征重要性Top25.png')


def chart6_v3v4_compare():
    """图6: V3 vs V4 指标对比"""
    metrics = {
        'V3': {'top_P': 94.0, 'top_R': 84.6, 'top_F1': 89.1, 'bot_P': 100, 'bot_R': 69.2, 'bot_F1': 81.8,
               'features': 153, 'samples': 3506, 'model': 'RF'},
        'V4': {'top_P': 100, 'top_R': 92.3, 'top_F1': 96.0, 'bot_P': 100, 'bot_R': 92.3, 'bot_F1': 96.0,
               'features': 63, 'samples': 7274, 'model': 'LGBM'},
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor('white')

    for i, (metric, title) in enumerate([('P', '精确率'), ('R', '召回率'), ('F1', 'F1 Score')]):
        ax = axes[i]
        x = np.arange(2); w = 0.3
        top_v = [metrics['V3'][f'top_{metric}'], metrics['V4'][f'top_{metric}']]
        bot_v = [metrics['V3'][f'bot_{metric}'], metrics['V4'][f'bot_{metric}']]
        ax.bar(x - w/2, top_v, w, color=C['top'], alpha=0.8, label='顶部')
        ax.bar(x + w/2, bot_v, w, color=C['bot'], alpha=0.8, label='底部')
        for j in range(2):
            ax.text(x[j]-w/2, top_v[j]+1, f'{top_v[j]:.1f}', ha='center', fontsize=10)
            ax.text(x[j]+w/2, bot_v[j]+1, f'{bot_v[j]:.1f}', ha='center', fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(['V3 (RF)', 'V4 (LGBM)'])
        ax.set_ylim(0, 115); ax.set_ylabel('%')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3, axis='y')

    fig.suptitle('V3 vs V4 模型性能对比', fontsize=16, fontweight='bold')
    fig.savefig(os.path.join(CHART_DIR, '35_V3vsV4对比.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图6: 35_V3vsV4对比.png')


def chart7_calibration():
    """图7: 校准曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')

    for ax, raw, cal, label, color in [
        (ax1, eval_nasdaq['pred_top'], eval_nasdaq['cal_top_prob'], '顶部', C['top']),
        (ax2, eval_nasdaq['pred_bot'], eval_nasdaq['cal_bot_prob'], '底部', C['bot']),
    ]:
        mask = raw > 0
        ax.scatter(raw[mask], cal[mask]*100, alpha=0.15, s=8, color=color)
        # 分箱均值线
        bins = np.linspace(0, raw.max(), 20)
        for j in range(len(bins)-1):
            m = (raw >= bins[j]) & (raw < bins[j+1])
            if m.sum() > 5:
                ax.plot([raw[m].mean()]*2, [0, cal[m].mean()*100], color=C['dark'], alpha=0.3, lw=1)
                ax.plot(raw[m].mean(), cal[m].mean()*100, 'o', color=C['dark'], ms=5)
        ax.set_xlabel('原始评分', fontsize=12)
        ax.set_ylabel('校准概率 %', fontsize=12)
        ax.set_title(f'{label}校准曲线', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_ylim(-2, 105)

    fig.suptitle('V4 概率校准曲线', fontsize=16, fontweight='bold')
    fig.savefig(os.path.join(CHART_DIR, '36_V4校准曲线.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图7: 36_V4校准曲线.png')


def chart8_feature_groups():
    """图8: 特征类别重要性"""
    groups = {
        '估值(PE/CAPE)': ['SP500_PE', 'Shiller_CAPE', 'PE_', 'CAPE_', 'PE_rank', 'CAPE_rank'],
        'VIX体系': ['VIX', 'VVIX', 'VIX_', 'VVIX_', 'VIX9D', 'VIX3M', 'VIX_Volatility'],
        '信用利差': ['HYG', 'LQD'],
        '国债利差': ['Spread'],
        '波动率': ['Volatility', 'ATR_', 'Vol_', 'Std_5d', 'Intraday', 'BB_'],
        '价格/均线': ['Dist_MA', 'MA', 'Close_above', 'Mom_', 'DD_depth', 'Days_since', 'Max_dd', 'Return_'],
        'RSI/MACD': ['RSI', 'MACD'],
        'SKEW': ['SKEW'],
        '统计': ['Return_skew', 'Return_kurtosis', 'Upper_shadow', 'Lower_shadow', 'High_low', 'Gap_'],
    }

    top_d = {str(k): v for k, v in zip(imp_top.index, imp_top['importance'])}
    bot_d = {str(k): v for k, v in zip(imp_bot.index, imp_bot['importance'])}

    g_top, g_bot = {}, {}
    for gname, prefixes in groups.items():
        t = sum(v for k, v in top_d.items() if any(k.startswith(p) or k == p for p in prefixes))
        b = sum(v for k, v in bot_d.items() if any(k.startswith(p) or k == p for p in prefixes))
        g_top[gname] = t; g_bot[gname] = b

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('white')
    for ax, gd, label, color in [(ax1, g_top, '顶部模型', C['top']), (ax2, g_bot, '底部模型', C['bot'])]:
        s = sorted(gd, key=gd.get, reverse=True)
        ax.bar(range(len(s)), [gd[g] for g in s], color=color, alpha=0.8)
        ax.set_xticks(range(len(s))); ax.set_xticklabels(s, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('累计重要性'); ax.set_title(f'{label} — 特征类别', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

    fig.suptitle('V4 特征类别重要性分布 (63维)', fontsize=16, fontweight='bold')
    fig.savefig(os.path.join(CHART_DIR, '37_V4特征类别.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图8: 37_V4特征类别.png')


def chart9_cv_heatmap():
    """图9: 滚动CV命中率热力图"""
    n_folds = len(cv_data)
    top_hit_30 = []; bot_hit_30 = []
    top_hit_50 = []; bot_hit_50 = []
    labels = []

    for fold in cv_data:
        pt = np.array(fold['pred_top']); pb = np.array(fold['pred_bot'])
        yt = np.array(fold['y_top']); yb = np.array(fold['y_bot'])

        for pred, y_true, store in [(pt, yt, top_hit_30), (pb, yb, bot_hit_30)]:
            m = pred >= 30
            store.append(float((y_true[m] >= 30).mean()) if m.sum() > 0 else np.nan)
        for pred, y_true, store in [(pt, yt, top_hit_50), (pb, yb, bot_hit_50)]:
            m = pred >= 50
            store.append(float((y_true[m] >= 30).mean()) if m.sum() > 0 else np.nan)
        labels.append(f"{fold['test_start'][2:7]}")

    fig, ax = plt.subplots(figsize=(20, 5))
    fig.patch.set_facecolor('white')

    x = np.arange(n_folds)
    ax.plot(x, top_hit_30, 'o-', color=C['top'], ms=3, lw=1, alpha=0.7, label='顶部>=30')
    ax.plot(x, bot_hit_30, 'o-', color=C['bot'], ms=3, lw=1, alpha=0.7, label='底部>=30')
    ax.axhline(0.5, color=C['warn'], ls='--', alpha=0.5)
    ax.fill_between(x, 0, top_hit_30, alpha=0.1, color=C['top'])
    ax.fill_between(x, 0, bot_hit_30, alpha=0.1, color=C['bot'])
    ax.set_xlim(0, n_folds-1); ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('命中率')
    ax.set_title(f'V4 滚动窗口 CV 命中率 ({n_folds}个窗口)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    # 每10个窗口标注日期
    for i in range(0, n_folds, max(1, n_folds//12)):
        ax.text(i, -0.08, labels[i], ha='center', fontsize=7, rotation=30)

    fig.savefig(os.path.join(CHART_DIR, '38_V4滚动CV.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图9: 38_V4滚动CV.png')


def chart10_pred_vs_actual():
    """图10: 预测 vs 实际标签散点图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')

    for ax, col_pred, col_label, label, color in [
        (ax1, 'pred_top', 'label_top', '顶部', C['top']),
        (ax2, 'pred_bot', 'label_bot', '底部', C['bot']),
    ]:
        raw = eval_nasdaq[col_pred]
        actual = eval_nasdaq[col_label]
        mask = actual > 0
        ax.scatter(actual[mask], raw[mask], c=color, alpha=0.3, s=10)
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, lw=1)
        ax.set_xlabel('实际标签', fontsize=12)
        ax.set_ylabel('预测评分', fontsize=12)
        ax.set_title(f'{label}模型: 预测 vs 实际 ({mask.sum()} 正样本)', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)

    fig.suptitle('V4 预测 vs 实际标签 (Nasdaq)', fontsize=16, fontweight='bold')
    fig.savefig(os.path.join(CHART_DIR, '39_V4预测vs实际.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  图10: 39_V4预测vs实际.png')


if __name__ == '__main__':
    print("生成 V4 评估图表...")
    chart1_timeseries()
    chart2_recent()
    chart3_distribution()
    chart4_per_event()
    chart5_importance()
    chart6_v3v4_compare()
    chart7_calibration()
    chart8_feature_groups()
    chart9_cv_heatmap()
    chart10_pred_vs_actual()
    print("\n全部 V4 图表生成完成!")
