#!/usr/bin/env python3
"""V6 模型评估结果可视化 — 6张对比图表。"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['Noto Serif CJK SC', 'Noto Serif CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CHART_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(CHART_DIR, exist_ok=True)

# ── 硬编码指标数据 ──
HORIZONS = ['1d', '3d', '5d', '7d', '10d']

# Base模型 OOS指标 (179特征, top_k=80/50)
OOS_DIR = [70.3, 63.2, 62.4, 64.1, 64.9]
OOS_IC = [0.529, 0.370, 0.352, 0.378, 0.408]
OOS_AUC = [0.760, 0.691, 0.680, 0.689, 0.715]
TRAIN_ACC = [75.0, 79.6, 79.3, 77.7, 79.8]
TRAIN_OOS_GAP = [4.7, 16.4, 16.9, 13.6, 14.8]

# Meta层AUC
META_TOP_AUC = 0.787
META_BOT_AUC = 0.707

# Holdout验证（最后20%）
HOLDOUT = {'1d': 73.2, '5d': 68.5, '10d': 72.3}

# 非重叠walk-forward OOS
NON_OVERLAP = {'1d': 71.6, '5d': 67.6, '10d': 68.9}

# 基准线
BASELINES = {'永远看多': 55.7, '1日动量': 49.7, '5日动量': 50.6}

# 版本对比
VERSION_DATA = {
    'V4\nML': {'top_f1': 0.000, 'bot_f1': 0.000, 'dir_acc': 0},
    'V5\n规则': {'top_f1': 0.220, 'bot_f1': 0.295, 'dir_acc': 0},
    'V6\n方向预测': {'top_f1': 0.000, 'bot_f1': 0.291, 'dir_acc': 65.4},
}

# 事件检测（13个事件OOS composite + ML底概率）
EVENTS = [
    ('#1 Euro危机\n-18.7%', None, 'N/A'),
    ('#2 人民币贬值\n-12.0%', None, 'N/A'),
    ('#3 加息恐慌\n-10.9%', None, 'N/A'),
    ('#4 贸易战\n-18.2%', +0.04, 'N/A'),
    ('#5 流动性\n-23.6%', +0.27, 'N/A'),
    ('#6 增长担忧\n-10.2%', -0.05, 'N/A'),
    ('#7 COVID\n-30.1%', +0.12, 'N/A'),
    ('#8 反弹回落\n-11.8%', +0.29, 'N/A'),
    ('#9 通胀\n-10.5%', -0.01, 'N/A'),
    ('#10 暴跌\n-36.4%', +0.09, 0.13),
    ('#11 加息熊市\n-13.1%', +0.21, 0.89),
    ('#12 二次探底\n-24.3%', -0.00, 0.37),
    ('#13 关税冲击\n-13.2%', -0.01, 0.38),
]

# 配色
C = {
    'v4': '#e74c3c', 'v5': '#f39c12', 'v6': '#2ecc71',
    'h': ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#1abc9c'],
    'base': '#95a5a6', 'dark': '#2c3e50', 'price': '#3498db',
    'good': '#27ae60', 'warn': '#f39c12', 'bad': '#e74c3c',
}


def save(fig, n, title):
    path = os.path.join(CHART_DIR, f'{n:02d}_{title}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Chart {n}: {path}')


def chart40_version_comparison():
    """Chart 40: V4/V5/V6 OOS对比。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor('white')

    versions = list(VERSION_DATA.keys())
    colors = [C['v4'], C['v5'], C['v6']]
    x = np.arange(len(versions))

    # 顶部F1
    vals = [VERSION_DATA[v]['top_f1'] for v in versions]
    axes[0].bar(x, vals, color=colors, edgecolor='white', width=0.6)
    axes[0].set_title('顶部检测 F1 (OOS)', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(versions, fontsize=11)
    axes[0].set_ylim(0, 0.35)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.008, f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')
    axes[0].axhline(0, color=C['dark'], lw=0.5)

    # 底部F1
    vals = [VERSION_DATA[v]['bot_f1'] for v in versions]
    axes[1].bar(x, vals, color=colors, edgecolor='white', width=0.6)
    axes[1].set_title('底部检测 F1 (OOS)', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(versions, fontsize=11)
    axes[1].set_ylim(0, 0.35)
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.008, f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')

    # 方向准确率
    vals = [VERSION_DATA[v]['dir_acc'] for v in versions]
    bars = axes[2].bar(x, vals, color=colors, edgecolor='white', width=0.6)
    axes[2].set_title('方向准确率 (OOS)', fontsize=13, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(versions, fontsize=11)
    axes[2].set_ylim(0, 80)
    axes[2].axhline(55.7, color=C['base'], ls='--', lw=1, label='基准(永远看多)')
    axes[2].legend(fontsize=9, loc='upper left')
    for i, v in enumerate(vals):
        if v > 0:
            axes[2].text(i, v + 1.5, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')
        else:
            axes[2].text(i, 2, 'N/A', ha='center', fontsize=11, color=C['base'])
    axes[2].annotate('完全过拟合\nOOS F1=0.000', xy=(0, 0), xytext=(0, 25),
                     fontsize=9, color=C['v4'], ha='center',
                     arrowprops=dict(arrowstyle='->', color=C['v4']))

    fig.suptitle('V4 → V5 → V6 泛化能力对比 (OOS)', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 40, 'V6_版本OOS对比')


def chart41_horizon_metrics():
    """Chart 41: 5个horizon OOS指标面板。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    x = np.arange(len(HORIZONS))

    # 方向准确率
    ax = axes[0, 0]
    bars = ax.bar(x, OOS_DIR, color=C['h'], edgecolor='white', width=0.6)
    ax.axhline(55.7, color=C['base'], ls='--', lw=1, label='基准(永远看多)')
    ax.axhline(50, color=C['dark'], ls=':', lw=0.5, label='随机')
    ax.set_title('方向准确率', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS, fontsize=11)
    ax.set_ylim(40, 80)
    ax.legend(fontsize=8)
    for i, v in enumerate(OOS_DIR):
        ax.text(i, v + 0.8, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

    # IC
    ax = axes[0, 1]
    ax.bar(x, OOS_IC, color=C['h'], edgecolor='white', width=0.6)
    ax.axhline(0.3, color=C['warn'], ls='--', lw=1, label='IC=0.3 阈值')
    ax.axhline(0, color=C['dark'], ls=':', lw=0.5)
    ax.set_title('IC (Spearman)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS, fontsize=11)
    ax.set_ylim(-0.1, 0.7)
    ax.legend(fontsize=8)
    for i, v in enumerate(OOS_IC):
        ax.text(i, v + 0.015, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    # AUC
    ax = axes[1, 0]
    ax.bar(x, OOS_AUC, color=C['h'], edgecolor='white', width=0.6)
    ax.axhline(0.5, color=C['dark'], ls=':', lw=0.5, label='随机(AUC=0.5)')
    ax.axhline(0.6, color=C['warn'], ls='--', lw=1, label='AUC=0.6 阈值')
    ax.set_title('AUC (方向二分类)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS, fontsize=11)
    ax.set_ylim(0.4, 0.85)
    ax.legend(fontsize=8)
    for i, v in enumerate(OOS_AUC):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    # Train/OOS差距
    ax = axes[1, 1]
    gap_colors = [C['good'] if g < 10 else (C['warn'] if g < 15 else C['bad']) for g in TRAIN_OOS_GAP]
    ax.bar(x, TRAIN_OOS_GAP, color=gap_colors, edgecolor='white', width=0.6)
    ax.axhline(10, color=C['good'], ls='--', lw=0.8, alpha=0.6, label='< 10% 优秀')
    ax.axhline(15, color=C['warn'], ls='--', lw=0.8, alpha=0.6, label='10-15% 可接受')
    ax.set_title('Train/OOS 差距 (过拟合程度)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS, fontsize=11)
    ax.set_ylim(0, 25)
    ax.legend(fontsize=8)
    for i, v in enumerate(TRAIN_OOS_GAP):
        label = '✓' if v < 10 else ('!' if v < 15 else '⚠')
        ax.text(i, v + 0.4, f'{v:.1f}% {label}', ha='center', fontsize=10, fontweight='bold')

    fig.suptitle('V6 Base模型 OOS指标 (5个Horizon, 179特征 top_k=80/50)',
                fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 41, 'V6_Horizon指标面板')


def chart42_baselines():
    """Chart 42: V6 vs 基准线。"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')

    x = np.arange(len(HORIZONS))
    width = 0.35

    # CV OOS
    bars1 = ax.bar(x - width/2, OOS_DIR, width, color=C['v6'], edgecolor='white',
                   label='V6 CV OOS', zorder=3)
    # Holdout
    hold_vals = [HOLDOUT.get(h.replace('d', '').strip(), None) for h in HORIZONS]
    # Convert HORIZONS to numeric keys for lookup
    h_keys = [1, 3, 5, 7, 10]
    hold_vals = [HOLDOUT.get(k, None) for k in h_keys]
    hold_x = [i for i, v in enumerate(hold_vals) if v is not None]
    hold_y = [v for v in hold_vals if v is not None]
    ax.bar([x[i] + width/2 for i in hold_x], hold_y, width,
           color='#1a8f4e', edgecolor='white', label='V6 Holdout (后20%)', zorder=3)

    # 基准线
    ax.axhline(55.7, color=C['v5'], ls='--', lw=1.5, label='基准: 永远看多 (55.7%)')
    ax.axhline(50.6, color=C['v4'], ls=':', lw=1, label='基准: 5日动量 (50.6%)')
    ax.axhline(49.7, color=C['base'], ls=':', lw=1, label='基准: 1日动量 (49.7%)')
    ax.axhline(50, color=C['dark'], ls='-', lw=0.3, alpha=0.3)

    # 标注
    for i, v in enumerate(OOS_DIR):
        ax.text(i - width/2, v + 0.8, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    for i, v in zip(hold_x, hold_y):
        ax.text(i + width/2, v + 0.8, f'{v:.1f}%', ha='center', fontsize=9,
                fontweight='bold', color='#1a8f4e')

    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS, fontsize=12)
    ax.set_ylabel('方向准确率 (%)', fontsize=12)
    ax.set_ylim(40, 80)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title('V6方向准确率 vs 基准线', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save(fig, 42, 'V6_vs_基准线')


def chart43_overfitting():
    """Chart 43: 训练vs OOS差距分析。"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')

    x = np.arange(len(HORIZONS))
    width = 0.35

    bars_train = ax.bar(x - width/2, TRAIN_ACC, width, color='#bdc3c7', edgecolor='white',
                        label='训练集准确率', zorder=3)
    bars_oos = ax.bar(x + width/2, OOS_DIR, width, color=C['v6'], edgecolor='white',
                      label='OOS准确率', zorder=3)

    # 差距标注
    for i in range(len(HORIZONS)):
        gap = TRAIN_OOS_GAP[i]
        color = C['good'] if gap < 10 else (C['warn'] if gap < 15 else C['bad'])
        mid_y = (TRAIN_ACC[i] + OOS_DIR[i]) / 2
        ax.annotate(f'差距 {gap:.1f}%',
                    xy=(i + width/2, OOS_DIR[i]),
                    xytext=(i, mid_y),
                    fontsize=9, fontweight='bold', color=color, ha='center',
                    arrowprops=dict(arrowstyle='<->', color=color, lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS, fontsize=12)
    ax.set_ylabel('方向准确率 (%)', fontsize=12)
    ax.set_ylim(40, 90)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_title('训练集 vs OOS — 过拟合差距分析', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 添加底部图例说明
    fig.text(0.5, -0.02,
             '绿色差距 < 10%（优秀） | 黄色 10-15%（可接受） | 红色 > 15%（需关注）',
             ha='center', fontsize=10, style='italic', color=C['dark'])
    fig.tight_layout()
    save(fig, 43, 'V6_过拟合差距')


def chart44_events():
    """Chart 44: 事件检测能力。"""
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor('white')

    labels = [e[0] for e in EVENTS]
    composites = [e[1] for e in EVENTS]
    ml_bots = [e[2] for e in EVENTS]

    x = np.arange(len(EVENTS))
    width = 0.35

    # OOS composite
    comp_vals = [v if v is not None else 0 for v in composites]
    comp_mask = [v is not None for v in composites]
    colors_comp = [C['v6'] if v is not None else '#ecf0f1' for v in composites]
    bars1 = ax.bar(x - width/2, comp_vals, width, color=colors_comp, edgecolor='white',
                   label='OOS Composite评分', zorder=3)

    # ML bottom probability
    ml_vals = []
    colors_ml = []
    for v in ml_bots:
        if isinstance(v, (int, float)):
            ml_vals.append(v)
            colors_ml.append(C['v4'] if v > 0.5 else C['v5'])
        else:
            ml_vals.append(0)
            colors_ml.append('#ecf0f1')
    bars2 = ax.bar(x + width/2, ml_vals, width, color=colors_ml, edgecolor='white',
                   label='ML底部概率', zorder=3)

    # N/A标注
    for i, (c, m) in enumerate(zip(composites, ml_bots)):
        if c is None:
            ax.text(i - width/2, 0.02, 'N/A\n(不在OOS区间)', ha='center',
                    fontsize=7, color=C['base'], rotation=45)
        if isinstance(m, str):
            ax.text(i + width/2, 0.02, 'N/A', ha='center', fontsize=7,
                    color=C['base'], rotation=45)
        elif isinstance(m, (int, float)) and m > 0.5:
            ax.text(i + width/2, m + 0.02, f'{m:.0%}', ha='center',
                    fontsize=9, fontweight='bold', color=C['v4'])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
    ax.axhline(0, color=C['dark'], lw=0.5)
    ax.set_ylabel('评分 / 概率', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_title('V6 事件检测能力 — 13个历史回撤事件 OOS分析', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    fig.text(0.5, -0.05,
             '绿色=Composite正值(看涨) | 橙/红色=ML底部概率高(底部信号) | 灰色=不在OOS区间',
             ha='center', fontsize=9, style='italic', color=C['dark'])
    fig.tight_layout()
    save(fig, 44, 'V6_事件检测')


def chart45_validation():
    """Chart 45: 三重验证对比。"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')

    x = np.arange(len(HORIZONS))
    width = 0.25

    # CV OOS
    ax.bar(x - width, OOS_DIR, width, color=C['v6'], edgecolor='white',
           label='CV OOS (非重叠)', zorder=3)

    # Holdout
    h_keys = [1, 3, 5, 7, 10]
    hold_vals = [HOLDOUT.get(k, 0) for k in h_keys]
    hold_mask = [k in HOLDOUT for k in h_keys]
    hold_display = [v if m else 0 for v, m in zip(hold_vals, hold_mask)]
    colors_hold = ['#1a8f4e' if m else '#ecf0f1' for m in hold_mask]
    ax.bar(x, hold_display, width, color=colors_hold, edgecolor='white',
           label='Holdout (后20%)', zorder=3)

    # 非重叠OOS
    no_vals = [NON_OVERLAP.get(k, 0) for k in h_keys]
    no_mask = [k in NON_OVERLAP for k in h_keys]
    no_display = [v if m else 0 for v, m in zip(no_vals, no_mask)]
    colors_no = ['#2980b9' if m else '#ecf0f1' for m in no_mask]
    ax.bar(x + width, no_display, width, color=colors_no, edgecolor='white',
           label='非重叠OOS (5折)', zorder=3)

    # 基准线
    ax.axhline(55.7, color=C['v5'], ls='--', lw=1.5, label='永远看多 (55.7%)')
    ax.axhline(50, color=C['dark'], ls='-', lw=0.3, alpha=0.3)

    # 标注
    for i, v in enumerate(OOS_DIR):
        ax.text(i - width, v + 0.5, f'{v:.1f}', ha='center', fontsize=8)
    for i, (v, m) in enumerate(zip(hold_display, hold_mask)):
        if m:
            ax.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=8, color='#1a8f4e')
    for i, (v, m) in enumerate(zip(no_display, no_mask)):
        if m:
            ax.text(i + width, v + 0.5, f'{v:.1f}', ha='center', fontsize=8, color='#2980b9')

    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS, fontsize=12)
    ax.set_ylabel('方向准确率 (%)', fontsize=12)
    ax.set_ylim(40, 80)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title('V6 三重泛化验证 — CV / Holdout / 非重叠OOS', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save(fig, 45, 'V6_三重验证')


if __name__ == '__main__':
    print('生成V6可视化图表:')
    chart40_version_comparison()
    chart41_horizon_metrics()
    chart42_baselines()
    chart43_overfitting()
    chart44_events()
    chart45_validation()
    print('完成!')
