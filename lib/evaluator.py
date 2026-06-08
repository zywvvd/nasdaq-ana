"""V4 模型评估模块 — 适配选中特征 + 滚动CV + 校准。"""
import pandas as pd
import numpy as np
import joblib
import json

from .config import (
    FEATURE_COLS, MODEL_TOP_PATH, MODEL_BOT_PATH, MODEL_META_PATH,
    MODEL_CAL_TOP_PATH, MODEL_CAL_BOT_PATH, MODEL_TYPE,
    CV_CONFIG, LABEL_CONFIG, CALIBRATION_CONFIG, data_path,
)
from .data_fetcher import fetch_merged_training_data
from .features import compute_features
from .labels import build_labels, compute_sample_weights
from .model_factory import create_model


def evaluate(model_top=None, model_bot=None):
    """加载模型并运行全面评估。"""
    meta = json.load(open(MODEL_META_PATH, encoding='utf-8'))
    selected_features = meta.get('features', FEATURE_COLS)

    if model_top is None:
        model_top = joblib.load(MODEL_TOP_PATH)
    if model_bot is None:
        model_bot = joblib.load(MODEL_BOT_PATH)

    print("=" * 60)
    print(f"模型评估 — {meta.get('model', 'Unknown')}")
    print(f"特征维度: {len(selected_features)}, 标签方案: {meta.get('label_scheme', '?')}")
    print("=" * 60)

    # 准备数据
    df = fetch_merged_training_data()
    df = compute_features(df)

    dd_csv = data_path('dd_definitions')
    top_labels, bot_labels = build_labels(df, dd_defs_csv=dd_csv)
    df['top_score_label'] = top_labels
    df['bottom_score_label'] = bot_labels

    keep = selected_features + ['top_score_label', 'bottom_score_label', 'Close']
    valid = df[keep].dropna(subset=selected_features + ['top_score_label', 'bottom_score_label'])
    print(f"有效样本: {len(valid)} 行\n")

    X = valid[selected_features].values
    pred_top = model_top.predict(X)
    pred_bot = model_bot.predict(X)

    # 预测分布
    print("─" * 40)
    print("预测分布")
    for side, pred in [('顶部', pred_top), ('底部', pred_bot)]:
        print(f"  {side}: [{pred.min():.1f}, {pred.max():.1f}] 均值={pred.mean():.1f} P90={np.percentile(pred, 90):.1f}")

    # 精确率 / 召回率 / F1
    dd_defs = pd.read_csv(dd_csv, parse_dates=['peak_date', 'trough_date'])
    print(f"\n{'─' * 40}")
    print("精确率 / 召回率 / F1 (30天窗口)")
    print("─" * 40)
    for threshold in [50, 60, 70]:
        for side, pred, date_col in [('顶部', pred_top, 'peak_date'), ('底部', pred_bot, 'trough_date')]:
            sig_mask = pred >= threshold
            sig_dates = valid.index[sig_mask]
            n_sig = int(sig_mask.sum())
            if n_sig == 0:
                continue
            event_dates = dd_defs[date_col].tolist()
            precision = sum(1 for d in sig_dates if any(abs((d - e).days) <= 30 for e in event_dates)) / n_sig
            recall = sum(1 for e in event_dates if any(abs((d - e).days) <= 30 for d in sig_dates)) / len(event_dates)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            print(f"  {side}>={threshold}: P={precision:.1%} R={recall:.1%} F1={f1:.3f}")

    # 滚动窗口 CV
    print(f"\n{'─' * 40}")
    print("滚动窗口 CV")
    print("─" * 40)
    y_top = df.loc[valid.index, 'top_score_label'].values
    y_bot = df.loc[valid.index, 'bottom_score_label'].values
    w_top = compute_sample_weights(y_top)
    w_bot = compute_sample_weights(y_bot)
    _rolling_cv_display(X, y_top, y_bot, w_top, w_bot, valid.index)

    # 特征重要性
    print(f"\n{'─' * 40}")
    print("特征重要性 Top 30")
    print("─" * 40)
    imp_top = pd.Series(model_top.feature_importances_, index=selected_features).sort_values(ascending=False)
    imp_bot = pd.Series(model_bot.feature_importances_, index=selected_features).sort_values(ascending=False)
    for label, imp in [('顶部模型', imp_top), ('底部模型', imp_bot)]:
        print(f"\n  {label}:")
        for i, (f, v) in enumerate(imp.head(30).items()):
            print(f"    {i+1:2d}. {f:35s} {v:.4f}")

    print(f"\n{'=' * 60}")
    print("评估完成!")


def _rolling_cv_display(X, y_top, y_bot, w_top, w_bot, dates):
    """滚动窗口 CV 显示。"""
    cfg = CV_CONFIG
    train_window = cfg['train_window']
    test_window = cfg['test_window']
    step = cfg['step']
    gap = cfg['label_gap']
    n = len(X)
    start = 0

    while start + train_window + gap + test_window <= n:
        tr_s = start
        tr_e = start + train_window
        te_s = tr_e + gap
        te_e = min(te_s + test_window, n)
        if te_e <= te_s:
            break

        mt = create_model(MODEL_TYPE)
        mt.fit(X[tr_s:tr_e], y_top[tr_s:tr_e], sample_weight=w_top[tr_s:tr_e])
        pt = mt.predict(X[te_s:te_e])

        mb = create_model(MODEL_TYPE)
        mb.fit(X[tr_s:tr_e], y_bot[tr_s:tr_e], sample_weight=w_bot[tr_s:tr_e])
        pb = mb.predict(X[te_s:te_e])

        dates_tr = dates[tr_s:tr_e]
        dates_te = dates[te_s:te_e]
        fold = start // step + 1

        for thresh in [30, 50]:
            for side, pred_f, y_true in [('顶部', pt, y_top[te_s:te_e]), ('底部', pb, y_bot[te_s:te_e])]:
                mask = pred_f >= thresh
                if mask.sum() > 0:
                    hit = (y_true[mask] >= 30).mean()
                    print(f"  Window {fold} {side}(>={thresh}): {mask.sum()}信号, 命中={hit:.1%}")

        start += step
