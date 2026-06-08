"""V4 模型训练模块 — LightGBM + 特征选择 + 滚动CV + 校准。"""
import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings

from .config import (
    FEATURE_COLS, N_FEATURES, MODEL_TYPE, MODEL_PARAMS,
    MODEL_TOP_PATH, MODEL_BOT_PATH, MODEL_META_PATH,
    MODEL_CAL_TOP_PATH, MODEL_CAL_BOT_PATH,
    PREDICTIONS_PATH, FEATURE_SELECTION, CV_CONFIG,
    LABEL_CONFIG, CALIBRATION_CONFIG, data_path,
)
from .data_fetcher import fetch_merged_training_data, fetch_training_data
from .features import compute_features
from .labels import build_labels, compute_sample_weights
from .model_factory import create_model

warnings.filterwarnings('ignore')


def train_and_evaluate():
    """V4 完整训练流水线。"""
    print("=" * 60)
    print(f"ML 模型 V4 训练流水线 ({MODEL_TYPE.upper()}, {LABEL_CONFIG['scheme']})")
    print("=" * 60)

    # ── 1. 加载数据 + 计算特征 ──
    df = fetch_merged_training_data()
    df = compute_features(df)
    print(f"  特征计算完成: {N_FEATURES} 维")

    # ── 2. 构建标签 ──
    print("\n构建标签...")
    dd_csv = data_path('dd_definitions')
    top_labels, bot_labels = build_labels(df, dd_defs_csv=dd_csv)
    df['top_score_label'] = top_labels
    df['bottom_score_label'] = bot_labels

    # ── 3. 准备训练数据 ──
    keep = FEATURE_COLS + ['top_score_label', 'bottom_score_label', 'Close']
    if 'source' in df.columns:
        keep.append('source')
    valid = df[keep].dropna(subset=FEATURE_COLS + ['top_score_label', 'bottom_score_label'])
    print(f"\n有效样本: {len(valid)} 行")

    X_all = valid[FEATURE_COLS].values
    y_top = valid['top_score_label'].values
    y_bot = valid['bottom_score_label'].values
    w_top = compute_sample_weights(y_top)
    w_bot = compute_sample_weights(y_bot)

    # ── 4. Stage 1: 全特征训练（特征选择用） ──
    print(f"\nStage 1: 训练全特征模型 ({len(FEATURE_COLS)} 维)...")
    fs_top = create_model(MODEL_TYPE)
    fs_top.fit(X_all, y_top, sample_weight=w_top)

    fs_bot = create_model(MODEL_TYPE)
    fs_bot.fit(X_all, y_bot, sample_weight=w_bot)

    # ── 5. 特征选择 ──
    selected_features = _select_features(fs_top, fs_bot, FEATURE_COLS)
    print(f"\n特征选择: {len(FEATURE_COLS)} → {len(selected_features)} 维")

    # ── 6. Stage 2: 用选中特征重新训练 ──
    print(f"\nStage 2: 训练选中特征模型 ({len(selected_features)} 维)...")
    X_sel = valid[selected_features].values

    model_top = create_model(MODEL_TYPE)
    model_top.fit(X_sel, y_top, sample_weight=w_top)

    model_bot = create_model(MODEL_TYPE)
    model_bot.fit(X_sel, y_bot, sample_weight=w_bot)

    pred_top = model_top.predict(X_sel)
    pred_bot = model_bot.predict(X_sel)
    print(f"  顶部预测范围: [{pred_top.min():.1f}, {pred_top.max():.1f}]")
    print(f"  底部预测范围: [{pred_bot.min():.1f}, {pred_bot.max():.1f}]")

    # ── 7. 滚动窗口交叉验证 ──
    cv_results = _rolling_cv(X_sel, y_top, y_bot, w_top, w_bot, valid.index)

    # ── 8. 概率校准 ──
    cal_top, cal_bot = None, None
    if CALIBRATION_CONFIG['enabled']:
        cal_top, cal_bot = _train_calibration(cv_results, valid.index, dd_csv)

    # ── 9. 评估 ──
    _evaluate(valid, pred_top, pred_bot, dd_csv)

    # ── 10. 特征重要性 ──
    _print_importance(model_top, model_bot, selected_features)

    # ── 11. 保存 ──
    _save_models(model_top, model_bot, cal_top, cal_bot, valid,
                 pred_top, pred_bot, selected_features)

    return model_top, model_bot


def _select_features(model_top, model_bot, all_features):
    """从两个模型的重要性中选出 top_k 特征的并集。"""
    cfg = FEATURE_SELECTION
    if not cfg['enabled']:
        return all_features

    imp_top = pd.Series(model_top.feature_importances_, index=all_features)
    imp_bot = pd.Series(model_bot.feature_importances_, index=all_features)

    selected = set()
    threshold = cfg['importance_threshold']
    top_k = cfg['top_k']

    for imp in [imp_top, imp_bot]:
        above = imp[imp >= threshold].index.tolist()
        selected.update(above)
        topk = imp.nlargest(top_k).index.tolist()
        selected.update(topk)

    return sorted(selected)


def _rolling_cv(X, y_top, y_bot, w_top, w_bot, dates):
    """前向滚动窗口交叉验证。"""
    cfg = CV_CONFIG
    train_window = cfg['train_window']
    test_window = cfg['test_window']
    step = cfg['step']
    gap = cfg['label_gap']
    n = len(X)

    print(f"\n滚动窗口 CV (train={train_window}, test={test_window}, step={step}, gap={gap})...")
    results = []
    start = 0

    while start + train_window + gap + test_window <= n:
        tr_s = start
        tr_e = start + train_window
        te_s = tr_e + gap
        te_e = min(te_s + test_window, n)

        if te_e <= te_s:
            break

        X_tr, X_te = X[tr_s:tr_e], X[te_s:te_e]
        y_top_tr, y_top_te = y_top[tr_s:tr_e], y_top[te_s:te_e]
        y_bot_tr, y_bot_te = y_bot[tr_s:tr_e], y_bot[te_s:te_e]
        w_top_tr, w_bot_tr = w_top[tr_s:tr_e], w_bot[tr_s:tr_e]

        mt = create_model(MODEL_TYPE)
        mt.fit(X_tr, y_top_tr, sample_weight=w_top_tr)
        pt = mt.predict(X_te)

        mb = create_model(MODEL_TYPE)
        mb.fit(X_tr, y_bot_tr, sample_weight=w_bot_tr)
        pb = mb.predict(X_te)

        dates_tr = dates[tr_s:tr_e]
        dates_te = dates[te_s:te_e]

        fold = {
            'fold': len(results) + 1,
            'train_start': str(dates_tr[0])[:10],
            'train_end': str(dates_tr[-1])[:10],
            'test_start': str(dates_te[0])[:10],
            'test_end': str(dates_te[-1])[:10],
            'train_size': len(X_tr),
            'test_size': len(X_te),
            'pred_top': pt.tolist(),
            'pred_bot': pb.tolist(),
            'y_top': y_top_te.tolist(),
            'y_bot': y_bot_te.tolist(),
            'test_indices': list(range(te_s, te_e)),
        }
        results.append(fold)

        for thresh in [30, 50]:
            for side, pred_f, y_true in [('顶部', pt, y_top_te), ('底部', pb, y_bot_te)]:
                mask = pred_f >= thresh
                if mask.sum() > 0:
                    hit = (y_true[mask] >= 30).mean()
                    print(f"  Window {fold['fold']} {side}(>={thresh}): "
                          f"{mask.sum()}信号, 命中={hit:.1%}")

        start += step

    print(f"  共 {len(results)} 个窗口")
    return results


def _train_calibration(cv_results, all_dates, dd_csv):
    """在 CV out-of-sample 预测上训练 IsotonicRegression 校准。"""
    from sklearn.isotonic import IsotonicRegression

    print("\n训练概率校准模型...")

    # 收集全部 OOS 预测
    oos_top, oos_bot, oos_idx = [], [], []
    for fold in cv_results:
        oos_top.extend(fold['pred_top'])
        oos_bot.extend(fold['pred_bot'])
        oos_idx.extend(fold['test_indices'])

    oos_dates = all_dates[oos_idx]

    if len(oos_top) == 0:
        print("  ⚠ 无 OOS 数据，跳过校准")
        return None, None

    oos_top = np.array(oos_top)
    oos_bot = np.array(oos_bot)
    oos_dates = pd.DatetimeIndex(oos_dates)

    # 构建二值标签: 是否在事件窗口内
    window = CALIBRATION_CONFIG['event_window']
    dd_defs = pd.read_csv(dd_csv, parse_dates=['peak_date', 'trough_date'])

    binary_top = np.zeros(len(oos_dates))
    binary_bot = np.zeros(len(oos_dates))
    for i, d in enumerate(oos_dates):
        for _, ev in dd_defs.iterrows():
            if abs((d - ev['peak_date']).days) <= window:
                binary_top[i] = 1
            if abs((d - ev['trough_date']).days) <= window:
                binary_bot[i] = 1

    print(f"  OOS 样本: {len(oos_top)}, 顶部正样本: {int(binary_top.sum())}, 底部正样本: {int(binary_bot.sum())}")

    cal_top = IsotonicRegression(out_of_bounds='clip')
    cal_top.fit(oos_top, binary_top)

    cal_bot = IsotonicRegression(out_of_bounds='clip')
    cal_bot.fit(oos_bot, binary_bot)

    print(f"  校准模型已训练")
    return cal_top, cal_bot


def _evaluate(valid, pred_top, pred_bot, dd_csv):
    """评估指标。"""
    dd_defs = pd.read_csv(dd_csv, parse_dates=['peak_date', 'trough_date'])

    print("\n" + "=" * 60)
    print("评估指标")
    print("=" * 60)

    # 精确率 / F1
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
            print(f"  {side}>={threshold}: P={precision:.1%}, R={recall:.1%}, F1={f1:.3f}")

    # 逐事件
    print("\n逐事件验证")
    eval_top = pd.Series(pred_top, index=valid.index)
    eval_bot = pd.Series(pred_bot, index=valid.index)
    for _, ev in dd_defs.iterrows():
        peak_date = ev['peak_date']
        trough_date = ev['trough_date']
        dd_pct = abs(ev['drawdown_pct'])

        mask_p = (valid.index <= peak_date) & (valid.index >= peak_date - pd.Timedelta(days=60))
        top_vals = eval_top[mask_p]
        max_top = top_vals.max() if len(top_vals) > 0 else 0
        delta_top = (peak_date - top_vals.idxmax()).days if max_top > 0 else 0

        mask_b = (valid.index <= trough_date) & (valid.index >= trough_date - pd.Timedelta(days=60))
        bot_vals = eval_bot[mask_b]
        max_bot = bot_vals.max() if len(bot_vals) > 0 else 0
        delta_bot = (trough_date - bot_vals.idxmax()).days if max_bot > 0 else 0

        top_label = f"提前{delta_top}天" if delta_top > 0 else f"滞后{abs(delta_top)}天"
        bot_label = f"提前{delta_bot}天" if delta_bot > 0 else f"滞后{abs(delta_bot)}天"

        print(f"  {peak_date.strftime('%Y-%m')}~{trough_date.strftime('%Y-%m')} ({dd_pct:.1f}%): "
              f"顶部={max_top:.0f}({top_label}), 底部={max_bot:.0f}({bot_label})")


def _print_importance(model_top, model_bot, features):
    """打印 Top 20 特征重要性。"""
    print("\n特征重要性 Top 20")
    imp_top = pd.Series(model_top.feature_importances_, index=features).sort_values(ascending=False)
    imp_bot = pd.Series(model_bot.feature_importances_, index=features).sort_values(ascending=False)

    for label, imp in [('顶部模型', imp_top), ('底部模型', imp_bot)]:
        print(f"\n  {label}:")
        for i, (f, v) in enumerate(imp.head(20).items()):
            print(f"    {i+1:2d}. {f:35s} {v:.4f}")


def _save_models(model_top, model_bot, cal_top, cal_bot, valid,
                 pred_top, pred_bot, selected_features):
    """保存模型和元数据。"""
    joblib.dump(model_top, MODEL_TOP_PATH)
    joblib.dump(model_bot, MODEL_BOT_PATH)
    if cal_top is not None:
        joblib.dump(cal_top, MODEL_CAL_TOP_PATH)
        joblib.dump(cal_bot, MODEL_CAL_BOT_PATH)

    imp_top = pd.Series(model_top.feature_importances_, index=selected_features).sort_values(ascending=False)
    imp_bot = pd.Series(model_bot.feature_importances_, index=selected_features).sort_values(ascending=False)

    meta = {
        "model": f"Dual {MODEL_TYPE.upper()} V4 ({len(selected_features)} features)",
        "model_type": MODEL_TYPE,
        "features": selected_features,
        "n_features": len(selected_features),
        "all_features": FEATURE_COLS,
        "label_scheme": LABEL_CONFIG['scheme'],
        "calibration": CALIBRATION_CONFIG['enabled'],
        "feature_importance_top": {k: round(float(v), 6) for k, v in imp_top.head(30).items()},
        "feature_importance_bot": {k: round(float(v), 6) for k, v in imp_bot.head(30).items()},
    }
    with open(MODEL_META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    out = valid[['Close']].copy()
    out['pred_top'] = pred_top
    out['pred_bot'] = pred_bot
    out.to_csv(PREDICTIONS_PATH)

    print(f"\n  模型已保存: {MODEL_TOP_PATH}")
    print(f"  特征数: {len(selected_features)}")
    if cal_top is not None:
        print(f"  校准模型: {MODEL_CAL_TOP_PATH}")
    print(f"  预测结果: {PREDICTIONS_PATH}")
