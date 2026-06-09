"""V6 训练器 — 6个horizon模型 + walk-forward CV + 元层聚合。"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from scipy.stats import spearmanr

from .config import FEATURE_COLS, data_path
from .data_fetcher import fetch_training_data
from .features import compute_features
from .model_factory import create_model
from .v6_config import (
    HORIZONS, BASE_MODEL_PARAMS, CV_CONFIG, FEATURE_SELECTION,
    META_CONFIG, MODEL_DIR,
)
from .v6_labels import build_all_labels


def train_v6():
    """V6 完整训练流程。"""
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 70)
    print("V6 多尺度方向预测系统 — 训练")
    print("=" * 70)

    # 1. 加载数据
    df = fetch_training_data()
    df = compute_features(df)
    print(f"数据: {len(df)} 行, {len(FEATURE_COLS)} 维特征\n")

    # 2. 构建标签
    print("构建方向标签:")
    labels = build_all_labels(df)
    print()

    # 3. 逐 horizon 训练
    all_oos = {}  # {horizon: DataFrame with pred, actual, date}
    all_meta = {}  # 模型元数据

    for horizon in HORIZONS:
        print(f"\n{'=' * 70}")
        print(f"Horizon = {horizon}d")
        print("=" * 70)
        label_col = f'label_{horizon}d'
        df_h = df.copy()
        df_h[label_col] = labels[horizon]

        # 准备数据
        valid = df_h[FEATURE_COLS + [label_col, 'Close']].dropna(subset=FEATURE_COLS + [label_col])
        print(f"  有效样本: {len(valid)}")

        X = valid[FEATURE_COLS].values
        y = valid[label_col].values
        dates = valid.index

        # 阶段1: 特征选择
        selected_idx, selected_names = _select_features(X, y, FEATURE_COLS)

        # 阶段2: 用选中特征训练
        X_sel = X[:, selected_idx]
        model = create_model('lgbm')
        model.set_params(**BASE_MODEL_PARAMS)
        model.fit(X_sel, y)

        # Walk-forward CV
        oos_df = _rolling_cv(X_sel, y, dates, horizon, selected_idx)

        # 评估
        _evaluate_horizon(oos_df, horizon)

        # 保存
        model_path = os.path.join(MODEL_DIR, f'model_{horizon}d.pkl')
        joblib.dump(model, model_path)

        meta = {
            'horizon': horizon,
            'features': selected_names,
            'n_features': len(selected_names),
            'params': BASE_MODEL_PARAMS,
        }
        all_meta[f'{horizon}d'] = meta
        all_oos[horizon] = oos_df

    # 4. 元层聚合
    print(f"\n{'=' * 70}")
    print("元层聚合")
    print("=" * 70)
    _meta_aggregation(all_oos)

    # 5. 保存元数据
    meta_path = os.path.join(MODEL_DIR, 'v6_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)
    print(f"\n元数据已保存: {meta_path}")


def _select_features(X, y, feature_names):
    """阶段1 特征选择：训练临时模型取 top_k。"""
    params = dict(BASE_MODEL_PARAMS, n_estimators=100, verbose=-1)
    tmp = create_model('lgbm')
    tmp.set_params(**params)
    tmp.fit(X, y)

    imp = pd.Series(tmp.feature_importances_, index=feature_names)
    top_k = FEATURE_SELECTION.get('top_k', 40)
    top = imp.sort_values(ascending=False).head(top_k)
    selected = top.index.tolist()
    idx = [list(feature_names).index(f) for f in selected]
    print(f"  特征选择: {len(feature_names)} → {len(selected)}")
    return idx, selected


def _rolling_cv(X, y, dates, horizon, selected_idx):
    """Walk-forward CV，返回 OOS DataFrame。"""
    cfg = CV_CONFIG
    train_w = cfg['train_window']
    test_w = cfg['test_window']
    step = cfg['step']
    n_folds = cfg['n_folds']
    smooth_w = max(5, horizon // 3)
    gap = horizon + smooth_w

    n = len(X)
    max_start = n - train_w - gap - test_w
    if max_start <= 0:
        return pd.DataFrame()

    starts = list(range(0, max_start + 1, step))[:n_folds]

    all_pred, all_actual, all_dates = [], [], []

    for start in starts:
        tr_e = start + train_w
        te_s = tr_e + gap
        te_e = min(te_s + test_w, n)
        if te_e <= te_s:
            continue

        X_tr, y_tr = X[start:tr_e], y[start:tr_e]
        X_te, y_te = X[te_s:te_e], y[te_s:te_e]
        dates_te = dates[te_s:te_e]

        model = create_model('lgbm')
        model.set_params(**BASE_MODEL_PARAMS)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        all_pred.extend(pred)
        all_actual.extend(y_te)
        all_dates.extend(dates_te.tolist())

    oos = pd.DataFrame({
        'date': all_dates,
        'pred': all_pred,
        'actual': all_actual,
    })
    # Deduplicate: keep last fold's prediction for each date
    oos = oos.drop_duplicates(subset='date', keep='last').reset_index(drop=True)
    return oos


def _evaluate_horizon(oos_df, horizon):
    """评估单个 horizon 的 OOS 表现。"""
    if oos_df.empty:
        print(f"  {horizon}d: 无OOS数据")
        return

    pred = oos_df['pred'].values
    actual = oos_df['actual'].values

    # 方向准确率
    dir_correct = (np.sign(pred) == np.sign(actual)).mean()

    # 做多准确率（预测>0时实际>0的比例）
    long_mask = pred > 0
    long_acc = (actual[long_mask] > 0).mean() if long_mask.sum() > 0 else 0

    # 做空准确率
    short_mask = pred < 0
    short_acc = (actual[short_mask] < 0).mean() if short_mask.sum() > 0 else 0

    # IC (rank correlation)
    ic, _ = spearmanr(pred, actual)

    # MAE
    mae = mean_absolute_error(actual, pred)

    # AUC (binary direction) — 需要正负两类都存在
    binary = (actual > 0).astype(int)
    if binary.sum() > 10 and (1 - binary).sum() > 10:
        auc = roc_auc_score(binary, pred)
    else:
        auc = 0

    print(f"  OOS: 方向准确率={dir_correct:.1%} | "
          f"做多准确={long_acc:.1%}(n={long_mask.sum()}) | "
          f"做空准确={short_acc:.1%}(n={short_mask.sum()})")
    print(f"       IC={ic:.4f} | AUC={auc:.3f} | MAE={mae:.3f}")
    print(f"       pred: [{pred.min():.3f}, {pred.max():.3f}] mean={pred.mean():.3f}")


def _meta_aggregation(all_oos):
    """元层聚合：合并6个horizon的OOS预测，生成综合评分。"""
    # 对齐所有 horizon 的日期
    merged = None
    for h, oos in all_oos.items():
        if oos.empty:
            continue
        df_h = oos.set_index('date')[['pred', 'actual']].rename(
            columns={'pred': f'pred_{h}d', 'actual': f'actual_{h}d'})
        if merged is None:
            merged = df_h
        else:
            merged = merged.join(df_h, how='inner')

    if merged is None or merged.empty:
        print("  元层: 无可用OOS数据")
        return

    print(f"  对齐样本: {len(merged)}")

    # 计算元特征
    pred_cols = [f'pred_{h}d' for h in HORIZONS if f'pred_{h}d' in merged.columns]
    actual_cols = [f'actual_{h}d' for h in HORIZONS if f'actual_{h}d' in merged.columns]

    preds = merged[pred_cols].values
    actuals = merged[actual_cols].values

    short_h = META_CONFIG['short_horizons']
    long_h = META_CONFIG['long_horizons']
    weights = META_CONFIG['weights']

    # 短期/长期均值
    short_cols = [f'pred_{h}d' for h in short_h if f'pred_{h}d' in merged.columns]
    long_cols = [f'pred_{h}d' for h in long_h if f'pred_{h}d' in merged.columns]

    merged['short_avg'] = merged[short_cols].mean(axis=1)
    merged['long_avg'] = merged[long_cols].mean(axis=1)
    merged['divergence'] = merged['short_avg'] - merged['long_avg']

    # 加权综合评分
    w = np.array([weights.get(h, 1.0) for h in HORIZONS if f'pred_{h}d' in merged.columns])
    pred_matrix = merged[pred_cols].values
    merged['composite'] = (pred_matrix * w).sum(axis=1) / w.sum()

    # 方向一致度
    signs = np.sign(preds)
    merged['agreement'] = (signs == signs[:, [0]]).all(axis=1).astype(float)
    # 更精细：同号比例
    merged['agreement'] = np.maximum(
        (signs > 0).sum(axis=1),
        (signs < 0).sum(axis=1)
    ) / len(pred_cols)

    # 极值
    merged['extreme'] = np.abs(preds).max(axis=1)

    # ── 综合评估 ──
    print(f"\n  {'─' * 60}")
    print("  元层综合评估")
    print(f"  {'─' * 60}")

    # 方向准确率
    comp_sign = np.sign(merged['composite'].values)
    actual_30d = merged.get('actual_30d', merged[actual_cols[-1]]).values
    dir_acc = (comp_sign == np.sign(actual_30d)).mean()
    print(f"  综合方向准确率: {dir_acc:.1%}")

    # 模拟交易：composite > 0 做多，< 0 做空
    ret_cols = [c for c in actual_cols]
    avg_actual = merged[ret_cols].mean(axis=1).values
    strategy_ret = comp_sign * avg_actual
    sharpe = strategy_ret.mean() / (strategy_ret.std() + 1e-10) * np.sqrt(252)
    print(f"  模拟Sharpe: {sharpe:.3f}")
    print(f"  策略日均收益: {strategy_ret.mean():.4f}")

    # 元层顶/底检测
    dd_csv = data_path('dd_definitions')
    dd_defs = pd.read_csv(dd_csv, parse_dates=['peak_date', 'trough_date'])
    event_peaks = dd_defs['peak_date'].tolist()
    event_troughs = dd_defs['trough_date'].tolist()

    # 顶部检测（composite > threshold）
    print(f"\n  顶部/底部检测（OOS）:")
    for thresh in [0.2, 0.3, 0.5]:
        # 顶部：composite 高 → 后续下跌
        top_sig = merged['composite'].values > thresh
        n_top = top_sig.sum()
        if n_top > 0:
            sig_dates = merged.index[top_sig]
            top_prec = sum(1 for d in sig_dates
                         if any(abs((d - e).days) <= 30 for e in event_peaks)) / n_top
            relevant = [e for e in event_peaks if merged.index.min() <= e <= merged.index.max()]
            top_rec = sum(1 for e in relevant
                        if any(abs((d - e).days) <= 30 for d in sig_dates)) / len(relevant) if relevant else 0
        else:
            top_prec, top_rec = 0, 0
        top_f1 = 2 * top_prec * top_rec / (top_prec + top_rec + 1e-10)

        # 底部：composite < -threshold
        bot_sig = merged['composite'].values < -thresh
        n_bot = bot_sig.sum()
        if n_bot > 0:
            sig_dates_b = merged.index[bot_sig]
            bot_prec = sum(1 for d in sig_dates_b
                         if any(abs((d - e).days) <= 30 for e in event_troughs)) / n_bot
            relevant_b = [e for e in event_troughs if merged.index.min() <= e <= merged.index.max()]
            bot_rec = sum(1 for e in relevant_b
                        if any(abs((d - e).days) <= 30 for d in sig_dates_b)) / len(relevant_b) if relevant_b else 0
        else:
            bot_prec, bot_rec = 0, 0
        bot_f1 = 2 * bot_prec * bot_rec / (bot_prec + bot_rec + 1e-10)

        print(f"    thresh={thresh:.1f}: "
              f"顶 P={top_prec:.1%} R={top_rec:.1%} F1={top_f1:.3f} (sig={n_top}) | "
              f"底 P={bot_prec:.1%} R={bot_rec:.1%} F1={bot_f1:.3f} (sig={n_bot})")

    # 逐事件
    print(f"\n  逐事件OOS检测:")
    for i, ev in dd_defs.iterrows():
        peak_date = ev['peak_date']
        trough_date = ev['trough_date']
        dd_pct = abs(ev['drawdown_pct'])

        peak_in = merged.index.min() <= peak_date <= merged.index.max()
        trough_in = merged.index.min() <= trough_date <= merged.index.max()

        top_comp = merged.loc[:peak_date, 'composite'].tail(20).max() if peak_in else None
        bot_comp = merged.loc[:trough_date, 'composite'].tail(20).min() if trough_in else None

        top_s = f"{top_comp:+.2f}" if top_comp is not None else "N/A"
        bot_s = f"{bot_comp:+.2f}" if bot_comp is not None else "N/A"
        print(f"    #{i+1} ({dd_pct:.1f}%): OOS顶composite={top_s}, OOS底composite={bot_s}")

    # 保存元层权重
    meta_path = os.path.join(MODEL_DIR, 'meta_weights.json')
    with open(meta_path, 'w') as f:
        json.dump(META_CONFIG, f, indent=2)
