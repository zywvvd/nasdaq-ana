"""V6 训练器 — 6个horizon模型 + walk-forward CV + 元层ML聚合。"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from .config import FEATURE_COLS, data_path
from .data_fetcher import fetch_training_data
from .features import compute_features
from .model_factory import create_model
from .v6_config import (
    HORIZONS, BASE_MODEL_PARAMS, HORIZON_PARAMS, CV_CONFIG, FEATURE_SELECTION,
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
    all_oos = {}
    all_meta = {}

    for horizon in HORIZONS:
        print(f"\n{'=' * 70}")
        print(f"Horizon = {horizon}d")
        print("=" * 70)
        label_col = f'label_{horizon}d'
        df_h = df.copy()
        df_h[label_col] = labels[horizon]

        valid = df_h[FEATURE_COLS + [label_col, 'Close']].dropna(subset=FEATURE_COLS + [label_col])
        print(f"  有效样本: {len(valid)}")

        X = valid[FEATURE_COLS].values
        y = valid[label_col].values
        dates = valid.index

        # 阶段1: 特征选择
        h_params = HORIZON_PARAMS.get(horizon, BASE_MODEL_PARAMS)
        selected_idx, selected_names = _select_features(X, y, FEATURE_COLS, h_params)

        # 阶段2: 用选中特征训练
        X_sel = X[:, selected_idx]
        model = create_model('lgbm')
        model.set_params(**h_params)
        model.fit(X_sel, y)

        # Walk-forward CV
        oos_df = _rolling_cv(X_sel, y, dates, horizon, selected_idx, h_params)

        # 评估
        _evaluate_horizon(oos_df, horizon)

        # 保存
        model_path = os.path.join(MODEL_DIR, f'model_{horizon}d.pkl')
        joblib.dump(model, model_path)

        meta = {
            'horizon': horizon,
            'features': selected_names,
            'n_features': len(selected_names),
            'params': h_params,
        }
        all_meta[f'{horizon}d'] = meta
        all_oos[horizon] = oos_df

    # 4. 元层聚合
    print(f"\n{'=' * 70}")
    print("元层聚合")
    print("=" * 70)
    _meta_aggregation(all_oos, df)

    # 5. 保存元数据
    meta_path = os.path.join(MODEL_DIR, 'v6_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)
    print(f"\n元数据已保存: {meta_path}")


def _select_features(X, y, feature_names, base_params=None):
    """阶段1 特征选择：训练临时模型取 top_k。"""
    if base_params is None:
        base_params = BASE_MODEL_PARAMS
    params = dict(base_params, n_estimators=100, verbose=-1)
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


def _rolling_cv(X, y, dates, horizon, selected_idx, params=None):
    """Walk-forward CV，返回 OOS DataFrame。"""
    cfg = CV_CONFIG
    train_w = cfg['train_window']
    test_w = cfg['test_window']
    step = cfg['step']
    n_folds = cfg['n_folds']
    smooth_w = max(5, horizon // 3)
    gap = horizon + smooth_w

    if params is None:
        params = BASE_MODEL_PARAMS

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
        model.set_params(**params)
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
    oos = oos.drop_duplicates(subset='date', keep='last').reset_index(drop=True)
    return oos


def _evaluate_horizon(oos_df, horizon):
    """评估单个 horizon 的 OOS 表现。"""
    if oos_df.empty:
        print(f"  {horizon}d: 无OOS数据")
        return

    pred = oos_df['pred'].values
    actual = oos_df['actual'].values

    dir_correct = (np.sign(pred) == np.sign(actual)).mean()

    long_mask = pred > 0
    long_acc = (actual[long_mask] > 0).mean() if long_mask.sum() > 0 else 0

    short_mask = pred < 0
    short_acc = (actual[short_mask] < 0).mean() if short_mask.sum() > 0 else 0

    ic, _ = spearmanr(pred, actual)
    mae = mean_absolute_error(actual, pred)

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


def _build_meta_features(merged, pred_cols):
    """从6个horizon预测计算元特征。"""
    short_h = META_CONFIG['short_horizons']
    long_h = META_CONFIG['long_horizons']
    weights = META_CONFIG['weights']

    preds = merged[pred_cols].values
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
    merged['agreement'] = np.maximum(
        (signs > 0).sum(axis=1),
        (signs < 0).sum(axis=1)
    ) / len(pred_cols)

    # 极值
    merged['extreme'] = np.abs(preds).max(axis=1)

    # 预测分散度
    merged['pred_std'] = merged[pred_cols].std(axis=1)

    # 全同号信号
    merged['all_positive'] = ((signs > 0).sum(axis=1) == len(pred_cols)).astype(float)
    merged['all_negative'] = ((signs < 0).sum(axis=1) == len(pred_cols)).astype(float)

    return merged


def _meta_aggregation(all_oos, full_df):
    """元层聚合：ML学习器 + 规则聚合双通道。"""
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

    pred_cols = [f'pred_{h}d' for h in HORIZONS if f'pred_{h}d' in merged.columns]
    actual_cols = [f'actual_{h}d' for h in HORIZONS if f'actual_{h}d' in merged.columns]

    # 构建元特征
    merged = _build_meta_features(merged, pred_cols)

    # ── 加入市场状态特征 ──
    state_cols = ['VIX', 'RSI', 'DD_depth', 'Dist_MA200', 'Dist_MA60',
                  'Volatility_20d', 'SKEW', 'Spread_10Y_2Y',
                  'HYG_TLT_Ratio', 'Mom_20d', 'VIX_rank_60d', 'RSI_rank_60d']
    available_state = [c for c in state_cols if c in full_df.columns]
    if available_state:
        state_df = full_df[available_state].copy()
        merged = merged.join(state_df, how='left')
        # Fill any remaining NaN with forward fill
        for c in available_state:
            if merged[c].isna().any():
                merged[c] = merged[c].ffill().bfill()

    # ── 构建 meta 标签 ──
    actual_30d = merged['actual_30d'] if 'actual_30d' in merged.columns else merged[actual_cols[-1]]
    merged['actual_30d'] = actual_30d

    q85 = actual_30d.quantile(0.85)
    q15 = actual_30d.quantile(0.15)
    merged['is_top'] = (actual_30d > q85).astype(int)
    merged['is_bottom'] = (actual_30d < q15).astype(int)
    print(f"  顶标签(n={merged['is_top'].sum()}), 底标签(n={merged['is_bottom'].sum()})")

    # ── ML Meta 学习器 ──
    meta_feat_cols = pred_cols + [
        'short_avg', 'long_avg', 'divergence', 'extreme',
        'agreement', 'pred_std', 'all_positive', 'all_negative',
    ] + available_state

    # 去除 NaN
    merged_clean = merged[meta_feat_cols + ['is_top', 'is_bottom']].dropna()
    if len(merged_clean) < 100:
        print("  ML Meta: 样本不足，跳过")
        merged_clean = merged

    X_meta = merged_clean[meta_feat_cols].values
    y_top = merged_clean['is_top'].values
    y_bot = merged_clean['is_bottom'].values

    # Walk-forward split
    n_meta = len(X_meta)
    meta_split = int(n_meta * 0.7)

    X_tr, X_te = X_meta[:meta_split], X_meta[meta_split:]
    y_top_tr, y_top_te = y_top[:meta_split], y_top[meta_split:]
    y_bot_tr, y_bot_te = y_bot[:meta_split], y_bot[meta_split:]
    dates_clean = merged_clean.index
    dates_te = dates_clean[meta_split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # LGBMClassifier for meta
    from sklearn.ensemble import GradientBoostingClassifier
    top_clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=50, random_state=42,
    )
    top_clf.fit(X_tr_s, y_top_tr)
    top_prob = top_clf.predict_proba(X_te_s)[:, 1] if len(top_clf.classes_) == 2 else np.zeros(len(X_te_s))

    bot_clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=50, random_state=42,
    )
    bot_clf.fit(X_tr_s, y_bot_tr)
    bot_prob = bot_clf.predict_proba(X_te_s)[:, 1] if len(bot_clf.classes_) == 2 else np.zeros(len(X_te_s))

    # ML AUC on test set
    if y_top_te.sum() > 5 and (1 - y_top_te).sum() > 5:
        top_auc = roc_auc_score(y_top_te, top_prob)
    else:
        top_auc = 0
    if y_bot_te.sum() > 5 and (1 - y_bot_te).sum() > 5:
        bot_auc = roc_auc_score(y_bot_te, bot_prob)
    else:
        bot_auc = 0
    print(f"  ML Meta OOS AUC: 顶={top_auc:.3f}, 底={bot_auc:.3f}")

    # 保存 meta 模型
    joblib.dump(top_clf, os.path.join(MODEL_DIR, 'meta_top_clf.pkl'))
    joblib.dump(bot_clf, os.path.join(MODEL_DIR, 'meta_bot_clf.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'meta_scaler.pkl'))
    joblib.dump(meta_feat_cols, os.path.join(MODEL_DIR, 'meta_features.pkl'))

    # ── 综合评估 ──
    print(f"\n  {'─' * 60}")
    print("  元层综合评估")
    print(f"  {'─' * 60}")

    comp_sign = np.sign(merged['composite'].values)
    dir_acc = (comp_sign == np.sign(actual_30d.values)).mean()
    print(f"  综合方向准确率: {dir_acc:.1%}")

    avg_actual = merged[actual_cols].mean(axis=1).values
    strategy_ret = comp_sign * avg_actual
    sharpe = strategy_ret.mean() / (strategy_ret.std() + 1e-10) * np.sqrt(252)
    print(f"  模拟Sharpe: {sharpe:.3f}")
    print(f"  策略日均收益: {strategy_ret.mean():.4f}")

    # ── 事件检测 ──
    dd_csv = data_path('dd_definitions')
    dd_defs = pd.read_csv(dd_csv, parse_dates=['peak_date', 'trough_date'])
    event_peaks = dd_defs['peak_date'].tolist()
    event_troughs = dd_defs['trough_date'].tolist()

    # ML 检测（meta test 区间）
    merged_te = merged_clean.iloc[meta_split:]
    print(f"\n  ML Meta 顶/底检测（OOS, 后30%数据）:")
    for thresh in [0.3, 0.4, 0.5]:
        top_sig_ml = top_prob > thresh
        bot_sig_ml = bot_prob > thresh

        ml_top_p, ml_top_r, ml_top_f1 = _eval_event_detection(
            dates_te[top_sig_ml], event_peaks, merged_te.index)
        ml_bot_p, ml_bot_r, ml_bot_f1 = _eval_event_detection(
            dates_te[bot_sig_ml], event_troughs, merged_te.index)

        print(f"    ML thresh={thresh:.1f}: "
              f"顶 P={ml_top_p:.1%} R={ml_top_r:.1%} F1={ml_top_f1:.3f} (sig={top_sig_ml.sum()}) | "
              f"底 P={ml_bot_p:.1%} R={ml_bot_r:.1%} F1={ml_bot_f1:.3f} (sig={bot_sig_ml.sum()})")

    # 规则聚合（全 OOS 区间）
    print(f"\n  规则聚合 顶/底检测（全OOS区间）:")
    for thresh in [0.1, 0.15, 0.2]:
        rule_top = merged['composite'].values > thresh
        rule_bot = merged['composite'].values < -thresh

        r_top_p, r_top_r, r_top_f1 = _eval_event_detection(
            merged.index[rule_top], event_peaks, merged.index)
        r_bot_p, r_bot_r, r_bot_f1 = _eval_event_detection(
            merged.index[rule_bot], event_troughs, merged.index)

        print(f"    Rule thresh={thresh:.1f}: "
              f"顶 P={r_top_p:.1%} R={r_top_r:.1%} F1={r_top_f1:.3f} (sig={rule_top.sum()}) | "
              f"底 P={r_bot_p:.1%} R={r_bot_r:.1%} F1={r_bot_f1:.3f} (sig={rule_bot.sum()})")

    # ── 逐事件OOS检测 ──
    print(f"\n  逐事件OOS检测:")
    for i, ev in dd_defs.iterrows():
        peak_date = ev['peak_date']
        trough_date = ev['trough_date']
        dd_pct = abs(ev['drawdown_pct'])

        peak_in = merged.index.min() <= peak_date <= merged.index.max()
        trough_in = merged.index.min() <= trough_date <= merged.index.max()

        top_comp = merged.loc[:peak_date, 'composite'].tail(20).max() if peak_in else None
        bot_comp = merged.loc[:trough_date, 'composite'].tail(20).min() if trough_in else None

        top_ml_s = "N/A"
        bot_ml_s = "N/A"
        if peak_in and len(dates_te) > 0 and peak_date >= dates_te[0]:
            idx_near = dates_clean.get_indexer([peak_date], method='nearest')[0]
            if meta_split <= idx_near < len(top_prob) + meta_split:
                top_ml_s = f"{top_prob[idx_near - meta_split]:.2f}"
        if trough_in and len(dates_te) > 0 and trough_date >= dates_te[0]:
            idx_near = dates_clean.get_indexer([trough_date], method='nearest')[0]
            if meta_split <= idx_near < len(bot_prob) + meta_split:
                bot_ml_s = f"{bot_prob[idx_near - meta_split]:.2f}"

        top_s = f"{top_comp:+.2f}" if top_comp is not None else "N/A"
        bot_s = f"{bot_comp:+.2f}" if bot_comp is not None else "N/A"
        print(f"    #{i+1} ({dd_pct:.1f}%): OOS顶={top_s} ML顶={top_ml_s} | "
              f"OOS底={bot_s} ML底={bot_ml_s}")

    # 保存元层配置
    meta_cfg = dict(META_CONFIG)
    meta_cfg['meta_features'] = meta_feat_cols
    meta_cfg['state_features'] = available_state
    meta_path = os.path.join(MODEL_DIR, 'meta_weights.json')
    with open(meta_path, 'w') as f:
        json.dump(meta_cfg, f, indent=2)


def _eval_event_detection(sig_dates, event_dates, full_range):
    """评估事件检测的精确率/召回率。"""
    n_sig = len(sig_dates)
    if n_sig == 0:
        return 0, 0, 0

    relevant = [e for e in event_dates if full_range.min() <= e <= full_range.max()]
    if not relevant:
        return 0, 0, 0

    precision = sum(1 for d in sig_dates
                   if any(abs((d - e).days) <= 30 for e in event_dates)) / n_sig
    recall = sum(1 for e in relevant
                if any(abs((d - e).days) <= 30 for d in sig_dates)) / len(relevant)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1
