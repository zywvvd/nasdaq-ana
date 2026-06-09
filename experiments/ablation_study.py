"""消融实验 — 系统测试不同配置的OOS泛化能力。

实验维度:
1. 特征数: 5, 10, 15, 20, 30, 63
2. 树深度: 2, 3, 4, 6
3. 树数量: 50, 100, 200, 500
4. min_child_samples: 20, 50, 100, 200
5. 正则化: alpha/lambda 组合
6. 标签阈值: 用二值化标签做分类
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
import time
from itertools import product
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from lib.config import MODEL_META_PATH, CV_CONFIG, data_path
from lib.data_fetcher import fetch_merged_training_data
from lib.features import compute_features
from lib.labels import build_labels, compute_sample_weights
from lib.model_factory import create_model

def get_top_features(meta_path, n):
    """获取前n个最重要的特征（顶部+底部模型并集）。"""
    meta = json.load(open(meta_path, encoding='utf-8'))
    imp_top = meta.get('feature_importance_top', {})
    imp_bot = meta.get('feature_importance_bot', {})
    all_features = meta.get('all_features', [])

    # 合并重要性（取两个模型中较大的值）
    combined = {}
    for f in all_features:
        combined[f] = max(imp_top.get(f, 0), imp_bot.get(f, 0))

    ranked = sorted(combined.items(), key=lambda x: -x[1])
    return [f for f, _ in ranked[:n]]


def run_rolling_cv(X, y_top, y_bot, dates, features_idx,
                   n_estimators=100, max_depth=3, min_child_samples=100,
                   reg_alpha=1.0, reg_lambda=5.0, subsample=0.7,
                   colsample_bytree=0.5, n_folds=8):
    """运行滚动窗口CV，返回OOS指标。"""
    n = len(X)
    cfg = CV_CONFIG
    train_window = cfg['train_window']
    test_window = cfg['test_window']
    gap = cfg['label_gap']

    max_start = n - train_window - gap - test_window
    if max_start <= 0:
        return None

    step = max(max_start // n_folds, 1)
    starts = list(range(0, max_start + 1, step))[:n_folds]

    oos_preds_top = []
    oos_preds_bot = []
    oos_y_top = []
    oos_y_bot = []
    oos_dates = []

    for start in starts:
        tr_s = start
        tr_e = start + train_window
        te_s = tr_e + gap
        te_e = min(te_s + test_window, n)
        if te_e <= te_s:
            continue

        X_tr = X[tr_s:tr_e, features_idx]
        X_te = X[te_s:te_e, features_idx]
        y_tr_top = y_top[tr_s:tr_e]
        y_tr_bot = y_bot[tr_s:tr_e]

        w_top = compute_sample_weights(y_tr_top)
        w_bot = compute_sample_weights(y_tr_bot)

        params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            learning_rate=0.05,
            random_state=42,
            n_jobs=4,
            verbose=-1,
        )

        try:
            mt = create_model('lgbm')
            mt.set_params(**params)
            mt.fit(X_tr, y_tr_top, sample_weight=w_top)
            pt = mt.predict(X_te)

            mb = create_model('lgbm')
            mb.set_params(**params)
            mb.fit(X_tr, y_tr_bot, sample_weight=w_bot)
            pb = mb.predict(X_te)
        except Exception:
            continue

        oos_preds_top.extend(pt)
        oos_preds_bot.extend(pb)
        oos_y_top.extend(y_top[te_s:te_e])
        oos_y_bot.extend(y_bot[te_s:te_e])
        oos_dates.extend(dates[te_s:te_e].tolist())

    if len(oos_preds_top) == 0:
        return None

    return {
        'pred_top': np.array(oos_preds_top),
        'pred_bot': np.array(oos_preds_bot),
        'y_top': np.array(oos_y_top),
        'y_bot': np.array(oos_y_bot),
        'dates': oos_dates,
    }


def compute_metrics(result, event_peak_dates, event_trough_dates):
    """计算OOS指标。"""
    if result is None:
        return None

    metrics = {}

    for side, pred, y, event_dates in [
        ('top', result['pred_top'], result['y_top'], event_peak_dates),
        ('bot', result['pred_bot'], result['y_bot'], event_trough_dates)
    ]:
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred) if y.std() > 0 else 0

        # Rank-based: 对label>=50的天，其预测排名百分位
        high_label = y >= 50
        if high_label.sum() > 0:
            rank_pct = (pred[high_label] > pred[:, None]).mean()
            metrics[f'{side}_high_rank'] = rank_pct
        else:
            metrics[f'{side}_high_rank'] = 0

        metrics[f'{side}_mae'] = mae
        metrics[f'{side}_r2'] = r2
        metrics[f'{side}_pred_max'] = pred.max()
        metrics[f'{side}_pred_mean'] = pred.mean()

        # 精确率/召回率
        for thresh in [30, 50]:
            sig = pred >= thresh
            n_sig = sig.sum()
            if n_sig > 0:
                sig_dates = np.array(result['dates'])[sig]
                prec = sum(1 for d in sig_dates
                          if any(abs((d - e).days) <= 30 for e in event_dates)) / n_sig
                relevant = [e for e in event_dates
                          if min(result['dates']) <= e <= max(result['dates'])]
                rec = sum(1 for e in relevant
                         if any(abs((d - e).days) <= 30 for d in sig_dates)) / len(relevant) if relevant else 0
            else:
                prec, rec = 0, 0
            metrics[f'{side}_p{thresh}'] = prec
            metrics[f'{side}_r{thresh}'] = rec
            metrics[f'{side}_f1_{thresh}'] = 2 * prec * rec / (prec + rec + 1e-10)
            metrics[f'{side}_sig{thresh}'] = n_sig

    return metrics


def main():
    print("=" * 80)
    print("消融实验 — 系统测试OOS泛化能力")
    print("=" * 80)

    # 准备数据
    df = fetch_merged_training_data()
    df = compute_features(df)
    dd_csv = data_path('dd_definitions')
    top_labels, bot_labels = build_labels(df, dd_defs_csv=dd_csv)
    df['top_score_label'] = top_labels
    df['bottom_score_label'] = bot_labels

    meta = json.load(open(MODEL_META_PATH, encoding='utf-8'))
    all_features = meta.get('all_features', [])
    selected_features = meta['features']

    keep = all_features + ['top_score_label', 'bottom_score_label', 'Close']
    valid = df[keep].dropna(subset=all_features + ['top_score_label', 'bottom_score_label'])
    print(f"有效样本: {len(valid)} 行, 全特征: {len(all_features)} 维\n")

    X = valid[all_features].values
    y_top = valid['top_score_label'].values
    y_bot = valid['bottom_score_label'].values
    dates = valid.index

    dd_defs = pd.read_csv(dd_csv, parse_dates=['peak_date', 'trough_date'])
    event_peak_dates = dd_defs['peak_date'].tolist()
    event_trough_dates = dd_defs['trough_date'].tolist()

    # 获取特征重要性排名
    imp_top = meta.get('feature_importance_top', {})
    imp_bot = meta.get('feature_importance_bot', {})
    combined_imp = {}
    for f in all_features:
        combined_imp[f] = max(imp_top.get(f, 0), imp_bot.get(f, 0))
    ranked_features = sorted(combined_imp.items(), key=lambda x: -x[1])
    feature_order = [f for f, _ in ranked_features]

    # 特征名到列索引的映射
    feat_to_idx = {f: i for i, f in enumerate(all_features)}

    results = []

    # ══════════════════════════════════════
    # 实验1: 特征数消融 (固定保守参数)
    # ══════════════════════════════════════
    print("\n" + "=" * 80)
    print("实验1: 特征数消融")
    print("参数: n_est=100, depth=3, min_child=100, alpha=1, lambda=5")
    print("=" * 80)

    base_params = dict(n_estimators=100, max_depth=3, min_child_samples=100,
                       reg_alpha=1.0, reg_lambda=5.0)

    for n_feat in [5, 10, 15, 20, 30, 63]:
        t0 = time.time()
        top_n = feature_order[:n_feat]
        feat_idx = [feat_to_idx[f] for f in top_n if f in feat_to_idx]
        if len(feat_idx) < n_feat:
            continue

        result = run_rolling_cv(X, y_top, y_bot, dates, feat_idx, **base_params)
        m = compute_metrics(result, event_peak_dates, event_trough_dates)

        if m:
            print(f"  n_feat={n_feat:3d} | "
                  f"top: R²={m['top_r2']:.3f} P50={m['top_p50']:.0%} F1={m['top_f1_50']:.3f} sig={m['top_sig50']} | "
                  f"bot: R²={m['bot_r2']:.3f} P50={m['bot_p50']:.0%} F1={m['bot_f1_50']:.3f} sig={m['bot_sig50']} | "
                  f"{time.time()-t0:.1f}s")
            m['n_feat'] = n_feat
            m['experiment'] = 'feature_count'
            results.append(m)

    # ══════════════════════════════════════
    # 实验2: 树深度消融
    # ══════════════════════════════════════
    print("\n" + "=" * 80)
    print("实验2: 树深度消融 (n_feat=15)")
    print("=" * 80)

    feat_15 = [feat_to_idx[f] for f in feature_order[:15] if f in feat_to_idx]

    for depth in [2, 3, 4, 5, 6]:
        t0 = time.time()
        result = run_rolling_cv(X, y_top, y_bot, dates, feat_15,
                                n_estimators=100, max_depth=depth,
                                min_child_samples=100, reg_alpha=1.0, reg_lambda=5.0)
        m = compute_metrics(result, event_peak_dates, event_trough_dates)
        if m:
            print(f"  depth={depth} | "
                  f"top: R²={m['top_r2']:.3f} P50={m['top_p50']:.0%} F1={m['top_f1_50']:.3f} | "
                  f"bot: R²={m['bot_r2']:.3f} P50={m['bot_p50']:.0%} F1={m['bot_f1_50']:.3f} | "
                  f"{time.time()-t0:.1f}s")
            m['depth'] = depth
            m['experiment'] = 'depth'
            results.append(m)

    # ══════════════════════════════════════
    # 实验3: 树数量消融
    # ══════════════════════════════════════
    print("\n" + "=" * 80)
    print("实验3: 树数量消融 (n_feat=15, depth=3)")
    print("=" * 80)

    for n_est in [20, 50, 100, 200, 500]:
        t0 = time.time()
        result = run_rolling_cv(X, y_top, y_bot, dates, feat_15,
                                n_estimators=n_est, max_depth=3,
                                min_child_samples=100, reg_alpha=1.0, reg_lambda=5.0)
        m = compute_metrics(result, event_peak_dates, event_trough_dates)
        if m:
            print(f"  n_est={n_est:4d} | "
                  f"top: R²={m['top_r2']:.3f} P50={m['top_p50']:.0%} F1={m['top_f1_50']:.3f} | "
                  f"bot: R²={m['bot_r2']:.3f} P50={m['bot_p50']:.0%} F1={m['bot_f1_50']:.3f} | "
                  f"{time.time()-t0:.1f}s")
            m['n_est'] = n_est
            m['experiment'] = 'n_estimators'
            results.append(m)

    # ══════════════════════════════════════
    # 实验4: 正则化消融
    # ══════════════════════════════════════
    print("\n" + "=" * 80)
    print("实验4: 正则化消融 (n_feat=15, depth=3, n_est=100)")
    print("=" * 80)

    for alpha, lam in [(0.1, 1.0), (1.0, 5.0), (5.0, 10.0), (10.0, 50.0), (50.0, 100.0)]:
        t0 = time.time()
        result = run_rolling_cv(X, y_top, y_bot, dates, feat_15,
                                n_estimators=100, max_depth=3,
                                min_child_samples=100, reg_alpha=alpha, reg_lambda=lam)
        m = compute_metrics(result, event_peak_dates, event_trough_dates)
        if m:
            print(f"  α={alpha:5.1f} λ={lam:6.1f} | "
                  f"top: R²={m['top_r2']:.3f} P50={m['top_p50']:.0%} F1={m['top_f1_50']:.3f} | "
                  f"bot: R²={m['bot_r2']:.3f} P50={m['bot_p50']:.0%} F1={m['bot_f1_50']:.3f} | "
                  f"{time.time()-t0:.1f}s")
            m['alpha'] = alpha
            m['lambda'] = lam
            m['experiment'] = 'regularization'
            results.append(m)

    # ══════════════════════════════════════
    # 实验5: min_child_samples 消融
    # ══════════════════════════════════════
    print("\n" + "=" * 80)
    print("实验5: min_child_samples 消融 (n_feat=15, depth=3)")
    print("=" * 80)

    for mcs in [20, 50, 100, 200, 500]:
        t0 = time.time()
        result = run_rolling_cv(X, y_top, y_bot, dates, feat_15,
                                n_estimators=100, max_depth=3,
                                min_child_samples=mcs, reg_alpha=1.0, reg_lambda=5.0)
        m = compute_metrics(result, event_peak_dates, event_trough_dates)
        if m:
            print(f"  mcs={mcs:4d} | "
                  f"top: R²={m['top_r2']:.3f} P50={m['top_p50']:.0%} F1={m['top_f1_50']:.3f} | "
                  f"bot: R²={m['bot_r2']:.3f} P50={m['bot_p50']:.0%} F1={m['bot_f1_50']:.3f} | "
                  f"{time.time()-t0:.1f}s")
            m['min_child'] = mcs
            m['experiment'] = 'min_child'
            results.append(m)

    # ══════════════════════════════════════
    # 实验6: 无样本权重
    # ══════════════════════════════════════
    print("\n" + "=" * 80)
    print("实验6: 样本权重消融 (n_feat=15, depth=3)")
    print("=" * 80)

    # 修改CV函数支持无权重
    def run_cv_no_weight(X, y_top, y_bot, dates, feat_idx, **params):
        n = len(X)
        cfg = CV_CONFIG
        tw, testw, gap = cfg['train_window'], cfg['test_window'], cfg['label_gap']
        max_start = n - tw - gap - testw
        if max_start <= 0:
            return None
        step = max(max_start // 8, 1)
        starts = list(range(0, max_start + 1, step))[:8]

        preds_t, preds_b, ys_t, ys_b, ds = [], [], [], [], []
        for start in starts:
            tr_e = start + tw
            te_s = tr_e + gap
            te_e = min(te_s + testw, n)
            if te_e <= te_s:
                continue
            X_tr = X[start:tr_e, feat_idx]
            X_te = X[te_s:te_e, feat_idx]

            p = dict(**params, learning_rate=0.05, random_state=42, n_jobs=4, verbose=-1)
            try:
                mt = create_model('lgbm'); mt.set_params(**p)
                mt.fit(X_tr, y_top[start:tr_e])  # 无权重
                preds_t.extend(mt.predict(X_te))
                ys_t.extend(y_top[te_s:te_e])

                mb = create_model('lgbm'); mb.set_params(**p)
                mb.fit(X_tr, y_bot[start:tr_e])
                preds_b.extend(mb.predict(X_te))
                ys_b.extend(y_bot[te_s:te_e])
                ds.extend(dates[te_s:te_e].tolist())
            except Exception:
                continue

        if not preds_t:
            return None
        return dict(pred_top=np.array(preds_t), pred_bot=np.array(preds_b),
                    y_top=np.array(ys_t), y_bot=np.array(ys_b), dates=ds)

    for use_weight in [True, False]:
        t0 = time.time()
        if use_weight:
            result = run_rolling_cv(X, y_top, y_bot, dates, feat_15,
                                    n_estimators=100, max_depth=3,
                                    min_child_samples=100, reg_alpha=1.0, reg_lambda=5.0)
        else:
            result = run_cv_no_weight(X, y_top, y_bot, dates, feat_15,
                                      n_estimators=100, max_depth=3,
                                      min_child_samples=100, reg_alpha=1.0, reg_lambda=5.0)
        m = compute_metrics(result, event_peak_dates, event_trough_dates)
        if m:
            label = "有权重" if use_weight else "无权重"
            print(f"  {label:6s} | "
                  f"top: R²={m['top_r2']:.3f} P50={m['top_p50']:.0%} F1={m['top_f1_50']:.3f} | "
                  f"bot: R²={m['bot_r2']:.3f} P50={m['bot_p50']:.0%} F1={m['bot_f1_50']:.3f} | "
                  f"{time.time()-t0:.1f}s")
            m['weighted'] = use_weight
            m['experiment'] = 'sample_weight'
            results.append(m)

    # ══════════════════════════════════════
    # 保存结果
    # ══════════════════════════════════════
    print(f"\n{'=' * 80}")
    print(f"实验完成，共 {len(results)} 组结果")
    print("=" * 80)

    # 找最佳配置
    best_top = max(results, key=lambda r: r.get('top_f1_50', 0))
    best_bot = max(results, key=lambda r: r.get('bot_f1_50', 0))
    print(f"\n最佳顶部F1(50): {best_top.get('top_f1_50', 0):.3f} — {best_top.get('experiment')}")
    print(f"最佳底部F1(50): {best_bot.get('bot_f1_50', 0):.3f} — {best_bot.get('experiment')}")

    # 保存详细结果
    df_results = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(__file__), 'ablation_results.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
