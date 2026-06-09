"""V6 逐 horizon 超参优化 — 随机搜索 + walk-forward CV。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import json

from lib.data_fetcher import fetch_training_data
from lib.features import compute_features
from lib.config import FEATURE_COLS
from lib.model_factory import create_model
from lib.v6_config import HORIZONS, CV_CONFIG, FEATURE_SELECTION, MODEL_DIR
from lib.v6_labels import build_all_labels

# ── 搜索空间 ──
PARAM_GRID = {
    'n_estimators': [200, 300, 500],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.03, 0.05, 0.08],
    'min_child_samples': [50, 100, 150, 200],
    'reg_alpha': [0.5, 1.0, 2.0, 4.0],
    'reg_lambda': [3.0, 5.0, 10.0, 15.0],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.4, 0.5, 0.6],
}


def sample_params(n=30):
    """随机采样 n 组参数。"""
    configs = []
    for _ in range(n):
        cfg = {k: np.random.choice(v) for k, v in PARAM_GRID.items()}
        cfg['random_state'] = 42
        cfg['n_jobs'] = 4
        cfg['verbose'] = -1
        configs.append(cfg)
    return configs


def _select_features(X, y, feature_names, params):
    """特征选择：训练临时模型取 top_k。"""
    tmp_params = dict(params, n_estimators=100)
    tmp = create_model('lgbm')
    tmp.set_params(**tmp_params)
    tmp.fit(X, y)
    imp = pd.Series(tmp.feature_importances_, index=feature_names)
    top_k = FEATURE_SELECTION.get('top_k', 40)
    top = imp.sort_values(ascending=False).head(top_k)
    selected = top.index.tolist()
    idx = [list(feature_names).index(f) for f in selected]
    return idx, selected


def _rolling_cv_fast(X, y, dates, horizon, selected_idx, params):
    """轻量 walk-forward CV，只返回方向准确率和 IC。"""
    train_w = 504
    test_w = 63
    step = 84
    smooth_w = max(5, horizon // 3)
    gap = horizon + smooth_w

    n = len(X)
    max_start = n - train_w - gap - test_w
    if max_start <= 0:
        return 0, 0, 0

    starts = list(range(0, max_start + 1, step))[:6]

    all_pred, all_actual = [], []
    for start in starts:
        tr_e = start + train_w
        te_s = tr_e + gap
        te_e = min(te_s + test_w, n)
        if te_e <= te_s:
            continue

        model = create_model('lgbm')
        model.set_params(**params)
        model.fit(X[start:tr_e], y[start:tr_e])
        pred = model.predict(X[te_s:te_e])

        all_pred.extend(pred)
        all_actual.extend(y[te_s:te_e])

    if not all_pred:
        return 0, 0, 0

    pred = np.array(all_pred)
    actual = np.array(all_actual)

    dir_acc = (np.sign(pred) == np.sign(actual)).mean()
    ic, _ = spearmanr(pred, actual)
    mae = mean_absolute_error(actual, pred)

    return dir_acc, ic, mae


def main():
    np.random.seed(42)
    n_trials = 40

    print("=" * 70)
    print(f"V6 超参优化 — 每个horizon {n_trials} 组随机搜索")
    print("=" * 70)

    # 加载数据
    df = fetch_training_data()
    df = compute_features(df)
    labels = build_all_labels(df)

    best_params = {}
    best_scores = {}

    for horizon in HORIZONS:
        print(f"\n{'─' * 60}")
        print(f"Horizon = {horizon}d")
        print(f"{'─' * 60}")

        label_col = f'label_{horizon}d'
        df_h = df.copy()
        df_h[label_col] = labels[horizon]

        valid = df_h[FEATURE_COLS + [label_col]].dropna(subset=FEATURE_COLS + [label_col])
        X = valid[FEATURE_COLS].values
        y = valid[label_col].values

        configs = sample_params(n_trials)

        best_dir_acc = 0
        best_ic = 0
        best_cfg = None
        best_mae = float('inf')

        for i, cfg in enumerate(configs):
            selected_idx, selected_names = _select_features(X, y, FEATURE_COLS, cfg)
            X_sel = X[:, selected_idx]

            dir_acc, ic, mae = _rolling_cv_fast(
                X_sel, y, valid.index, horizon, selected_idx, cfg
            )

            # 综合评分：方向准确率权重 60%，IC 权重 40%
            score = dir_acc * 0.6 + max(0, ic) * 0.4

            if i < 5 or score > best_dir_acc * 0.6 + max(0, best_ic) * 0.4:
                marker = " ← best" if score > best_dir_acc * 0.6 + max(0, best_ic) * 0.4 else ""
                print(f"  #{i+1:2d} dir={dir_acc:.1%} IC={ic:.4f} MAE={mae:.3f} "
                      f"depth={cfg['max_depth']} lr={cfg['learning_rate']} "
                      f"n_est={cfg['n_estimators']} reg=({cfg['reg_alpha']},{cfg['reg_lambda']})"
                      f"{marker}")

            composite = dir_acc * 0.6 + max(0, ic) * 0.4
            if composite > best_dir_acc * 0.6 + max(0, best_ic) * 0.4:
                best_dir_acc = dir_acc
                best_ic = ic
                best_mae = mae
                best_cfg = cfg

        print(f"\n  最优: dir={best_dir_acc:.1%} IC={best_ic:.4f} MAE={best_mae:.3f}")
        print(f"  参数: depth={best_cfg['max_depth']} lr={best_cfg['learning_rate']} "
              f"n_est={best_cfg['n_estimators']} "
              f"reg=({best_cfg['reg_alpha']},{best_cfg['reg_lambda']}) "
              f"min_child={best_cfg['min_child_samples']} "
              f"subsample={best_cfg['subsample']} colsample={best_cfg['colsample_bytree']}")

        best_params[f'{horizon}d'] = {k: v for k, v in best_cfg.items()
                                       if k not in ('random_state', 'n_jobs', 'verbose')}
        best_scores[f'{horizon}d'] = {
            'dir_acc': float(best_dir_acc),
            'ic': float(best_ic),
            'mae': float(best_mae),
        }

    # 保存结果
    result_path = os.path.join(MODEL_DIR, 'optimal_params.json')
    with open(result_path, 'w') as f:
        json.dump({'params': best_params, 'scores': best_scores}, f, indent=2)
    print(f"\n最优参数已保存: {result_path}")

    # 输出可粘贴到 v6_config.py 的格式
    print(f"\n{'=' * 70}")
    print("最优参数汇总 (可粘贴到 v6_config.py):")
    print(f"{'=' * 70}")
    for h in HORIZONS:
        key = f'{h}d'
        p = best_params[key]
        print(f"\n# {h}d: dir={best_scores[key]['dir_acc']:.1%}, IC={best_scores[key]['ic']:.4f}")
        print(f"OPTIMAL_PARAMS_{h}D = {{")
        for k, v in p.items():
            if isinstance(v, str):
                print(f"    '{k}': '{v}',")
            else:
                print(f"    '{k}': {v},")
        print(f"}}")


if __name__ == '__main__':
    main()
