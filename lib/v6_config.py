"""V6 多尺度方向预测系统配置。"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "v6_models")

# ── 前向预测窗口（交易日）──
HORIZONS = [1, 3, 5, 7, 10, 30]

# ── 标签配置 ──
LABEL_CONFIG = {
    'rank_window': 504,
    'smooth_type': 'uniform',       # 'uniform' 或 'median'
    'smooth_window': lambda h: max(5, h // 3),
    'clip_range': (-1.0, 1.0),
}

# ── 逐 horizon 最优参数 ──
HORIZON_PARAMS = {
    1: {
        'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.01,
        'reg_alpha': 2.0, 'reg_lambda': 10.0, 'min_child_samples': 100,
        'subsample': 0.8, 'colsample_bytree': 0.6,
    },
    3: {
        'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
        'reg_alpha': 2.0, 'reg_lambda': 15.0, 'min_child_samples': 150,
        'subsample': 0.8, 'colsample_bytree': 0.5,
    },
    5: {
        'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.01,
        'reg_alpha': 0.5, 'reg_lambda': 10.0, 'min_child_samples': 150,
        'subsample': 0.7, 'colsample_bytree': 0.6,
    },
    7: {
        'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.01,
        'reg_alpha': 2.0, 'reg_lambda': 3.0, 'min_child_samples': 150,
        'subsample': 0.8, 'colsample_bytree': 0.5,
    },
    10: {
        'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05,
        'reg_alpha': 1.0, 'reg_lambda': 5.0, 'min_child_samples': 150,
        'subsample': 0.7, 'colsample_bytree': 0.6,
    },
    30: {
        'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.08,
        'reg_alpha': 4.0, 'reg_lambda': 3.0, 'min_child_samples': 50,
        'subsample': 0.8, 'colsample_bytree': 0.4,
    },
}

# 默认参数（用于特征选择阶段）
BASE_MODEL_PARAMS = {
    'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05,
    'reg_alpha': 1.0, 'reg_lambda': 5.0, 'min_child_samples': 100,
    'subsample': 0.8, 'colsample_bytree': 0.5,
}

# ── 特征选择 ──
FEATURE_SELECTION = {
    'enabled': True,
    'top_k': 40,
}

# ── Walk-forward CV ──
CV_CONFIG = {
    'train_window': 504,
    'test_window': 63,
    'step': 42,
    'n_folds': 80,
}

# ── 元层配置 ──
META_CONFIG = {
    'short_horizons': [1, 3, 5],
    'long_horizons': [7, 10, 30],
    'weights': {1: 0.5, 3: 0.8, 5: 1.0, 7: 1.2, 10: 1.5, 30: 2.0},
}
