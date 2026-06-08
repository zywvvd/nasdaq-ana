"""模型工厂 — 统一创建 RF / LightGBM 回归模型。"""
from sklearn.ensemble import RandomForestRegressor

RF_DEFAULTS = {
    'n_estimators': 1500,
    'max_depth': 10,
    'min_samples_leaf': 8,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,
}

LGBM_DEFAULTS = {
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.05,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}


def create_model(model_type='lgbm', params=None):
    """创建回归模型实例。

    Parameters
    ----------
    model_type : str — 'rf' 或 'lgbm'
    params : dict, optional — 覆盖默认参数

    Returns
    -------
    sklearn-compatible regressor
    """
    if model_type == 'rf':
        defaults = RF_DEFAULTS.copy()
    elif model_type == 'lgbm':
        from lightgbm import LGBMRegressor
        defaults = LGBM_DEFAULTS.copy()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if params:
        defaults.update(params)

    if model_type == 'rf':
        return RandomForestRegressor(**defaults)
    return LGBMRegressor(**defaults)
