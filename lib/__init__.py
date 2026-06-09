"""纳斯达克多尺度方向预测系统 (V6) — 5个horizon LightGBM + GBC元分类器。"""
from .config import FEATURE_COLS, N_FEATURES, MODEL_PARAMS
from .features import compute_features
from .data_fetcher import fetch_training_data, fetch_live_data
from .model_factory import create_model
from .v6_labels import build_direction_labels, build_all_labels
from .v6_trainer import train_v6
from .v6_scorer import predict_v6
