"""V6 预测模块 — 加载6个模型 + 元层ML聚合。"""
import os
import json
import numpy as np
import pandas as pd
import joblib

from .config import FEATURE_COLS
from .v6_config import HORIZONS, MODEL_DIR, META_CONFIG


def predict_v6(df):
    """对 DataFrame 生成 V6 多尺度方向预测。

    Parameters
    ----------
    df : DataFrame — 需包含 FEATURE_COLS 中所有特征列

    Returns
    -------
    list[dict] — 每行一个 dict
    """
    meta_path = os.path.join(MODEL_DIR, 'v6_meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"V6 模型不存在: {meta_path}\n请先运行 python run_v6_pipeline.py --train")

    all_meta = json.load(open(meta_path))
    weights = META_CONFIG['weights']

    # 尝试加载 ML meta 模型
    meta_top_clf = None
    meta_bot_clf = None
    meta_scaler = None
    meta_feat_cols = None

    clf_path = os.path.join(MODEL_DIR, 'meta_top_clf.pkl')
    if os.path.exists(clf_path):
        meta_top_clf = joblib.load(clf_path)
        meta_bot_clf = joblib.load(os.path.join(MODEL_DIR, 'meta_bot_clf.pkl'))
        meta_scaler = joblib.load(os.path.join(MODEL_DIR, 'meta_scaler.pkl'))
        meta_feat_cols = joblib.load(os.path.join(MODEL_DIR, 'meta_features.pkl'))

    results = []
    for idx, row in df.iterrows():
        x = row[FEATURE_COLS].values.astype(float).reshape(1, -1)
        if np.isnan(x).any():
            continue

        preds = {}
        for h in HORIZONS:
            key = f'{h}d'
            if key not in all_meta:
                continue
            m = all_meta[key]
            model_path = os.path.join(MODEL_DIR, f'model_{h}d.pkl')
            if not os.path.exists(model_path):
                continue
            model = joblib.load(model_path)
            feat_idx = [FEATURE_COLS.index(f) for f in m['features'] if f in FEATURE_COLS]
            pred = float(model.predict(x[:, feat_idx])[0])
            preds[h] = pred

        if len(preds) < 3:
            continue

        # 元特征
        w = np.array([weights.get(h, 1.0) for h in preds])
        pred_vals = np.array([preds[h] for h in preds])
        composite = float((pred_vals * w).sum() / w.sum())

        short_h = [h for h in META_CONFIG['short_horizons'] if h in preds]
        long_h = [h for h in META_CONFIG['long_horizons'] if h in preds]
        short_avg = float(np.mean([preds[h] for h in short_h])) if short_h else 0
        long_avg = float(np.mean([preds[h] for h in long_h])) if long_h else 0
        divergence = short_avg - long_avg

        signs = np.sign(pred_vals)
        agreement = float(max((signs > 0).sum(), (signs < 0).sum())) / len(preds)
        extreme = float(np.abs(pred_vals).max())
        pred_std = float(np.std(pred_vals))
        all_positive = float((signs > 0).sum() == len(preds))
        all_negative = float((signs < 0).sum() == len(preds))

        # 规则评分
        top_risk = 0.0
        if composite > 0.2:
            top_risk = min(100, composite * (1 + max(0, divergence)) * agreement * 100)

        bottom_opportunity = 0.0
        if composite < -0.2:
            bottom_opportunity = min(100, abs(composite) * (1 + max(0, -divergence)) * agreement * 100)

        # ML meta 评分
        ml_top_prob = 0.0
        ml_bot_prob = 0.0
        if meta_top_clf is not None and meta_feat_cols is not None:
            feat_dict = {}
            for h in HORIZONS:
                feat_dict[f'pred_{h}d'] = preds.get(h, 0.0)
            feat_dict.update({
                'short_avg': short_avg, 'long_avg': long_avg,
                'divergence': divergence, 'extreme': extreme,
                'agreement': agreement, 'pred_std': pred_std,
                'all_positive': all_positive, 'all_negative': all_negative,
            })
            # 加入市场状态特征
            state_cols = ['VIX', 'RSI', 'DD_depth', 'Dist_MA200', 'Dist_MA60',
                          'Volatility_20d', 'SKEW', 'Spread_10Y_2Y',
                          'HYG_TLT_Ratio', 'Mom_20d', 'VIX_rank_60d', 'RSI_rank_60d']
            for c in state_cols:
                feat_dict[c] = float(row.get(c, 0)) if c in row.index and pd.notna(row.get(c, None)) else 0.0
            x_meta = np.array([[feat_dict.get(c, 0) for c in meta_feat_cols]])
            x_meta_s = meta_scaler.transform(x_meta)

            if meta_top_clf.classes_.size == 2:
                ml_top_prob = float(meta_top_clf.predict_proba(x_meta_s)[0, 1])
            if meta_bot_clf.classes_.size == 2:
                ml_bot_prob = float(meta_bot_clf.predict_proba(x_meta_s)[0, 1])

        r = {
            'date': idx,
            'close': float(row.get('Close', 0)),
        }
        for h in HORIZONS:
            r[f'pred_{h}d'] = preds.get(h, 0.0)
        r.update({
            'composite': composite,
            'short_avg': short_avg,
            'long_avg': long_avg,
            'divergence': divergence,
            'extreme': extreme,
            'agreement': agreement,
            'top_risk': top_risk,
            'bottom_opportunity': bottom_opportunity,
            'ml_top_prob': ml_top_prob,
            'ml_bot_prob': ml_bot_prob,
        })
        results.append(r)

    return results


def print_prediction(result):
    """打印单日预测结果。"""
    print(f"\n{'=' * 60}")
    print(f"V6 多尺度方向预测 — {result['date'].strftime('%Y-%m-%d')}")
    print(f"{'=' * 60}")
    print(f"  收盘价: {result['close']:,.0f}")
    print(f"\n  方向预测:")
    for h in HORIZONS:
        p = result.get(f'pred_{h}d', 0)
        arrow = "↑" if p > 0.1 else ("↓" if p < -0.1 else "→")
        print(f"    {h:2d}d: {p:+.3f} {arrow}")

    comp = result['composite']
    arrow = "↑" if comp > 0.1 else ("↓" if comp < -0.1 else "→")
    print(f"\n  综合评分: {comp:+.3f} {arrow}")
    print(f"  短期({'+'.join(str(h) for h in META_CONFIG['short_horizons'])}d): {result['short_avg']:+.3f}")
    print(f"  长期({'+'.join(str(h) for h in META_CONFIG['long_horizons'])}d): {result['long_avg']:+.3f}")
    print(f"  短长分歧: {result['divergence']:+.3f}")
    print(f"  方向一致度: {result['agreement']:.1%}")

    print(f"\n  规则评分: 顶部风险={result['top_risk']:.1f}/100 | 底部机会={result['bottom_opportunity']:.1f}/100")

    ml_top = result.get('ml_top_prob', 0)
    ml_bot = result.get('ml_bot_prob', 0)
    if ml_top > 0 or ml_bot > 0:
        print(f"  ML评分: 顶部概率={ml_top:.1%} | 底部概率={ml_bot:.1%}")

    if ml_top > 0.4:
        print(f"  ⚠️  ML顶部风险升高 ({ml_top:.0%})")
    elif ml_bot > 0.4:
        print(f"  ⚠️  ML底部机会升高 ({ml_bot:.0%})")
    elif result['top_risk'] > 30:
        print(f"  ⚠️  规则顶部风险升高")
    elif result['bottom_opportunity'] > 30:
        print(f"  ⚠️  规则底部机会升高")
    else:
        print(f"  ✅ 市场中性")
