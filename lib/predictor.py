"""V4 每日预测模块 — 选中特征 + 校准概率。"""
import pandas as pd
import numpy as np
import joblib
import json
import os

from .config import FEATURE_COLS, MODEL_TOP_PATH, MODEL_BOT_PATH, MODEL_META_PATH
from .config import MODEL_CAL_TOP_PATH, MODEL_CAL_BOT_PATH, CALIBRATION_CONFIG
from .data_fetcher import fetch_live_data
from .features import compute_features


def predict(df=None):
    """运行每日预测。输出原始评分 + 校准概率。"""
    meta = json.load(open(MODEL_META_PATH, encoding='utf-8'))
    model_top = joblib.load(MODEL_TOP_PATH)
    model_bot = joblib.load(MODEL_BOT_PATH)

    # 选中特征
    selected_features = meta.get('features', FEATURE_COLS)

    # 校准模型
    cal_enabled = meta.get('calibration', False) and os.path.exists(MODEL_CAL_TOP_PATH)
    cal_top = joblib.load(MODEL_CAL_TOP_PATH) if cal_enabled else None
    cal_bot = joblib.load(MODEL_CAL_BOT_PATH) if cal_enabled else None

    if df is None:
        df = fetch_live_data()

    df = compute_features(df)

    latest = df[selected_features].dropna().iloc[-1:]
    date_str = latest.index[0].strftime('%Y-%m-%d')
    close = df.loc[latest.index[0], 'Close']

    top_score = model_top.predict(latest.values)[0]
    bot_score = model_bot.predict(latest.values)[0]

    # 校准
    top_prob = float(cal_top.transform([top_score])[0]) if cal_top else None
    bot_prob = float(cal_bot.transform([bot_score])[0]) if cal_bot else None

    # 输出
    print(f"\n{'=' * 60}")
    print(f"纳斯达克顶部/底部评分 V4 — {date_str}")
    print(f"{'=' * 60}")
    print(f"  模型: {meta.get('model', 'Unknown')}")
    print(f"  收盘价: {close:,.0f}")
    print(f"  顶部风险评分: {top_score:.1f}/100", end="")
    if top_prob is not None:
        print(f"  (校准概率: {top_prob:.1%})")
    else:
        print()
    print(f"  底部机会评分: {bot_score:.1f}/100", end="")
    if bot_prob is not None:
        print(f"  (校准概率: {bot_prob:.1%})")
    else:
        print()

    # 信号解读
    print(f"\n  信号解读:")
    if top_prob is not None:
        if top_prob >= 0.5:
            print(f"    ⚠️ 顶部风险概率 {top_prob:.0%} — 高风险")
        elif top_prob >= 0.2:
            print(f"    ⚡ 顶部风险概率 {top_prob:.0%} — 需警惕")
        else:
            print(f"    ✅ 顶部风险概率 {top_prob:.0%} — 暂无风险")

        if bot_prob >= 0.5:
            print(f"    🟢 底部机会概率 {bot_prob:.0%} — 机会较大")
        elif bot_prob >= 0.2:
            print(f"    📊 底部机会概率 {bot_prob:.0%} — 可关注")
        else:
            print(f"    ⚠️ 底部机会概率 {bot_prob:.0%} — 暂无机会")
    else:
        if top_score >= 60:
            print(f"    ⚠️ 高度顶部风险区域 (评分{top_score:.0f})")
        elif top_score >= 50:
            print(f"    ⚡ 存在顶部风险信号 (评分{top_score:.0f})")
        else:
            print(f"    ✅ 暂无明显顶部风险 (评分{top_score:.0f})")

        if bot_score >= 50:
            print(f"    🟢 存在底部机会信号 (评分{bot_score:.0f})")
        else:
            print(f"    ⚠️ 暂无明显底部机会 (评分{bot_score:.0f})")

    return {
        'date': date_str,
        'close': float(close),
        'top_score': float(top_score),
        'bottom_score': float(bot_score),
        'top_probability': top_prob,
        'bottom_probability': bot_prob,
    }
