"""过拟合诊断 — 滚动CV OOS评估 + 训练集vs OOS对比。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib, json
from sklearn.metrics import mean_absolute_error, r2_score
from lib.config import FEATURE_COLS, MODEL_TOP_PATH, MODEL_BOT_PATH, MODEL_META_PATH, \
    CV_CONFIG, LABEL_CONFIG, data_path
from lib.data_fetcher import fetch_merged_training_data
from lib.features import compute_features
from lib.labels import build_labels, compute_sample_weights
from lib.model_factory import create_model

def main():
    # 加载数据
    print("=" * 70)
    print("V4 过拟合诊断")
    print("=" * 70)

    df = fetch_merged_training_data()
    df = compute_features(df)
    dd_csv = data_path('dd_definitions')
    top_labels, bot_labels = build_labels(df, dd_defs_csv=dd_csv)
    df['top_score_label'] = top_labels
    df['bottom_score_label'] = bot_labels

    meta = json.load(open(MODEL_META_PATH, encoding='utf-8'))
    features = meta['features']

    keep = features + ['top_score_label', 'bottom_score_label', 'Close']
    valid = df[keep].dropna(subset=features + ['top_score_label', 'bottom_score_label'])
    print(f"有效样本: {len(valid)} 行, 特征: {len(features)} 维\n")

    X = valid[features].values
    y_top = valid['top_score_label'].values
    y_bot = valid['bottom_score_label'].values
    dates = valid.index

    # ── 1. 训练集性能 ──
    print("=" * 70)
    print("1. 训练集性能 (当前V4模型)")
    print("=" * 70)
    model_top = joblib.load(MODEL_TOP_PATH)
    model_bot = joblib.load(MODEL_BOT_PATH)

    pred_top_train = model_top.predict(X)
    pred_bot_train = model_bot.predict(X)

    for name, pred, y in [('顶部', pred_top_train, y_top), ('底部', pred_bot_train, y_bot)]:
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        corr = np.corrcoef(y, pred)[0, 1]
        print(f"  {name}: MAE={mae:.2f}, R²={r2:.4f}, Corr={corr:.4f}")
        print(f"    pred范围: [{pred.min():.2f}, {pred.max():.2f}], 均值={pred.mean():.2f}")
        print(f"    label范围: [{y.min():.2f}, {y.max():.2f}], 均值={y.mean():.2f}")
        # 正样本(>0)表现
        pos = y > 0
        if pos.sum() > 0:
            mae_pos = mean_absolute_error(y[pos], pred[pos])
            print(f"    正样本MAE: {mae_pos:.2f} (n={pos.sum()})")

    # ── 2. 滚动窗口 CV (OOS) ──
    print(f"\n{'=' * 70}")
    print("2. 滚动窗口 CV (OOS评估)")
    print("=" * 70)

    cfg = CV_CONFIG
    train_window = cfg['train_window']
    test_window = cfg['test_window']
    step = cfg['step']
    gap = cfg['label_gap']
    n = len(X)

    dd_defs = pd.read_csv(dd_csv, parse_dates=['peak_date', 'trough_date'])
    event_peak_dates = dd_defs['peak_date'].tolist()
    event_trough_dates = dd_defs['trough_date'].tolist()

    all_oos_pred_top = []
    all_oos_pred_bot = []
    all_oos_y_top = []
    all_oos_y_bot = []
    all_oos_dates = []

    fold_results = []
    fold_idx = 0
    start = 0

    while start + train_window + gap + test_window <= n:
        tr_s = start
        tr_e = start + train_window
        te_s = tr_e + gap
        te_e = min(te_s + test_window, n)
        if te_e <= te_s:
            break

        X_tr, y_tr_top, y_tr_bot = X[tr_s:tr_e], y_top[tr_s:tr_e], y_bot[tr_s:tr_e]
        X_te, y_te_top, y_te_bot = X[te_s:te_e], y_top[te_s:te_e], y_bot[te_s:te_e]
        dates_te = dates[te_s:te_e]

        w_tr_top = compute_sample_weights(y_tr_top)
        w_tr_bot = compute_sample_weights(y_tr_bot)

        mt = create_model('lgbm')
        mt.fit(X_tr, y_tr_top, sample_weight=w_tr_top)
        pt = mt.predict(X_te)

        mb = create_model('lgbm')
        mb.fit(X_tr, y_tr_bot, sample_weight=w_tr_bot)
        pb = mb.predict(X_te)

        # OOS 指标
        mae_top = mean_absolute_error(y_te_top, pt)
        mae_bot = mean_absolute_error(y_te_bot, pb)

        # 检查测试窗口内有几个事件
        te_dates = dates_te
        peak_in = sum(1 for d in event_peak_dates if te_dates[0] <= d <= te_dates[-1])
        trough_in = sum(1 for d in event_trough_dates if te_dates[0] <= d <= te_dates[-1])

        # 信号命中分析
        for thresh in [30, 50, 70]:
            sig_top = pt >= thresh
            sig_bot = pb >= thresh
            hit_top = (y_te_top[sig_top] >= 30).mean() if sig_top.sum() > 0 else 0
            hit_bot = (y_te_bot[sig_bot] >= 30).mean() if sig_bot.sum() > 0 else 0

            if fold_idx == 0 or thresh == 50:
                print(f"  Fold {fold_idx+1} [{dates_te[0].strftime('%Y-%m')} ~ {dates_te[-1].strftime('%Y-%m')}] "
                      f"thresh={thresh}: top_sig={sig_top.sum()}, hit={hit_top:.1%} | "
                      f"bot_sig={sig_bot.sum()}, hit={hit_bot:.1%} | "
                      f"MAE_top={mae_top:.2f}, MAE_bot={mae_bot:.2f} | "
                      f"events: peak={peak_in}, trough={trough_in}")

        fold_results.append({
            'fold': fold_idx + 1,
            'train_start': str(dates[tr_s].date()),
            'train_end': str(dates[tr_e - 1].date()),
            'test_start': str(dates[te_s].date()),
            'test_end': str(dates[te_e - 1].date()),
            'mae_top': mae_top,
            'mae_bot': mae_bot,
            'peaks_in_test': peak_in,
            'troughs_in_test': trough_in,
        })

        all_oos_pred_top.extend(pt)
        all_oos_pred_bot.extend(pb)
        all_oos_y_top.extend(y_te_top)
        all_oos_y_bot.extend(y_te_bot)
        all_oos_dates.extend(dates_te.tolist())

        start += step
        fold_idx += 1

    # 汇总OOS
    print(f"\n{'─' * 70}")
    print("OOS 汇总")
    print("─" * 70)
    oos_pt = np.array(all_oos_pred_top)
    oos_pb = np.array(all_oos_pred_bot)
    oos_yt = np.array(all_oos_y_top)
    oos_yb = np.array(all_oos_y_bot)

    for name, pred, y in [('顶部', oos_pt, oos_yt), ('底部', oos_pb, oos_yb)]:
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        corr = np.corrcoef(y, pred)[0, 1] if len(y) > 1 else 0
        print(f"  {name} OOS: MAE={mae:.2f}, R²={r2:.4f}, Corr={corr:.4f}")
        print(f"    pred范围: [{pred.min():.2f}, {pred.max():.2f}], 均值={pred.mean():.2f}")
        print(f"    label范围: [{y.min():.2f}, {y.max():.2f}], 均值={y.mean():.2f}")
        pos = y > 0
        if pos.sum() > 0:
            mae_pos = mean_absolute_error(y[pos], pred[pos])
            print(f"    正样本MAE: {mae_pos:.2f} (n={pos.sum()})")
            # 高分正样本
            high = y >= 50
            if high.sum() > 0:
                mae_high = mean_absolute_error(y[high], pred[high])
                print(f"    高分(>=50)MAE: {mae_high:.2f} (n={high.sum()})")

    # OOS 精确率/召回率
    print(f"\n{'─' * 70}")
    print("OOS 精确率/召回率 (30天窗口)")
    print("─" * 70)
    for thresh in [30, 50, 70]:
        for side, pred, y_true, event_dates in [
            ('顶部', oos_pt, oos_yt, event_peak_dates),
            ('底部', oos_pb, oos_yb, event_trough_dates)
        ]:
            sig_mask = pred >= thresh
            n_sig = int(sig_mask.sum())
            if n_sig == 0:
                print(f"  {side}>={thresh}: 0信号")
                continue
            # 精确率: 信号中有多少在事件30天内
            sig_dates_arr = np.array(all_oos_dates)[sig_mask]
            precision = sum(1 for d in sig_dates_arr
                          if any(abs((d - e).days) <= 30 for e in event_dates)) / n_sig
            # 召回率: 事件有多少被信号覆盖
            test_start = min(all_oos_dates)
            test_end = max(all_oos_dates)
            relevant_events = [e for e in event_dates if test_start <= e <= test_end]
            if len(relevant_events) > 0:
                recall = sum(1 for e in relevant_events
                           if any(abs((d - e).days) <= 30 for d in sig_dates_arr)) / len(relevant_events)
            else:
                recall = 0
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            print(f"  {side}>={thresh}: P={precision:.1%} R={recall:.1%} F1={f1:.3f} (sig={n_sig})")

    # ── 3. 训练 vs OOS 差距 ──
    print(f"\n{'=' * 70}")
    print("3. 过拟合差距分析")
    print("=" * 70)
    for name, p_train, y_train, p_oos, y_oos in [
        ('顶部', pred_top_train, y_top, oos_pt, oos_yt),
        ('底部', pred_bot_train, y_bot, oos_pb, oos_yb)
    ]:
        r2_tr = r2_score(y_train, p_train)
        r2_oos = r2_score(y_oos, p_oos)
        mae_tr = mean_absolute_error(y_train, p_train)
        mae_oos = mean_absolute_error(y_oos, p_oos)
        gap = r2_tr - r2_oos
        print(f"  {name}:")
        print(f"    训练 R²={r2_tr:.4f}, MAE={mae_tr:.2f}")
        print(f"    OOS  R²={r2_oos:.4f}, MAE={mae_oos:.2f}")
        print(f"    R²差距={gap:.4f} {'⚠️ 严重过拟合' if gap > 0.3 else '✅ 差距可控' if gap < 0.1 else '⚠️ 存在过拟合'}")

    # ── 4. 逐事件OOS分析 ──
    print(f"\n{'=' * 70}")
    print("4. 逐事件OOS检测 (信号在OOS预测中的表现)")
    print("=" * 70)
    oos_df = pd.DataFrame({
        'date': all_oos_dates,
        'pred_top': oos_pt,
        'pred_bot': oos_pb,
        'label_top': oos_yt,
        'label_bot': oos_yb,
    })
    oos_df = oos_df.set_index('date')

    for i, ev in dd_defs.iterrows():
        peak_date = ev['peak_date']
        trough_date = ev['trough_date']
        dd_pct = abs(ev['drawdown_pct'])

        # 检查事件是否在OOS范围内
        top_in_oos = (oos_df.index.min() <= peak_date <= oos_df.index.max())
        bot_in_oos = (oos_df.index.min() <= trough_date <= oos_df.index.max())

        top_score = oos_df.loc[:peak_date, 'pred_top'].tail(30).max() if top_in_oos else None
        bot_score = oos_df.loc[:trough_date, 'pred_bot'].tail(30).max() if bot_in_oos else None

        top_str = f"{top_score:.0f}" if top_score is not None else "不在OOS"
        bot_str = f"{bot_score:.0f}" if bot_score is not None else "不在OOS"
        print(f"  #{i+1} ({dd_pct:.1f}%): OOS顶部={top_str}, OOS底部={bot_str}")

    print(f"\n总fold数: {fold_idx}")


if __name__ == '__main__':
    main()
