"""快速过拟合诊断 — 减少fold数，限制n_jobs。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_absolute_error, r2_score
from lib.config import FEATURE_COLS, MODEL_META_PATH, CV_CONFIG, data_path
from lib.data_fetcher import fetch_merged_training_data
from lib.features import compute_features
from lib.labels import build_labels, compute_sample_weights
from lib.model_factory import create_model

def main():
    print("=" * 70)
    print("V4 快速过拟合诊断")
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

    # ── 滚动窗口 CV ──
    cfg = CV_CONFIG
    train_window = cfg['train_window']
    test_window = cfg['test_window']
    step = cfg['step']
    gap = cfg['label_gap']
    n = len(X)

    # 减少fold: 只取5个等间距fold
    max_start = n - train_window - gap - test_window
    n_folds = 5
    starts = np.linspace(0, max_start, n_folds + 2, dtype=int)[1:-1]

    print(f"数据总长: {n}, 训练窗口: {train_window}, 测试窗口: {test_window}")
    print(f"采样 {n_folds} 个fold (等间距)\n")

    dd_defs = pd.read_csv(dd_csv, parse_dates=['peak_date', 'trough_date'])
    event_peak_dates = dd_defs['peak_date'].tolist()
    event_trough_dates = dd_defs['trough_date'].tolist()

    all_oos_pred_top = []
    all_oos_pred_bot = []
    all_oos_y_top = []
    all_oos_y_bot = []
    all_oos_dates = []

    for fi, start in enumerate(starts):
        tr_s = start
        tr_e = start + train_window
        te_s = tr_e + gap
        te_e = min(te_s + test_window, n)

        X_tr = X[tr_s:tr_e]
        X_te = X[te_s:te_e]
        y_tr_top = y_top[tr_s:tr_e]
        y_tr_bot = y_bot[tr_s:tr_e]
        y_te_top = y_top[te_s:te_e]
        y_te_bot = y_bot[te_s:te_e]
        dates_te = dates[te_s:te_e]

        w_top = compute_sample_weights(y_tr_top)
        w_bot = compute_sample_weights(y_tr_bot)

        # 用n_jobs=4限制并行度
        mt = create_model('lgbm')
        mt.set_params(n_jobs=4)
        mt.fit(X_tr, y_tr_top, sample_weight=w_top)
        pt = mt.predict(X_te)

        mb = create_model('lgbm')
        mb.set_params(n_jobs=4)
        mb.fit(X_tr, y_tr_bot, sample_weight=w_bot)
        pb = mb.predict(X_te)

        peak_in = sum(1 for d in event_peak_dates if dates_te[0] <= d <= dates_te[-1])
        trough_in = sum(1 for d in event_trough_dates if dates_te[0] <= d <= dates_te[-1])

        for thresh in [30, 50, 70]:
            sig_top = pt >= thresh
            sig_bot = pb >= thresh
            hit_top = (y_te_top[sig_top] >= 30).mean() if sig_top.sum() > 0 else 0
            hit_bot = (y_te_bot[sig_bot] >= 30).mean() if sig_bot.sum() > 0 else 0
            if thresh == 50:
                print(f"  Fold {fi+1} [{dates_te[0].strftime('%Y-%m')} ~ {dates_te[-1].strftime('%Y-%m')}] "
                      f"top_sig={sig_top.sum()} hit={hit_top:.1%} | "
                      f"bot_sig={sig_bot.sum()} hit={hit_bot:.1%} | "
                      f"events: {peak_in}P/{trough_in}T")

        all_oos_pred_top.extend(pt)
        all_oos_pred_bot.extend(pb)
        all_oos_y_top.extend(y_te_top)
        all_oos_y_bot.extend(y_te_bot)
        all_oos_dates.extend(dates_te.tolist())

    # 汇总
    print(f"\n{'=' * 70}")
    print("OOS 汇总")
    print("=" * 70)
    oos_pt = np.array(all_oos_pred_top)
    oos_pb = np.array(all_oos_pred_bot)
    oos_yt = np.array(all_oos_y_top)
    oos_yb = np.array(all_oos_y_bot)

    for name, pred, y in [('顶部', oos_pt, oos_yt), ('底部', oos_pb, oos_yb)]:
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        corr = np.corrcoef(y, pred)[0, 1] if len(y) > 1 else 0
        print(f"  {name} OOS: MAE={mae:.2f}, R²={r2:.4f}, Corr={corr:.4f}")
        print(f"    pred: [{pred.min():.2f}, {pred.max():.2f}], mean={pred.mean():.2f}")
        pos = y > 0
        if pos.sum() > 0:
            print(f"    正样本(n={pos.sum()}) MAE: {mean_absolute_error(y[pos], pred[pos]):.2f}")

    # OOS精确率/召回率
    print(f"\n{'─' * 70}")
    print("OOS 精确率/召回率 (30天窗口)")
    print("─" * 70)
    for thresh in [30, 50, 70]:
        for side, pred, event_dates in [
            ('顶部', oos_pt, event_peak_dates),
            ('底部', oos_pb, event_trough_dates)
        ]:
            sig_mask = pred >= thresh
            n_sig = int(sig_mask.sum())
            if n_sig == 0:
                print(f"  {side}>={thresh}: 0信号")
                continue
            sig_dates_arr = np.array(all_oos_dates)[sig_mask]
            precision = sum(1 for d in sig_dates_arr
                          if any(abs((d - e).days) <= 30 for e in event_dates)) / n_sig
            relevant = [e for e in event_dates if min(all_oos_dates) <= e <= max(all_oos_dates)]
            recall = sum(1 for e in relevant
                       if any(abs((d - e).days) <= 30 for d in sig_dates_arr)) / len(relevant) if relevant else 0
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            print(f"  {side}>={thresh}: P={precision:.1%} R={recall:.1%} F1={f1:.3f} (sig={n_sig})")

    # 逐事件OOS
    print(f"\n{'─' * 70}")
    print("逐事件OOS检测")
    print("─" * 70)
    oos_df = pd.DataFrame({
        'date': all_oos_dates,
        'pred_top': oos_pt,
        'pred_bot': oos_pb,
        'label_top': oos_yt,
        'label_bot': oos_yb,
    }).set_index('date')

    for i, ev in dd_defs.iterrows():
        peak_date = ev['peak_date']
        trough_date = ev['trough_date']
        dd_pct = abs(ev['drawdown_pct'])
        top_in = (oos_df.index.min() <= peak_date <= oos_df.index.max())
        bot_in = (oos_df.index.min() <= trough_date <= oos_df.index.max())
        top_score = oos_df.loc[:peak_date, 'pred_top'].tail(30).max() if top_in else None
        bot_score = oos_df.loc[:trough_date, 'pred_bot'].tail(30).max() if bot_in else None
        print(f"  #{i+1} ({dd_pct:.1f}%): OOS顶={'%.0f' % top_score if top_score else 'N/A'}, "
              f"OOS底={'%.0f' % bot_score if bot_score else 'N/A'}")


if __name__ == '__main__':
    main()
