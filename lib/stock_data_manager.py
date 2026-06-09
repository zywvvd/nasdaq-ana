"""Nasdaq 100 成分股数据下载与缓存管理。"""
import os
import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import DATA_DIR

STOCKS_DIR = os.path.join(DATA_DIR, 'stocks')


def _download_single(ticker, period='15y'):
    """下载单只股票 OHLCV，失败返回 None。"""
    import yfinance as yf
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"  {ticker}: 下载失败 - {e}")
        return None


def download_stock_data(tickers, period='15y', max_workers=5, delay=0.3):
    """批量下载股票数据，返回 {ticker: DataFrame}。"""
    os.makedirs(STOCKS_DIR, exist_ok=True)
    results = {}
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_single, t, period): t for t in tickers}

        for i, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            df = future.result()
            if df is not None and len(df) > 100:
                results[ticker] = df
                _save_stock_csv(ticker, df)
            else:
                failed.append(ticker)

            if i % 10 == 0:
                print(f"  进度: {i}/{len(tickers)} (成功={len(results)}, 失败={len(failed)})")

            if delay > 0:
                time.sleep(delay)

    print(f"  下载完成: {len(results)} 成功, {len(failed)} 失败")
    if failed:
        print(f"  失败tickers: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    return results


def _stock_csv_path(ticker):
    return os.path.join(STOCKS_DIR, f"{ticker}.csv")


def _save_stock_csv(ticker, df):
    df.to_csv(_stock_csv_path(ticker))


def _load_stock_csv(ticker):
    path = _stock_csv_path(ticker)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    if len(df) < 100:
        return None
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def get_all_stock_data(tickers, period='15y', force_refresh=False):
    """获取所有股票数据（优先从缓存，否则下载）。

    Returns {ticker: DataFrame(OHLCV)}
    """
    os.makedirs(STOCKS_DIR, exist_ok=True)
    results = {}
    to_download = []

    if not force_refresh:
        for ticker in tickers:
            df = _load_stock_csv(ticker)
            if df is not None:
                results[ticker] = df
            else:
                to_download.append(ticker)
    else:
        to_download = list(tickers)

    print(f"  缓存命中: {len(results)}, 需下载: {len(to_download)}")

    if to_download:
        downloaded = download_stock_data(to_download, period=period)
        results.update(downloaded)

    return results
