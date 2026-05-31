import os
import joblib
import pandas as pd
from typing import Tuple
from pathlib import Path

from src.data_processing import load_sales_data, preprocess_time_series, train_test_split_ts
from src.modeling import train_linear_trend, train_arima
import json
from datetime import datetime


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def retrain_and_save(data_path: str = 'data/sales_data.csv', save_dir: str = 'models', test_size: int = 30, arima_order=(1,1,1)) -> Tuple[str, str]:
    """Train models on full data (optionally holdout) and save artifacts.

    Returns paths to saved linear and arima models.
    """
    ensure_dir(save_dir)
    # load and preprocess
    df = load_sales_data(data_path)
    ts = preprocess_time_series(df, freq='D')

    # train on full series (production use) or on train portion
    if len(ts) <= test_size:
        raise ValueError('Not enough data to hold out test set')

    train = ts.iloc[:-test_size]

    # linear model
    lin_model, _ = train_linear_trend(train)
    lin_path = Path(save_dir) / 'linear_model.joblib'
    joblib.dump(lin_model, lin_path)

    # arima model
    arima_model = train_arima(train, order=arima_order)
    arima_path = Path(save_dir) / 'arima_model.pkl'
    try:
        arima_model.save(arima_path)
    except Exception:
        # fallback to joblib/pickle
        joblib.dump(arima_model, arima_path)

    return str(lin_path), str(arima_path)


def load_models(save_dir: str = 'models'):
    lin_path = Path(save_dir) / 'linear_model.joblib'
    arima_path = Path(save_dir) / 'arima_model.pkl'
    lin = joblib.load(lin_path)
    # try statsmodels loader first
    try:
        from statsmodels.tsa.arima.model import ARIMAResults
        arima = ARIMAResults.load(arima_path)
    except Exception:
        arima = joblib.load(arima_path)
    return lin, arima


def save_versioned_models(lin_model, arima_model, ts, save_dir: str = 'models', prefix: str = 'model') -> Tuple[str, str]:
    """Save models with timestamped filenames and write metadata.json"""
    ensure_dir(save_dir)
    ts_end = str(ts.index[-1])
    ts_len = len(ts)
    ts_now = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')

    lin_name = f"{prefix}_linear_{ts_now}.joblib"
    arima_name = f"{prefix}_arima_{ts_now}.pkl"

    lin_path = Path(save_dir) / lin_name
    arima_path = Path(save_dir) / arima_name

    joblib.dump(lin_model, lin_path)
    try:
        arima_model.save(arima_path)
    except Exception:
        joblib.dump(arima_model, arima_path)

    metadata = {
        'timestamp_utc': ts_now,
        'train_end': ts_end,
        'train_length': ts_len,
        'linear_path': str(lin_path),
        'arima_path': str(arima_path)
    }
    meta_path = Path(save_dir) / f"{prefix}_metadata_{ts_now}.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return str(lin_path), str(arima_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Retrain models and save artifacts')
    parser.add_argument('--data', default='data/sales_data.csv')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--test-size', type=int, default=30)
    parser.add_argument('--arima-order', nargs=3, type=int, default=[1,1,1])
    args = parser.parse_args()

    lin_path, arima_path = retrain_and_save(data_path=args.data, save_dir=args.save_dir, test_size=args.test_size, arima_order=tuple(args.arima_order))
    print('Saved linear model to', lin_path)
    print('Saved ARIMA model to', arima_path)
