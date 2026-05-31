import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional


def plot_forecasts(train: pd.Series, test: pd.Series, pred_lin: pd.Series, pred_arima: pd.Series, arima_model=None):
    plt.figure(figsize=(14,6))
    plt.plot(train.index, train.values, label='train')
    plt.plot(test.index, test.values, label='actual')
    plt.plot(pred_lin.index, pred_lin.values, label='linear_pred')
    plt.plot(pred_arima.index, pred_arima.values, label='arima_pred')

    # linear prediction interval using train residuals
    try:
        resid = train - (np.arange(len(train)).reshape(-1,1) @ np.array([0]).reshape(1,))
    except Exception:
        resid = train - train.mean()
    std = resid.std()
    cis_low = pred_lin - 1.96 * std
    cis_high = pred_lin + 1.96 * std
    plt.fill_between(pred_lin.index, cis_low, cis_high, color='C2', alpha=0.2)

    # ARIMA conf int if available
    if arima_model is not None:
        try:
            fc = arima_model.get_forecast(steps=len(pred_arima))
            ci = fc.conf_int()
            low = pd.Series(ci.iloc[:,0].values, index=pred_arima.index)
            high = pd.Series(ci.iloc[:,1].values, index=pred_arima.index)
            plt.fill_between(pred_arima.index, low, high, color='C3', alpha=0.2)
        except Exception:
            pass

    plt.legend()
    plt.title('Forecasts with Prediction Intervals')
    plt.show()


def plot_residuals(y_true: pd.Series, y_pred: pd.Series):
    resid = y_true - y_pred
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    sns.histplot(resid, kde=True, ax=axes[0])
    axes[0].set_title('Residuals distribution')
    axes[1].plot(resid.index, resid.values)
    axes[1].axhline(0, color='k', linestyle='--')
    axes[1].set_title('Residuals over time')
    plt.show()


def plot_rolling_error(y_true: pd.Series, y_pred: pd.Series, window: int = 7):
    err = (y_true - y_pred).abs()
    rolling = err.rolling(window=window).mean()
    plt.figure(figsize=(10,4))
    plt.plot(rolling.index, rolling.values)
    plt.title(f'Rolling MAE (window={window})')
    plt.show()
