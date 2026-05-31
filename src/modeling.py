import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple


def train_linear_trend(train: pd.Series) -> Tuple[LinearRegression, float]:
    # use integer time index as regressor
    X = np.arange(len(train)).reshape(-1, 1)
    y = train.values
    model = LinearRegression()
    model.fit(X, y)
    return model, model.intercept_


def forecast_linear(model: LinearRegression, steps: int, start_index: int = 0) -> np.ndarray:
    Xf = np.arange(start_index, start_index + steps).reshape(-1, 1)
    return model.predict(Xf)


def train_arima(train: pd.Series, order=(1, 1, 1)) -> ARIMA:
    model = ARIMA(train, order=order).fit()
    return model


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    # compute RMSE in a way compatible with older sklearn versions
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = (np.mean(np.abs((y_true - y_pred) / (y_true.replace(0, np.nan)))) * 100)
    return {'mae': mae, 'rmse': rmse, 'mape_percent': mape}
