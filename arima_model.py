"""
arima_model.py — ARIMA model: fit, forecast, evaluate
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import config

def fit_and_forecast(train, steps: int):
    model = ARIMA(train, order=config.ARIMA_ORDER).fit()
    fc    = model.forecast(steps=steps)
    return model, fc

def evaluate(actual, predicted) -> dict:
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE%": round(mape, 2)}
