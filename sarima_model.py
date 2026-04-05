"""
sarima_model.py — SARIMA model: fit, forecast, evaluate, future projection
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import config

def fit_and_forecast(train, steps: int):
    model = SARIMAX(train, order=config.SARIMA_ORDER,
                    seasonal_order=config.SARIMA_SEASONAL).fit(disp=False)
    fc    = model.forecast(steps=steps)
    return model, fc

def future_forecast(full_series, steps: int):
    model = SARIMAX(full_series, order=config.SARIMA_ORDER,
                    seasonal_order=config.SARIMA_SEASONAL).fit(disp=False)
    fc    = model.forecast(steps=steps)
    future_idx = pd.date_range(full_series.index[-1], periods=steps + 1, freq="MS")[1:]
    fc.index   = future_idx
    return fc

def evaluate(actual, predicted) -> dict:
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE%": round(mape, 2)}
