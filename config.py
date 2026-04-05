"""
config.py — Sales Forecasting (ARIMA / SARIMA)
"""

N_MONTHS     = 60          # 5 years of monthly data
HOLDOUT_MONTHS = 12        # last 12 months as test set
FORECAST_MONTHS = 6        # future forecast horizon

ARIMA_ORDER  = (1, 1, 1)
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL = (1, 1, 1, 12)

CHART_RAW      = "sales_raw_series.png"
CHART_ACF_PACF = "sales_acf_pacf.png"
CHART_FORECAST = "sales_forecast.png"
CHART_DPI      = 150
