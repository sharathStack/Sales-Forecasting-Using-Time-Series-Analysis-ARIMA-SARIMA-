"""
dashboard.py — Raw series, forecast comparison, future projection charts
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import config

def plot_raw(series: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series.index, series.values, color="#2c3e50", linewidth=1.5)
    ax.fill_between(series.index, series.values, alpha=0.12, color="#3498db")
    ax.set_title("Historical Monthly Sales", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sales ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(config.CHART_RAW, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"Raw series saved → {config.CHART_RAW}")

def plot_forecast(train, test, arima_fc, sarima_fc, future_fc,
                  arima_metrics, sarima_metrics) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train.index, train.values, color="#2c3e50", linewidth=1.5, label="Train")
    ax.plot(test.index,  test.values,  color="#7f8c8d", linewidth=1.5,
            linestyle="--", label="Actual (test)")
    ax.plot(arima_fc.index, arima_fc.values, color="#e74c3c", linewidth=2,
            label=f"ARIMA  MAE=${arima_metrics['MAE']:,.0f}")
    ax.plot(sarima_fc.index, sarima_fc.values, color="#27ae60", linewidth=2,
            label=f"SARIMA MAE=${sarima_metrics['MAE']:,.0f}")
    ax.plot(future_fc.index, future_fc.values, color="#8e44ad", linewidth=2,
            linestyle=":", label=f"{config.FORECAST_MONTHS}-month Forecast")
    ax.axvline(test.index[0], color="gray", linestyle=":", alpha=0.5,
               label="Train/Test split")
    ax.set_title("Sales Forecasting — ARIMA vs SARIMA", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sales ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(config.CHART_FORECAST, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"Forecast chart saved → {config.CHART_FORECAST}")
