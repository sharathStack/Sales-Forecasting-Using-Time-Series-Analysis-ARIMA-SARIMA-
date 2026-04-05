"""
stationarity.py — ADF test + ACF/PACF diagnostic plots
"""

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import config

def adf_test(series, title=""):
    result = adfuller(series.dropna())
    print(f"\n{'─'*40}")
    print(f"ADF Test — {title}")
    print(f"  Statistic : {result[0]:.4f}")
    print(f"  p-value   : {result[1]:.4f}")
    print(f"  Stationary: {'Yes ✓' if result[1] < 0.05 else 'No ✗ (needs differencing)'}")
    return result[1] < 0.05

def plot_diagnostics(series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_acf(series.diff().dropna(),  lags=20, ax=axes[0],
             title="ACF — First-Differenced Sales")
    plot_pacf(series.diff().dropna(), lags=20, ax=axes[1],
             title="PACF — First-Differenced Sales")
    plt.tight_layout()
    plt.savefig(config.CHART_ACF_PACF, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"ACF/PACF saved → {config.CHART_ACF_PACF}")
