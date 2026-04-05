"""
data_gen.py — 60-month synthetic e-commerce sales with trend, seasonality, COVID shock
"""

import numpy as np
import pandas as pd
import config

def generate() -> pd.Series:
    np.random.seed(42)
    n = config.N_MONTHS
    dates   = pd.date_range("2019-01-01", periods=n, freq="MS")
    trend   = np.linspace(5000, 9000, n)
    seasonal= 1500 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise   = np.random.normal(0, 300, n)
    sales   = trend + seasonal + noise
    sales[14:18] -= 2000          # COVID shock
    s = pd.Series(np.round(sales, 2), index=dates, name="sales")
    print(f"Generated {n} months  [{s.index[0].date()} → {s.index[-1].date()}]")
    print(f"Mean: ${s.mean():,.0f}   Std: ${s.std():,.0f}")
    return s
