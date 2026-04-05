"""
main.py — Sales Forecasting entry point
"""

import config
from data_gen       import generate
from stationarity   import adf_test, plot_diagnostics
from arima_model    import fit_and_forecast as arima_fit, evaluate as arima_eval
from sarima_model   import (fit_and_forecast as sarima_fit,
                            future_forecast, evaluate as sarima_eval)
from dashboard      import plot_raw, plot_forecast

def main():
    print("=" * 55)
    print("  SALES FORECASTING — ARIMA / SARIMA")
    print("=" * 55)

    sales = generate()

    print("\n[1] Plotting raw series...")
    plot_raw(sales)

    print("\n[2] Stationarity tests...")
    adf_test(sales,        "Original Series")
    adf_test(sales.diff(), "First-Differenced")
    plot_diagnostics(sales)

    train = sales.iloc[:-config.HOLDOUT_MONTHS]
    test  = sales.iloc[-config.HOLDOUT_MONTHS:]
    print(f"\n[3] Split: Train={len(train)}  Test={len(test)}")

    print("\n[4] Fitting ARIMA...")
    _, arima_fc = arima_fit(train, len(test))
    arima_fc.index = test.index
    a_metrics = arima_eval(test.values, arima_fc.values)
    print(f"   ARIMA  → {a_metrics}")

    print("\n[5] Fitting SARIMA...")
    _, sarima_fc = sarima_fit(train, len(test))
    sarima_fc.index = test.index
    s_metrics = sarima_eval(test.values, sarima_fc.values)
    print(f"   SARIMA → {s_metrics}")

    print(f"\n[6] {config.FORECAST_MONTHS}-month future forecast...")
    future = future_forecast(sales, config.FORECAST_MONTHS)
    print(future.rename("Forecast ($)").to_string())

    print("\n[7] Generating charts...")
    plot_forecast(train, test, arima_fc, sarima_fc, future, a_metrics, s_metrics)

    print("\n  Done ✓")

if __name__ == "__main__":
    main()
