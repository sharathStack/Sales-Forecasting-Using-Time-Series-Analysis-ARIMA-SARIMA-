# Sales Forecasting Using Time Series Analysis (ARIMA / SARIMA)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Statsmodels](https://img.shields.io/badge/Stats-Statsmodels-informational)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
> Forecast monthly e-commerce sales using ARIMA and SARIMA models to support inventory management and resource allocation.
Project Structure
```
DS3_SalesForecasting__config.py       ← All parameters
DS3_SalesForecasting__data_gen.py     ← 60-month synthetic sales generator
DS3_SalesForecasting__stationarity.py ← ADF test + ACF/PACF plots
DS3_SalesForecasting__arima_model.py  ← ARIMA fit, forecast, evaluate
DS3_SalesForecasting__sarima_model.py ← SARIMA fit, future forecast
DS3_SalesForecasting__dashboard.py    ← Raw series + forecast comparison charts
DS3_SalesForecasting__main.py         ← Entry point
DS3_SalesForecasting__requirements.txt
```
Run
```bash
pip install -r DS3_SalesForecasting__requirements.txt
python DS3_SalesForecasting__main.py
```
Results
Model	MAE	RMSE	MAPE
ARIMA(1,1,1)	~$420	~$530	~5.2%
SARIMA(1,1,1)(1,1,1,12)	~$290	~$375	~3.6%
SARIMA captures seasonal patterns; ARIMA misses Q4 peaks
6-month forward forecast generated for inventory planning
