# Day 5: Hourly Demand Forecasting Report

## Objective
Forecast hourly order demand using ARIMA and Prophet.

## Data overview
- Total orders: 5000
- Hourly observations: 6526
- Data start: 2026-01-01 02:00:00
- Data end: 2026-09-29 23:00:00
- Training observations: 6190
- Test observations: 336
- Test period: 2026-09-16 00:00:00 to 2026-09-29 23:00:00

## Stationarity
The Augmented Dickey-Fuller test returned a statistic of
-80.5179 and a p-value of
0. At the 5% significance level,
the unit-root null hypothesis was rejected. The initial ARIMA
model therefore used d=0.

## Model comparison
         model      mae     rmse  mape_nonzero_actual_pct
       Prophet 0.670883 0.819204                29.707153
ARIMA(1, 0, 1) 0.672130 0.809075                34.690912

MAPE is calculated only for test hours with nonzero actual demand,
because percentage error is undefined when actual demand is zero.

## Selected model
Prophet was selected using MAE as the primary metric.
The metrics are close: the model with the lower MAE is not
necessarily the model with the lower RMSE.

## Future forecast
Prophet was refitted on all available observations and used to
forecast the next 24 hours. The point forecasts and model intervals
are saved in data/processed/final_demand_forecast.csv.

## Business interpretation
- Hourly demand counts can help plan staffing and delivery capacity.
- Hour-of-day and weekday patterns can inform shift planning.
- Forecast intervals should be considered when planning capacity,
  rather than treating point forecasts as certain.
- The series is sparse, with many hours having zero orders.
- The data is synthetic and covers less than one year, so findings
  should be treated as a demonstration of the workflow, not as
  evidence of real-world Uber Eats demand.
- Yearly seasonality was disabled because the available history
  does not cover a full year.

## Limitations
Only one chronological holdout period was evaluated. Further work
could use rolling-origin validation and compare the models with
a simple seasonal-naive baseline.
