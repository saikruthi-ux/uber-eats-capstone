
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from prophet import Prophet

# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "orders.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load and validate orders
# --------------------------------------------------
orders = pd.read_csv(RAW_FILE)

orders["order_timestamp"] = pd.to_datetime(
    orders["order_timestamp"], errors="coerce"
)

invalid_timestamps = orders["order_timestamp"].isna().sum()
orders = orders.dropna(subset=["order_timestamp"]).copy()

orders = orders.sort_values("order_timestamp")

print("=" * 60)
print("DAY 5: HOURLY DEMAND FORECASTING")
print("=" * 60)
print(f"Valid orders: {len(orders):,}")
print(f"Invalid timestamps removed: {invalid_timestamps}")
print(f"Date range: {orders['order_timestamp'].min()} to "
      f"{orders['order_timestamp'].max()}")

print("\nOrder status counts:")
print(orders["order_status"].value_counts())

# --------------------------------------------------
# 3. Aggregate orders into hourly demand
# --------------------------------------------------
orders["order_hour"] = orders["order_timestamp"].dt.floor("h")

hourly = (
    orders.groupby("order_hour")
    .size()
    .rename("order_count")
    .to_frame()
)

# Include missing hours as zero recorded orders
hourly = hourly.asfreq("h", fill_value=0)

hourly.index.name = "order_hour"

hourly_file = PROCESSED_DIR / "hourly_order_counts.csv"
hourly.to_csv(hourly_file)

print("\nHourly time-series summary:")
print(f"Number of hourly observations: {len(hourly):,}")
print(f"Total orders counted: {hourly['order_count'].sum():,}")
print(hourly["order_count"].describe())

print("\nFirst 10 hourly observations:")
print(hourly.head(10))

# --------------------------------------------------
# 4. Plot the full hourly time series
# --------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(hourly.index, hourly["order_count"], linewidth=0.8)
plt.title("Hourly Uber Eats Order Demand")
plt.xlabel("Date")
plt.ylabel("Number of orders")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "hourly_order_counts.png",
    dpi=150
)
plt.close()

# --------------------------------------------------
# 5. Explore order volume by hour of day
# --------------------------------------------------
orders["hour_of_day"] = orders["order_timestamp"].dt.hour

by_hour = (
    orders.groupby("hour_of_day")
    .size()
    .reindex(range(24), fill_value=0)
    .rename("total_orders")
)

by_hour.index.name = "hour_of_day"
by_hour.to_csv(PROCESSED_DIR / "orders_by_hour_of_day.csv")

plt.figure(figsize=(10, 5))
plt.bar(by_hour.index, by_hour.values)
plt.xticks(range(24))
plt.title("Order Volume by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Total Orders")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "orders_by_hour_of_day.png",
    dpi=150
)
plt.close()

# --------------------------------------------------
# 6. Explore order volume by day of week
# --------------------------------------------------
orders["day_of_week"] = orders["order_timestamp"].dt.day_of_week

weekday_names = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

by_weekday = (
    orders.groupby("day_of_week")
    .size()
    .reindex(range(7), fill_value=0)
)

by_weekday.index = weekday_names
by_weekday.index.name = "day_of_week"
by_weekday = by_weekday.rename("total_orders")

by_weekday.to_csv(PROCESSED_DIR / "orders_by_weekday.csv")

plt.figure(figsize=(10, 5))
plt.bar(by_weekday.index, by_weekday.values)
plt.title("Order Volume by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Total Orders")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "orders_by_weekday.png",
    dpi=150
)
plt.close()

# --------------------------------------------------
# 7. Explore weekend versus weekday demand
# --------------------------------------------------
orders["is_weekend"] = orders["day_of_week"] >= 5

weekend_summary = (
    orders.groupby("is_weekend")
    .agg(
        total_orders=("order_id", "count"),
        unique_days=("order_timestamp",
                     lambda x: x.dt.normalize().nunique())
    )
)

weekend_summary["orders_per_day"] = (
    weekend_summary["total_orders"]
    / weekend_summary["unique_days"]
)

weekend_summary.index = ["Weekday", "Weekend"]
weekend_summary.index.name = "day_type"

weekend_summary.to_csv(
    PROCESSED_DIR / "weekday_weekend_summary.csv"
)

print("\nAverage order volume per calendar day:")
print(weekend_summary)

# --------------------------------------------------
# 8. Identify peak ordering hour and weekday
# --------------------------------------------------
peak_hour = by_hour.idxmax()
peak_weekday = by_weekday.idxmax()

print(f"\nPeak hour by total orders: {peak_hour}:00 "
      f"({by_hour.max():,} orders)")
print(f"Highest-volume weekday: {peak_weekday} "
      f"({by_weekday.max():,} orders)")

print("\nFiles created:")
print(f"- {hourly_file.relative_to(ROOT)}")
print("- data/processed/orders_by_hour_of_day.csv")
print("- data/processed/orders_by_weekday.csv")
print("- data/processed/weekday_weekend_summary.csv")
print("- reports/figures/hourly_order_counts.png")
print("- reports/figures/orders_by_hour_of_day.png")
print("- reports/figures/orders_by_weekday.png")


# --------------------------------------------------
# 9. Time-series decomposition (Part 3)
# --------------------------------------------------

print("\n" + "=" * 60)
print("PART 3: TIME-SERIES DECOMPOSITION")
print("=" * 60)

# A 24-hour period represents a daily repeating pattern
decomposition = seasonal_decompose(
    hourly["order_count"],
    model="additive",
    period=24
)

fig = decomposition.plot()
fig.set_size_inches(14, 10)
fig.suptitle(
    "Hourly Demand Decomposition (24-Hour Seasonality)",
    y=1.02
)
fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "time_series_decomposition.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close(fig)

print("Saved: reports/figures/time_series_decomposition.png")

# Save the individual components for further analysis
decomposition_df = pd.DataFrame({
    "observed": hourly["order_count"],
    "trend": decomposition.trend,
    "seasonal": decomposition.seasonal,
    "residual": decomposition.resid
})

decomposition_df.to_csv(
    PROCESSED_DIR / "time_series_decomposition.csv"
)

print("Saved: data/processed/time_series_decomposition.csv")

# --------------------------------------------------
# 10. Stationarity testing with ADF (Part 4)
# --------------------------------------------------
print("\n" + "=" * 60)
print("PART 4: STATIONARITY AND ADF TEST")
print("=" * 60)

# ADF null hypothesis: the series has a unit root
def run_adf(series, series_name):
    series = series.dropna()

    result = adfuller(series, autolag="AIC")

    return {
        "series": series_name,
        "adf_statistic": result[0],
        "p_value": result[1],
        "used_lags": result[2],
        "observations": result[3],
        "critical_value_1_percent": result[4]["1%"],
        "critical_value_5_percent": result[4]["5%"],
        "critical_value_10_percent": result[4]["10%"],
        "stationary_at_5_percent": result[1] < 0.05
    }

adf_results = []

# Test the original hourly series
adf_results.append(
    run_adf(hourly["order_count"], "Original hourly demand")
)

# If needed, test first-order differencing
original_p_value = adf_results[0]["p_value"]

if original_p_value >= 0.05:
    print("Original series is not stationary at 5%.")
    print("Testing first-order differencing...")

    differenced = hourly["order_count"].diff().dropna()

    adf_results.append(
        run_adf(differenced, "First-differenced hourly demand")
    )
else:
    print("Original series is stationary at the 5% significance level.")
    print("First differencing is not required based on this ADF result.")

adf_df = pd.DataFrame(adf_results)
adf_df.to_csv(
    PROCESSED_DIR / "adf_test_results.csv",
    index=False
)

print("\nADF test results:")
print(adf_df.to_string(index=False))

print("\nSaved: data/processed/adf_test_results.csv")


# ============================================================
# PART 5: ARIMA DEMAND FORECASTING
# ============================================================

print("\n" + "=" * 60)
print("PART 5: ARIMA DEMAND FORECASTING")
print("=" * 60)

# Use the final 14 days as the test period.
# All earlier observations are used for training.
test_hours = 14 * 24

train = hourly["order_count"].iloc[:-test_hours]
test = hourly["order_count"].iloc[-test_hours:]

print(f"Training observations: {len(train)}")
print(f"Testing observations:  {len(test)}")
print(f"Training period: {train.index.min()} to {train.index.max()}")
print(f"Testing period:  {test.index.min()} to {test.index.max()}")

# Fit a baseline ARIMA model.
# d=0 is the starting choice based on the ADF test.
arima_order = (1, 0, 1)

print(f"\nFitting ARIMA{arima_order}...")

arima_model = ARIMA(
    train,
    order=arima_order
)

arima_fit = arima_model.fit()

# Forecast exactly the number of hours in the test period.
arima_predictions = arima_fit.forecast(steps=len(test))

# Match forecast timestamps to the test timestamps.
arima_predictions = pd.Series(
    np.asarray(arima_predictions),
    index=test.index,
    name="arima_forecast"
)

# Order counts cannot be negative.
arima_predictions = arima_predictions.clip(lower=0)

# Calculate evaluation metrics.
mae = mean_absolute_error(test, arima_predictions)
rmse = np.sqrt(mean_squared_error(test, arima_predictions))

# MAPE is undefined when actual demand is zero.
nonzero_actual = test != 0

if nonzero_actual.any():
    mape = (
        np.abs(
            (test[nonzero_actual] - arima_predictions[nonzero_actual])
            / test[nonzero_actual]
        ).mean()
        * 100
    )
else:
    mape = np.nan

arima_metrics = pd.DataFrame([{
    "model": f"ARIMA{arima_order}",
    "train_observations": len(train),
    "test_observations": len(test),
    "mae": mae,
    "rmse": rmse,
    "mape_nonzero_actual_pct": mape,
    "test_hours_with_nonzero_actual": int(nonzero_actual.sum()),
    "test_hours_with_zero_actual": int((~nonzero_actual).sum())
}])

print("\nARIMA evaluation results:")
print(arima_metrics.to_string(index=False))

# Save predictions and actual values.
arima_forecast_df = pd.DataFrame({
    "actual_orders": test,
    "arima_forecast": arima_predictions
})

arima_forecast_df.to_csv(
    PROCESSED_DIR / "arima_forecast.csv",
    index_label="order_hour"
)

arima_metrics.to_csv(
    PROCESSED_DIR / "arima_model_metrics.csv",
    index=False
)

# Plot actual demand versus the ARIMA forecast.
plt.figure(figsize=(15, 6))
plt.plot(
    test.index,
    test.values,
    label="Actual orders",
    linewidth=1.5
)
plt.plot(
    arima_predictions.index,
    arima_predictions.values,
    label="ARIMA forecast",
    linewidth=1.5
)

plt.title("ARIMA Hourly Demand Forecast — Final 14 Days")
plt.xlabel("Date and hour")
plt.ylabel("Number of orders")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "arima_forecast.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("\nFiles saved:")
print("- data/processed/arima_forecast.csv")
print("- data/processed/arima_model_metrics.csv")
print("- reports/figures/arima_forecast.png")
print("\nARIMA forecasting section completed.")


# ============================================================
# PART 6: PROPHET DEMAND FORECASTING
# ============================================================

print("\n" + "=" * 60)
print("PART 6: PROPHET DEMAND FORECASTING")
print("=" * 60)

# Prophet requires columns named ds (timestamp) and y (target).
prophet_data = hourly.reset_index().rename(
    columns={
        "order_hour": "ds",
        "order_count": "y"
    }
)

# Use exactly the same chronological split as ARIMA.
prophet_train = prophet_data.iloc[:-test_hours].copy()
prophet_test = prophet_data.iloc[-test_hours:].copy()

print(f"Training observations: {len(prophet_train)}")
print(f"Testing observations:  {len(prophet_test)}")

# Model daily and weekly seasonality.
# Yearly seasonality is disabled because the dataset covers less than a full year.
prophet_model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False,
    interval_width=0.95
)

print("\nFitting Prophet model...")
prophet_model.fit(prophet_train)

# Forecast the held-out timestamps.
future = prophet_test[["ds"]].copy()
prophet_forecast = prophet_model.predict(future)

# Keep predictions non-negative because order volume cannot be negative.
prophet_predictions = pd.Series(
    prophet_forecast["yhat"].clip(lower=0).to_numpy(),
    index=test.index,
    name="prophet_forecast"
)

# Evaluate on the same test observations as ARIMA.
prophet_mae = mean_absolute_error(test, prophet_predictions)
prophet_rmse = np.sqrt(
    mean_squared_error(test, prophet_predictions)
)

# MAPE is undefined for actual values of zero.
nonzero_actual = test != 0

if nonzero_actual.any():
    prophet_mape = (
        np.abs(
            (
                test[nonzero_actual]
                - prophet_predictions[nonzero_actual]
            ) / test[nonzero_actual]
        ).mean()
        * 100
    )
else:
    prophet_mape = np.nan

prophet_metrics = pd.DataFrame([{
    "model": "Prophet",
    "train_observations": len(prophet_train),
    "test_observations": len(prophet_test),
    "mae": prophet_mae,
    "rmse": prophet_rmse,
    "mape_nonzero_actual_pct": prophet_mape,
    "test_hours_with_nonzero_actual": int(nonzero_actual.sum()),
    "test_hours_with_zero_actual": int((~nonzero_actual).sum())
}])

print("\nProphet evaluation results:")
print(prophet_metrics.to_string(index=False))

# Save actual values and predictions.
prophet_forecast_df = pd.DataFrame({
    "actual_orders": test,
    "prophet_forecast": prophet_predictions
})

prophet_forecast_df.to_csv(
    PROCESSED_DIR / "prophet_forecast.csv",
    index_label="order_hour"
)

prophet_metrics.to_csv(
    PROCESSED_DIR / "prophet_model_metrics.csv",
    index=False
)

# Plot actual demand against Prophet predictions.
plt.figure(figsize=(15, 6))
plt.plot(
    test.index,
    test.values,
    label="Actual orders",
    linewidth=1.5
)
plt.plot(
    prophet_predictions.index,
    prophet_predictions.values,
    label="Prophet forecast",
    linewidth=1.5
)

plt.title("Prophet Hourly Demand Forecast — Final 14 Days")
plt.xlabel("Date and hour")
plt.ylabel("Number of orders")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "prophet_forecast.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("\nFiles saved:")
print("- data/processed/prophet_forecast.csv")
print("- data/processed/prophet_model_metrics.csv")
print("- reports/figures/prophet_forecast.png")
print("\nProphet forecasting section completed.")


# ============================================================
# PART 7: MODEL COMPARISON AND FINAL DEMAND FORECAST
# ============================================================

print("\n" + "=" * 60)
print("PART 7: MODEL COMPARISON AND FINAL DEMAND FORECAST")
print("=" * 60)

# Combine model evaluation results from the same test period.
comparison = pd.concat(
    [arima_metrics, prophet_metrics],
    ignore_index=True
)

comparison = comparison.sort_values("mae").reset_index(drop=True)

comparison.to_csv(
    PROCESSED_DIR / "model_comparison.csv",
    index=False
)

print("\nARIMA vs Prophet comparison:")
print(
    comparison[
        ["model", "mae", "rmse", "mape_nonzero_actual_pct"]
    ].to_string(index=False)
)

# Select the model with the lower MAE.
# Prophet currently has a marginally lower MAE.
best_model_name = comparison.loc[0, "model"]
print(f"\nSelected model based on MAE: {best_model_name}")

# Refit Prophet on the full available dataset for future forecasting.
# The held-out test results above remain the unbiased comparison results.
full_prophet_data = prophet_data[["ds", "y"]].copy()

final_prophet_model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False,
    interval_width=0.95
)

print("\nRefitting Prophet on all available hourly observations...")
final_prophet_model.fit(full_prophet_data)

# Forecast the next 24 hours after the final observed timestamp.
future_24h = final_prophet_model.make_future_dataframe(
    periods=24,
    freq="h",
    include_history=False
)

future_forecast = final_prophet_model.predict(future_24h)

final_forecast = future_forecast[
    ["ds", "yhat", "yhat_lower", "yhat_upper"]
].copy()

# Order demand cannot be negative.
final_forecast["yhat"] = final_forecast["yhat"].clip(lower=0)
final_forecast["yhat_lower"] = final_forecast["yhat_lower"].clip(lower=0)
final_forecast["yhat_upper"] = final_forecast["yhat_upper"].clip(lower=0)

# Save the future forecast.
final_forecast.to_csv(
    PROCESSED_DIR / "final_demand_forecast.csv",
    index=False
)

print("\nNext 24-hour demand forecast:")
print(final_forecast.to_string(index=False))

print(
    "\nTotal predicted orders over the next 24 hours: "
    f"{final_forecast['yhat'].sum():.2f}"
)

# Plot the future forecast and its uncertainty interval.
plt.figure(figsize=(14, 6))

plt.plot(
    final_forecast["ds"],
    final_forecast["yhat"],
    label="Predicted demand",
    linewidth=2
)

plt.fill_between(
    final_forecast["ds"],
    final_forecast["yhat_lower"],
    final_forecast["yhat_upper"],
    alpha=0.2,
    label="95% prediction interval"
)

plt.title("Final Hourly Demand Forecast — Next 24 Hours")
plt.xlabel("Date and hour")
plt.ylabel("Predicted order count")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "final_demand_forecast.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# Create a business interpretation report.
report = f"""# Day 5: Hourly Demand Forecasting Report

## Objective
Forecast hourly order demand using ARIMA and Prophet.

## Data overview
- Total orders: {int(hourly["order_count"].sum())}
- Hourly observations: {len(hourly)}
- Data start: {hourly.index.min()}
- Data end: {hourly.index.max()}
- Training observations: {len(train)}
- Test observations: {len(test)}
- Test period: {test.index.min()} to {test.index.max()}

## Stationarity
The Augmented Dickey-Fuller test returned a statistic of
{adf_results[0]["adf_statistic"]:.4f} and a p-value of
{adf_results[0]["p_value"]:.6g}. At the 5% significance level,
the unit-root null hypothesis was rejected. The initial ARIMA
model therefore used d=0.

## Model comparison
{comparison[["model", "mae", "rmse", "mape_nonzero_actual_pct"]].to_string(index=False)}

MAPE is calculated only for test hours with nonzero actual demand,
because percentage error is undefined when actual demand is zero.

## Selected model
{best_model_name} was selected using MAE as the primary metric.
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
"""

report_path = ROOT / "docs" / "day5_forecasting_report.md"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(report, encoding="utf-8")

print("\nFinal files saved:")
print("- data/processed/model_comparison.csv")
print("- data/processed/final_demand_forecast.csv")
print("- reports/figures/final_demand_forecast.png")
print("- docs/day5_forecasting_report.md")
print("\nDay 5 forecasting workflow completed.")

