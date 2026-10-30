# Day 6 – Integrated ML Pipeline & Business Insights

## Objective
Combine the saved results from segmentation, sentiment analysis, delivery-time prediction, and hourly demand forecasting.

## End-to-end workflow
Raw data → Cleaning and validation → Feature engineering → Modeling → Evaluation → Prediction → Business insight

This integration step consumes saved artifacts. It does not retrain the component models or regenerate raw data.

## Model comparison
See `../data/processed/integrated_model_comparison.csv`. Metrics are task-specific and must not be compared across different task types.

## Business insights

- **Customer segmentation:** KMeans k=6, silhouette=0.262, records=1000. Treat clusters as exploratory groups and validate their business value.
- **Restaurant segmentation:** KMeans k=7, silhouette=0.305, records=100. Treat clusters as exploratory groups and validate their business value.
- **Customer retention:** cluster 3 (High-Value Frequent Customers) has the highest average spending (6949.61) and averages 8.47 orders. Test retention initiatives and measure repeat orders and margin.
- **Restaurant intervention:** cluster 1 has the highest average cancellation rate (22.0% in the source scale). Investigate stock availability, preparation times, and cancellation reasons before taking action.
- **Sentiment analysis:** VADER has the highest weighted F1 (0.800) among the evaluated models. Results are preliminary because the reviews contain only 15 unique templates and labels may be rating-derived.
- **Delivery-time prediction:** Linear Regression has the lowest test RMSE (5.26 minutes), test MAE 4.17 minutes, and R² 0.867.
- **Reported delivery-time features:** delivery_distance_km (9.20), traffic_condition (4.79), weather (1.98). Importance indicates predictive contribution, not causation.
- **Negative-review terms:** food (113), experience (89), much (84), much better (84), could (84). These are term counts, not verified root causes. Review complete comments and account for repeated templates.
- **Demand forecasting:** Prophet has the lowest MAE (0.671 orders/hour); ARIMA(1, 0, 1) has the lowest RMSE (0.809). Results are close, so compare forecast stability and operational usefulness.

## Recommendations

1. **Customers:** test loyalty and reorder campaigns for high-spending segments; measure incremental orders and profitability.
2. **Restaurants:** investigate high-cancellation clusters using restaurant-level records and operational reasons.
3. **Customer experience:** review complete negative comments before deciding which issues to address.
4. **Delivery operations:** use delivery-time predictions as estimates and consider distance, traffic, and weather when planning dispatch.
5. **Capacity planning:** use hourly forecasts to plan flexible courier capacity, then compare forecasts against actual demand.

## Limitations

- Segmentation silhouette scores are moderate and do not prove commercial value.
- Sentiment results may not generalize because there are only 15 unique review templates and labels may be rating-derived.
- Evaluation scores depend on the original data splits and may not represent future production performance.
- Feature importance does not establish causal relationships.
- Frequent complaint terms are not necessarily root causes.
- Synthetic data and limited time coverage may not represent real marketplace behavior.
- This project predicts delivery time; it does not build a separate tip-prediction model.

## Future improvements

- Add automated data validation and schema tests.
- Validate segments against repeat orders, margins, and outcomes.
- Evaluate sentiment using more varied reviews and reviewed labels.
- Validate predictions on future data and monitor model drift.
- Compare forecast errors by hour, weekday, and location where data supports those breakdowns.

## Artifact validation
See `../data/processed/integrated_pipeline_artifact_inventory.csv`.
