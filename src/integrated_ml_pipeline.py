
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

# Existing outputs from Days 2–5
FILES = {
    "segmentation_metrics": "segmentation_metrics.csv",
    "sentiment_models": "sentiment_model_comparison.csv",
    "regression_models": "regression_model_comparison.csv",
    "delivery_importance": "regression_feature_importance.csv",
    "customers": "customer_cluster_profiles.csv",
    "restaurants": "restaurant_cluster_profiles.csv",
    "complaints": "negative_complaint_themes.csv",
    "forecasts": "model_comparison.csv",
}

# Stage 1: validate and load artifacts
data = {}
inventory = []

for name, filename in FILES.items():
    path = DATA / filename
    if path.exists():
        try:
            df = pd.read_csv(path)
            data[name] = df
            inventory.append({
                "artifact": name,
                "file": filename,
                "status": "LOADED",
                "rows": len(df),
                "columns": len(df.columns),
            })
        except Exception as exc:
            inventory.append({
                "artifact": name, "file": filename,
                "status": f"READ ERROR: {exc}",
                "rows": None, "columns": None,
            })
    else:
        inventory.append({
            "artifact": name, "file": filename,
            "status": "MISSING", "rows": None, "columns": None,
        })

pd.DataFrame(inventory).to_csv(
    DATA / "integrated_pipeline_artifact_inventory.csv",
    index=False,
)

missing = [name for name in FILES if name not in data]
if missing:
    print("Missing or unreadable artifacts:", missing)
    print(pd.DataFrame(inventory).to_string(index=False))
    raise SystemExit("Resolve missing artifacts before continuing.")

# Stage 2: consolidated, task-specific model comparison
comparison = []

for _, row in data["sentiment_models"].iterrows():
    comparison.append({
        "task": "Sentiment classification",
        "model": row["model"],
        "primary_metric": "weighted_f1",
        "primary_score": row["weighted_f1"],
        "additional_metric": "accuracy",
        "additional_score": row["accuracy"],
    })

for _, row in data["regression_models"].iterrows():
    comparison.append({
        "task": "Delivery-time prediction",
        "model": row["model"],
        "primary_metric": "test_rmse_minutes",
        "primary_score": row["test_rmse"],
        "additional_metric": "test_mae_minutes",
        "additional_score": row["test_mae"],
    })

for _, row in data["forecasts"].iterrows():
    comparison.append({
        "task": "Hourly demand forecasting",
        "model": row["model"],
        "primary_metric": "test_mae_orders_per_hour",
        "primary_score": row["mae"],
        "additional_metric": "test_rmse_orders_per_hour",
        "additional_score": row["rmse"],
    })

comparison_df = pd.DataFrame(comparison)
comparison_df.to_csv(
    DATA / "integrated_model_comparison.csv",
    index=False,
)

# Stage 3: derive insights from the complete saved tables
insights = []

# Segmentation
for _, row in data["segmentation_metrics"].iterrows():
    insights.append(
        f"- **{row['entity'].title()} segmentation:** "
        f"KMeans k={row['selected_k']}, silhouette="
        f"{row['kmeans_silhouette']:.3f}, records={row['records']}. "
        "Treat clusters as exploratory groups and validate their business value."
    )

# Customer segment with highest average spending
customers = data["customers"].copy()
customers["average_spending"] = pd.to_numeric(
    customers["average_spending"], errors="coerce"
)
customers = customers.dropna(subset=["average_spending"])

if not customers.empty:
    top = customers.loc[customers["average_spending"].idxmax()]
    insights.append(
        f"- **Customer retention:** cluster {top['kmeans_cluster']} "
        f"({top['business_segment']}) has the highest average spending "
        f"({top['average_spending']:.2f}) and averages "
        f"{top['average_orders']:.2f} orders. Test retention initiatives "
        "and measure repeat orders and margin."
    )

# Restaurant cluster with the highest cancellation rate
restaurants = data["restaurants"].copy()
restaurants["average_cancellation_rate"] = pd.to_numeric(
    restaurants["average_cancellation_rate"], errors="coerce"
)
restaurants = restaurants.dropna(subset=["average_cancellation_rate"])

if not restaurants.empty:
    worst = restaurants.loc[
        restaurants["average_cancellation_rate"].idxmax()
    ]
    rate = worst["average_cancellation_rate"]
    # Convert to percent only when the source stores a proportion.
    rate_text = f"{rate * 100:.1f}%" if rate <= 1 else f"{rate:.1f}"
    insights.append(
        f"- **Restaurant intervention:** cluster {worst['kmeans_cluster']} "
        f"has the highest average cancellation rate ({rate_text} in the "
        "source scale). Investigate stock availability, preparation times, "
        "and cancellation reasons before taking action."
    )

# Sentiment model with highest weighted F1
sentiment = data["sentiment_models"].copy()
sentiment["weighted_f1"] = pd.to_numeric(
    sentiment["weighted_f1"], errors="coerce"
)
sentiment = sentiment.dropna(subset=["weighted_f1"])

if not sentiment.empty:
    best = sentiment.loc[sentiment["weighted_f1"].idxmax()]
    insights.append(
        f"- **Sentiment analysis:** {best['model']} has the highest "
        f"weighted F1 ({best['weighted_f1']:.3f}) among the evaluated "
        "models. Results are preliminary because the reviews contain "
        "only 15 unique templates and labels may be rating-derived."
    )

# Best delivery-time regression model by test RMSE
regression = data["regression_models"].copy()
regression["test_rmse"] = pd.to_numeric(
    regression["test_rmse"], errors="coerce"
)
regression = regression.dropna(subset=["test_rmse"])

if not regression.empty:
    best = regression.loc[regression["test_rmse"].idxmin()]
    insights.append(
        f"- **Delivery-time prediction:** {best['model']} has the lowest "
        f"test RMSE ({best['test_rmse']:.2f} minutes), test MAE "
        f"{best['test_mae']:.2f} minutes, and R² {best['test_r2']:.3f}."
    )

# Highest-ranked reported delivery-time features
importance = data["delivery_importance"].copy()
importance["importance_mean"] = pd.to_numeric(
    importance["importance_mean"], errors="coerce"
)
importance = importance.dropna(subset=["importance_mean"])
importance = importance.sort_values(
    "importance_mean", ascending=False
).head(3)

if not importance.empty:
    features = ", ".join(
        f"{row['feature']} ({row['importance_mean']:.2f})"
        for _, row in importance.iterrows()
    )
    insights.append(
        f"- **Reported delivery-time features:** {features}. "
        "Importance indicates predictive contribution, not causation."
    )

# Frequent terms in negative reviews
complaints = data["complaints"].copy()
complaints["count"] = pd.to_numeric(
    complaints["count"], errors="coerce"
)
complaints = complaints.dropna(subset=["count"]).sort_values(
    "count", ascending=False
).head(5)

if not complaints.empty:
    terms = ", ".join(
        f"{row['term_or_phrase']} ({int(row['count'])})"
        for _, row in complaints.iterrows()
    )
    insights.append(
        f"- **Negative-review terms:** {terms}. These are term counts, "
        "not verified root causes. Review complete comments and account "
        "for repeated templates."
    )

# Demand forecasting: MAE and RMSE can favor different models
forecast = data["forecasts"].copy()
forecast["mae"] = pd.to_numeric(forecast["mae"], errors="coerce")
forecast["rmse"] = pd.to_numeric(forecast["rmse"], errors="coerce")
forecast = forecast.dropna(subset=["mae", "rmse"])

if not forecast.empty:
    mae_best = forecast.loc[forecast["mae"].idxmin()]
    rmse_best = forecast.loc[forecast["rmse"].idxmin()]
    insights.append(
        f"- **Demand forecasting:** {mae_best['model']} has the lowest "
        f"MAE ({mae_best['mae']:.3f} orders/hour); "
        f"{rmse_best['model']} has the lowest RMSE "
        f"({rmse_best['rmse']:.3f}). Results are close, so compare "
        "forecast stability and operational usefulness."
    )

# Stage 4: produce the Day 6 report
report = [
    "# Day 6 – Integrated ML Pipeline & Business Insights",
    "",
    "## Objective",
    "Combine the saved results from segmentation, sentiment analysis, "
    "delivery-time prediction, and hourly demand forecasting.",
    "",
    "## End-to-end workflow",
    "Raw data → Cleaning and validation → Feature engineering → Modeling "
    "→ Evaluation → Prediction → Business insight",
    "",
    "This integration step consumes saved artifacts. It does not retrain "
    "the component models or regenerate raw data.",
    "",
    "## Model comparison",
    "See `../data/processed/integrated_model_comparison.csv`. Metrics are "
    "task-specific and must not be compared across different task types.",
    "",
    "## Business insights",
    "",
    *insights,
    "",
    "## Recommendations",
    "",
    "1. **Customers:** test loyalty and reorder campaigns for high-spending "
    "segments; measure incremental orders and profitability.",
    "2. **Restaurants:** investigate high-cancellation clusters using "
    "restaurant-level records and operational reasons.",
    "3. **Customer experience:** review complete negative comments before "
    "deciding which issues to address.",
    "4. **Delivery operations:** use delivery-time predictions as estimates "
    "and consider distance, traffic, and weather when planning dispatch.",
    "5. **Capacity planning:** use hourly forecasts to plan flexible "
    "courier capacity, then compare forecasts against actual demand.",
    "",
    "## Limitations",
    "",
    "- Segmentation silhouette scores are moderate and do not prove "
    "commercial value.",
    "- Sentiment results may not generalize because there are only 15 "
    "unique review templates and labels may be rating-derived.",
    "- Evaluation scores depend on the original data splits and may not "
    "represent future production performance.",
    "- Feature importance does not establish causal relationships.",
    "- Frequent complaint terms are not necessarily root causes.",
    "- Synthetic data and limited time coverage may not represent real "
    "marketplace behavior.",
    "- This project predicts delivery time; it does not build a separate "
    "tip-prediction model.",
    "",
    "## Future improvements",
    "",
    "- Add automated data validation and schema tests.",
    "- Validate segments against repeat orders, margins, and outcomes.",
    "- Evaluate sentiment using more varied reviews and reviewed labels.",
    "- Validate predictions on future data and monitor model drift.",
    "- Compare forecast errors by hour, weekday, and location where data "
    "supports those breakdowns.",
    "",
    "## Artifact validation",
    "See `../data/processed/integrated_pipeline_artifact_inventory.csv`.",
]

report_path = DOCS / "day6_integrated_ml_report.md"
report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

print("Day 6 integrated pipeline completed.")
print("\nArtifact inventory:")
print(pd.DataFrame(inventory).to_string(index=False))
print("\nIntegrated model comparison:")
print(comparison_df.to_string(index=False))
print(f"\nReport saved to: {report_path}")
print(f"Model comparison saved to: {DATA / 'integrated_model_comparison.csv'}")
print(f"Artifact inventory saved to: "
      f"{DATA / 'integrated_pipeline_artifact_inventory.csv'}")

