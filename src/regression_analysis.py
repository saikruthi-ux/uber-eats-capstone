from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. Paths and reproducibility
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"

for folder in [PROCESSED, FIGURES, MODELS, DOCS]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "delivery_time_minutes"

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------
# 2. Load and join the datasets
# ---------------------------------------------------------
print("\n[1/9] Loading datasets...")

deliveries = pd.read_csv(RAW / "deliveries.csv")
orders = pd.read_csv(RAW / "orders.csv")
restaurants = pd.read_csv(RAW / "restaurants.csv")

# A delivery record should correspond to one order.
data = deliveries.merge(
    orders,
    on="order_id",
    how="left",
    validate="one_to_one",
    indicator="_order_merge",
)

if not (data["_order_merge"] == "both").all():
    raise ValueError("Some deliveries do not match an order.")

data = data.drop(columns="_order_merge")

data = data.merge(
    restaurants,
    on="restaurant_id",
    how="left",
    validate="many_to_one",
    indicator="_restaurant_merge",
)

if not (data["_restaurant_merge"] == "both").all():
    raise ValueError("Some orders do not match a restaurant.")

data = data.drop(columns="_restaurant_merge")

print("Joined dataset shape:", data.shape)
print("Delivery records:", len(data))
print("Missing values after joins:", int(data.isna().sum().sum()))


# ---------------------------------------------------------
# 3. Feature engineering and leakage prevention
# ---------------------------------------------------------
print("\n[2/9] Engineering features...")

data["order_timestamp"] = pd.to_datetime(
    data["order_timestamp"], errors="coerce"
)

if data["order_timestamp"].isna().any():
    raise ValueError("Some order timestamps could not be parsed.")

data["order_hour"] = data["order_timestamp"].dt.hour
data["order_day_of_week"] = data["order_timestamp"].dt.day_of_week
data["is_weekend"] = (
    data["order_timestamp"].dt.day_of_week >= 5
).astype(int)

# Keep only features intended to be available when estimating delivery time.
# Exclude delivery_time_minutes (target), tip_amount, and identifiers.
numeric_features = [
    "delivery_distance_km",
    "order_amount",
    "num_items",
    "rating",
    "avg_prep_time",
    "order_hour",
    "order_day_of_week",
    "is_weekend",
]

categorical_features = [
    "weather",
    "traffic_condition",
    "category",
]

feature_columns = numeric_features + categorical_features

model_data = data[feature_columns + [TARGET]].copy()

# Validate the modeling table.
if model_data.isna().any().any():
    missing = model_data.isna().sum()
    raise ValueError(f"Missing values in modeling table:\n{missing}")

if not np.isfinite(
    model_data[numeric_features + [TARGET]].to_numpy(dtype=float)
).all():
    raise ValueError("Non-finite numeric values found.")

model_data.to_csv(
    PROCESSED / "delivery_regression_dataset.csv",
    index=False,
)

print("Modeling dataset shape:", model_data.shape)
print("Features:", feature_columns)
print("Target:", TARGET)
print("Saved processed modeling dataset.")


# ---------------------------------------------------------
# 4. Exploratory data analysis
# ---------------------------------------------------------
print("\n[3/9] Creating EDA plots...")

# Target distribution
plt.figure(figsize=(9, 5))
sns.histplot(data=model_data, x=TARGET, kde=True, bins=25)
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery time (minutes)")
plt.ylabel("Number of deliveries")
plt.tight_layout()
plt.savefig(FIGURES / "delivery_time_distribution.png", dpi=150)
plt.close()

# Target outliers
plt.figure(figsize=(9, 4))
sns.boxplot(x=model_data[TARGET])
plt.title("Delivery Time: Outlier Inspection")
plt.xlabel("Delivery time (minutes)")
plt.tight_layout()
plt.savefig(FIGURES / "delivery_time_boxplot.png", dpi=150)
plt.close()

# Distance versus delivery time
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=model_data,
    x="delivery_distance_km",
    y=TARGET,
    alpha=0.5,
)
sns.regplot(
    data=model_data,
    x="delivery_distance_km",
    y=TARGET,
    scatter=False,
    color="red",
)
plt.title("Delivery Distance vs Delivery Time")
plt.tight_layout()
plt.savefig(FIGURES / "distance_vs_delivery_time.png", dpi=150)
plt.close()

# Correlations among numeric variables
correlation_columns = numeric_features + [TARGET]
correlations = model_data[correlation_columns].corr(numeric_only=True)

correlations.to_csv(PROCESSED / "regression_correlations.csv")

plt.figure(figsize=(11, 8))
sns.heatmap(
    correlations,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
)
plt.title("Numeric Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(FIGURES / "regression_correlation_heatmap.png", dpi=150)
plt.close()

# Compare delivery time across traffic and weather categories
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.boxplot(
    data=model_data,
    x="traffic_condition",
    y=TARGET,
    ax=axes[0],
)
axes[0].set_title("Delivery Time by Traffic Condition")

sns.boxplot(
    data=model_data,
    x="weather",
    y=TARGET,
    ax=axes[1],
)
axes[1].set_title("Delivery Time by Weather")
axes[1].tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.savefig(FIGURES / "delivery_time_by_conditions.png", dpi=150)
plt.close()

print("EDA plots and correlation table saved.")


# ---------------------------------------------------------
# 5. Train/test split and preprocessing
# ---------------------------------------------------------
print("\n[4/9] Preparing training and test sets...")

X = model_data[feature_columns]
y = model_data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ]
)

print("Training records:", len(X_train))
print("Test records:", len(X_test))


# ---------------------------------------------------------
# 6. Train candidate models and run cross-validation
# ---------------------------------------------------------
print("\n[5/9] Training models and running cross-validation...")

models = {
    "Dummy Baseline": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=20000),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

scoring = {
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "r2": "r2",
}

results = []
fitted_models = {}

for name, estimator in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=1,
    )

    pipeline.fit(X_train, y_train)
    fitted_models[name] = pipeline

    test_predictions = pipeline.predict(X_test)

    test_mae = mean_absolute_error(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_r2 = r2_score(y_test, test_predictions)

    cv_mae = -cv_results["test_mae"].mean()
    cv_rmse = np.sqrt(-cv_results["test_mse"].mean())
    cv_r2 = cv_results["test_r2"].mean()

    train_mae = -cv_results["train_mae"].mean()
    train_rmse = np.sqrt(-cv_results["train_mse"].mean())
    train_r2 = cv_results["train_r2"].mean()

    results.append(
        {
            "model": name,
            "cv_mae": cv_mae,
            "cv_rmse": cv_rmse,
            "cv_r2": cv_r2,
            "cv_train_mae": train_mae,
            "cv_train_rmse": train_rmse,
            "cv_train_r2": train_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "mae_train_test_gap": test_mae - train_mae,
            "rmse_train_test_gap": test_rmse - train_rmse,
            "r2_train_test_gap": train_r2 - test_r2,
        }
    )

    print(
        f"{name:20s} | "
        f"CV MAE: {cv_mae:.3f} | "
        f"CV RMSE: {cv_rmse:.3f} | "
        f"CV R2: {cv_r2:.3f} | "
        f"Test MAE: {test_mae:.3f} | "
        f"Test R2: {test_r2:.3f}"
    )

comparison = pd.DataFrame(results).sort_values("cv_rmse")
comparison.to_csv(
    PROCESSED / "regression_model_comparison.csv",
    index=False,
)

print("\nModel comparison saved.")


# ---------------------------------------------------------
# 7. Select the best model and evaluate it
# ---------------------------------------------------------
print("\n[6/9] Selecting the model with the lowest CV RMSE...")

best_model_name = comparison.iloc[0]["model"]
best_model = fitted_models[best_model_name]
best_predictions = best_model.predict(X_test)

print("Selected model:", best_model_name)
print(
    "Selection criterion: lowest five-fold cross-validation RMSE."
)
print(f"Test MAE:  {mean_absolute_error(y_test, best_predictions):.3f}")
print(
    "Test RMSE: "
    f"{np.sqrt(mean_squared_error(y_test, best_predictions)):.3f}"
)
print(f"Test R2:   {r2_score(y_test, best_predictions):.3f}")

predictions = pd.DataFrame(
    {
        "actual_delivery_time_minutes": y_test.to_numpy(),
        "predicted_delivery_time_minutes": best_predictions,
    }
)
predictions["prediction_error_minutes"] = (
    predictions["predicted_delivery_time_minutes"]
    - predictions["actual_delivery_time_minutes"]
)
predictions.to_csv(
    PROCESSED / "delivery_time_predictions.csv",
    index=False,
)

# Actual vs predicted
plt.figure(figsize=(7, 6))
sns.scatterplot(
    x=y_test.to_numpy(),
    y=best_predictions,
    alpha=0.6,
)
min_value = min(y_test.min(), best_predictions.min())
max_value = max(y_test.max(), best_predictions.max())
plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
)
plt.xlabel("Actual delivery time (minutes)")
plt.ylabel("Predicted delivery time (minutes)")
plt.title(f"Actual vs Predicted — {best_model_name}")
plt.tight_layout()
plt.savefig(FIGURES / "actual_vs_predicted_delivery_time.png", dpi=150)
plt.close()

# Residual plot
residuals = y_test.to_numpy() - best_predictions

plt.figure(figsize=(8, 5))
sns.scatterplot(x=best_predictions, y=residuals, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted delivery time (minutes)")
plt.ylabel("Residual: actual - predicted (minutes)")
plt.title("Residual Analysis")
plt.tight_layout()
plt.savefig(FIGURES / "delivery_time_residuals.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# 8. Feature importance and save model
# ---------------------------------------------------------
print("\n[7/9] Calculating feature importance...")

importance_result = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=1,
)

importance = pd.DataFrame(
    {
        "feature": X_test.columns,
        "importance_mean": importance_result.importances_mean,
        "importance_std": importance_result.importances_std,
    }
).sort_values("importance_mean", ascending=False)

importance.to_csv(
    PROCESSED / "regression_feature_importance.csv",
    index=False,
)

plt.figure(figsize=(9, 6))
plot_importance = importance.sort_values("importance_mean")
sns.barplot(
    data=plot_importance,
    x="importance_mean",
    y="feature",
)
plt.xlabel("Increase in MAE when feature is shuffled (minutes)")
plt.ylabel("Feature")
plt.title(f"Permutation Feature Importance — {best_model_name}")
plt.tight_layout()
plt.savefig(FIGURES / "regression_feature_importance.png", dpi=150)
plt.close()

model_path = MODELS / "delivery_time_regression.joblib"
joblib.dump(best_model, model_path)

print("Saved model:", model_path)


# ---------------------------------------------------------
# 9. Write report and finish
# ---------------------------------------------------------
print("\n[8/9] Writing the Day 4 report...")

best_row = comparison.iloc[0]
test_mae = mean_absolute_error(y_test, best_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, best_predictions))
test_r2 = r2_score(y_test, best_predictions)

top_features = importance.head(5)

report = f"""# Day 4: Delivery Time Regression Analysis

## Objective
Predict delivery duration in minutes using order, restaurant, distance,
weather, traffic, and order-time features.

## Dataset
- Joined delivery records: {len(data):,}
- Modeling records: {len(model_data):,}
- Training records: {len(X_train):,}
- Test records: {len(X_test):,}
- Target: `{TARGET}`
- Target mean: {y.mean():.2f} minutes
- Target median: {y.median():.2f} minutes
- Target minimum / maximum: {y.min():.0f} / {y.max():.0f} minutes

## Feature engineering and leakage prevention
The delivery, order, and restaurant datasets were joined using their keys.
Order hour, day of week, and weekend status were derived from the order
timestamp. Numeric features were standardized for regression models, and
categorical features were one-hot encoded.

The target itself, identifiers, and `tip_amount` were excluded from the
predictors. Tip amount was excluded because it may not be known when the
delivery estimate is made. The available dataset does not establish whether
every other predictor is available at the exact time a real-world estimate
would be generated; this should be confirmed before deployment.

## Validation strategy
A fixed 80/20 train/test split was used. Five-fold shuffled cross-validation
was performed on the training set. Models were ranked by mean cross-validation
RMSE; lower RMSE is preferred. MAE, RMSE, and R-squared were also reported.

## Model comparison
See `data/processed/regression_model_comparison.csv` for all models.

Selected model: **{best_model_name}**

- Cross-validation MAE: {best_row['cv_mae']:.3f} minutes
- Cross-validation RMSE: {best_row['cv_rmse']:.3f} minutes
- Cross-validation R-squared: {best_row['cv_r2']:.3f}
- Test MAE: {test_mae:.3f} minutes
- Test RMSE: {test_rmse:.3f} minutes
- Test R-squared: {test_r2:.3f}

MAE describes the average absolute prediction error in minutes. RMSE gives
greater weight to larger errors. R-squared describes the proportion of
target variation explained relative to a mean-only reference, and may be
negative when predictions perform poorly.

## Overfitting checks
Compare the training-fold and validation-fold metrics in the comparison CSV,
alongside test performance. A large difference between training and validation
performance may indicate overfitting. The train/test gaps are diagnostic and
should be interpreted alongside the cross-validation scores.

## Most influential features
Permutation importance measures the change in test-set MAE when each feature
is shuffled. Larger positive values indicate that shuffling the feature made
the predictions worse.

{top_features[['feature', 'importance_mean', 'importance_std']].to_markdown(index=False)}

These are model-specific associations, not proof of causation.

## Business recommendations
1. Use the model as a decision-support estimate only after confirming that
   each input is available at prediction time.
2. Review the MAE in minutes to decide whether model accuracy is adequate for
   customer-facing ETA communication.
3. Investigate systematic underprediction or overprediction using the
   residual plot before operational use.
4. Monitor error by traffic, weather, distance, and time-of-day groups to
   identify conditions where predictions are less reliable.
5. Reassess the model as new delivery records become available.

## Limitations
- The data is synthetic and may not represent real-world delivery behavior.
- The available dataset contains only records with delivery information.
- Performance on a random holdout may differ from performance on future dates.
- Feature importance indicates predictive usefulness, not causal impact.
- Model metrics must be reviewed before the system is used for live estimates.

## Deliverables
- `data/processed/delivery_regression_dataset.csv`
- `data/processed/regression_correlations.csv`
- `data/processed/regression_model_comparison.csv`
- `data/processed/delivery_time_predictions.csv`
- `data/processed/regression_feature_importance.csv`
- `models/delivery_time_regression.joblib`
- EDA, actual-versus-predicted, residual, and feature-importance figures
"""

(DOCS / "day4_regression_report.md").write_text(report, encoding="utf-8")

print("Report saved:", DOCS / "day4_regression_report.md")
print("\n[9/9] Day 4 regression pipeline completed successfully.")
print("\nTop model results:")
print(
    comparison[
        ["model", "cv_mae", "cv_rmse", "cv_r2", "test_mae", "test_rmse", "test_r2"]
    ].round(3).to_string(index=False)
)
print("\nTop features:")
print(importance.head(5).round(3).to_string(index=False))
