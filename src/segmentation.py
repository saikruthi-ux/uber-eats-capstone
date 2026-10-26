
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

for folder in [PROCESSED, FIGURES, MODELS]:
    folder.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the seven synthetic datasets."""
    names = [
        "customers", "restaurants", "orders", "deliveries",
        "drivers", "reviews", "payments"
    ]
    return {
        name: pd.read_csv(RAW / f"{name}.csv")
        for name in names
    }


def build_customer_features(data):
    """Aggregate order behavior into one row per customer."""
    customers = data["customers"].copy()
    orders = data["orders"].copy()
    reviews = data["reviews"].copy()

    orders["order_timestamp"] = pd.to_datetime(
        orders["order_timestamp"], errors="coerce"
    )
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )

    # Weekend: Saturday/Sunday. Late night: 10 PM–5:59 AM.
    orders["is_weekend"] = (
        orders["order_timestamp"].dt.day_of_week >= 5
    ).astype(int)

    hours = orders["order_timestamp"].dt.hour
    orders["is_late_night"] = (
        (hours >= 22) | (hours < 6)
    ).astype(int)

    # All orders contribute to order count and spending.
    order_features = orders.groupby("customer_id").agg(
        total_orders=("order_id", "nunique"),
        total_spending=("order_amount", "sum"),
        average_order_value=("order_amount", "mean"),
        weekend_orders=("is_weekend", "sum"),
        late_night_orders=("is_late_night", "sum"),
    ).reset_index()

    # The data covers January–September 2026 (nine calendar months).
    # This is orders per observed calendar month, not per active month.
    order_features["ordering_frequency"] = (
        order_features["total_orders"] / 9
    )

    # Link review ratings to the customer who placed the order.
    review_data = reviews.merge(
        orders[["order_id", "customer_id"]],
        on="order_id",
        how="left",
        validate="many_to_one",
    )
    rating_features = review_data.groupby("customer_id").agg(
        average_rating_given=("rating", "mean")
    ).reset_index()

    features = (
        customers
        .merge(order_features, on="customer_id", how="left")
        .merge(rating_features, on="customer_id", how="left")
    )

    count_cols = [
        "total_orders", "weekend_orders", "late_night_orders"
    ]
    for col in count_cols:
        features[col] = features[col].fillna(0)

    features["total_spending"] = features["total_spending"].fillna(0)
    features["average_order_value"] = features[
        "average_order_value"
    ].fillna(0)
    features["ordering_frequency"] = features[
        "ordering_frequency"
    ].fillna(0)

    features["signup_date"] = pd.to_datetime(
        features["signup_date"], errors="coerce"
    )
    features["age"] = pd.to_numeric(features["age"], errors="coerce")

    # Keep IDs for reporting, but exclude them from model features.
    features.to_csv(
        PROCESSED / "customer_features.csv", index=False
    )
    return features


def build_restaurant_features(data):
    """Aggregate orders, reviews and deliveries per restaurant."""
    restaurants = data["restaurants"].copy()
    orders = data["orders"].copy()
    deliveries = data["deliveries"].copy()
    reviews = data["reviews"].copy()

    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )

    order_features = orders.groupby("restaurant_id").agg(
        total_orders=("order_id", "nunique"),
        revenue=("order_amount", "sum"),
        cancelled_orders=(
            "order_status",
            lambda s: (s == "Cancelled").sum(),
        ),
    ).reset_index()

    order_features["cancellation_rate"] = np.where(
        order_features["total_orders"] > 0,
        order_features["cancelled_orders"]
        / order_features["total_orders"],
        0,
    )

    # Average customer review rating, mapped through order_id.
    restaurant_reviews = (
        reviews
        .merge(
            orders[["order_id", "restaurant_id"]],
            on="order_id",
            how="left",
            validate="many_to_one",
        )
        .groupby("restaurant_id")
        .agg(average_rating=("rating", "mean"))
        .reset_index()
    )

    restaurant_deliveries = (
        deliveries
        .merge(
            orders[["order_id", "restaurant_id"]],
            on="order_id",
            how="left",
            validate="many_to_one",
        )
        .groupby("restaurant_id")
        .agg(
            average_delivery_time=(
                "delivery_time_minutes", "mean"
            )
        )
        .reset_index()
    )

    features = (
        restaurants
        .merge(order_features, on="restaurant_id", how="left")
        .merge(restaurant_reviews, on="restaurant_id", how="left")
        .merge(
            restaurant_deliveries,
            on="restaurant_id",
            how="left",
        )
    )

    # Retain the supplied restaurant rating as a separate feature.
    features["catalog_rating"] = features["rating"]
    features = features.rename(
        columns={"avg_prep_time": "average_prep_time"}
    )

    features["total_orders"] = features["total_orders"].fillna(0)
    features["revenue"] = features["revenue"].fillna(0)
    features["cancelled_orders"] = features[
        "cancelled_orders"
    ].fillna(0)
    features["cancellation_rate"] = features[
        "cancellation_rate"
    ].fillna(0)

    features.to_csv(
        PROCESSED / "restaurant_features.csv", index=False
    )
    return features


def prepare_model_matrix(features, entity):
    """Select behavioral features and encode categorical context."""
    if entity == "customer":
        numeric_cols = [
            "age",
            "total_orders",
            "average_order_value",
            "total_spending",
            "ordering_frequency",
            "average_rating_given",
            "weekend_orders",
            "late_night_orders",
        ]
        categorical_cols = ["city"]
        id_col = "customer_id"
    else:
        numeric_cols = [
            "total_orders",
            "revenue",
            "average_rating",
            "catalog_rating",
            "average_prep_time",
            "cancellation_rate",
            "average_delivery_time",
        ]
        categorical_cols = ["category"]
        id_col = "restaurant_id"

    # One-hot encode city/category. Do not train on IDs or dates.
    model_data = features[numeric_cols + categorical_cols].copy()
    model_data = pd.get_dummies(
        model_data,
        columns=categorical_cols,
        dtype=int,
    )

    # Explicit missing-value treatment is performed by the pipeline.
    model_data = model_data.replace([np.inf, -np.inf], np.nan)

    # All-missing columns cannot be median-imputed; remove them.
    model_data = model_data.dropna(axis=1, how="all")

    return id_col, model_data


def name_customer_segment(row, medians):
    high_orders = row["average_orders"] >= medians["average_orders"]
    high_spending = row["average_spending"] >= medians["average_spending"]
    high_rating = row["average_rating_given"] >= medians["average_rating_given"]

    if high_orders and high_spending:
        return "High-Value Frequent Customers"
    elif high_orders:
        return "Frequent Orderers"
    elif high_spending:
        return "Occasional High-Spend Customers"
    elif high_rating:
        return "Lower-Activity, Higher-Rating Customers"
    else:
        return "Lower-Activity Customers"


def name_restaurant_segment(row, medians):
    high_revenue = row["average_revenue"] >= medians["average_revenue"]
    high_rating = row["average_rating"] >= medians["average_rating"]
    high_cancellation = (
        row["average_cancellation_rate"]
        >= medians["average_cancellation_rate"]
    )
    fast_prep = row["average_prep_time"] < medians["average_prep_time"]

    if high_cancellation and high_revenue:
        return "High-Revenue Cancellation Watchlist"
    elif high_cancellation and not high_rating:
        return "Cancellation and Rating Watchlist"
    elif high_cancellation:
        return "Cancellation Review Candidates"
    elif high_rating:
        return "Higher-Rated Restaurants"
    elif fast_prep:
        return "Faster-Preparation Restaurants"
    elif high_revenue:
        return "Revenue Leaders"
    else:
        return "Steady-Performance Restaurants"


def run_segmentation(features, entity):
    """Fit K-Means and DBSCAN, evaluate, plot and save outputs."""
    id_col, X = prepare_model_matrix(features, entity)
    ids = features[id_col].to_numpy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    # K-Means: evaluate k=2..8 using silhouette score.
    max_k = min(8, len(X) - 1)
    candidate_ks = range(2, max_k + 1)
    inertias = []
    silhouettes = []

    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        inertias.append(model.inertia_)

        if k >= 2:
            silhouettes.append(
                silhouette_score(X_scaled, labels)
            )

    best_k = list(candidate_ks)[int(np.argmax(silhouettes))]
    kmeans = KMeans(
        n_clusters=best_k, random_state=42, n_init=10
    )
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

    # Save reusable preprocessing and K-Means model.
    kmeans_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(
            n_clusters=best_k, random_state=42, n_init=10
        )),
    ])
    kmeans_pipeline.fit(X)
    joblib.dump(
        {"pipeline": kmeans_pipeline, "feature_columns": X.columns.tolist()},
        MODELS / f"{entity}_kmeans.joblib",
    )

    # Elbow plot.
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title(f"{entity.title()} K-Means Elbow Method")
    plt.tight_layout()
    plt.savefig(
        FIGURES / f"{entity}_elbow.png", dpi=150
    )
    plt.close()

    # Silhouette scores by k.
    plt.figure(figsize=(8, 5))
    plt.plot(list(candidate_ks), silhouettes, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title(f"{entity.title()} K-Means Silhouette Scores")
    plt.tight_layout()
    plt.savefig(
        FIGURES / f"{entity}_silhouette.png", dpi=150
    )
    plt.close()

    # PCA visualization of the K-Means clusters.
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        components[:, 0],
        components[:, 1],
        c=kmeans_labels,
        cmap="tab10",
        alpha=0.75,
        s=25,
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"{entity.title()} K-Means Clusters (PCA)")
    plt.colorbar(scatter, label="K-Means cluster")
    plt.tight_layout()
    plt.savefig(
        FIGURES / f"{entity}_pca_clusters.png", dpi=150
    )
    plt.close()


    # DBSCAN: test multiple parameter combinations.
    dbscan_candidates = []

    for eps in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        for min_samples in [3, 4, 5, 6, 8]:
            candidate = DBSCAN(
                eps=eps,
                min_samples=min_samples
            )
            labels = candidate.fit_predict(X_scaled)

            cluster_ids = set(labels) - {-1}
            n_clusters = len(cluster_ids)
            noise_fraction = float(np.mean(labels == -1))
            score = np.nan

            non_noise = labels != -1

            if (
                n_clusters >= 2
                and non_noise.sum() > n_clusters
            ):
                score = silhouette_score(
                    X_scaled[non_noise],
                    labels[non_noise]
                )

            dbscan_candidates.append({
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "noise_percentage": noise_fraction * 100,
                "silhouette_excluding_noise": score,
                "labels": labels,
            })

    candidate_report = pd.DataFrame([
        {
            key: value
            for key, value in item.items()
            if key != "labels"
        }
        for item in dbscan_candidates
    ])

    print("\nDBSCAN PARAMETER SEARCH")
    print(candidate_report.round(3).to_string(index=False))

    valid_candidates = [
        item for item in dbscan_candidates
        if item["n_clusters"] >= 2
        and np.isfinite(item["silhouette_excluding_noise"])
        and item["noise_percentage"] <= 30
    ]

    if valid_candidates:
        best_dbscan = max(
            valid_candidates,
            key=lambda item: (
                item["silhouette_excluding_noise"],
                -item["noise_percentage"]
            )
        )

        dbscan_labels = best_dbscan["labels"]
        best_eps = best_dbscan["eps"]
        best_min_samples = best_dbscan["min_samples"]
        dbscan_silhouette = best_dbscan[
            "silhouette_excluding_noise"
        ]

        print(
            f"\nSelected DBSCAN parameters: "
            f"eps={best_eps}, min_samples={best_min_samples}"
        )
    else:
        best_dbscan = min(
            dbscan_candidates,
            key=lambda item: item["noise_percentage"]
        )

        dbscan_labels = best_dbscan["labels"]
        best_eps = best_dbscan["eps"]
        best_min_samples = best_dbscan["min_samples"]
        dbscan_silhouette = np.nan

        print(
            "\nNo tested DBSCAN configuration formed at least "
            "two valid clusters. Report this limitation honestly."
        )

    dbscan_cluster_count = len(set(dbscan_labels) - {-1})
    noise_pct = float(np.mean(dbscan_labels == -1) * 100)

    # Save a pipeline using the selected DBSCAN parameters.
    dbscan_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("dbscan", DBSCAN(
            eps=best_eps,
            min_samples=best_min_samples
        )),
    ])

    dbscan_pipeline.fit(X)

    joblib.dump(
        {
            "pipeline": dbscan_pipeline,
            "feature_columns": X.columns.tolist(),
            "eps": best_eps,
            "min_samples": best_min_samples,
        },
        MODELS / f"{entity}_dbscan.joblib",
    )


    plt.figure(figsize=(8, 6))
    plt.scatter(
        components[:, 0],
        components[:, 1],
        c=dbscan_labels,
        cmap="tab10",
        alpha=0.75,
        s=25,
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"{entity.title()} DBSCAN Clusters (PCA; -1 = Noise)")
    plt.tight_layout()
    plt.savefig(
        FIGURES / f"{entity}_dbscan_pca.png", dpi=150
    )
    plt.close()

    # Save record-level cluster assignments.
    output = features.copy()
    output["kmeans_cluster"] = kmeans_labels
    output["dbscan_cluster"] = dbscan_labels

    # Generate transparent, profile-based business labels.
    if entity == "customer":
        profiles = output.groupby("kmeans_cluster").agg(
            customer_count=(id_col, "count"),
            average_orders=("total_orders", "mean"),
            average_order_value=("average_order_value", "mean"),
            average_spending=("total_spending", "mean"),
            average_rating_given=("average_rating_given", "mean"),
            weekend_orders=("weekend_orders", "mean"),
            late_night_orders=("late_night_orders", "mean"),
        )
        medians = profiles[
            [
                "average_orders",
                "average_spending",
                "average_rating_given",
            ]
        ].median()
        names = {
            cluster: name_customer_segment(row, medians)
            for cluster, row in profiles.iterrows()
        }
    
    else:
        profiles = output.groupby("kmeans_cluster").agg(
            restaurant_count=(id_col, "count"),
            average_orders=("total_orders", "mean"),
            average_revenue=("revenue", "mean"),
            average_rating=("average_rating", "mean"),
            average_prep_time=("average_prep_time", "mean"),
            average_cancellation_rate=("cancellation_rate", "mean"),
            average_delivery_time=("average_delivery_time", "mean"),
        )

        medians = profiles[
            [
                "average_revenue",
                "average_rating",
                "average_cancellation_rate",
                "average_prep_time",
            ]
        ].median()

        names = {
            cluster: name_restaurant_segment(row, medians)
            for cluster, row in profiles.iterrows()
        }

    output["business_segment"] = output["kmeans_cluster"].map(names)
    output.to_csv(
        PROCESSED / f"{entity}_clusters.csv", index=False
    )

    profiles = profiles.reset_index()
    profiles["business_segment"] = profiles[
        "kmeans_cluster"
    ].map(names)
    profiles.to_csv(
        PROCESSED / f"{entity}_cluster_profiles.csv",
        index=False,
    )

    metrics = {
        "entity": entity,
        "selected_k": best_k,
        "kmeans_silhouette": kmeans_silhouette,
        "dbscan_cluster_count": dbscan_cluster_count,
        "dbscan_noise_percentage": noise_pct,
        "dbscan_silhouette_excluding_noise": dbscan_silhouette,
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "number_of_model_features": X.shape[1],
        "records": len(X),
    }

    print(f"\n{entity.upper()} SEGMENTATION")
    print(f"Records: {len(X)}")
    print(f"Model features (IDs excluded): {X.shape[1]}")
    print(f"Selected K-Means k: {best_k}")
    print(f"K-Means silhouette score: {kmeans_silhouette:.3f}")
    print(f"DBSCAN clusters (excluding noise): {dbscan_cluster_count}")
    print(f"DBSCAN noise: {noise_pct:.1f}%")
    print(f"PCA variance explained by 2 components: {metrics['pca_variance_explained']:.1%}")
    print("\nK-Means cluster profiles:")
    print(profiles.to_string(index=False))

    return metrics


def main():
    data = load_data()

    customer_features = build_customer_features(data)
    restaurant_features = build_restaurant_features(data)

    metrics = [
        run_segmentation(customer_features, "customer"),
        run_segmentation(restaurant_features, "restaurant"),
    ]

    pd.DataFrame(metrics).to_csv(
        PROCESSED / "segmentation_metrics.csv", index=False
    )

    print("\nSegmentation complete.")
    print(f"Processed outputs: {PROCESSED}")
    print(f"Figures: {FIGURES}")
    print(f"Models: {MODELS}")


if __name__ == "__main__":
    main()
