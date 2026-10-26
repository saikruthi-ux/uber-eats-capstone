# Day 2: Customer and Restaurant Segmentation

## 1. Objective

Segment customers and restaurants using operational and behavioral
features to identify patterns that can support business decisions.

The analysis uses synthetic Uber Eats project data.

## 2. Data Preparation

Customer and restaurant feature datasets were created from the raw
customer, order, delivery, restaurant, and review data.

Preparation included:
- Feature engineering for customer purchasing behavior.
- Feature engineering for restaurant performance.
- Missing-value imputation using the median.
- One-hot encoding of categorical variables.
- StandardScaler for feature scaling.
- Exclusion of identifier columns from model training.

## 3. Customer Segmentation

- Records analyzed: 1,000
- Model features: 13
- Selected K-Means clusters: 6
- K-Means silhouette score: 0.262
- DBSCAN clusters: 5
- DBSCAN noise: 9.0%
- DBSCAN parameters: eps=2.0, min_samples=8
- PCA variance explained by two components: 38.5%

### Customer Business Groups

- High-Value Frequent Customers: 483 customers
- Lower-Activity, Higher-Rating Customers: 346 customers
- Lower-Activity Customers: 171 customers

### Interpretation

One K-Means cluster has notably higher average order frequency
and spending than the other customer clusters. This may represent
an opportunity for loyalty benefits and customer retention campaigns.

The business labels group clusters with similar characteristics;
they do not replace the six original K-Means cluster IDs.

## 4. Restaurant Segmentation

- Records analyzed: 100
- Model features: 14
- Selected K-Means clusters: 7
- K-Means silhouette score: 0.305
- DBSCAN clusters: 6
- DBSCAN noise: 19.0%
- DBSCAN parameters: eps=3.5, min_samples=8
- PCA variance explained by two components: 25.8%

### Restaurant Business Groups

- High-Revenue Cancellation Watchlist: 48 restaurants
- Higher-Rated Restaurants: 38 restaurants
- Cancellation Review Candidates: 14 restaurants

### Interpretation

Restaurants in cancellation-focused groups may warrant investigation
of cancellation patterns and operational causes. Higher-rated
restaurants may provide useful examples of customer experience
practices.

These are exploratory findings from synthetic data and should not
be interpreted as verified real-world business performance.

## 5. Model Evaluation

K-Means selected k using silhouette comparisons, alongside elbow
plots to inspect within-cluster variation.

The customer silhouette score of 0.262 and restaurant score of
0.305 suggest that cluster separation is limited to moderate.
The clusters are useful for exploratory analysis but are not
perfectly distinct.

DBSCAN was evaluated across multiple eps and min_samples
combinations. Noise percentages and silhouette scores were
considered when selecting parameters. DBSCAN noise points are
records that were not assigned to a density-based cluster under
the selected configuration.

PCA visualizations show the first two principal components.
They explain 38.5% of customer-feature variance and 25.8% of
restaurant-feature variance, respectively. Therefore, the plots
do not capture all variation in the original feature spaces.

## 6. Deliverables

- Customer and restaurant feature datasets
- Customer and restaurant K-Means cluster assignments
- Customer and restaurant DBSCAN labels
- Cluster profile CSV files
- Segmentation metrics CSV
- Saved K-Means and DBSCAN model pipelines
- Elbow plots
- Silhouette plots
- K-Means PCA plots
- DBSCAN PCA plots

## 7. Limitations

- The datasets are synthetic and may not reflect real marketplace behavior.
- Silhouette scores indicate overlapping clusters.
- DBSCAN results depend on the selected distance and density parameters.
- Business labels are rule-based summaries of cluster profiles.
- PCA visualizations show only two dimensions of the feature space.
- Results should be validated against real data before operational use.

## 8. Conclusion

Day 2 produced customer and restaurant feature datasets, K-Means
and DBSCAN cluster assignments, model artifacts, evaluation metrics,
and visualizations. The results provide a starting point for exploring
customer engagement and restaurant operational performance.
