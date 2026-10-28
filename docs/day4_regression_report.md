# Day 4: Delivery Time Regression Analysis

## Objective

Predict delivery duration in minutes using order, restaurant, distance,
weather, traffic, and order-time features.

## Dataset

- Joined delivery records: 2,879
- Modeling records: 2,879
- Training records: 2,303
- Test records: 576
- Target: `delivery_time_minutes`
- Target mean: 44.17 minutes
- Target median: 45.00 minutes
- Target minimum / maximum: 10 / 87 minutes

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

Selected model: **Ridge Regression**

- Cross-validation MAE: 4.031 minutes
- Cross-validation RMSE: 5.010 minutes
- Cross-validation R-squared: 0.875
- Test MAE: 4.172 minutes
- Test RMSE: 5.262 minutes
- Test R-squared: 0.867

MAE describes the average absolute prediction error in minutes. RMSE gives
greater weight to larger errors. R-squared describes the proportion of
target variation explained relative to a mean-only reference, and may be
negative when predictions perform poorly.

## Overfitting and Generalization Analysis

Five-fold cross-validation was used to assess model generalization. Ridge Regression achieved a training RMSE of 4.953 minutes and a validation RMSE of 5.010 minutes, with corresponding R² values of 0.878 and 0.875. The small performance gap indicates consistent generalization.

Linear Regression and Lasso Regression also showed small training-validation gaps. In contrast, Random Forest achieved a training RMSE of 2.490 minutes but a validation RMSE of 5.461 minutes, indicating potential overfitting.

Ridge Regression was selected based on its lowest cross-validation RMSE, although Linear Regression performed almost identically. On the held-out test set, Ridge achieved an MAE of 4.172 minutes, an RMSE of 5.262 minutes, and an R² of 0.867.

Delivery distance, traffic condition, and weather were the most influential features according to permutation importance. These results support using a linear regression approach as a baseline for delivery-time estimation on this synthetic dataset. Further validation on time-separated or real-world data is recommended before operational deployment.

## Most influential features

Permutation importance measures the change in test-set MAE when each feature
is shuffled. Larger positive values indicate that shuffling the feature made
the predictions worse.

| feature              | importance_mean | importance_std |
| :------------------- | --------------: | -------------: |
| delivery_distance_km |         9.19874 |       0.297198 |
| traffic_condition    |         4.78503 |       0.142332 |
| weather              |         1.97543 |       0.164089 |
| category             |      0.00780252 |      0.0134686 |
| rating               |      0.00142554 |     0.00222387 |

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
