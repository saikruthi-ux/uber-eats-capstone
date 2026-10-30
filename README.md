
# Uber Eats – Operational Analytics & Machine Learning

## Project Overview

This project focuses on analyzing an Uber Eats-style food delivery marketplace using data analytics and machine learning techniques.

The objective is to understand customer behavior, evaluate restaurant performance, analyze customer feedback, predict delivery times, and forecast hourly order demand.

Synthetic datasets are generated using Python to simulate marketplace operations. The project applies data preprocessing, exploratory analysis, machine learning, natural language processing (NLP), regression, and time-series forecasting to generate actionable business insights.

## Business Objectives

The project addresses the following business questions:

- Which customer segments are most valuable?
- Which restaurant segments require operational improvement?
- What patterns can be identified from negative customer reviews?
- Which factors contribute most to delivery-time predictions?
- How accurately can delivery times be estimated?
- When is order demand likely to increase?
- How can forecasting support delivery capacity planning?

## Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Natural Language Processing:** NLTK, TF-IDF, VADER
- **Time-Series Forecasting:** ARIMA, Prophet
- **Model Persistence:** Joblib
- **Development Environment:** Jupyter Notebook, VS Code
- **Version Control:** Git and GitHub

## Dataset Overview

Seven related synthetic datasets were generated to represent the food delivery marketplace.

| Dataset     | Description                                    |
| ----------- | ---------------------------------------------- |
| Customers   | Customer details and purchasing behavior       |
| Restaurants | Restaurant information and performance         |
| Orders      | Order transactions and order details           |
| Deliveries  | Delivery records and delivery-time information |
| Drivers     | Delivery partner information                   |
| Reviews     | Customer ratings and textual feedback          |
| Payments    | Payment and transaction information            |

The datasets contain unique identifiers and relationships between entities to support integrated analysis.

**Note:** The datasets are synthetic and are intended for learning, experimentation, and demonstration. They do not represent actual Uber Eats customer or operational records.

Refer to `docs/data_dictionary.md` for dataset and column descriptions.

## Project Structure

```text
uber-eats-capstone/
│
├── data/
│   ├── raw/
│   │   └── Synthetic source datasets
│   │
│   └── processed/
│       └── Cleaned datasets, analytical outputs,
│           model metrics, predictions and forecasts
│
├── src/
│   ├── generate_data.py
│   ├── validate_data.py
│   ├── segmentation.py
│   ├── sentiment_analysis.py
│   ├── regression_analysis.py
│   ├── hourly_order_forecasting.py
│   └── integrated_ml_pipeline.py
│
├── models/
│   ├── customer_kmeans.joblib
│   ├── customer_dbscan.joblib
│   ├── restaurant_kmeans.joblib
│   ├── restaurant_dbscan.joblib
│   ├── sentiment_naive_bayes.joblib
│   └── delivery_time_regression.joblib
│
├── docs/
│   ├── data_dictionary.md
│   ├── day2_segmentation_report.md
│   ├── day3_sentiment_report.md
│   ├── day4_regression_report.md
│   ├── day5_forecasting_report.md
│   ├── day6_integrated_ml_report.md
│   └── prompt_engineering.md
│
├── README.md
└── requirements.txt
```

*Note: The structure above describes the main project components. Some generated files and folders may vary depending on the execution environment.*

## Project Workflow

The project follows an end-to-end analytics and machine learning workflow:

1. Synthetic Data Generation
2. Data Understanding and Validation
3. Data Cleaning and Preprocessing
4. Exploratory Data Analysis
5. Feature Engineering
6. Machine Learning Model Development
7. Model Evaluation and Comparison
8. Prediction and Forecasting
9. Business Insights and Recommendations

## Machine Learning Implementation

### 1. Customer and Restaurant Segmentation

**Objective:** Identify groups of customers and restaurants based on behavioral and operational characteristics.

**Algorithms:**

- K-Means Clustering
- DBSCAN
- Principal Component Analysis (PCA)

**Key Results:**

| Entity      | K-Means Clusters | Silhouette Score |
| ----------- | ---------------: | ---------------: |
| Customers   |                6 |            0.262 |
| Restaurants |                7 |            0.305 |

DBSCAN was also evaluated to identify dense groups and potential outliers.

**Business Application:**

- Identify high-value customer groups.
- Support customer retention strategies.
- Identify restaurant groups requiring operational attention.

### 2. Sentiment Analysis

**Objective:** Analyze customer reviews and classify feedback into sentiment categories.

**Techniques:**

- Text preprocessing
- TF-IDF vectorization
- Multinomial Naive Bayes
- VADER Sentiment Analysis

**Model Comparison:**

| Model                            | Accuracy | Weighted F1 |
| -------------------------------- | -------: | ----------: |
| TF-IDF + Multinomial Naive Bayes |   43.59% |      47.85% |
| VADER                            |   84.86% |      79.97% |

VADER achieved the highest weighted F1 score among the evaluated models.

**Business Application:**

- Understand customer feedback.
- Identify recurring negative-review themes.
- Support customer experience improvement initiatives.

### 3. Delivery-Time Prediction

**Objective:** Predict delivery duration using delivery and operational features.

**Features include:**

- Delivery distance
- Traffic condition
- Weather
- Restaurant category
- Rating
- Other engineered features

**Models Evaluated:**

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Dummy Regression Baseline

**Model Comparison:**

| Model             | Test MAE (minutes) | Test RMSE (minutes) |        Test R² |
| ----------------- | -----------------: | ------------------: | --------------: |
| Linear Regression |              4.171 |               5.260 |           0.867 |
| Ridge Regression  |              4.172 |               5.262 |           0.867 |
| Lasso Regression  |              4.217 |               5.351 |           0.862 |
| Random Forest     |              4.565 |               5.781 |           0.839 |
| Dummy Baseline    |             11.844 |              14.406 | Approximately 0 |

Linear Regression achieved the lowest test MAE and RMSE, with Ridge Regression performing almost identically.

**Business Application:**

- Support delivery-time estimation.
- Improve dispatch planning.
- Help identify important delivery-time predictors.

### 4. Hourly Demand Forecasting

**Objective:** Analyze historical order patterns and forecast future hourly order demand.

**Techniques:**

- Hourly order aggregation
- Time-series decomposition
- Augmented Dickey-Fuller (ADF) stationarity test
- ARIMA forecasting
- Prophet forecasting

**Model Comparison:**

| Metric                  | Prophet |  ARIMA |
| ----------------------- | ------: | -----: |
| MAE                     |   0.671 |  0.672 |
| RMSE                    |   0.819 |  0.809 |
| MAPE on nonzero actuals |  29.71% | 34.69% |

Prophet achieved a slightly lower MAE, while ARIMA achieved a slightly lower RMSE.

**Business Application:**

- Understand hourly demand patterns.
- Support delivery-partner capacity planning.
- Help identify periods that may require additional delivery capacity.

## Integrated ML Pipeline

The Day 6 integration component brings together the saved outputs from the four machine learning tasks.

**Workflow:**

Raw Data → Cleaning → Feature Engineering → Modeling → Evaluation → Prediction → Business Insights

The integration script:

- Validates the availability of existing analytical artifacts.
- Loads saved model evaluation results.
- Generates a consolidated model comparison.
- Extracts business insights from segmentation and model outputs.
- Produces an integrated project report.

**Generated outputs:**

- `data/processed/integrated_model_comparison.csv`
- `data/processed/integrated_pipeline_artifact_inventory.csv`
- `docs/day6_integrated_ml_report.md`

The integration script uses previously generated results; it does not retrain all component models in a single execution.

## Setup and Installation

### Prerequisites

- Python 3.10 or a compatible Python version.
- Git.
- VS Code or another Python development environment.

### 1. Clone the Repository

```bash
git clone https://github.com/saikruthi-ux/uber-eats-capstone.git

cd uber-eats-capstone
```

### 2. Create a Virtual Environment

On Linux/macOS:

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### 3. Install Dependencies

If the project includes `requirements.txt`, install the required packages:

```bash
pip install -r requirements.txt
```

If dependencies are not listed in that file, install the packages required by the individual scripts before running them.

## Execution Guide

Run the scripts from the project root directory with the virtual environment activated.

### Step 1: Generate Synthetic Data

```bash
python src/generate_data.py
```

### Step 2: Validate Data

```bash
python src/validate_data.py
```

### Step 3: Customer and Restaurant Segmentation

```bash
python src/segmentation.py
```

### Step 4: Sentiment Analysis

```bash
python src/sentiment_analysis.py
```

### Step 5: Delivery-Time Regression

```bash
python src/regression_analysis.py
```

### Step 6: Hourly Demand Forecasting

```bash
python src/hourly_order_forecasting.py
```

### Step 7: Integrated ML Pipeline

```bash
python src/integrated_ml_pipeline.py
```

**Execution note:** The scripts depend on the datasets and outputs generated by earlier stages. Run them in the sequence above. The Day 6 integration script expects the saved analytical artifacts from Days 2–5 to be available.

## Business Insights and Recommendations

Based on the analysis, the following recommendations were identified:

1. **Customer Retention:** Focus on high-spending and frequent customer segments through targeted loyalty initiatives.
2. **Restaurant Operations:** Investigate restaurant groups with higher cancellation rates and identify operational bottlenecks.
3. **Customer Experience:** Review recurring negative feedback themes and address service-quality concerns.
4. **Delivery Operations:** Use delivery-time predictions to support ETA estimation and dispatch planning.
5. **Demand Planning:** Use hourly demand forecasts to support delivery-partner allocation and capacity planning.

These recommendations are derived from the synthetic dataset and analytical results. They have not been validated through a live operational deployment.

## Limitations

- The project uses synthetic data rather than actual production data.
- The sentiment dataset contains only 15 unique review templates, limiting generalization.
- Clustering results require further business validation.
- Model evaluation results depend on the available dataset and evaluation splits.
- Feature importance indicates predictive contribution, not causation.
- Forecasting performance may differ with real-world demand patterns.
- The regression component predicts delivery time; a separate tip-prediction model was not developed.
- The project does not include a live production deployment.

## Future Scope

- Work with larger and more diverse real-world datasets.
- Improve data validation and automated testing.
- Explore additional clustering and classification algorithms.
- Improve sentiment analysis using more varied and independently labelled reviews.
- Evaluate models on future unseen operational data.
- Develop location-based and more granular demand forecasting.
- Build an interactive analytics dashboard.
- Deploy models through APIs and monitor model performance.

## Detailed Documentation

Additional analysis and technical details are available in the `docs/` directory:

| Document                         | Description                                 |
| -------------------------------- | ------------------------------------------- |
| `data_dictionary.md`           | Dataset and column definitions              |
| `day2_segmentation_report.md`  | Customer and restaurant segmentation        |
| `day3_sentiment_report.md`     | NLP and sentiment analysis                  |
| `day4_regression_report.md`    | Delivery-time prediction                    |
| `day5_forecasting_report.md`   | Hourly demand forecasting                   |
| `day6_integrated_ml_report.md` | Integrated ML results and business insights |
| `prompt_engineering.md`        | Prompt engineering documentation            |

## GitHub Repository

**Repository:** https://github.com/saikruthi-ux/uber-eats-capstone

## Conclusion

This project demonstrates an end-to-end application of data analytics and machine learning to a simulated food delivery marketplace.

By combining customer and restaurant segmentation, sentiment analysis, delivery-time prediction, and demand forecasting, the project illustrates how data-driven insights can support operational planning and business decision-making.

**Project Status:** Completed – 7-Day Analytics and Machine Learning Capstone.
