
# Uber Eats Operational & Customer Intelligence System

## Synthetic Dataset — Data Dictionary

### 1. Overview

This data dictionary describes the seven synthetic datasets used in the Uber Eats Operational & Customer Intelligence System capstone project.

The datasets represent customers, restaurants, orders, deliveries, drivers, customer reviews, and payments. They are designed to support exploratory data analysis (EDA), customer and restaurant segmentation, sentiment analysis, regression modeling, and hourly demand forecasting.

All records are synthetically generated for learning and analytical purposes.

---

### 2. Customers Dataset

**File:** `customers.csv`

**Purpose:** Stores customer information to analyze customer demographics, ordering behavior, spending patterns, and customer segments.

**Primary Key:** `customer_id`

| Column      | Description                                  |
| ----------- | -------------------------------------------- |
| customer_id | Unique identifier assigned to each customer  |
| age         | Age of the customer                          |
| city        | Customer's city                              |
| signup_date | Date the customer registered on the platform |

**Relationships:**

- `customer_id` is referenced by `orders.csv`.

**Analytical Use:**

- Customer segmentation
- Customer spending and ordering behavior
- Identification of high-value customers

---

### 3. Restaurants Dataset

**File:** `restaurants.csv`

**Purpose:** Stores restaurant information to analyze restaurant performance, popularity, preparation efficiency, and customer satisfaction.

**Primary Key:** `restaurant_id`

| Column        | Description                                   |
| ------------- | --------------------------------------------- |
| restaurant_id | Unique identifier assigned to each restaurant |
| category      | Cuisine or restaurant category                |
| rating        | Average restaurant rating                     |
| avg_prep_time | Average food preparation time in minutes      |

**Relationships:**

- `restaurant_id` is referenced by `orders.csv`.

**Analytical Use:**

- Restaurant segmentation
- Restaurant performance analysis
- Revenue and order-volume analysis
- Identification of restaurants requiring operational improvements

---

### 4. Orders Dataset

**File:** `orders.csv`

**Purpose:** Stores order-level information and serves as the central dataset connecting customers and restaurants with order activity.

**Primary Key:** `order_id`

| Column          | Description                                                   |
| --------------- | ------------------------------------------------------------- |
| order_id        | Unique identifier assigned to each order                      |
| customer_id     | Identifier of the customer who placed the order               |
| restaurant_id   | Identifier of the restaurant receiving the order              |
| order_timestamp | Date and time when the order was placed                       |
| order_amount    | Total order amount                                            |
| num_items       | Number of items included in the order                         |
| order_status    | Current or final order status, such as delivered or cancelled |

**Relationships:**

- `customer_id` references `customers.csv`.
- `restaurant_id` references `restaurants.csv`.
- `order_id` is referenced by `deliveries.csv`, `reviews.csv`, and `payments.csv`.

**Analytical Use:**

- Order-volume and revenue analysis
- Customer and restaurant feature engineering
- Hourly demand forecasting
- Analysis of ordering patterns by day and time

---

### 5. Deliveries Dataset

**File:** `deliveries.csv`

**Purpose:** Stores delivery-level information to analyze delivery performance, operational delays, delivery conditions, and tip behavior.

**Primary Key:** `delivery_id`

| Column                | Description                                 |
| --------------------- | ------------------------------------------- |
| delivery_id           | Unique identifier assigned to each delivery |
| order_id              | Identifier of the associated order          |
| driver_id             | Identifier of the assigned delivery driver  |
| delivery_distance_km  | Delivery distance in kilometres             |
| delivery_time_minutes | Total delivery duration in minutes          |
| weather               | Weather conditions during delivery          |
| traffic_condition     | Traffic level during delivery               |
| tip_amount            | Tip amount associated with the delivery     |

**Relationships:**

- `order_id` references `orders.csv`.
- `driver_id` references `drivers.csv`.

**Analytical Use:**

- Delivery-time prediction
- Tip prediction
- Analysis of factors contributing to delivery delays
- Evaluation of weather and traffic effects

---

### 6. Drivers Dataset

**File:** `drivers.csv`

**Purpose:** Stores delivery driver information to support analysis of driver characteristics and delivery operations.

**Primary Key:** `driver_id`

| Column           | Description                               |
| ---------------- | ----------------------------------------- |
| driver_id        | Unique identifier assigned to each driver |
| vehicle_type     | Type of vehicle used for deliveries       |
| experience_years | Number of years of delivery experience    |

**Relationships:**

- `driver_id` is referenced by `deliveries.csv`.

**Analytical Use:**

- Driver and delivery performance analysis
- Exploration of vehicle type and delivery characteristics

---

### 7. Reviews Dataset

**File:** `reviews.csv`

**Purpose:** Stores customer feedback and ratings to understand customer sentiment, identify common complaints, and evaluate customer satisfaction.

**Primary Key:** `review_id`

| Column      | Description                                        |
| ----------- | -------------------------------------------------- |
| review_id   | Unique identifier assigned to each review          |
| order_id    | Identifier of the order associated with the review |
| review_text | Written customer feedback                          |
| rating      | Numerical rating provided by the customer          |

**Relationships:**

- `order_id` references `orders.csv`.

**Analytical Use:**

- Text preprocessing and NLP
- TF-IDF feature extraction
- Sentiment classification
- Identification of recurring customer complaints
- Comparison of review sentiment with numerical ratings

---

### 8. Payments Dataset

**File:** `payments.csv`

**Purpose:** Stores payment-level information to analyze transaction amounts, payment methods, and payment outcomes.

**Primary Key:** `payment_id`

| Column         | Description                                |
| -------------- | ------------------------------------------ |
| payment_id     | Unique identifier assigned to each payment |
| order_id       | Identifier of the associated order         |
| payment_method | Method used to make the payment            |
| payment_status | Outcome of the payment transaction         |
| payment_amount | Amount recorded for the payment            |

**Relationships:**

- `order_id` references `orders.csv`.

**Analytical Use:**

- Payment-method analysis
- Payment-status analysis
- Reconciliation of order amounts and payment amounts

---

### 9. Dataset Relationships Summary

| Dataset     | Primary Key   | Foreign Keys               |
| ----------- | ------------- | -------------------------- |
| Customers   | customer_id   | None                       |
| Restaurants | restaurant_id | None                       |
| Orders      | order_id      | customer_id, restaurant_id |
| Deliveries  | delivery_id   | order_id, driver_id        |
| Drivers     | driver_id     | None                       |
| Reviews     | review_id     | order_id                   |
| Payments    | payment_id    | order_id                   |

### 10. Data Quality and Validation Considerations

The synthetic data-generation process should aim to maintain the following conditions:

- Primary keys should be unique and non-null.
- Foreign keys should reference valid records in the corresponding datasets.
- Numerical values should fall within reasonable ranges.
- Timestamps should use a consistent datetime format.
- Ratings should follow a consistent numerical scale.
- Order, delivery, review, and payment records should maintain logical relationships.
- Missing values and unusual observations, if included, should be intentional and documented.
- Synthetic data should be reproducible by using a fixed random seed.

### 11. Notes and Assumptions

- The datasets are synthetic and do not represent actual Uber Eats customer or operational records.
- The column definitions describe the planned schema and should be updated if the implemented generator changes.
- Additional columns may be introduced during feature engineering and model development.
- The final analytical datasets may combine information from multiple source datasets.
- Identifiers are intended for joining datasets and should not be used as numerical model features.
