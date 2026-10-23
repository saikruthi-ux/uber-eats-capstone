
# Prompt Engineering – Uber Eats Operational & Customer Intelligence System

## 1. Objective

The objective of this document is to define and evaluate prompt-engineering techniques for a synthetic data-generation pipeline developed for the Uber Eats Operational & Customer Intelligence System.

The project contains seven related datasets: customers, restaurants, orders, deliveries, drivers, reviews, and payments. These datasets are intended to support exploratory data analysis, customer and restaurant segmentation, review sentiment analysis, delivery-time prediction, and order-volume forecasting.

Four prompting techniques are considered:

1. P.T.C.F. prompting
2. Few-shot prompting
3. Role-based prompting
4. Structured-output prompting

The prompts specify the expected requirements and provide a framework for reviewing the Python implementation and its generated datasets.

## 2. P.T.C.F. Prompting

### Objective

Structure a technical request using Persona, Task, Context, and Format to make the requirements clear.

### Prompt

**Persona:** Act as a Python data engineer experienced in synthetic data generation, pandas, NumPy, and relational data modelling.

**Task:** Develop a Python pipeline that generates seven related datasets for a food-delivery analytics project: customers, restaurants, orders, deliveries, drivers, reviews, and payments.

**Context:** The datasets must support exploratory data analysis, customer and restaurant segmentation, review sentiment analysis, delivery-time regression, and hourly order forecasting. They must contain unique identifiers, appropriate data types, valid relationships, and useful analytical features.

Customers require demographic and signup information. Restaurants require categories, ratings, and preparation times. Orders require timestamps, amounts, item counts, and statuses. Deliveries require distance, time, weather, traffic, and tips. Reviews require text and ratings, while payments require payment methods, statuses, and amounts.

**Format:** Organize the Python code into separate functions for each dataset. Use pandas and NumPy, set a fixed random seed, save the generated CSV files in `data/raw/` without an index column, and include a main execution block. Explain the functions and document assumptions.

### Evaluation criteria

- Are all seven datasets included?
- Is the implementation organized into reusable functions?
- Is the random seed fixed?
- Are output paths and CSV formatting specified?
- Are the analytical objectives and data relationships clear?

## 3. Few-Shot Prompting

### Objective

Provide illustrative examples that demonstrate the expected dataset structure and conventions.

### Prompt

Generate synthetic data for a food-delivery analytics project. Follow these examples when defining dataset-generation functions.

**Example 1 – Customer records**

Requirements:

- Generate three customers.
- Include `customer_id`, `age`, `city`, and `signup_date`.
- Use unique customer identifiers.
- Restrict ages to 18–65.

| customer_id | age | city      | signup_date |
| ----------: | --: | --------- | ----------- |
|           1 |  28 | Bengaluru | 2025-02-10  |
|           2 |  41 | Mumbai    | 2024-11-18  |
|           3 |  35 | Chennai   | 2025-06-22  |

**Example 2 – Restaurant records**

Requirements:

- Generate two restaurants.
- Include `restaurant_id`, `category`, `rating`, and `avg_prep_time`.
- Use unique identifiers, ratings from 1 to 5, and positive preparation times.

| restaurant_id | category | rating | avg_prep_time |
| ------------: | -------- | -----: | ------------: |
|             1 | Indian   |    4.5 |            25 |
|             2 | Italian  |    4.2 |            30 |

The rows above are illustrative examples, not actual project records.

**Task:** Use the same conventions to define functions for orders, drivers, deliveries, reviews, and payments. Preserve relationships between datasets, use pandas and NumPy, ensure reproducibility, and explain how foreign keys can be validated.

### Evaluation criteria

- Are column names consistent and descriptive?
- Are identifiers unique?
- Are data types and value ranges sensible?
- Do the examples communicate the desired output clearly?
- Are the relationships between datasets preserved?

## 4. Role-Based Prompting

### Objective

Use a specialist role to emphasize data integrity, maintainability, and analytical usefulness.

### Prompt

Act as a senior data engineer mentoring a trainee data analyst on a food-delivery analytics project.

Design a synthetic data pipeline containing customers, restaurants, orders, deliveries, drivers, reviews, and payments.

Prioritize data integrity and analytical usefulness over simply generating records. Ensure that identifiers link datasets correctly, timestamps are valid, numerical fields have sensible ranges, and business rules are documented.

Deliveries should reference valid orders and drivers, and reviews should reference valid orders. Delivery distance, weather, and traffic should be usable as potential explanatory variables for delivery-time analysis. Review text and ratings should be designed with their intended sentiment-analysis use in mind.

Present the implementation in manageable functions. Explain design decisions, identify assumptions, and recommend validation tests. Distinguish basic data-integrity checks from statistical realism. Do not assume that synthetic data is realistic merely because it passes validation.

### Evaluation criteria

- Is the code modular and understandable?
- Are foreign-key relationships valid?
- Are assumptions and limitations documented?
- Are the generated features useful for the intended analyses?
- Are the validation checks appropriate for the business rules?

## 5. Structured-Output Prompting

### Objective

Specify schemas, record counts, constraints, and output requirements explicitly.

### Prompt

Develop a Python script that generates synthetic food-delivery analytics data according to the following specification.

### Required schemas

| Dataset     | Required columns                                                                                                                                      |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Customers   | `customer_id`, `age`, `city`, `signup_date`                                                                                                   |
| Restaurants | `restaurant_id`, `category`, `rating`, `avg_prep_time`                                                                                        |
| Orders      | `order_id`, `customer_id`, `restaurant_id`, `order_timestamp`, `order_amount`, `num_items`, `order_status`                              |
| Drivers     | `driver_id`, `vehicle_type`, `experience_years`                                                                                                 |
| Deliveries  | `delivery_id`, `order_id`, `driver_id`, `delivery_distance_km`, `delivery_time_minutes`, `weather`, `traffic_condition`, `tip_amount` |
| Reviews     | `review_id`, `order_id`, `review_text`, `rating`                                                                                              |
| Payments    | `payment_id`, `order_id`, `payment_method`, `payment_status`, `payment_amount`                                                              |

### Implementation requirements

1. Use Python, pandas, and NumPy.
2. Use a fixed random seed.
3. Generate 1,000 customers, 100 restaurants, 5,000 orders, and 300 drivers.
4. Generate deliveries only for orders marked `Delivered`.
5. Generate reviews for a subset of delivered orders.
6. Generate exactly one payment record per order.
7. Use unique, non-null primary keys and valid foreign keys.
8. Apply appropriate ranges to numerical fields and define allowed categorical values.
9. Save each dataset to `data/raw/` as a CSV without an additional index column.
10. Use separate generation functions and a main execution block.
11. Document assumptions and include validation checks.
12. Return the implementation followed by an explanation of the functions, relationships, and validation requirements.

The datasets must be identified as synthetic rather than actual Uber Eats records.

### Evaluation criteria

- Do the generated datasets match the required schemas?
- Are the expected record counts satisfied?
- Are all primary and foreign keys valid?
- Are missing values, categories, numerical ranges, and dates checked?
- Are business rules and cross-table relationships enforced?
- Does the validation process report failures clearly?

## 6. Project Review and Validation

The project includes `src/generate_data.py` for dataset generation and `src/validate_data.py` for data-quality checks.

The generation pipeline uses a fixed random seed, separates dataset generation into functions, and saves seven CSV files without an index column.

Delivery-time generation incorporates distance, traffic conditions, weather, and random variation. Review text is generated according to the assigned rating. These are modelling assumptions for synthetic data and do not represent measured Uber Eats behaviour.

### Generated dataset sizes

| Dataset     |  Rows |
| ----------- | ----: |
| Customers   | 1,000 |
| Restaurants |   100 |
| Orders      | 5,000 |
| Drivers     |   300 |
| Deliveries  | 2,879 |
| Reviews     | 1,871 |
| Payments    | 5,000 |

### Validation results

The updated validation script checks schemas, expected record counts where specified, missing values, primary keys, foreign keys, allowed categories, business rules, numerical ranges, date ranges, review text, and payment amounts.

**Observed result from the validation run:**

- Checks passed: 71
- Checks failed: 0
- Final status: All validation checks passed

The result establishes that the generated datasets satisfy the implemented validation rules. It does not establish that every statistical distribution or business assumption matches real-world food-delivery operations.

## 7. Limitations and Further Improvements

- The datasets are synthetic and should not be interpreted as actual business records.
- Delivery-time relationships are based on assumed delay factors and random variation.
- Review text is deliberately associated with star ratings, which may make sentiment analysis easier than analysis of genuine customer feedback.
- Cancelled orders are assigned a `Refunded` payment status as a simplified payment rule.
- Further analysis should examine distributions, correlations, outliers, and whether the data is appropriate for each intended modelling task.

## 8. Conclusion

P.T.C.F., few-shot, role-based, and structured-output prompting offer complementary ways to express technical requirements. They help define the task, demonstrate expected structures, establish the desired perspective, and make implementation constraints explicit.

For this project, success is assessed through schema consistency, relational integrity, reproducibility, business-rule checks, and analytical usefulness. The validation results provide evidence that the current synthetic datasets meet the implemented data-quality rules, while the documented limitations identify areas that require caution during subsequent analysis.
