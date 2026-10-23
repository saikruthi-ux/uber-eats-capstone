
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_COLUMNS = {
    "customers": [
        "customer_id", "age", "city", "signup_date"
    ],
    "restaurants": [
        "restaurant_id", "category", "rating", "avg_prep_time"
    ],
    "orders": [
        "order_id", "customer_id", "restaurant_id",
        "order_timestamp", "order_amount", "num_items",
        "order_status"
    ],
    "drivers": [
        "driver_id", "vehicle_type", "experience_years"
    ],
    "deliveries": [
        "delivery_id", "order_id", "driver_id",
        "delivery_distance_km", "delivery_time_minutes",
        "weather", "traffic_condition", "tip_amount"
    ],
    "reviews": [
        "review_id", "order_id", "review_text", "rating"
    ],
    "payments": [
        "payment_id", "order_id", "payment_method",
        "payment_status", "payment_amount"
    ],
}

EXPECTED_ROWS = {
    "customers": 1000,
    "restaurants": 100,
    "orders": 5000,
    "drivers": 300,
    "payments": 5000,
}

PRIMARY_KEYS = {
    "customers": "customer_id",
    "restaurants": "restaurant_id",
    "orders": "order_id",
    "drivers": "driver_id",
    "deliveries": "delivery_id",
    "reviews": "review_id",
    "payments": "payment_id",
}

errors = []
checks_passed = 0


def check(condition, description):
    """Record and print the result of a validation check."""
    global checks_passed

    if condition:
        checks_passed += 1
        print(f"PASS: {description}")
    else:
        errors.append(description)
        print(f"FAIL: {description}")


def load_tables():
    """Load all expected CSV files and report missing files."""
    tables = {}

    for name in EXPECTED_COLUMNS:
        path = RAW_DATA_DIR / f"{name}.csv"

        if not path.exists():
            errors.append(f"Missing file: {path}")
            print(f"FAIL: Missing file: {path}")
            continue

        try:
            tables[name] = pd.read_csv(path)
        except Exception as exc:
            errors.append(f"Could not read {path}: {exc}")
            print(f"FAIL: Could not read {path}: {exc}")

    return tables


def validate_schema_and_keys(tables):
    print("\n" + "=" * 60)
    print("1. SCHEMA, ROW COUNTS AND PRIMARY KEYS")
    print("=" * 60)

    for name, expected_columns in EXPECTED_COLUMNS.items():
        if name not in tables:
            continue

        df = tables[name]

        check(
            list(df.columns) == expected_columns,
            f"{name}: columns match the expected schema"
        )

        if name in EXPECTED_ROWS:
            check(
                len(df) == EXPECTED_ROWS[name],
                f"{name}: expected {EXPECTED_ROWS[name]} rows, "
                f"found {len(df)}"
            )

        if name in PRIMARY_KEYS:
            pk = PRIMARY_KEYS[name]

            if pk in df.columns:
                check(
                    df[pk].notna().all(),
                    f"{name}: primary key {pk} has no nulls"
                )
                check(
                    not df[pk].duplicated().any(),
                    f"{name}: primary key {pk} is unique"
                )

        if len(df.columns):
            check(
                not df.isna().any().any(),
                f"{name}: no missing values in any column"
            )


def validate_foreign_keys(tables):
    print("\n" + "=" * 60)
    print("2. FOREIGN KEY VALIDATION")
    print("=" * 60)

    relationships = [
        ("orders", "customer_id", "customers", "customer_id"),
        ("orders", "restaurant_id", "restaurants", "restaurant_id"),
        ("deliveries", "order_id", "orders", "order_id"),
        ("deliveries", "driver_id", "drivers", "driver_id"),
        ("reviews", "order_id", "orders", "order_id"),
        ("payments", "order_id", "orders", "order_id"),
    ]

    for child, child_col, parent, parent_col in relationships:
        if child not in tables or parent not in tables:
            continue

        child_df = tables[child]
        parent_df = tables[parent]

        if child_col not in child_df or parent_col not in parent_df:
            check(
                False,
                f"{child}.{child_col} references "
                f"{parent}.{parent_col}"
            )
            continue

        invalid = ~child_df[child_col].isin(parent_df[parent_col])

        check(
            not invalid.any(),
            f"{child}.{child_col} contains only valid "
            f"{parent}.{parent_col} references"
        )


def validate_categories(tables):
    print("\n" + "=" * 60)
    print("3. CATEGORY AND BUSINESS RULE VALIDATION")
    print("=" * 60)

    allowed_categories = {
        "customers": {
            "city": {
                "Bengaluru", "Mumbai", "Delhi",
                "Hyderabad", "Chennai"
            }
        },
        "restaurants": {
            "category": {
                "Indian", "Chinese", "Italian",
                "Fast Food", "Mexican", "Cafe", "Desserts"
            }
        },
        "orders": {
            "order_status": {
                "Delivered", "Cancelled", "In Progress"
            }
        },
        "drivers": {
            "vehicle_type": {"Bike", "Scooter", "Car"}
        },
        "deliveries": {
            "weather": {"Clear", "Rainy", "Cloudy", "Stormy"},
            "traffic_condition": {"Low", "Medium", "High"}
        },
        "payments": {
            "payment_method": {
                "UPI", "Credit Card", "Debit Card",
                "Wallet", "Cash"
            },
            "payment_status": {"Paid", "Refunded"}
        },
    }

    for table_name, column_rules in allowed_categories.items():
        if table_name not in tables:
            continue

        df = tables[table_name]

        for column, allowed in column_rules.items():
            if column not in df.columns:
                check(False, f"{table_name}: {column} exists")
                continue

            check(
                df[column].isin(allowed).all(),
                f"{table_name}.{column} contains allowed values"
            )

    if "orders" not in tables:
        return

    orders = tables["orders"]

    if "order_status" not in orders.columns:
        return

    delivered_ids = set(
        orders.loc[
            orders["order_status"] == "Delivered", "order_id"
        ]
    )

    for table_name in ["deliveries", "reviews"]:
        if table_name not in tables:
            continue

        df = tables[table_name]

        if "order_id" not in df.columns:
            continue

        check(
            df["order_id"].isin(delivered_ids).all(),
            f"{table_name}: every record belongs to a delivered order"
        )

        check(
            not df["order_id"].duplicated().any(),
            f"{table_name}: at most one record per order"
        )

    if "payments" in tables:
        payments = tables["payments"]

        check(
            payments["order_id"].is_unique,
            "payments: one payment record per order"
        )

        check(
            set(payments["order_id"]) == set(orders["order_id"]),
            "payments: every order has exactly one payment record"
        )

        expected_status = orders.set_index("order_id")[
            "order_status"
        ].map(
            lambda status: "Refunded"
            if status == "Cancelled" else "Paid"
        )

        actual_status = payments.set_index("order_id")[
            "payment_status"
        ]

        if expected_status.index.is_unique and actual_status.index.is_unique:
            check(
                actual_status.reindex(expected_status.index).equals(
                    expected_status
                ),
                "payments: status follows the documented order-status rule"
            )


def validate_ranges_and_dates(tables):
    print("\n" + "=" * 60)
    print("4. NUMERICAL RANGES AND DATE VALIDATION")
    print("=" * 60)

    range_rules = {
        "customers": {
            "age": (18, 65),
        },
        "restaurants": {
            "rating": (1, 5),
            "avg_prep_time": (10, 45),
        },
        "orders": {
            "order_amount": (0, float("inf")),
            "num_items": (1, 8),
        },
        "drivers": {
            "experience_years": (0, 10),
        },
        "deliveries": {
            "delivery_distance_km": (0, float("inf")),
            "delivery_time_minutes": (10, float("inf")),
            "tip_amount": (0, float("inf")),
        },
        "reviews": {
            "rating": (1, 5),
        },
        "payments": {
            "payment_amount": (0, float("inf")),
        },
    }

    for table_name, columns in range_rules.items():
        if table_name not in tables:
            continue

        df = tables[table_name]

        for column, (minimum, maximum) in columns.items():
            if column not in df.columns:
                check(False, f"{table_name}: {column} exists")
                continue

            values = pd.to_numeric(df[column], errors="coerce")

            valid = (
                values.notna()
                & np.isfinite(values)
                & values.ge(minimum)
                & values.le(maximum)
            )

            check(
                valid.all(),
                f"{table_name}.{column} is within "
                f"[{minimum}, {maximum}]"
            )

    date_rules = [
        ("customers", "signup_date", "2024-01-01", "2026-09-30"),
        ("orders", "order_timestamp", "2026-01-01", "2026-09-30 23:59:59"),
    ]

    for table_name, column, start, end in date_rules:
        if table_name not in tables:
            continue

        df = tables[table_name]

        if column not in df.columns:
            check(False, f"{table_name}: {column} exists")
            continue

        dates = pd.to_datetime(df[column], errors="coerce")

        check(
            dates.notna().all(),
            f"{table_name}.{column} contains valid dates"
        )

        check(
            dates.between(pd.Timestamp(start), pd.Timestamp(end)).all(),
            f"{table_name}.{column} falls within the expected date range"
        )

    if "reviews" in tables and "review_text" in tables["reviews"].columns:
        reviews = tables["reviews"]

        check(
            reviews["review_text"].astype(str).str.strip().ne("").all(),
            "reviews: review text is not empty"
        )


def validate_payment_amounts(tables):
    print("\n" + "=" * 60)
    print("5. CROSS-TABLE AMOUNT VALIDATION")
    print("=" * 60)

    if "orders" not in tables or "payments" not in tables:
        return

    orders = tables["orders"]
    payments = tables["payments"]

    if not {"order_id", "order_amount"}.issubset(orders.columns):
        check(False, "orders: payment comparison columns exist")
        return

    if not {"order_id", "payment_amount"}.issubset(payments.columns):
        check(False, "payments: payment comparison columns exist")
        return

    merged = payments.merge(
        orders[["order_id", "order_amount"]],
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    amounts_match = np.isclose(
        merged["payment_amount"],
        merged["order_amount"],
        rtol=0,
        atol=0.01
    )

    check(
        amounts_match.all(),
        "payments: amounts match their corresponding order amounts"
    )


def main():
    print("=" * 60)
    print("UBER EATS DATA VALIDATION")
    print("=" * 60)

    tables = load_tables()

    if len(tables) != len(EXPECTED_COLUMNS):
        print("\nValidation stopped: one or more CSV files could not be loaded.")
        sys.exit(1)

    validate_schema_and_keys(tables)
    validate_foreign_keys(tables)
    validate_categories(tables)
    validate_ranges_and_dates(tables)
    validate_payment_amounts(tables)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Checks passed: {checks_passed}")
    print(f"Checks failed: {len(errors)}")

    if errors:
        print("\nIssues requiring attention:")
        for error in errors:
            print(f"- {error}")

        print("\nVALIDATION FAILED")
        sys.exit(1)

    print("\nALL VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
