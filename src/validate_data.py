
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Load all seven tables
customers = pd.read_csv(RAW_DATA_DIR / "customers.csv")
restaurants = pd.read_csv(RAW_DATA_DIR / "restaurants.csv")
orders = pd.read_csv(RAW_DATA_DIR / "orders.csv")
drivers = pd.read_csv(RAW_DATA_DIR / "drivers.csv")
deliveries = pd.read_csv(RAW_DATA_DIR / "deliveries.csv")
reviews = pd.read_csv(RAW_DATA_DIR / "reviews.csv")
payments = pd.read_csv(RAW_DATA_DIR / "payments.csv")

tables = {
    "customers": (customers, "customer_id"),
    "restaurants": (restaurants, "restaurant_id"),
    "orders": (orders, "order_id"),
    "drivers": (drivers, "driver_id"),
    "deliveries": (deliveries, "delivery_id"),
    "reviews": (reviews, "review_id"),
    "payments": (payments, "payment_id")
}

print("=" * 50)
print("1. TABLE SIZE AND PRIMARY KEY VALIDATION")
print("=" * 50)

for name, (df, pk) in tables.items():
    print(f"\n{name.upper()}")
    print(f"Rows: {len(df)}")
    print(f"Null primary keys: {df[pk].isna().sum()}")
    print(f"Duplicate primary keys: {df[pk].duplicated().sum()}")

print("\n" + "=" * 50)
print("2. FOREIGN KEY VALIDATION")
print("=" * 50)


def check_foreign_key(child_df, child_column, parent_df, parent_column, label):
    invalid = ~child_df[child_column].isin(parent_df[parent_column])
    print(f"{label}: {invalid.sum()} invalid references")


check_foreign_key(
    orders, "customer_id", customers, "customer_id",
    "Orders -> Customers"
)

check_foreign_key(
    orders, "restaurant_id", restaurants, "restaurant_id",
    "Orders -> Restaurants"
)

check_foreign_key(
    deliveries, "order_id", orders, "order_id",
    "Deliveries -> Orders"
)

check_foreign_key(
    deliveries, "driver_id", drivers, "driver_id",
    "Deliveries -> Drivers"
)

check_foreign_key(
    reviews, "order_id", orders, "order_id",
    "Reviews -> Orders"
)

check_foreign_key(
    payments, "order_id", orders, "order_id",
    "Payments -> Orders"
)

print("\n" + "=" * 50)
print("3. BUSINESS RULE VALIDATION")
print("=" * 50)

delivered_order_ids = set(
    orders.loc[orders["order_status"] == "Delivered", "order_id"]
)

delivery_order_ids = set(deliveries["order_id"])
review_order_ids = set(reviews["order_id"])

print(
    "Deliveries linked to non-delivered orders:",
    len(delivery_order_ids - delivered_order_ids)
)

print(
    "Reviews linked to non-delivered orders:",
    len(review_order_ids - delivered_order_ids)
)

print(
    "Orders without a payment record:",
    len(set(orders["order_id"]) - set(payments["order_id"]))
)

print(
    "Duplicate payment order IDs:",
    payments["order_id"].duplicated().sum()
)

print(
    "Payment amounts differing from order amounts:",
    (
        payments.merge(
            orders[["order_id", "order_amount"]],
            on="order_id",
            how="left"
        )["payment_amount"]
        !=
        payments.merge(
            orders[["order_id", "order_amount"]],
            on="order_id",
            how="left"
        )["order_amount"]
    ).sum()
)

print("\n" + "=" * 50)
print("4. NUMERICAL RANGE VALIDATION")
print("=" * 50)

print("Invalid customer ages:", (~customers["age"].between(18, 65)).sum())
print("Invalid restaurant ratings:", (~restaurants["rating"].between(1, 5)).sum())
print("Invalid review ratings:", (~reviews["rating"].between(1, 5)).sum())
print("Non-positive order amounts:", (orders["order_amount"] <= 0).sum())
print("Non-positive delivery distances:", (deliveries["delivery_distance_km"] <= 0).sum())
print("Non-positive delivery times:", (deliveries["delivery_time_minutes"] <= 0).sum())
print("Negative tips:", (deliveries["tip_amount"] < 0).sum())

print("\nValidation checks completed.")
