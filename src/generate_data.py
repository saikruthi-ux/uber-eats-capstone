
import pandas as pd
import numpy as np
from pathlib import Path

# Keep generated data reproducible
np.random.seed(42)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_customers(n_customers=1000):
    """Generate synthetic customer records."""

    cities = ["Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Chennai"]

    customers = pd.DataFrame({
        "customer_id": range(1, n_customers + 1),
        "age": np.random.randint(18, 66, n_customers),
        "city": np.random.choice(cities, n_customers),
        "signup_date": pd.to_datetime(
            np.random.choice(
                pd.date_range("2024-01-01", "2026-09-30"),
                n_customers
            )
        )
    })

    return customers



def generate_restaurants(n_restaurants=100):
    """Generate synthetic restaurant records."""

    categories = [
        "Indian",
        "Chinese",
        "Italian",
        "Fast Food",
        "Mexican",
        "Cafe",
        "Desserts"
    ]

    restaurants = pd.DataFrame({
        "restaurant_id": range(1, n_restaurants + 1),
        "category": np.random.choice(categories, n_restaurants),
        "rating": np.round(
            np.random.uniform(2.5, 5.0, n_restaurants), 1
        ),
        "avg_prep_time": np.random.randint(10, 46, n_restaurants)
    })

    return restaurants



def generate_orders(customers, restaurants, n_orders=5000):
    """Generate synthetic order records linked to customers and restaurants."""

    order_statuses = [
        "Delivered",
        "Delivered",
        "Delivered",
        "Cancelled",
        "In Progress"
    ]

    start_date = pd.Timestamp("2026-01-01")
    end_date = pd.Timestamp("2026-09-30")

    timestamps = pd.to_datetime(
        np.random.choice(
            pd.date_range(start_date, end_date, freq="min"),
            n_orders
        )
    )

    orders = pd.DataFrame({
        "order_id": range(1, n_orders + 1),
        "customer_id": np.random.choice(
            customers["customer_id"], n_orders
        ),
        "restaurant_id": np.random.choice(
            restaurants["restaurant_id"], n_orders
        ),
        "order_timestamp": timestamps,
        "order_amount": np.round(
            np.random.uniform(100, 1500, n_orders), 2
        ),
        "num_items": np.random.randint(1, 9, n_orders),
        "order_status": np.random.choice(
            order_statuses, n_orders
        )
    })

    return orders



def generate_drivers(n_drivers=300):
    """Generate synthetic driver records."""

    vehicle_types = ["Bike", "Scooter", "Car"]

    drivers = pd.DataFrame({
        "driver_id": range(1, n_drivers + 1),
        "vehicle_type": np.random.choice(
            vehicle_types, n_drivers, p=[0.4, 0.4, 0.2]
        ),
        "experience_years": np.round(
            np.random.uniform(0, 10, n_drivers), 1
        )
    })

    return drivers



def generate_deliveries(orders, drivers):
    """Generate deliveries for delivered orders using realistic factors."""

    delivered_orders = orders[
        orders["order_status"] == "Delivered"
    ].copy()

    n_deliveries = len(delivered_orders)

    # Assign delivery characteristics first
    distances = np.round(
        np.random.uniform(1, 20, n_deliveries), 2
    )

    weather = np.random.choice(
        ["Clear", "Rainy", "Cloudy", "Stormy"],
        n_deliveries,
        p=[0.5, 0.2, 0.25, 0.05]
    )

    traffic = np.random.choice(
        ["Low", "Medium", "High"],
        n_deliveries,
        p=[0.3, 0.45, 0.25]
    )

    # Model delivery time using distance, traffic, weather and noise
    traffic_delay = pd.Series(traffic).map({
        "Low": 0,
        "Medium": 8,
        "High": 18
    }).to_numpy()

    weather_delay = pd.Series(weather).map({
        "Clear": 0,
        "Cloudy": 3,
        "Rainy": 8,
        "Stormy": 15
    }).to_numpy()

    random_variation = np.random.normal(
        0, 5, n_deliveries
    )

    delivery_times = np.maximum(
        10,
        np.round(
            12 + distances * 2 + traffic_delay
            + weather_delay + random_variation
        )
    ).astype(int)

    deliveries = pd.DataFrame({
        "delivery_id": range(1, n_deliveries + 1),
        "order_id": delivered_orders["order_id"].to_numpy(),
        "driver_id": np.random.choice(
            drivers["driver_id"], n_deliveries
        ),
        "delivery_distance_km": distances,
        "delivery_time_minutes": delivery_times,
        "weather": weather,
        "traffic_condition": traffic,
        "tip_amount": np.random.choice(
            [0, 10, 20, 30, 50, 75, 100],
            n_deliveries
        )
    })

    return deliveries





def generate_reviews(orders):
    """Generate reviews with text broadly consistent with ratings."""

    delivered_orders = orders[
        orders["order_status"] == "Delivered"
    ]

    # A subset of delivered orders receives reviews
    reviewed_orders = delivered_orders.sample(
        frac=0.65, random_state=42
    )

    review_texts_by_rating = {
        1: [
            "Terrible experience. The food was very poor.",
            "Very late delivery and disappointing food.",
            "The order arrived cold and incomplete."
        ],
        2: [
            "Below average experience.",
            "Delivery was late and food quality was poor.",
            "Could have been much better."
        ],
        3: [
            "Average experience, nothing special.",
            "Food was okay, but there is room for improvement.",
            "Neither great nor bad."
        ],
        4: [
            "Good food and a pleasant experience.",
            "Delivery was good and the food was tasty.",
            "Satisfied with the overall service."
        ],
        5: [
            "Excellent experience. Will order again!",
            "Amazing food and great service.",
            "Fast delivery and delicious food."
        ]
    }

    n_reviews = len(reviewed_orders)

    ratings = np.random.choice(
        [1, 2, 3, 4, 5],
        n_reviews,
        p=[0.05, 0.10, 0.15, 0.35, 0.35]
    )

    review_texts = [
        np.random.choice(review_texts_by_rating[rating])
        for rating in ratings
    ]

    reviews = pd.DataFrame({
        "review_id": range(1, n_reviews + 1),
        "order_id": reviewed_orders["order_id"].to_numpy(),
        "review_text": review_texts,
        "rating": ratings
    })

    return reviews



def generate_payments(orders):
    """Generate payment records for every order."""

    payment_methods = [
        "UPI", "Credit Card", "Debit Card",
        "Wallet", "Cash"
    ]

    payments = pd.DataFrame({
        "payment_id": range(1, len(orders) + 1),
        "order_id": orders["order_id"].to_numpy(),
        "payment_method": np.random.choice(
            payment_methods, len(orders)
        ),
        "payment_status": np.where(
            orders["order_status"].to_numpy() == "Cancelled",
            "Refunded",
            "Paid"
        ),
        "payment_amount": orders["order_amount"].to_numpy()
    })

    return payments





if __name__ == "__main__":
    customers = generate_customers()
    restaurants = generate_restaurants()
    orders = generate_orders(customers, restaurants)
    drivers = generate_drivers()
    deliveries = generate_deliveries(orders, drivers)
    reviews = generate_reviews(orders)
    payments = generate_payments(orders)

    tables = {
        "customers": customers,
        "restaurants": restaurants,
        "orders": orders,
        "drivers": drivers,
        "deliveries": deliveries,
        "reviews": reviews,
        "payments": payments
    }

    for table_name, dataframe in tables.items():
        output_path = RAW_DATA_DIR / f"{table_name}.csv"
        dataframe.to_csv(output_path, index=False)
        print(f"Generated {len(dataframe)} rows: {output_path}")

    print("\nReviews preview:")
    print(reviews.head())

    print("\nPayments preview:")
    print(payments.head())
