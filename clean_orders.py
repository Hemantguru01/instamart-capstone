
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
CLEAN = Path("data/clean")
CLEAN.mkdir(parents=True, exist_ok=True)
customers = pd.read_csv(CLEAN / "customers.csv", keep_default_na=False)

orders = pd.read_csv(RAW / "orders.csv", keep_default_na=False)
products = pd.read_csv(RAW / "products.csv", keep_default_na=False)
customers = pd.read_csv(
    CLEAN / "customers.csv",
    keep_default_na=False
)

# 1. Convert order timestamp
orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")

# 2. Remove exact duplicate rows
orders = orders.drop_duplicates().copy()

# 3. Keep latest record for each order_id
orders = (
    orders.sort_values("order_ts")
    .drop_duplicates(subset="order_id", keep="last")
    .copy()
)

# Convert numerical fields
for col in ["order_amount", "quantity", "discount_pct", "delivery_minutes"]:
    orders[col] = pd.to_numeric(orders[col], errors="coerce")

# 4. Fill missing amounts using product price
orders = orders.merge(
    products[["product_id", "price_inr"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

orders["price_inr"] = pd.to_numeric(orders["price_inr"], errors="coerce")
missing_amount = orders["order_amount"].isna()

orders.loc[missing_amount, "order_amount"] = (
    orders.loc[missing_amount, "price_inr"]
    * orders.loc[missing_amount, "quantity"]
    * (1 - orders.loc[missing_amount, "discount_pct"] / 100)
).round(2)

orders = orders.drop(columns=["price_inr"])

# 5. Negative delivery times become missing
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = float("nan")

# Attach customer city to each order
orders = orders.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left",
    validate="many_to_one"
)


# 6. Fill missing delivery times for completed orders only
completed = orders["status"].astype(str).str.strip().str.lower().eq("completed")
city_median = orders.loc[completed].groupby("city")["delivery_minutes"].transform("median")

# Align medians to the original row index
orders.loc[completed, "delivery_minutes"] = (
    orders.loc[completed, "delivery_minutes"]
    .fillna(city_median)
)

# Keep cancelled-order delivery times missing
cancelled = orders["status"].astype(str).str.strip().str.lower().eq("cancelled")
orders.loc[cancelled, "delivery_minutes"] = float("nan")

# Save cleaned orders
orders.to_csv(CLEAN / "orders.csv", index=False)

# Validate
status = orders["status"].astype(str).str.strip().str.lower()
print("Rows:", len(orders))
print("Distinct order_id:", orders["order_id"].nunique())
print("Completed:", status.eq("completed").sum())
print("Cancelled:", status.eq("cancelled").sum())
print("Pending:", status.eq("pending").sum())
print("Missing order_amount:", orders["order_amount"].isna().sum())
print("Missing completed delivery times:",
      orders.loc[status.eq("completed"), "delivery_minutes"].isna().sum())
print("Missing cancelled delivery times:",
      orders.loc[status.eq("cancelled"), "delivery_minutes"].isna().sum())
print("Saved:", CLEAN / "orders.csv")
