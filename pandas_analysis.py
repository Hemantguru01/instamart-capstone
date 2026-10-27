
from pathlib import Path
import sqlite3
import pandas as pd

BASE = Path(__file__).resolve().parent
DB = BASE / "data" / "instamart.db"

with sqlite3.connect(DB) as conn:
    orders = pd.read_sql_query("SELECT * FROM orders", conn)
    customers = pd.read_sql_query(
        "SELECT customer_id, city FROM customers", conn
    )
    products = pd.read_sql_query(
        "SELECT product_id, category FROM products", conn
    )

# Completed orders only
completed = orders.loc[orders["status"] == "completed"].copy()
completed["order_amount"] = pd.to_numeric(
    completed["order_amount"], errors="coerce"
)

# Add city and category to each completed order
joined = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left",
    validate="many_to_one"
)
# Remove any city column already present in orders
completed = completed.drop(
    columns=["city"],
    errors="ignore"
)

# Join city from customers and category from products
joined = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left",
    validate="many_to_one"
)

joined = joined.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

# Confirm join columns exist before pivoting
print("Joined columns:", joined.columns.tolist())
assert "city" in joined.columns, "City column missing after customer join"
assert "category" in joined.columns, "Category column missing after product join"

# Q2 pandas pivot cross-check
pivot = pd.pivot_table(
    joined,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="All"
)

print("\n=== Q2 PIVOT TABLE (INR) ===")
print(pivot.round(2).to_string())

pivot_total = round(float(pivot.loc["All", "All"]), 2)
print("\nPivot grand total:", pivot_total)
assert pivot_total == 2998668.80, f"Unexpected total: {pivot_total}"

# Per-city named aggregation
city_summary = joined.groupby("city")["order_amount"].agg(
    total_spend="sum",
    mean_order_amount="mean",
    completed_order_count="count"
)

print("\n=== PER-CITY SUMMARY ===")
print(city_summary.round(2).to_string())

# Distribution and IQR outliers
amounts = joined["order_amount"].dropna()
q1 = amounts.quantile(0.25)
median = amounts.median()
q3 = amounts.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

print("\n=== ORDER AMOUNT DISTRIBUTION (INR) ===")
print("Count:", len(amounts))
print("Mean:", round(amounts.mean(), 2))
print("Median:", round(median, 2))
print("Skewness:", round(amounts.skew(), 4))
print("Q1:", round(q1, 2))
print("Q3:", round(q3, 2))
print("IQR:", round(iqr, 2))
print("Lower fence:", round(lower_fence, 2))
print("Upper fence:", round(upper_fence, 2))
print("Outliers above upper fence:", int((amounts > upper_fence).sum()))
print("Outliers below lower fence:", int((amounts < lower_fence).sum()))

# Correlations
numeric_orders = completed[
    ["order_amount", "quantity", "discount_pct"]
].apply(pd.to_numeric, errors="coerce")

print("\n=== CORRELATIONS ===")
print(numeric_orders.corr().round(4).to_string())

print("\nPANDAS CROSS-CHECK COMPLETED")
