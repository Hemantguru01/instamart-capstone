
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
CLEAN = BASE / "data" / "clean"

customers = pd.read_csv(CLEAN / "customers.csv", keep_default_na=False)
products = pd.read_csv(CLEAN / "products.csv", keep_default_na=False)
orders = pd.read_csv(CLEAN / "orders.csv", keep_default_na=False)
tickets = pd.read_csv(CLEAN / "tickets.csv", keep_default_na=False)

print("=== CLEAN DATA VALIDATION ===")
print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))
print("Tickets:", len(tickets))

print("\n=== UNIQUENESS ===")
print("Customer IDs unique:", customers["customer_id"].is_unique)
print("Product IDs unique:", products["product_id"].is_unique)
print("Order IDs unique:", orders["order_id"].is_unique)
print("Ticket IDs unique:", tickets["ticket_id"].is_unique)

print("\n=== RELATIONSHIPS ===")
print("Orders with unknown customers:",
      (~orders["customer_id"].isin(customers["customer_id"])).sum())
print("Orders with unknown products:",
      (~orders["product_id"].isin(products["product_id"])).sum())
print("Tickets with unknown customers:",
      (~tickets["customer_id"].isin(customers["customer_id"])).sum())

print("\n=== ORDER STATUS ===")
print(orders["status"].value_counts(dropna=False).to_string())

print("\n=== TICKET CATEGORIES ===")
print(tickets["true_category"].value_counts().to_string())
