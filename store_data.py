
from pathlib import Path
import sqlite3
import pandas as pd

BASE = Path(__file__).resolve().parent
CLEAN = BASE / "data" / "clean"
DB_PATH = BASE / "data" / "instamart.db"

customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

with sqlite3.connect(DB_PATH) as conn:
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    orders.to_sql("orders", conn, if_exists="replace", index=False)

    for table in ["customers", "products", "orders"]:
        count = pd.read_sql_query(
            f"SELECT COUNT(*) AS n FROM {table}", conn
        )["n"].iloc[0]
        print(f"{table}: {count} rows")

print(f"Database saved: {DB_PATH}")
