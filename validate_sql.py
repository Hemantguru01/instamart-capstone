
from pathlib import Path
import sqlite3
import pandas as pd

BASE = Path(__file__).resolve().parent
DB = BASE / "data" / "instamart.db"

with sqlite3.connect(DB) as conn:
    # Cross-check completed-order revenue
    sql_revenue = pd.read_sql_query(
        """
        SELECT ROUND(SUM(order_amount), 2) AS revenue
        FROM orders
        WHERE status = 'completed'
        """,
        conn
    )["revenue"].iloc[0]

    orders = pd.read_sql_query(
        "SELECT status, order_amount FROM orders",
        conn
    )

    pandas_revenue = round(
        orders.loc[
            orders["status"] == "completed",
            "order_amount"
        ].sum(),
        2
    )

    assert sql_revenue == pandas_revenue, (
        f"Revenue mismatch: SQL={sql_revenue}, "
        f"pandas={pandas_revenue}"
    )

    # Verify cleaned table counts
    counts = {}
    for table in ["customers", "products", "orders"]:
        counts[table] = pd.read_sql_query(
            f"SELECT COUNT(*) AS n FROM {table}",
            conn
        )["n"].iloc[0]

    assert counts["customers"] == 1200
    assert counts["products"] == 40
    assert counts["orders"] == 12309

    # Verify order statuses
    statuses = pd.read_sql_query(
        "SELECT status, COUNT(*) AS n FROM orders GROUP BY status",
        conn
    ).set_index("status")["n"].to_dict()

    assert statuses.get("completed") == 11110
    assert statuses.get("cancelled") == 1199
    assert "pending" not in statuses

print("SQL revenue:", sql_revenue)
print("Pandas revenue:", pandas_revenue)
print("Table counts:", counts)
print("Order statuses:", statuses)
print("ALL CHECKPOINTS PASSED")
