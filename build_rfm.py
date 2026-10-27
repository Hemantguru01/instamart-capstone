
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"
SNAPSHOT = pd.Timestamp("2025-10-01")

def main():
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")

    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )

    # Only completed orders strictly before the snapshot are eligible.
    completed = orders.loc[
        orders["status"].astype(str).str.lower().eq("completed")
        & orders["order_ts"].notna()
        & (orders["order_ts"] < SNAPSHOT)
    ].copy()

    rfm = (
        completed.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )

    rfm["recency"] = (
    SNAPSHOT - rfm["last_order_ts"].dt.normalize()
).dt.days


    rfm = rfm[
        ["customer_id", "recency", "frequency", "monetary"]
    ].sort_values("customer_id").reset_index(drop=True)

    # Include only customers with at least one eligible completed order.
    assert len(rfm) == 1175, f"Expected 1175 customers, got {len(rfm)}"
    assert rfm["customer_id"].is_unique
    assert (rfm["recency"] >= 0).all()
    assert (rfm["frequency"] >= 1).all()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "rfm.csv"
    rfm.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Customers: {len(rfm)}")
    print("\nRFM reference checks:")
    print("Recency median:", rfm["recency"].median())
    print("Recency min/max:", rfm["recency"].min(), rfm["recency"].max())
    print("Frequency mean/max:", round(rfm["frequency"].mean(), 2), rfm["frequency"].max())
    print("Monetary mean/max:", round(rfm["monetary"].mean(), 2), round(rfm["monetary"].max(), 2))
    print("\nPreview:")
    print(rfm.head())

if __name__ == "__main__":
    main()
