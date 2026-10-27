
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"

CUTOFF = pd.Timestamp("2025-09-01")
LABEL_END = pd.Timestamp("2025-10-01")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    orders = pd.read_csv(DATA_DIR / "orders.csv")
    customers = pd.read_csv(DATA_DIR / "customers.csv")

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    )

    # Only completed orders before the cutoff are allowed in features.
    before = orders.loc[
        orders["order_ts"].notna()
        & (orders["order_ts"] < CUTOFF)
    ].copy()

    completed_before = before.loc[
        before["status"].astype(str).str.lower().eq("completed")
    ].copy()

    # Cohort: customers with at least one completed order before cutoff.
    cohort_ids = completed_before["customer_id"].unique()
    cohort = customers.loc[
        customers["customer_id"].isin(cohort_ids)
    ].copy()

    # Behaviour features from completed orders only.
    rfm = (
        completed_before.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
            avg_discount_pct=("discount_pct", "mean"),
            avg_delivery_minutes=("delivery_minutes", "mean"),
        )
        .reset_index()
    )
    rfm["recency"] = (
        CUTOFF - rfm["last_order_ts"].dt.normalize()
    ).dt.days

    # Cancellation rate: cancelled / all final-status orders before cutoff.
    final_orders = before.loc[
        before["status"].astype(str).str.lower().isin(
            ["completed", "cancelled"]
        )
    ]
    status_counts = (
        final_orders.assign(
            is_cancelled=final_orders["status"]
            .astype(str).str.lower().eq("cancelled").astype(int)
        )
        .groupby("customer_id")
        .agg(
            final_order_count=("order_id", "nunique"),
            cancelled_count=("is_cancelled", "sum"),
        )
        .reset_index()
    )
    status_counts["cancellation_rate"] = (
        status_counts["cancelled_count"]
        / status_counts["final_order_count"]
    )

    # Most frequent product category, using only pre-cutoff completed orders.
    products = pd.read_csv(DATA_DIR / "products.csv")
    completed_with_category = completed_before.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    favourite_category = (
        completed_with_category.groupby(
            ["customer_id", "category"]
        ).size().rename("n").reset_index()
        .sort_values(
            ["customer_id", "n", "category"],
            ascending=[True, False, True],
        )
        .drop_duplicates("customer_id")
        [["customer_id", "category"]]
        .rename(columns={"category": "favourite_category"})
    )

    # Build features and calculate the label separately.
    data = (
        cohort.merge(rfm, on="customer_id", how="left", validate="one_to_one")
        .merge(
            status_counts[["customer_id", "cancellation_rate"]],
            on="customer_id", how="left", validate="one_to_one",
        )
        .merge(favourite_category, on="customer_id", how="left",
               validate="one_to_one")
    )

    data["tenure_days"] = (
        CUTOFF - data["signup_date"].dt.normalize()
    ).dt.days

    # Label window: 2025-09-01 through 2025-09-30.
    september_completed = orders.loc[
        orders["order_ts"].ge(CUTOFF)
        & orders["order_ts"].lt(LABEL_END)
        & orders["status"].astype(str).str.lower().eq("completed")
    ]
    september_customer_ids = set(september_completed["customer_id"])
    data["lapsed"] = (
        ~data["customer_id"].isin(september_customer_ids)
    ).astype(int)

    # Confirm no post-cutoff orders entered feature construction.
    assert (before["order_ts"] < CUTOFF).all()
    assert not data["customer_id"].duplicated().any()

    # Missing values are handled explicitly.
    numeric_features = [
        "recency", "frequency", "monetary", "avg_discount_pct",
        "cancellation_rate", "avg_delivery_minutes", "tenure_days", "age",
    ]
    for col in numeric_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median())

    for col in ["city", "gender", "favourite_category"]:
        data[col] = data[col].fillna("unknown").astype(str)

    feature_columns = numeric_features + [
        "city", "gender", "favourite_category"
    ]
    assert not data[feature_columns].isna().any().any(), (
        "Missing feature values remain"
    )

    # Fixed split, as required by the brief.
    X = data[feature_columns]
    y = data["lapsed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Cohort customers:", len(data))
    print("Lapsed:", int(y.sum()))
    print("Lapsed rate (%):", round(100 * y.mean(), 2))
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Feature columns:", feature_columns)
    print("Missing feature values:", int(data[feature_columns].isna().sum().sum()))

    # Save intermediate data for the modelling step.
    data.to_csv(OUTPUT_DIR / "lapse_cohort.csv", index=False)
    X_train.assign(lapsed=y_train).to_csv(
        OUTPUT_DIR / "lapse_train_raw.csv", index=True, index_label="row_id"
    )
    X_test.assign(lapsed=y_test).to_csv(
        OUTPUT_DIR / "lapse_test_raw.csv", index=True, index_label="row_id"
    )


if __name__ == "__main__":
    main()
