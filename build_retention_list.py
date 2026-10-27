
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

NUMERIC = [
    "recency", "frequency", "monetary", "avg_discount_pct",
    "cancellation_rate", "avg_delivery_minutes", "tenure_days", "age",
]
CATEGORICAL = ["city", "gender", "favourite_category"]
FEATURES = NUMERIC + CATEGORICAL


def main():
    data = pd.read_csv(OUTPUT_DIR / "lapse_cohort.csv")
    segments = pd.read_csv(OUTPUT_DIR / "customer_segments.csv")

    X = data[FEATURES]
    y = data["lapsed"].astype(int)

    # Recreate exactly the same stratified split.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    test_rows = data.loc[X_test.index, ["customer_id"]].copy()
    test_rows["actually_lapsed"] = y_test.to_numpy()

    model = joblib.load(OUTPUT_DIR / "lapse_model.joblib")
    probabilities = model.predict_proba(X_test)[:, 1]
    test_rows["lapse_probability"] = probabilities

    # Premium is a post-hoc whole-window segment label, not a model feature.
    test_rows = test_rows.merge(
        segments[["customer_id", "is_premium"]],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    retention_list = (
        test_rows.loc[test_rows["is_premium"].eq(True)]
        .sort_values("lapse_probability", ascending=False)
        .head(10)
        [["customer_id", "lapse_probability", "actually_lapsed"]]
    )

    retention_list.to_csv(
        OUTPUT_DIR / "retention_list.csv", index=False
    )

    print("Premium customers in test set:", int(
        test_rows["is_premium"].eq(True).sum()
    ))
    print("\nTop retention candidates:")
    print(retention_list.to_string(index=False))
    print("\nTruly lapsed in retention list:", int(
        retention_list["actually_lapsed"].sum()
    ))
    print("Saved:", OUTPUT_DIR / "retention_list.csv")


if __name__ == "__main__":
    main()
