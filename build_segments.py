
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"

FEATURES = ["recency", "frequency", "monetary"]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    STATE_DIR.mkdir(exist_ok=True)

    rfm = pd.read_csv(OUTPUT_DIR / "rfm.csv")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[FEATURES])

    model = KMeans(
        n_clusters=4,
        n_init=10,
        random_state=42,
    )
    rfm["segment_id"] = model.fit_predict(X_scaled)

    profiles = rfm.groupby("segment_id")[FEATURES].mean()
    print("Segment profiles:")
    print(profiles.round(2))

    # Assign business names based on the observed cluster profiles.
    names = {}
    for segment_id, profile in profiles.iterrows():
        if profile["monetary"] == profiles["monetary"].max():
            name = "Premium High-Spend"
        elif (
            profile["frequency"] == profiles["frequency"].max()
        ):
            name = "Loyal Frequent"
        elif profile["recency"] == profiles["recency"].max():
            name = "At-Risk Inactive"
        else:
            name = "Occasional Value"
        names[int(segment_id)] = name

    rfm["segment_name"] = rfm["segment_id"].map(names)
    premium_id = int(profiles["monetary"].idxmax())
    rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

    result = rfm[
        [
            "customer_id", "recency", "frequency", "monetary",
            "segment_id", "segment_name", "is_premium",
        ]
    ]
    result.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)

    joblib.dump(
        {"scaler": scaler, "model": model, "features": FEATURES},
        STATE_DIR / "segmentation.joblib",
    )

    print("\nSegment counts:")
    print(result["segment_id"].value_counts().sort_index())
    print("\nSegment names:")
    print(result.groupby(
        ["segment_id", "segment_name", "is_premium"]
    ).size())
    print("\nSaved:", OUTPUT_DIR / "customer_segments.csv")
    print("Premium segment ID:", premium_id)


if __name__ == "__main__":
    main()
