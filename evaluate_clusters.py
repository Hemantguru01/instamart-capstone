
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent


def main():
    rfm = pd.read_csv(ROOT / "outputs" / "rfm.csv")
    features = ["recency", "frequency", "monetary"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[features])

    results = []
    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42,
        )
        labels = model.fit_predict(X_scaled)

        results.append({
            "k": k,
            "inertia": round(model.inertia_, 1),
            "silhouette": round(
                silhouette_score(X_scaled, labels), 3
            ),
        })

    table = pd.DataFrame(results)
    print("K-means comparison:")
    print(table.to_string(index=False))

    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    ratios = pca.explained_variance_ratio_

    print("\nPCA explained variance ratios:")
    print([round(float(x), 3) for x in ratios])
    print("Total:", round(float(ratios.sum()), 3))


if __name__ == "__main__":
    main()
