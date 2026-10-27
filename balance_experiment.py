
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from train_lapse_models import make_pipeline, NUMERIC, CATEGORICAL
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def report(name, y_true, predictions):
    print(name)
    print("Precision:", round(
        precision_score(y_true, predictions, zero_division=0), 4
    ))
    print("Recall:", round(
        recall_score(y_true, predictions, zero_division=0), 4
    ))
    print("F1:", round(
        f1_score(y_true, predictions, zero_division=0), 4
    ))


def main():
    data = pd.read_csv(OUTPUT_DIR / "lapse_cohort.csv")
    X = data[NUMERIC + CATEGORICAL]
    y = data["lapsed"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Load the best C found by the earlier logistic grid search.
    # This reuses the model choice from the required tuning stage.
    import joblib
    selected = joblib.load(OUTPUT_DIR / "lapse_model.joblib")
    best_C = selected.named_steps["model"].C

    # Unbalanced baseline: same tuned logistic model.
    unbalanced = make_pipeline(LogisticRegression(
        C=best_C, max_iter=1000
    ))
    unbalanced.fit(X_train, y_train)
    original_pred = unbalanced.predict(X_test)
    report("UNBALANCED MODEL", y_test, original_pred)

    # Oversample the lapsed class in training data only.
    train = X_train.copy()
    train["lapsed"] = y_train.to_numpy()

    counts = train["lapsed"].value_counts()
    target_count = int(counts.max())
    minority = train.loc[train["lapsed"] == 1]

    duplicates = minority.sample(
        n=target_count - int(counts.loc[1]),
        replace=True,
        random_state=42,
    )
    balanced = pd.concat([train, duplicates], ignore_index=True)

    print("\nTraining counts before:")
    print(counts.sort_index().to_dict())
    print("Training counts after:")
    print(balanced["lapsed"].value_counts().sort_index().to_dict())

    X_balanced = balanced[NUMERIC + CATEGORICAL]
    y_balanced = balanced["lapsed"].astype(int)

    balanced_model = make_pipeline(LogisticRegression(
        C=best_C, max_iter=1000
    ))
    balanced_model.fit(X_balanced, y_balanced)
    balanced_pred = balanced_model.predict(X_test)
    report("BALANCED TRAINING MODEL", y_test, balanced_pred)


if __name__ == "__main__":
    main()
