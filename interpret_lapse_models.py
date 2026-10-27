
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from train_lapse_models import (
    make_pipeline, NUMERIC, CATEGORICAL
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

FEATURES = NUMERIC + CATEGORICAL


def main():
    data = pd.read_csv(OUTPUT_DIR / "lapse_cohort.csv")
    X = data[FEATURES]
    y = data["lapsed"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Refit the same required random-forest grid search on training data.
    rf = GridSearchCV(
        make_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=42)
        ),
        {
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [5, 10, 20],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    best_rf = rf.best_estimator_
    prep_rf = best_rf.named_steps["prep"]
    model_rf = best_rf.named_steps["model"]
    feature_names = prep_rf.get_feature_names_out()

    importances = pd.Series(
        model_rf.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)

    print("Best Random Forest params:", rf.best_params_)
    print("\nTop 5 Random Forest features:")
    print(importances.head(5).round(4).to_string())

    # Refit the same required logistic regression grid search.
    lr = GridSearchCV(
        make_pipeline(LogisticRegression(max_iter=1000)),
        {"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    lr.fit(X_train, y_train)

    best_lr = lr.best_estimator_
    prep_lr = best_lr.named_steps["prep"]
    model_lr = best_lr.named_steps["model"]
    lr_names = prep_lr.get_feature_names_out()
    coefficients = pd.Series(
        model_lr.coef_[0],
        index=lr_names,
    )

    print("\nBest Logistic Regression params:", lr.best_params_)
    print("\nSelected coefficient signs:")
    for term in [
        "num__recency",
        "num__frequency",
        "num__monetary",
    ]:
        if term in coefficients.index:
            print(f"{term}: {coefficients[term]:+.4f}")

    print("\nLargest absolute logistic coefficients:")
    print(
        coefficients.reindex(
            coefficients.abs().sort_values(ascending=False).index
        ).head(10).round(4).to_string()
    )


if __name__ == "__main__":
    main()
