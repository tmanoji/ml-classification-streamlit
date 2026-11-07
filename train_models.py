"""
Train 5 classification models on the Breast Cancer Wisconsin (Diagnostic) dataset,
evaluate them on a held-out test split, and save the models + test data.

Run:  python train_models.py
Outputs:
    model/<model_name>.pkl     trained pipelines (preprocessing + classifier)
    model/metrics.csv          evaluation metrics for all models
    test_data.csv              held-out test split (features + target) for the Streamlit app
    data/breast_cancer.csv     full dataset used for training/testing
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET = "diagnosis"  # 1 = malignant, 0 = benign

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"


def load_dataset() -> pd.DataFrame:
    raw = load_breast_cancer(as_frame=True)
    df = raw.data.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    # sklearn encodes 0 = malignant, 1 = benign; flip so malignant is the positive class
    df[TARGET] = 1 - raw.target
    return df


def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=RANDOM_STATE)),
        ]),
    }


def positive_scores(model, X):
    """Scores for the positive class, used for AUC (SVC has no predict_proba)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X, y) -> dict:
    y_pred = model.predict(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, positive_scores(model, X)),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def model_filename(name: str) -> str:
    return name.lower().replace(" ", "_") + ".pkl"


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    df.to_csv(DATA_DIR / "breast_cancer.csv", index=False)
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1] - 1} features")
    print(f"Class balance: {df[TARGET].value_counts().to_dict()} (1 = malignant)")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(BASE_DIR / "test_data.csv", index=False)

    rows = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / model_filename(name))
        rows.append({"ML Model Name": name, **evaluate(model, X_test, y_test)})

    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.pkl")

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
