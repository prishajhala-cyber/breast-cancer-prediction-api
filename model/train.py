"""
Train a breast cancer classifier and save it for the FastAPI server.

Run from the project root:
    python model/train.py
"""

from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Paths are built relative to this file, so the script works from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "model.joblib"
SAMPLE_CSV_PATH = PROJECT_ROOT / "data" / "sample_input.csv"

RANDOM_STATE = 42


def load_data():
    """Load the dataset and convert column names to snake_case for easier API use."""
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    X.columns = [c.replace(" ", "_") for c in X.columns]  # "mean radius" -> "mean_radius"
    y = data.target  # 0 = malignant, 1 = benign
    return X, y, list(data.target_names)


def build_model():
    """Scale features, then fit a logistic regression. Bundling both in a Pipeline
    means the server applies the exact same scaling at prediction time."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def main():
    X, y, target_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = build_model()
    model.fit(X_train, y_train)

    # Evaluate on held-out data
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC AUC:  {roc_auc:.3f}\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Save the model together with the metadata the API will need
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_names": list(X.columns),
        "target_names": target_names,
        "metrics": {"accuracy": round(accuracy, 4), "roc_auc": round(roc_auc, 4)},
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # Save some unseen test rows (without labels) to use when testing the API later
    SAMPLE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    X_test.head(20).to_csv(SAMPLE_CSV_PATH, index=False)
    print(f"Saved sample input to {SAMPLE_CSV_PATH}")


if __name__ == "__main__":
    main()