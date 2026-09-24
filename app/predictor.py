"""
Prediction logic shared by the single and batch endpoints.

Keeping this in one place guarantees /predict and /predict/batch
always produce identical results for the same input.
"""

import numpy as np
import pandas as pd


def predict_frame(artifact: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Run the model on any number of rows at once.

    Returns a DataFrame with the same index as X and three columns:
    prediction, probability_malignant, confidence.
    """
    model = artifact["model"]
    target_names = list(artifact["target_names"])  # ["malignant", "benign"]
    X = X[artifact["feature_names"]]  # enforce training column order

    probabilities = model.predict_proba(X)  # shape: (n_rows, 2)
    predicted_index = probabilities.argmax(axis=1)

    return pd.DataFrame(
        {
            "prediction": np.array(target_names)[predicted_index],
            "probability_malignant": probabilities[:, target_names.index("malignant")].round(4),
            "confidence": probabilities.max(axis=1).round(4),
        },
        index=X.index,
    )