"""
Tests for the Breast Cancer Prediction API.

Run from the project root:
    pytest
    pytest -v          (show each test name)
    pytest -k batch    (run only tests with "batch" in the name)

Requires a trained model: run `python model/train.py` first.
"""

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from app.main import app


# ---------- fixtures (shared setup) ----------

@pytest.fixture(scope="module")
def client():
    """A test client that talks to the app in memory, no running server needed.
    The `with` block triggers the lifespan function, so the model gets loaded."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def samples():
    """20 real samples with snake_case column names, built directly from the
    dataset so tests don't depend on files in data/."""
    data = load_breast_cancer(as_frame=True)
    df = data.data.head(20).copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


@pytest.fixture
def one_sample(samples):
    """A single valid sample as a dict, ready to send as JSON."""
    return samples.iloc[0].to_dict()


def upload(df: pd.DataFrame, filename: str = "test.csv") -> dict:
    """Turn a DataFrame into the `files` argument for a CSV upload."""
    return {"file": (filename, df.to_csv(index=False).encode(), "text/csv")}


# ---------- /health ----------

def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


# ---------- /predict ----------

def test_predict_returns_valid_response(client, one_sample):
    response = client.post("/predict", json=one_sample)
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in ("malignant", "benign")
    assert 0 <= body["probability_malignant"] <= 1
    assert 0.5 <= body["confidence"] <= 1


def test_predict_known_malignant_sample(client, one_sample):
    # The first sample in the dataset is a well-known malignant case
    response = client.post("/predict", json=one_sample)
    assert response.json()["prediction"] == "malignant"


@pytest.mark.parametrize(
    "change, description",
    [
        ({"mean_radius": -5}, "negative value"),
        ({"mean_area": "abc"}, "non-numeric value"),
        ({"unknown_field": 1.0}, "extra field"),
    ],
)
def test_predict_rejects_invalid_values(client, one_sample, change, description):
    bad = {**one_sample, **change}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422, f"Should reject {description}"


def test_predict_rejects_missing_field(client, one_sample):
    bad = {k: v for k, v in one_sample.items() if k != "mean_texture"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


# ---------- /analyze ----------

def test_analyze_clean_file(client, samples):
    response = client.post("/analyze", files=upload(samples))
    assert response.status_code == 200

    body = response.json()
    assert body["total_rows"] == 20
    assert body["valid_rows"] == 20
    assert body["rejected_rows"] == 0


def test_analyze_reports_each_bad_row(client, samples):
    messy = samples.head(5).astype(object).copy()
    messy.loc[1, "mean_texture"] = np.nan   # missing
    messy.loc[2, "mean_area"] = "abc"       # not a number
    messy.loc[3, "worst_radius"] = -1.0     # negative

    body = client.post("/analyze", files=upload(messy)).json()
    assert body["valid_rows"] == 2
    assert body["rejected_rows"] == 3

    issues_by_row = {r["row_number"]: r["issues"] for r in body["rejected"]}
    # Row numbers count the header as row 1, so index 1 -> row 3
    assert issues_by_row[3] == ["mean_texture: missing value"]
    assert "not a number" in issues_by_row[4][0]
    assert "negative" in issues_by_row[5][0]


def test_analyze_accepts_messy_headers_and_ignores_extra_columns(client, samples):
    df = samples.rename(columns={"mean_radius": "Mean Radius"})
    df["patient_notes"] = "follow-up"

    body = client.post("/analyze", files=upload(df)).json()
    assert body["valid_rows"] == 20
    assert any("patient_notes" in w for w in body["warnings"])


def test_analyze_rejects_missing_column(client, samples):
    response = client.post("/analyze", files=upload(samples.drop(columns=["worst_symmetry"])))
    assert response.status_code == 400
    assert "worst_symmetry" in response.json()["detail"]


@pytest.mark.parametrize(
    "filename, content",
    [
        ("notes.txt", b"not a csv"),  # wrong file type
        ("empty.csv", b""),           # empty file
    ],
)
def test_analyze_rejects_bad_files(client, filename, content):
    response = client.post("/analyze", files={"file": (filename, content, "text/csv")})
    assert response.status_code == 400


# ---------- /predict/batch ----------

def test_batch_accounts_for_every_row(client, samples):
    messy = samples.astype(object).copy()
    messy.loc[0, "mean_area"] = "abc"

    body = client.post("/predict/batch", files=upload(messy)).json()
    assert body["predicted_rows"] + body["rejected_rows"] == body["total_rows"] == 20
    assert sum(body["class_counts"].values()) == body["predicted_rows"]


def test_batch_matches_single_predictions(client, samples):
    """The same sample must get the same answer from both endpoints."""
    batch = client.post("/predict/batch", files=upload(samples)).json()["predictions"]

    for i in range(5):
        single = client.post("/predict", json=samples.iloc[i].to_dict()).json()
        assert batch[i]["prediction"] == single["prediction"]
        assert batch[i]["probability_malignant"] == pytest.approx(single["probability_malignant"])


def test_batch_flags_outliers(client, samples):
    df = samples.copy()
    df.loc[0, "mean_area"] = df.loc[0, "mean_area"] * 100

    predictions = client.post("/predict/batch", files=upload(df)).json()["predictions"]
    assert predictions[0]["outlier_warning"] is True
    assert predictions[1]["outlier_warning"] is False


def test_batch_handles_all_rows_rejected(client, samples):
    df = samples.head(3).astype(object).copy()
    df["mean_area"] = "abc"

    response = client.post("/predict/batch", files=upload(df))
    assert response.status_code == 200
    assert response.json()["predicted_rows"] == 0
    assert response.json()["rejected_rows"] == 3


def test_batch_csv_download(client, samples):
    messy = samples.head(5).astype(object).copy()
    messy.loc[2, "mean_area"] = "abc"

    response = client.post("/predict/batch?format=csv", files=upload(messy))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]

    result = pd.read_csv(io.StringIO(response.text))
    assert len(result) == 5                                  # one line per input row
    assert list(result["row_number"]) == [2, 3, 4, 5, 6]     # original order
    assert (result["status"] == "rejected").sum() == 1