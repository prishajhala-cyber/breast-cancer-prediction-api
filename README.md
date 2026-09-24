---
title: Breast Cancer Prediction API
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Breast Cancer Prediction API

A FastAPI service that validates tabular clinical data and classifies tumor samples as malignant or benign. Upload a CSV and get back predictions for every valid row, plus a row-by-row report explaining exactly why any row was rejected.

**Live demo:** `https://YOUR-USERNAME-YOUR-SPACE-NAME.hf.space/docs`

## What it does

| Endpoint | Purpose |
|---|---|
| `GET /health` | Confirms the server is running and the model is loaded |
| `POST /predict` | Classifies one sample sent as JSON, with strict input validation |
| `POST /analyze` | Data-quality report on a CSV: rejected rows with reasons, per-column issue counts, outlier warnings, summary stats |
| `POST /predict/batch` | Validates a CSV and predicts every valid row. Returns JSON, or a downloadable CSV with one line per input row (`?format=csv`) |

## How it works

**Model.** A scikit-learn pipeline (standard scaling + logistic regression) trained on the Breast Cancer Wisconsin dataset: 569 samples, 30 features computed from digitized images of fine needle aspirates. On a stratified 20% held-out test set it reaches 98.2% accuracy and 0.995 ROC AUC, with 98% recall on malignant cases.

**Validation pipeline.** Uploaded CSVs pass through a pandas pipeline that separates two kinds of problems:

- *File-level* issues (empty file, wrong type, missing required columns) reject the whole upload, since nothing in it can be trusted.
- *Row-level* issues (missing, non-numeric, negative, or infinite values) reject only the affected rows, each with a specific reason, while valid rows continue to prediction.

Rows with missing values are rejected rather than imputed, because silently guessing a clinical measurement would hide uncertainty. Values more than 5 standard deviations from the training distribution are flagged as outliers but still predicted, so users know which results to treat with caution. Column headers are normalized (`Mean Radius` → `mean_radius`) and extra columns are ignored with a warning.

**Consistency.** Single and batch endpoints share one prediction function, and a test confirms they return identical results for the same sample.

## Project structure

```
app/
  main.py        App setup; loads the model once at startup
  routes.py      API endpoints
  schemas.py     Pydantic request/response models
  pipeline.py    CSV parsing, validation, and outlier detection
  predictor.py   Shared prediction logic
model/
  train.py       Trains and saves the model
scripts/
  make_test_files.py   Generates deliberately broken CSVs for testing
tests/
  test_api.py    pytest suite covering every endpoint and edge case
Dockerfile       Trains the model and runs tests during the build
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python model/train.py
fastapi dev app/main.py
```

Then open http://127.0.0.1:8000/docs.

## Run tests

```bash
pytest -v
```

## Run with Docker

```bash
docker build -t cancer-api .
docker run --rm -p 7860:7860 cancer-api
```

Then open http://localhost:7860/docs. The Docker build trains the model and runs the full test suite, so a failing test stops the build.

## Try it

Sample files are in `data/`: `sample_input.csv` (clean), `messy_input.csv` (deliberate errors), and `missing_column.csv`. Upload any of them to `/analyze` or `/predict/batch` from the `/docs` page.

*This project is a portfolio demonstration and is not intended for clinical use.*