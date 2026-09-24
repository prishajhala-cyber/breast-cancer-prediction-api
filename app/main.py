"""
Entry point for the API.

Run from the project root:
    fastapi dev app/main.py
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.routes import router
from app.schemas import PredictionInput

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts (before `yield`) and once when it stops (after).
    Loading the model here means it's read from disk once, not on every request."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run: python model/train.py")

    artifact = joblib.load(MODEL_PATH)

    # Fail fast if the API schema and the trained model disagree on the inputs
    schema_fields = list(PredictionInput.model_fields)
    if schema_fields != artifact["feature_names"]:
        raise RuntimeError("PredictionInput fields do not match the model's feature names.")

    app.state.artifact = artifact
    print(f"Model loaded. Metrics: {artifact['metrics']}")
    yield
    # Nothing to clean up on shutdown for now


app = FastAPI(
    title="Breast Cancer Prediction API",
    description="Classifies tumor samples as malignant or benign using a logistic regression model.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    """Send visitors at the base URL to the interactive docs.
    Hugging Face shows the base URL on the Space page, so this makes the demo usable there."""
    return RedirectResponse(url="/docs")