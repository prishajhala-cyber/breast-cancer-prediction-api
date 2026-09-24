"""API endpoints."""

from typing import Literal

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

from app.pipeline import PipelineError, ValidationResult, outlier_mask, run_pipeline, summary_stats
from app.predictor import predict_frame
from app.schemas import (
    AnalyzeResponse,
    BatchResponse,
    HealthResponse,
    PredictionInput,
    PredictionOutput,
)

router = APIRouter()


# ---------- helpers ----------

async def process_upload(file: UploadFile, artifact: dict) -> tuple:
    """Shared by /analyze and /predict/batch: check the file type, read it,
    and run the validation pipeline. Returns (filename, ValidationResult)."""
    filename = file.filename or "upload.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    content = await file.read()
    try:
        result = run_pipeline(
            content,
            feature_names=artifact["feature_names"],
            scaler=artifact["model"].named_steps["scaler"],
        )
    except PipelineError as e:
        # File-level problem: tell the user exactly what's wrong with the file
        raise HTTPException(status_code=400, detail=str(e))
    return filename, result


def build_results_csv(predictions: pd.DataFrame, result: ValidationResult) -> str:
    """One row per input row, predicted or rejected, sorted to match the original file."""
    predicted = predictions.assign(status="predicted", issues="")
    rejected = pd.DataFrame(
        [
            {"row_number": r["row_number"], "status": "rejected", "issues": "; ".join(r["issues"])}
            for r in result.rejected
        ]
    )
    combined = pd.concat([predicted, rejected], ignore_index=True).sort_values("row_number")
    columns = [
        "row_number", "status", "prediction", "probability_malignant",
        "confidence", "outlier_warning", "issues",
    ]
    return combined.reindex(columns=columns).to_csv(index=False)


# ---------- endpoints ----------

@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request):
    """Confirms the server is running and the model is loaded."""
    artifact = getattr(request.app.state, "artifact", None)
    return HealthResponse(
        status="ok",
        model_loaded=artifact is not None,
        model_metrics=artifact["metrics"] if artifact else None,
    )


@router.post("/predict", response_model=PredictionOutput, tags=["prediction"])
def predict(payload: PredictionInput, request: Request):
    """Classify a single tumor sample as malignant or benign."""
    artifact = request.app.state.artifact
    X = pd.DataFrame([payload.model_dump()])
    row = predict_frame(artifact, X).iloc[0]
    return PredictionOutput(
        prediction=str(row["prediction"]),
        probability_malignant=float(row["probability_malignant"]),
        confidence=float(row["confidence"]),
    )


@router.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
async def predict_batch(
    request: Request,
    file: UploadFile = File(..., description="CSV file with one sample per row"),
    output_format: Literal["json", "csv"] = Query(
        "json", alias="format", description="json for an API response, csv for a downloadable file"
    ),
):
    """Validate a CSV, predict every valid row, and report rejected rows."""
    artifact = request.app.state.artifact
    filename, result = await process_upload(file, artifact)

    # Predict all valid rows in one call (much faster than looping row by row)
    if result.valid.empty:
        predictions = pd.DataFrame(
            columns=["row_number", "prediction", "probability_malignant", "confidence", "outlier_warning"]
        )
    else:
        predictions = predict_frame(artifact, result.valid)
        predictions["outlier_warning"] = outlier_mask(
            result.valid, artifact["model"].named_steps["scaler"]
        )
        predictions.insert(0, "row_number", predictions.index + 2)  # same numbering as rejections

    if output_format == "csv":
        return Response(
            content=build_results_csv(predictions, result),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="predictions_{filename}"'},
        )

    class_counts = {name: 0 for name in artifact["target_names"]}
    class_counts.update(predictions["prediction"].value_counts().to_dict())

    return BatchResponse(
        filename=filename,
        total_rows=result.total_rows,
        predicted_rows=len(predictions),
        rejected_rows=len(result.rejected),
        class_counts=class_counts,
        predictions=predictions.to_dict(orient="records"),
        rejected=result.rejected,
        warnings=result.warnings,
    )


@router.post("/analyze", response_model=AnalyzeResponse, tags=["data quality"])
async def analyze(
    request: Request,
    file: UploadFile = File(..., description="CSV file with one sample per row"),
):
    """Check a CSV for data quality problems without running predictions."""
    artifact = request.app.state.artifact
    filename, result = await process_upload(file, artifact)

    return AnalyzeResponse(
        filename=filename,
        total_rows=result.total_rows,
        valid_rows=len(result.valid),
        rejected_rows=len(result.rejected),
        rejected=result.rejected,
        column_issues=result.column_issues,
        warnings=result.warnings,
        summary_stats=summary_stats(result.valid),
    )