"""
Request and response models.

Pydantic checks every request against these before your code runs:
missing fields, wrong types, negative values, and unknown fields all
get rejected automatically with a clear 422 error.
"""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# All measurements in this dataset are non-negative
NonNegFloat = Annotated[float, Field(ge=0)]


class PredictionInput(BaseModel):
    """One tumor sample: 30 measurements computed from a digitized image
    of a fine needle aspirate. Field order must match the training data."""

    model_config = ConfigDict(
        extra="forbid",  # reject fields the model doesn't know about (catches typos)
        json_schema_extra={
            "example": {
                "mean_radius": 17.99,
                "mean_texture": 10.38,
                "mean_perimeter": 122.8,
                "mean_area": 1001.0,
                "mean_smoothness": 0.1184,
                "mean_compactness": 0.2776,
                "mean_concavity": 0.3001,
                "mean_concave_points": 0.1471,
                "mean_symmetry": 0.2419,
                "mean_fractal_dimension": 0.07871,
                "radius_error": 1.095,
                "texture_error": 0.9053,
                "perimeter_error": 8.589,
                "area_error": 153.4,
                "smoothness_error": 0.006399,
                "compactness_error": 0.04904,
                "concavity_error": 0.05373,
                "concave_points_error": 0.01587,
                "symmetry_error": 0.03003,
                "fractal_dimension_error": 0.006193,
                "worst_radius": 25.38,
                "worst_texture": 17.33,
                "worst_perimeter": 184.6,
                "worst_area": 2019.0,
                "worst_smoothness": 0.1622,
                "worst_compactness": 0.6656,
                "worst_concavity": 0.7119,
                "worst_concave_points": 0.2654,
                "worst_symmetry": 0.4601,
                "worst_fractal_dimension": 0.1189,
            }
        },
    )

    # Mean values across cell nuclei
    mean_radius: NonNegFloat
    mean_texture: NonNegFloat
    mean_perimeter: NonNegFloat
    mean_area: NonNegFloat
    mean_smoothness: NonNegFloat
    mean_compactness: NonNegFloat
    mean_concavity: NonNegFloat
    mean_concave_points: NonNegFloat
    mean_symmetry: NonNegFloat
    mean_fractal_dimension: NonNegFloat

    # Standard error of each measurement
    radius_error: NonNegFloat
    texture_error: NonNegFloat
    perimeter_error: NonNegFloat
    area_error: NonNegFloat
    smoothness_error: NonNegFloat
    compactness_error: NonNegFloat
    concavity_error: NonNegFloat
    concave_points_error: NonNegFloat
    symmetry_error: NonNegFloat
    fractal_dimension_error: NonNegFloat

    # "Worst" values (mean of the three largest)
    worst_radius: NonNegFloat
    worst_texture: NonNegFloat
    worst_perimeter: NonNegFloat
    worst_area: NonNegFloat
    worst_smoothness: NonNegFloat
    worst_compactness: NonNegFloat
    worst_concavity: NonNegFloat
    worst_concave_points: NonNegFloat
    worst_symmetry: NonNegFloat
    worst_fractal_dimension: NonNegFloat


class PredictionOutput(BaseModel):
    prediction: Literal["malignant", "benign"]
    probability_malignant: float = Field(description="Model's probability that the sample is malignant (0 to 1)")
    confidence: float = Field(description="Probability of the predicted class (0.5 to 1)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_metrics: Optional[dict] = None


class RejectedRow(BaseModel):
    row_number: int = Field(description="Row number in the file, counting the header as row 1")
    issues: list[str]


class AnalyzeResponse(BaseModel):
    filename: str
    total_rows: int
    valid_rows: int
    rejected_rows: int
    rejected: list[RejectedRow]
    column_issues: dict[str, dict[str, int]] = Field(
        description="Per-column counts of each issue type, only for columns with problems"
    )
    warnings: list[str]
    summary_stats: dict[str, dict[str, float]] = Field(
        description="Mean, min, and max of each column across valid rows"
    )


class BatchPrediction(BaseModel):
    row_number: int = Field(description="Row number in the file, counting the header as row 1")
    prediction: Literal["malignant", "benign"]
    probability_malignant: float
    confidence: float
    outlier_warning: bool = Field(
        description="True if any value in this row is far outside the training data"
    )


class BatchResponse(BaseModel):
    filename: str
    total_rows: int
    predicted_rows: int
    rejected_rows: int
    class_counts: dict[str, int] = Field(description="Number of rows predicted as each class")
    predictions: list[BatchPrediction]
    rejected: list[RejectedRow]
    warnings: list[str]