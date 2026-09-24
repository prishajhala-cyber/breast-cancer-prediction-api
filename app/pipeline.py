"""
CSV ingestion and validation pipeline.

Flow:
    raw bytes -> read_csv -> check_columns -> validate_rows -> ValidationResult

Two kinds of problems:
    - File-level (unreadable, empty, missing columns): the whole upload is
      rejected with a PipelineError, because nothing in it can be trusted.
    - Row-level (missing, non-numeric, negative, infinite values): only the
      bad rows are rejected, each with a list of reasons. Good rows continue.
"""

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_ROWS = 10_000
OUTLIER_Z = 5.0  # flag values more than 5 standard deviations from the training mean


class PipelineError(Exception):
    """A problem with the file as a whole. The entire upload is rejected."""


@dataclass
class ValidationResult:
    valid: pd.DataFrame  # clean numeric rows, ready for the model
    rejected: list = field(default_factory=list)  # [{"row_number": int, "issues": [str]}]
    total_rows: int = 0
    extra_columns: list = field(default_factory=list)
    column_issues: dict = field(default_factory=dict)  # {column: {issue_type: count}}
    warnings: list = field(default_factory=list)


def normalize_column(name) -> str:
    """'Mean Radius ' -> 'mean_radius', so small formatting differences don't fail the upload."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def read_csv(content: bytes) -> pd.DataFrame:
    """Parse raw bytes into a DataFrame of strings. Reading everything as text
    lets us tell the difference between a blank cell and a cell containing 'abc'."""
    if len(content) == 0:
        raise PipelineError("The file is empty.")
    if len(content) > MAX_FILE_BYTES:
        raise PipelineError(f"The file is larger than the {MAX_FILE_BYTES // (1024 * 1024)} MB limit.")

    try:
        df = pd.read_csv(io.BytesIO(content), dtype=str, skipinitialspace=True)
    except pd.errors.EmptyDataError:
        raise PipelineError("The file has no data.")
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        raise PipelineError(f"Could not read the file as CSV: {e}")

    if df.empty:
        raise PipelineError("The file has a header row but no data rows.")
    if len(df) > MAX_ROWS:
        raise PipelineError(f"The file has {len(df)} rows; the limit is {MAX_ROWS}.")
    return df


def check_columns(df: pd.DataFrame, feature_names: list) -> tuple:
    """Normalize column names, confirm every required column is present,
    and drop (but report) any extra columns."""
    df = df.copy()
    df.columns = [normalize_column(c) for c in df.columns]

    duplicates = sorted(set(df.columns[df.columns.duplicated()]))
    if duplicates:
        raise PipelineError(f"Duplicate column(s) after normalizing names: {', '.join(duplicates)}")

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise PipelineError(f"Missing {len(missing)} required column(s): {', '.join(missing)}")

    extra = [c for c in df.columns if c not in feature_names]
    return df[feature_names], extra


def validate_rows(raw: pd.DataFrame, feature_names: list) -> tuple:
    """Check every cell. Returns (valid numeric rows, rejected rows, per-column issue counts)."""
    numeric = raw.apply(pd.to_numeric, errors="coerce").astype(float)

    is_missing = raw.isna()                      # blank, 'NA', 'n/a', etc.
    is_non_numeric = raw.notna() & numeric.isna()  # had text that isn't a number
    is_infinite = np.isinf(numeric)
    is_negative = numeric < 0

    bad_cells = is_missing | is_non_numeric | is_infinite | is_negative
    bad_rows = bad_cells.any(axis=1)

    rejected = []
    for idx in raw.index[bad_rows]:
        issues = []
        for col in feature_names:
            if is_missing.at[idx, col]:
                issues.append(f"{col}: missing value")
            elif is_non_numeric.at[idx, col]:
                issues.append(f"{col}: '{raw.at[idx, col]}' is not a number")
            elif is_infinite.at[idx, col]:
                issues.append(f"{col}: infinite value")
            elif is_negative.at[idx, col]:
                issues.append(f"{col}: {numeric.at[idx, col]} is negative")
        # +2 converts the 0-based index to the row number in a spreadsheet (header is row 1)
        rejected.append({"row_number": int(idx) + 2, "issues": issues})

    column_issues = {}
    for col in feature_names:
        counts = {
            "missing": int(is_missing[col].sum()),
            "non_numeric": int(is_non_numeric[col].sum()),
            "infinite": int(is_infinite[col].sum()),
            "negative": int(is_negative[col].sum()),
        }
        counts = {k: v for k, v in counts.items() if v > 0}
        if counts:
            column_issues[col] = counts

    return numeric.loc[~bad_rows], rejected, column_issues


def find_outliers(valid: pd.DataFrame, scaler) -> list:
    """Warn (but don't reject) when values sit far outside what the model saw in training.
    Uses the mean and standard deviation the StandardScaler learned during training."""
    if valid.empty or scaler is None:
        return []
    z_scores = (valid.to_numpy() - scaler.mean_) / scaler.scale_
    counts = (np.abs(z_scores) > OUTLIER_Z).sum(axis=0)

    warnings = []
    for col, n in zip(valid.columns, counts):
        if n > 0:
            warnings.append(
                f"{col}: {n} value(s) more than {OUTLIER_Z:g} standard deviations from the "
                f"training data. Predictions for these rows may be unreliable."
            )
    return warnings


def outlier_mask(valid: pd.DataFrame, scaler) -> pd.Series:
    """True for each row that has at least one value far outside the training data."""
    if valid.empty or scaler is None:
        return pd.Series(False, index=valid.index)
    z_scores = (valid.to_numpy() - scaler.mean_) / scaler.scale_
    return pd.Series((np.abs(z_scores) > OUTLIER_Z).any(axis=1), index=valid.index)


def summary_stats(valid: pd.DataFrame) -> dict:
    """Mean, min, and max of each column across valid rows."""
    if valid.empty:
        return {}
    stats = valid.agg(["mean", "min", "max"]).T.round(4)
    return stats.to_dict(orient="index")


def run_pipeline(content: bytes, feature_names: list, scaler=None) -> ValidationResult:
    raw = read_csv(content)
    raw, extra = check_columns(raw, feature_names)
    valid, rejected, column_issues = validate_rows(raw, feature_names)

    warnings = []
    if extra:
        warnings.append(f"Ignored {len(extra)} column(s) not used by the model: {', '.join(extra)}")
    if valid.empty:
        warnings.append("No rows passed validation.")
    warnings.extend(find_outliers(valid, scaler))

    return ValidationResult(
        valid=valid,
        rejected=rejected,
        total_rows=len(raw),
        extra_columns=extra,
        column_issues=column_issues,
        warnings=warnings,
    )