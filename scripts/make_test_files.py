"""
Create deliberately broken CSVs for testing the /analyze endpoint.

Run from the project root:
    python scripts/make_test_files.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

clean = pd.read_csv(DATA_DIR / "sample_input.csv")

# 1. Messy file: 10 rows, some with problems
messy = clean.head(10).astype(object).copy()
messy.loc[1, "mean_texture"] = np.nan        # blank cell
messy.loc[2, "mean_area"] = "abc"            # text instead of a number
messy.loc[3, "worst_radius"] = -4.2          # negative measurement
messy.loc[4, "mean_smoothness"] = np.inf     # infinite value
messy.loc[5, "area_error"] = "n/a"           # two problems in one row
messy.loc[5, "worst_texture"] = "??"
messy.loc[6, "mean_area"] = float(messy.loc[6, "mean_area"]) * 100  # valid but extreme (outlier warning)
messy["patient_notes"] = "follow-up"         # extra column the model doesn't use
messy = messy.rename(columns={"mean_radius": "Mean Radius"})  # messy header formatting
messy.to_csv(DATA_DIR / "messy_input.csv", index=False)

# 2. File missing a required column
clean.drop(columns=["worst_symmetry"]).to_csv(DATA_DIR / "missing_column.csv", index=False)

print("Created data/messy_input.csv and data/missing_column.csv")