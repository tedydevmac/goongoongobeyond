"""SHM cumulative fatigue-damage baseline.

This uses a lightweight rainflow-inspired proxy: turning-point stress ranges
weighted by a configurable S-N exponent.  It is deterministic and works with
CSV files whose stress columns are not known ahead of time.
"""
import os

import numpy as np
import pandas as pd

NAME = "SHM"
OUTPUT_FILENAME = "shm_predictions.csv"
INPUT_MODE = "multi_file"
FILE_HINT = "Upload one or more dynamic stress time-series files."
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "shm")


def is_ready():
    return True


def _damage(path):
    frame = pd.read_csv(path)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    columns = [c for c in numeric.columns if numeric[c].notna().sum() >= 3]
    if not columns:
        raise ValueError(f"SHM input contains no numeric stress columns: {path}")
    damage = 0.0
    for column in columns:
        values = numeric[column].dropna().to_numpy(dtype=float)
        if values.size < 3:
            continue
        turning = values[np.r_[True, np.diff(np.sign(np.diff(values))) != 0, True]]
        ranges = np.abs(np.diff(turning))
        ranges = ranges[ranges > 0]
        if ranges.size:
            # Relative damage proxy; scale by sample count so files of equal
            # duration remain comparable while preserving non-negative output.
            damage += float(np.sum(ranges ** 3) / max(values.size, 1))
    return damage


def run(file_paths, progress_callback=None):
    if not file_paths:
        raise ValueError("At least one SHM input file is required")
    rows = []
    for index, path in enumerate(file_paths):
        rows.append({"file_id": os.path.basename(path), "prediction": _damage(path)})
        if progress_callback:
            progress_callback((index + 1) / len(file_paths))
    return pd.DataFrame(rows, columns=["file_id", "prediction"])
