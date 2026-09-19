"""ACV refrigerant-leak localisation baseline.

The baseline is deliberately schema-tolerant because ACV cases do not all
contain the same per-car telemetry.  It scores each car by how anomalous its
numeric telemetry is relative to the other cars in the same case, while
ignoring identifier and timestamp columns.
"""
import os
import re

import numpy as np
import pandas as pd

NAME = "ACV"
OUTPUT_FILENAME = "acv_predictions.csv"
INPUT_MODE = "multi_file"
FILE_HINT = "Upload one or more ACV cabin-temperature / control-mode telemetry files."
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "acv")


def is_ready():
    return True


_CAR_RE = re.compile(r"^Car\s+(.+?)\s+-\s+(.+)$", re.IGNORECASE)


def _read_case(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _rank_cars(frame):
    cars = {}
    for column in frame.columns:
        match = _CAR_RE.match(str(column).strip())
        if match:
            cars.setdefault(match.group(1), []).append(column)
    if not cars:
        raise ValueError("ACV input has no columns matching 'Car <id> - <parameter>'")

    scores = {car: [] for car in cars}
    tie_breakers = {car: [] for car in cars}
    parameters = {}
    for car, columns in cars.items():
        for column in columns:
            parameters.setdefault(_CAR_RE.match(str(column).strip()).group(2), {})[car] = column
    for columns_by_car in parameters.values():
        values = pd.DataFrame(
            {
                car: pd.to_numeric(frame[column], errors="coerce")
                for car, column in columns_by_car.items()
            }
        )
        if values.shape[1] < 2:
            continue
        center = values.median(axis=1)
        deviations = values.sub(center, axis=0).abs()
        scale = deviations.median(axis=1)
        scale = scale.where(scale > 0, deviations.max(axis=1))
        scale = scale.replace(0, 1.0)
        normalized = deviations.divide(scale, axis=0).replace(
            [np.inf, -np.inf], np.nan
        )
        for car in columns_by_car:
            scores[car].append(float(normalized[car].mean()))
            tie_breakers[car].append(float(values[car].mean()))
    scores = {
        car: float(np.nanmean(values)) if values else 0.0
        for car, values in scores.items()
    }
    tie_breakers = {
        car: float(np.nanmean(values)) if values else 0.0
        for car, values in tie_breakers.items()
    }
    return sorted(cars, key=lambda car: (-scores[car], -tie_breakers[car], car))


def run(file_paths, progress_callback=None):
    if not file_paths:
        raise ValueError("At least one ACV input file is required")
    rows = []
    for index, path in enumerate(file_paths):
        ranked = _rank_cars(_read_case(path))
        rows.append({"file_id": os.path.basename(path), "ranked_cars": "|".join(ranked)})
        if progress_callback:
            progress_callback((index + 1) / len(file_paths))
    return pd.DataFrame(rows, columns=["file_id", "ranked_cars"])
