"""Door cycle segmentation and abnormal-resistance baseline."""
import os

import numpy as np
import pandas as pd

NAME = "Door"
OUTPUT_FILENAME = "door_predictions.csv"
INPUT_MODE = "single_stream"
FILE_HINT = "Upload the continuous door-cycle stream file (Test.csv)."
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "door")


def is_ready():
    return True


def _column(frame, name):
    matches = [c for c in frame.columns if str(c).strip().lower() == name]
    if not matches:
        raise ValueError(f"Door input is missing required column '{name}'")
    return matches[0]


def _segments(frame):
    opening = pd.to_numeric(frame[_column(frame, "opening")], errors="coerce").fillna(0)
    closing = pd.to_numeric(frame[_column(frame, "closing")], errors="coerce").fillna(0)
    active = (opening > 0) | (closing > 0)
    indices = np.flatnonzero(active.to_numpy())
    if indices.size == 0:
        return []
    groups = np.split(indices, np.where(np.diff(indices) > 2)[0] + 1)
    return [group for group in groups if group.size >= 2]


def _classify(frame, groups):
    current = pd.to_numeric(
        frame[_column(frame, "motor current (ma)")], errors="coerce"
    )
    metrics = []
    for group in groups:
        values = current.iloc[group].fillna(0).to_numpy()
        metrics.append(float(np.mean(np.abs(values))))
    if len(metrics) < 3:
        return ["Normal"] * len(metrics)
    energy = np.asarray(metrics)
    median = np.median(energy)
    mad = np.median(np.abs(energy - median))
    threshold = median + max(3.0 * mad, 0.25 * max(median, 1.0))
    return [
        "Abnormal resistance" if value > threshold else "Normal"
        for value in energy
    ]


def run(file_paths, progress_callback=None):
    if len(file_paths) != 1:
        raise ValueError("Door requires exactly one continuous stream CSV")
    frame = pd.read_csv(file_paths[0])
    if frame.empty:
        raise ValueError("Door input stream is empty")
    time_column = frame.columns[0]
    groups = _segments(frame)
    labels = _classify(frame, groups)
    rows = [
        {
            "start_time": frame.iloc[group[0]][time_column],
            "end_time": frame.iloc[group[-1]][time_column],
            "prediction": label,
        }
        for group, label in zip(groups, labels)
    ]
    if progress_callback:
        progress_callback(1.0)
    return pd.DataFrame(rows, columns=["start_time", "end_time", "prediction"])
