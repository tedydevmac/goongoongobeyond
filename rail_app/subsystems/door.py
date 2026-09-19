"""Door cycle segmentation and trained abnormal-resistance inference."""
import os
from functools import lru_cache

import numpy as np
import pandas as pd

NAME = "Door"
OUTPUT_FILENAME = "door_predictions.csv"
INPUT_MODE = "single_stream"
FILE_HINT = "Upload the continuous door-cycle stream file (Test.csv)."
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "door")
MODEL_PATH = os.path.join(MODEL_DIR, "status_model.keras")
SCALING_PATH = os.path.join(MODEL_DIR, "status_model_scaling.npz")
MAX_SEQUENCE_LENGTH = 190


def is_ready():
    """Return whether the Door model, normalization, and TensorFlow are available."""
    if not (os.path.isfile(MODEL_PATH) and os.path.isfile(SCALING_PATH)):
        return False
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def _load_artifacts():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "The Door model requires TensorFlow. Install the dependencies from "
            "rail_app/requirements.txt and restart the app."
        ) from error
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Door model artifact not found: {MODEL_PATH}")
    if not os.path.isfile(SCALING_PATH):
        raise FileNotFoundError(f"Door scaling artifact not found: {SCALING_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaling = np.load(SCALING_PATH)
    if "mean" not in scaling or "std" not in scaling:
        raise ValueError("Door scaling artifact must contain 'mean' and 'std' arrays")
    return model, scaling["mean"].astype(np.float32), scaling["std"].astype(np.float32)


def _parse_timestamps(values):
    parts = values.astype(str).str.split("-", expand=True)
    if parts.shape[1] != 7:
        raise ValueError(
            "Door Datetime values must use "
            "Year-Month-Day-Hour-Minute-Second-Millisecond format"
        )
    try:
        components = parts.astype(int)
    except ValueError as error:
        raise ValueError("Door Datetime contains a non-numeric timestamp component") from error
    return pd.to_datetime(
        {
            "year": components[0],
            "month": components[1],
            "day": components[2],
            "hour": components[3],
            "minute": components[4],
            "second": components[5],
        }
    ) + pd.to_timedelta(components[6], unit="ms")


def _create_segments(frame):
    if "Datetime" not in frame.columns:
        raise ValueError("Door input is missing the required 'Datetime' column")
    timestamps = _parse_timestamps(frame["Datetime"])
    starts_segment = timestamps.diff().gt(pd.Timedelta(milliseconds=200))
    result = frame.copy()
    result.insert(0, "id", starts_segment.fillna(True).cumsum().astype(int))
    result["timestamp"] = timestamps
    result["elapsed_ms"] = (
        result.groupby("id")["timestamp"]
        .transform(lambda values: (values - values.iloc[0]).dt.total_seconds() * 1000)
    )
    return result


def _make_sequences(data, feature_columns):
    sequences = [
        segment[feature_columns].to_numpy(dtype=np.float32)
        for _, segment in data.groupby("id", sort=True)
    ]
    lengths = [len(sequence) for sequence in sequences]
    if not sequences:
        raise ValueError("Door input contains no segments")
    feature_count = len(feature_columns)
    padded = np.zeros(
        (len(sequences), MAX_SEQUENCE_LENGTH, feature_count), dtype=np.float32
    )
    for index, sequence in enumerate(sequences):
        if len(sequence) > MAX_SEQUENCE_LENGTH:
            sequence = sequence[:MAX_SEQUENCE_LENGTH]
        padded[index, : len(sequence)] = sequence
    return padded, lengths


def _operation_from_commands(segment):
    close_fraction = segment["Close command"].eq(1).mean()
    open_fraction = segment["Open command"].eq(1).mean()
    if close_fraction >= 0.90:
        return "Close"
    if open_fraction >= 0.90:
        return "Open"
    return "Unknown"


def _predict_statuses(data):
    model, mean, std = _load_artifacts()
    feature_columns = data.select_dtypes(include="number").columns.tolist()
    feature_columns.remove("id")
    sequences, lengths = _make_sequences(data, feature_columns)
    expected_features = mean.shape[-1]
    if sequences.shape[-1] != expected_features:
        raise ValueError(
            f"CSV has {sequences.shape[-1]} numeric features, but the Door model "
            f"expects {expected_features}."
        )
    if std.shape != mean.shape or np.any(std == 0):
        raise ValueError("Door scaling artifact has invalid mean/std arrays")
    sequences = (sequences - mean) / std
    for sequence, length in enumerate(lengths):
        sequences[sequence, min(length, MAX_SEQUENCE_LENGTH) :] = 0
    probabilities = np.asarray(model(sequences, training=False)).reshape(-1)
    statuses = np.where(probabilities >= 0.5, "Abnormal resistance", "Normal")
    return statuses


def run(file_paths, progress_callback=None):
    if len(file_paths) != 1:
        raise ValueError("Door requires exactly one continuous stream CSV")
    data = _create_segments(pd.read_csv(file_paths[0]))
    statuses = _predict_statuses(data)
    rows = []
    for index, (_, segment) in enumerate(data.groupby("id", sort=True)):
        rows.append(
            {
                "start_time": segment["Datetime"].iloc[0],
                "end_time": segment["Datetime"].iloc[-1],
                "prediction": statuses[index],
            }
        )
    if progress_callback:
        progress_callback(1.0)
    return pd.DataFrame(rows, columns=["start_time", "end_time", "prediction"])
