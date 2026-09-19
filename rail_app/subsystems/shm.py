"""SHM cumulative fatigue-damage inference using the trained Keras model."""
import os
from functools import lru_cache

import numpy as np
import pandas as pd

NAME = "SHM"
OUTPUT_FILENAME = "shm_predictions.csv"
INPUT_MODE = "multi_file"
FILE_HINT = "Upload one or more dynamic stress time-series CSV files."
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "shm")
MODEL_PATH = os.path.join(MODEL_DIR, "regression_model.keras")


def is_ready():
    """Return whether the model artifact and its runtime are available."""
    if not os.path.isfile(MODEL_PATH):
        return False
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def _load_model():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "The SHM model requires TensorFlow. Install the dependencies from "
            "rail_app/requirements.txt and restart the app."
        ) from error
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"SHM model artifact not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def _match_length(signal, n_points):
    if signal.size == n_points:
        return signal
    if signal.size == 0:
        raise ValueError("SHM input contains no readings")
    old_index = np.linspace(0.0, 1.0, signal.size)
    new_index = np.linspace(0.0, 1.0, n_points)
    return np.interp(new_index, old_index, signal).astype(np.float32)


def _prepare_input(frame, model):
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    invalid = int(numeric.isna().sum().sum())
    if invalid:
        raise ValueError(
            f"SHM input contains {invalid} non-numeric or missing value(s). "
            "Upload a CSV containing only stress readings."
        )
    raw = numeric.to_numpy(dtype=np.float32).reshape(-1)
    input_shape = tuple(model.input_shape)
    if len(input_shape) == 3:
        _, expected_timesteps, expected_features = input_shape
        features = expected_features or 1
        if features != 1:
            raise ValueError(
                f"Unsupported SHM model input feature count: {features}"
            )
        if expected_timesteps is not None:
            raw = _match_length(raw, int(expected_timesteps))
        return raw.reshape(1, -1, features)
    if len(input_shape) == 2:
        _, expected_features = input_shape
        if expected_features is not None:
            raw = _match_length(raw, int(expected_features))
        return raw.reshape(1, -1)
    raise ValueError(f"Unexpected SHM model input shape: {input_shape}")


def _predict(path):
    frame = pd.read_csv(path)
    model = _load_model()
    model_input = _prepare_input(frame, model)
    prediction = np.asarray(model.predict(model_input, verbose=0)).squeeze()
    if not np.isfinite(prediction):
        raise ValueError(f"SHM model returned a non-finite prediction for {path}")
    return float(prediction)


def run(file_paths, progress_callback=None):
    if not file_paths:
        raise ValueError("At least one SHM input file is required")
    rows = []
    for index, path in enumerate(file_paths):
        rows.append(
            {
                "file_id": os.path.basename(path),
                "prediction": _predict(path),
            }
        )
        if progress_callback:
            progress_callback((index + 1) / len(file_paths))
    return pd.DataFrame(rows, columns=["file_id", "prediction"])
