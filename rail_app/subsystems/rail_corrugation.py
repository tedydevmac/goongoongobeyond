"""
Rail Corrugation subsystem plugin.

Wraps the two-stage model developed earlier (features.py / train.py /
predict.py) behind the app's plugin contract — see subsystems/__init__.py.
"""
import json
import os

import pandas as pd
import xgboost as xgb

from ml import features as feat

NAME = "Rail Corrugation"
OUTPUT_FILENAME = "rail_predictions.csv"
INPUT_MODE = "multi_file"
FILE_HINT = "Upload one or more 1-second axle-box vibration/shock CSV files (e.g. Test1.csv, Test2.csv, ...)."
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "rail_corrugation")

_REQUIRED_FILES = ["stage1_model.json", "stage2_model.json", "config.json"]


def is_ready():
    return all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in _REQUIRED_FILES)


def _load_artifacts():
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)
    stage1_bst = xgb.Booster()
    stage1_bst.load_model(os.path.join(MODEL_DIR, "stage1_model.json"))
    stage2_bst = xgb.Booster()
    stage2_bst.load_model(os.path.join(MODEL_DIR, "stage2_model.json"))
    return config, stage1_bst, stage2_bst


def run(file_paths, progress_callback=None):
    config, stage1_bst, stage2_bst = _load_artifacts()

    rows = []
    for i, path in enumerate(file_paths):
        rows.append(feat.extract_file_features(path, use_cwt=config.get("use_cwt", True)))
        if progress_callback:
            progress_callback((i + 1) / len(file_paths))

    feat_df = pd.DataFrame(rows)
    feat_df.insert(0, "filename", [os.path.basename(p) for p in file_paths])

    stage1_cols = config["stage1_feature_cols"]
    stage2_cols = config["stage2_feature_cols"]
    thr1, thr2 = config["threshold1"], config["threshold2"]

    p1 = stage1_bst.predict(xgb.DMatrix(feat_df[stage1_cols]))
    p2 = stage2_bst.predict(xgb.DMatrix(feat_df[stage2_cols]))

    predictions = []
    for fault_prob, sideI_prob in zip(p1, p2):
        if fault_prob <= thr1:
            predictions.append("Normal")
        elif sideI_prob > thr2:
            predictions.append("Side I")
        else:
            predictions.append("Side II")

    return pd.DataFrame({"file_id": feat_df["filename"], "prediction": predictions})
