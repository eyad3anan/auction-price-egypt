"""
utils.py — Shared helper functions for the Egyptian Auction Price Prediction project.
"""
import os
import numpy as np
import pandas as pd
import joblib


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "auction_dataset_egypt.csv")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def rmsle(y_true, y_pred):
    """Root Mean Squared Log Error — robust to large price swings."""
    y_pred_clipped = np.maximum(y_pred, 0)          # guard against negatives
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred_clipped)) ** 2))


def r2(y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)


def report_metrics(y_true, y_pred, label=""):
    rs = r2(y_true, y_pred)
    rl = rmsle(y_true, y_pred)
    print(f"  [{label}]  R²={rs:.4f}   RMSLE={rl:.4f}")
    return {"R2": rs, "RMSLE": rl}


# ─────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────
def save_object(obj, name):
    path = os.path.join(MODEL_DIR, name)
    joblib.dump(obj, path)
    print(f"  Saved → {path}")
    return path


def load_object(name):
    path = os.path.join(MODEL_DIR, name)
    return joblib.load(path)


# ─────────────────────────────────────────────
# Condition ordering map
# ─────────────────────────────────────────────
CONDITION_ORDER = {
    "For Parts": 0,
    "Poor":      1,
    "Fair":      2,
    "Good":      3,
    "Excellent": 4,
    "Very Good": 5,   # re-mapped so ordinal is monotone
    "Like New":  6,
    "New":       7,
}

DAY_ORDER = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}