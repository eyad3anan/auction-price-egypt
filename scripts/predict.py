"""
predict.py — Phase 6: Final Prediction using the saved best model (ensemble).

Pipeline order:
  Raw input → strip leakage → drop text → format → encode → outlier cap →
  log-transform → feature engineering → split (reproduce test) → scale →
  feature selection → ensemble predict (RF + LGBM + XGB log-blend)

All transformers: .transform() only — nothing is refit on test data.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from scripts.utils import load_object, report_metrics, OUTPUT_DIR
from scripts.preprocessing import (
    load_raw, step1_drop_duplicates, step2_drop_leakage,
    step3_drop_text_cols, step4_fix_formatting, step5_handle_missing,
    step6_encode, step7_handle_outliers, step8_log_transform
)
from scripts.feature_engineering import engineer_features

TARGET = "final_selling_price"


def load_pipeline():
    encoders          = load_object("encoders.pkl")
    caps              = load_object("outlier_caps.pkl")
    log_cols          = load_object("log_cols.pkl")
    scaler            = load_object("scaler.pkl")
    selected_features = load_object("selected_features.pkl")
    best_bundle       = load_object("best_model.pkl")
    best_model_name   = load_object("best_model_name.pkl")
    return encoders, caps, log_cols, scaler, selected_features, best_bundle, best_model_name


def ensemble_predict(bundle, X):
    """Log-blend of RF + LGBM + XGB predictions (all trained on log-target)."""
    rf, lgbm, xgb = bundle["rf"], bundle["lgbm"], bundle["xgb"]
    p_rf   = rf.predict(X)
    p_lgbm = lgbm.predict(X)
    p_xgb  = xgb.predict(X)
    blended_log = (p_rf + p_lgbm + p_xgb) / 3
    return np.maximum(np.expm1(blended_log), 0)


def run_predictions():
    print(f"\n{'='*60}")
    print(" PHASE 6 — FINAL PREDICTION")
    print(f"{'='*60}")

    encoders, caps, log_cols, scaler, selected_features, bundle, model_name = load_pipeline()
    print(f"\n  Model type  : {model_name}")
    print(f"  Components  : RF + LightGBM + XGBoost (log-space blend)")
    print(f"  Selected features ({len(selected_features)}): {selected_features}")

    # Reproduce dataset → preprocessing → FE → split (same random_state=42)
    df = load_raw()
    df = step1_drop_duplicates(df)
    df = step2_drop_leakage(df)
    df = step3_drop_text_cols(df)
    df = step4_fix_formatting(df)
    df = step5_handle_missing(df)
    df, _ = step6_encode(df, fit=False, encoders=encoders)
    df, _ = step7_handle_outliers(df, fit=False, caps=caps)
    df, _ = step8_log_transform(df, log_cols=log_cols)
    df    = engineer_features(df)

    X_all = df.drop(columns=[TARGET])
    y_all = df[TARGET]
    _, X_test_raw, _, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    # Scale (transform only)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=X_test_raw.columns, index=X_test_raw.index
    )
    X_test_final = X_test_scaled[selected_features]

    # Ensemble predict
    y_pred = ensemble_predict(bundle, X_test_final)

    # Metrics
    print("\n  Final Test Performance:")
    metrics = report_metrics(y_test, y_pred, "Ensemble_Test")

    # Pct errors
    pct = np.abs(y_test.values - y_pred) / (y_test.values + 1e-6) * 100
    print(f"  Median % error : {np.median(pct):.1f}%")
    print(f"  Mean   % error : {np.mean(pct):.1f}%")

    # Comparison table
    comparison = pd.DataFrame({
        "True_Value": y_test.values[:30],
        "Predicted":  y_pred[:30].round(0).astype(int),
        "Abs_Error":  np.abs(y_test.values[:30] - y_pred[:30]).round(0).astype(int),
        "Pct_Error":  pct[:30].round(1)
    })
    print("\n  True vs Predicted (first 30 test samples):")
    print(comparison.to_string(index=False))
    comparison.to_csv(os.path.join(OUTPUT_DIR, "predictions_comparison.csv"), index=False)

    # Actual vs Predicted plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.1, s=5, color="steelblue")
    mn, mx = y_test.min(), y_test.max()
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
    axes[0].set_xlabel("Actual Final Selling Price (EGP)")
    axes[0].set_ylabel("Predicted Final Selling Price (EGP)")
    axes[0].set_title(f"Actual vs Predicted — Ensemble\nR²={metrics['R2']:.4f}  RMSLE={metrics['RMSLE']:.4f}")
    axes[0].legend()

    residuals = y_test.values - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.1, s=5, color="salmon")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_predictions.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot  → outputs/final_predictions.png")
    print(f"  CSV   → outputs/predictions_comparison.csv")
    print(f"\n{'='*60}\n")
    return y_test, y_pred, metrics


def predict_single(input_dict: dict) -> float:
    """
    Accept raw feature dict → apply full fitted pipeline → return predicted price.
    Used by FastAPI. All transformers are transform-only, never refit.
    """
    encoders, caps, log_cols, scaler, selected_features, bundle, _ = load_pipeline()

    df_input = pd.DataFrame([input_dict])

    # Apply pipeline — transform only (fit=False)
    df_input, _ = step6_encode(df_input, fit=False, encoders=encoders)
    df_input, _ = step7_handle_outliers(df_input, fit=False, caps=caps)
    df_input, _ = step8_log_transform(df_input, log_cols=log_cols)
    df_input    = engineer_features(df_input)
    df_input    = df_input.drop(columns=[TARGET], errors="ignore")

    X_scaled = pd.DataFrame(
        scaler.transform(df_input),
        columns=df_input.columns
    )
    X_final = X_scaled[selected_features]
    return round(float(ensemble_predict(bundle, X_final)[0]), 2)


if __name__ == "__main__":
    run_predictions()