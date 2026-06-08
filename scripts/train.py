"""
train.py — Phase 5: Modelling.

Models chosen:
  1. Random Forest    — robust ensemble, handles non-linearities, low tuning risk
  2. XGBoost          — gradient boosting, excellent on tabular data, fast
  3. LightGBM         — faster than XGBoost, great on medium-sized datasets

Metrics:
  R²     — proportion of variance explained; higher = better
  RMSLE  — Root Mean Squared Log Error; preferred for price prediction
            because it penalises under-prediction more and is scale-invariant.
            We choose RMSLE (not MAE/RMSE) because price ranges from ~200 to
            100,000 EGP — a fixed MAE would be dominated by expensive items.
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
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score

from scripts.utils import report_metrics, rmsle, r2, save_object, OUTPUT_DIR

FIGDIR = os.path.join(OUTPUT_DIR, "model_figures")
os.makedirs(FIGDIR, exist_ok=True)


# ─────────────────────────────────────────────
#  Helper: cross-val RMSLE score
# ─────────────────────────────────────────────
def cv_rmsle(model, X, y, cv=5):
    from sklearn.metrics import make_scorer
    scorer = make_scorer(lambda yt, yp: -rmsle(yt, yp), greater_is_better=False)
    scores = -cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
    return scores.mean()


# ─────────────────────────────────────────────
#  1. RANDOM FOREST
# ─────────────────────────────────────────────
def train_rf(X_train, X_test, y_train, y_test):
    print("\n── Model 1: Random Forest ──")

    # Baseline
    rf_base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_base.fit(X_train, y_train)
    print("  Baseline:")
    m_tr = report_metrics(y_train, rf_base.predict(X_train), "Train")
    m_te = report_metrics(y_test,  rf_base.predict(X_test),  "Test ")

    # Optuna tuning
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 400),
            "max_depth":         trial.suggest_int("max_depth", 5, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_float("max_features", 0.3, 1.0),
            "n_jobs": -1, "random_state": 42
        }
        model = RandomForestRegressor(**params)
        return cv_rmsle(model, X_train, y_train, cv=3)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=25, show_progress_bar=False)
    best_params = study.best_params
    print(f"  Best params: {best_params}")

    rf_tuned = RandomForestRegressor(**best_params, n_jobs=-1, random_state=42)
    rf_tuned.fit(X_train, y_train)
    print("  Tuned:")
    m_tr_t = report_metrics(y_train, rf_tuned.predict(X_train), "Train")
    m_te_t = report_metrics(y_test,  rf_tuned.predict(X_test),  "Test ")

    return {
        "model": rf_tuned,
        "results": {
            "RF_baseline_train": m_tr,  "RF_baseline_test":  m_te,
            "RF_tuned_train":    m_tr_t, "RF_tuned_test":    m_te_t,
        }
    }


# ─────────────────────────────────────────────
#  2. XGBOOST
# ─────────────────────────────────────────────
def train_xgb(X_train, X_test, y_train, y_test):
    print("\n── Model 2: XGBoost ──")

    xgb_base = XGBRegressor(n_estimators=200, random_state=42,
                             n_jobs=-1, verbosity=0)
    xgb_base.fit(X_train, y_train,
                 eval_set=[(X_test, y_test)],
                 verbose=False)
    print("  Baseline:")
    m_tr = report_metrics(y_train, xgb_base.predict(X_train), "Train")
    m_te = report_metrics(y_test,  xgb_base.predict(X_test),  "Test ")

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-5, 10, log=True),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42, "n_jobs": -1, "verbosity": 0
        }
        model = XGBRegressor(**params)
        return cv_rmsle(model, X_train, y_train, cv=3)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    best_params = study.best_params
    print(f"  Best params: {best_params}")

    xgb_tuned = XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0)
    xgb_tuned.fit(X_train, y_train, verbose=False)
    print("  Tuned:")
    m_tr_t = report_metrics(y_train, xgb_tuned.predict(X_train), "Train")
    m_te_t = report_metrics(y_test,  xgb_tuned.predict(X_test),  "Test ")

    return {
        "model": xgb_tuned,
        "results": {
            "XGB_baseline_train": m_tr,  "XGB_baseline_test":  m_te,
            "XGB_tuned_train":    m_tr_t, "XGB_tuned_test":    m_te_t,
        }
    }


# ─────────────────────────────────────────────
#  3. LIGHTGBM
# ─────────────────────────────────────────────
def train_lgbm(X_train, X_test, y_train, y_test):
    print("\n── Model 3: LightGBM ──")

    lgbm_base = LGBMRegressor(n_estimators=200, random_state=42,
                               n_jobs=-1, verbose=-1)
    lgbm_base.fit(X_train, y_train)
    print("  Baseline:")
    m_tr = report_metrics(y_train, lgbm_base.predict(X_train), "Train")
    m_te = report_metrics(y_test,  lgbm_base.predict(X_test),  "Test ")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 600),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 150),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-5, 10, log=True),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
            "random_state": 42, "n_jobs": -1, "verbose": -1
        }
        model = LGBMRegressor(**params)
        return cv_rmsle(model, X_train, y_train, cv=3)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    best_params = study.best_params
    print(f"  Best params: {best_params}")

    lgbm_tuned = LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
    lgbm_tuned.fit(X_train, y_train)
    print("  Tuned:")
    m_tr_t = report_metrics(y_train, lgbm_tuned.predict(X_train), "Train")
    m_te_t = report_metrics(y_test,  lgbm_tuned.predict(X_test),  "Test ")

    return {
        "model": lgbm_tuned,
        "results": {
            "LGBM_baseline_train": m_tr,  "LGBM_baseline_test":  m_te,
            "LGBM_tuned_train":    m_tr_t, "LGBM_tuned_test":    m_te_t,
        }
    }


# ─────────────────────────────────────────────
#  COMPARE & SAVE BEST
# ─────────────────────────────────────────────
def compare_and_save(rf_out, xgb_out, lgbm_out, X_test, y_test):
    print(f"\n{'='*60}")
    print(" MODEL COMPARISON TABLE")
    print(f"{'='*60}")

    all_results = {}
    all_results.update(rf_out["results"])
    all_results.update(xgb_out["results"])
    all_results.update(lgbm_out["results"])

    rows = []
    for name, metrics in all_results.items():
        rows.append({"Run": name, "R²": metrics["R2"], "RMSLE": metrics["RMSLE"]})
    results_df = pd.DataFrame(rows).set_index("Run")
    print(results_df.to_string())
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_results.csv"))

    # Select best by test RMSLE (lower = better)
    test_runs = {k: v for k, v in all_results.items() if "test" in k.lower()}
    best_run  = min(test_runs, key=lambda k: test_runs[k]["RMSLE"])
    print(f"\n  Best run: {best_run}")

    model_map = {
        "RF":   rf_out["model"],
        "XGB":  xgb_out["model"],
        "LGBM": lgbm_out["model"],
    }
    best_model_key = [k for k in model_map if k in best_run][0]
    best_model = model_map[best_model_key]
    save_object(best_model, "best_model.pkl")
    save_object(best_model_key, "best_model_name.pkl")
    print(f"  Saved best model ({best_model_key}) → models/best_model.pkl")

    # Residual plot
    preds = best_model.predict(X_test)
    residuals = y_test.values - preds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(preds, residuals, alpha=0.1, s=5, color="steelblue")
    axes[0].axhline(0, color="red", lw=1)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Residual Plot (test set)")
    axes[1].scatter(y_test, preds, alpha=0.1, s=5, color="salmon")
    mn, mx = y_test.min(), y_test.max()
    axes[1].plot([mn, mx], [mn, mx], "k--", lw=1)
    axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
    axes[1].set_title("Actual vs Predicted (test set)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "residuals.png"), dpi=120)
    plt.close()

    return best_model, best_run


def run_training(X_train, X_test, y_train, y_test):
    print(f"\n{'='*60}")
    print(" PHASE 5 — MODELLING")
    print(f"{'='*60}")

    rf_out   = train_rf(X_train, X_test, y_train, y_test)
    xgb_out  = train_xgb(X_train, X_test, y_train, y_test)
    lgbm_out = train_lgbm(X_train, X_test, y_train, y_test)
    best_model, best_run = compare_and_save(rf_out, xgb_out, lgbm_out, X_test, y_test)
    return best_model, best_run


if __name__ == "__main__":
    from scripts.preprocessing import run_preprocessing
    from scripts.feature_engineering import engineer_features
    from scripts.feature_selection import select_features
    print("Use the Jupyter notebook or run predict.py for end-to-end execution.")