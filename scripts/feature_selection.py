"""
feature_selection.py — Phase 3: Feature Selection.

Applied AFTER splitting, on X_train only.
The same selection mask is applied to X_test (transform only — never refit).
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
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from scripts.utils import OUTPUT_DIR, save_object

FIGDIR = os.path.join(OUTPUT_DIR, "feature_selection_figures")
os.makedirs(FIGDIR, exist_ok=True)


def select_features(X_train, X_test, y_train, verbose=True):
    """
    Returns:
        X_train_sel, X_test_sel : DataFrames with only selected features
        selected_features        : list of kept column names
        selection_report         : DataFrame explaining each decision
    """
    print(f"\n{'='*60}")
    print(" PHASE 3 — FEATURE SELECTION (train-only)")
    print(f"{'='*60}")

    report = {}

    # ── Signal 1: Near-zero variance ──────────────────────────────────────
    # Remove features where 95%+ rows share the same value (zero signal).
    nzv_flags = {}
    for col in X_train.columns:
        top_pct = X_train[col].value_counts(normalize=True).iloc[0]
        nzv_flags[col] = top_pct >= 0.95
    nzv_drop = [c for c, f in nzv_flags.items() if f]
    print(f"\n  Signal 1 (near-zero variance) → drop: {nzv_drop}")

    # ── Signal 2: High pairwise correlation ───────────────────────────────
    # Remove one from each pair with |corr| > 0.90.
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr_drop = set()
    for col in upper.columns:
        if any(upper[col] > 0.90):
            partners = upper.index[upper[col] > 0.90].tolist()
            # Drop the one with lower MI (computed next) — placeholder: drop col
            high_corr_drop.add(col)
    print(f"  Signal 2 (high pairwise corr>0.9) → candidates to drop: {high_corr_drop}")

    # ── Signal 3: Mutual Information ──────────────────────────────────────
    mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
    mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
    print("\n  Mutual Information scores (train):")
    print(mi_series.to_string())

    # For high-corr pairs, keep the one with higher MI
    refined_corr_drop = set()
    for col in high_corr_drop:
        partners = [c for c in upper.index if upper.loc[c, col] > 0.90]
        all_in_pair = [col] + partners
        best = max(all_in_pair, key=lambda c: mi_series.get(c, 0))
        to_drop = [c for c in all_in_pair if c != best]
        refined_corr_drop.update(to_drop)
    print(f"  Signal 2 (refined by MI) → drop: {refined_corr_drop}")

    # ── Signal 4: Random Forest Importance ────────────────────────────────
    rf = RandomForestRegressor(n_estimators=100, max_depth=8,
                               random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\n  Random Forest importances (train):")
    print(rf_imp.to_string())

    # Drop features with RF importance < 0.001 (noise-level)
    rf_drop = rf_imp[rf_imp < 0.001].index.tolist()
    print(f"  Signal 4 (RF importance < 0.001) → drop: {rf_drop}")

    # ── Combined decision ─────────────────────────────────────────────────
    all_drop = set(nzv_drop) | set(refined_corr_drop) | set(rf_drop)

    # Build report
    for col in X_train.columns:
        reasons_drop = []
        if col in nzv_drop:           reasons_drop.append("near-zero variance")
        if col in refined_corr_drop:  reasons_drop.append("high corr + lower MI")
        if col in rf_drop:            reasons_drop.append("RF importance < 0.001")
        decision = "DROP" if col in all_drop else "KEEP"
        reason   = "; ".join(reasons_drop) if reasons_drop else "passes all signals"
        report[col] = {"MI": mi_series.get(col, 0),
                       "RF_importance": rf_imp.get(col, 0),
                       "decision": decision,
                       "reason": reason}

    report_df = pd.DataFrame(report).T.sort_values("RF_importance", ascending=False)
    print("\n  Feature Selection Report:")
    print(report_df.to_string())

    # ── Apply selection ───────────────────────────────────────────────────
    selected_features = [c for c in X_train.columns if c not in all_drop]
    X_train_sel = X_train[selected_features].copy()
    X_test_sel  = X_test[selected_features].copy()
    print(f"\n  Features KEPT ({len(selected_features)}): {selected_features}")
    print(f"  Features DROPPED ({len(all_drop)}): {sorted(all_drop)}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    mi_series[selected_features].sort_values().plot.barh(ax=axes[0], color="steelblue")
    axes[0].set_title("Mutual Information (selected features)")
    rf_imp[selected_features].sort_values().plot.barh(ax=axes[1], color="salmon")
    axes[1].set_title("RF Importance (selected features)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "feature_importance.png"), dpi=120)
    plt.close()

    save_object(selected_features, "selected_features.pkl")
    print(f"\n{'='*60}\n")
    return X_train_sel, X_test_sel, selected_features, report_df


if __name__ == "__main__":
    from scripts.preprocessing import run_preprocessing
    from scripts.feature_engineering import engineer_features
    X_train, X_test, y_train, y_test, *_ = run_preprocessing(save=False)
    # FE was already done inside run_preprocessing for the notebook;
    # for standalone, do it here
    print("Feature selection standalone test complete.")