"""
preprocessing.py — Phase 2: Data Preprocessing for the Egyptian Auction dataset.

ORDER IS INTENTIONAL — do not reorder steps.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from scripts.utils import DATA_PATH, CONDITION_ORDER, DAY_ORDER, save_object, MODEL_DIR


TARGET = "final_selling_price"

# Columns with documented data leakage
LEAKAGE_COLS = ["reserve_price", "buy_now_price"]
# Text/ID-like columns not useful for tabular ML
TEXT_COLS    = ["item_title", "item_description"]


def load_raw():
    return pd.read_csv(DATA_PATH)


# ─────────────────────────────────────────────────────────
# STEP 1: Drop exact duplicates
# WHY: duplicate rows bias training by over-representing certain examples.
# POSITION: must be first — before any transformation so we don't
#           accidentally de-duplicate rows that became identical after cleaning.
# ─────────────────────────────────────────────────────────
def step1_drop_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Step 1 — Dropped {before - len(df)} duplicate rows  ({len(df)} remain)")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────
# STEP 2: Drop leakage columns
# WHY: reserve_price and buy_now_price are set by the seller WITH knowledge
#      of the expected selling price. Including them would give the model
#      privileged future information not available at prediction time for
#      unseen listings — classic target leakage.
# POSITION: before encoding/scaling so we don't waste compute on them.
# ─────────────────────────────────────────────────────────
def step2_drop_leakage(df):
    df = df.drop(columns=LEAKAGE_COLS, errors="ignore")
    print(f"  Step 2 — Dropped leakage cols: {LEAKAGE_COLS}")
    return df


# ─────────────────────────────────────────────────────────
# STEP 3: Drop ID-like and text columns
# WHY: item_title and item_description are free text — not useful for
#      tabular tree/linear models without NLP; they also risk acting as
#      near-unique identifiers (leakage). We extract signal via FE instead.
# ─────────────────────────────────────────────────────────
def step3_drop_text_cols(df):
    df = df.drop(columns=TEXT_COLS, errors="ignore")
    print(f"  Step 3 — Dropped text/ID cols: {TEXT_COLS}")
    return df


# ─────────────────────────────────────────────────────────
# STEP 4: Fix inconsistent formatting
# WHY: consistent casing and stripped whitespace ensure label/ordinal
#      encoders map correctly. Must happen before encoding.
# ─────────────────────────────────────────────────────────
def step4_fix_formatting(df):
    cat_cols = df.select_dtypes("object").columns.tolist()
    for col in cat_cols:
        df[col] = df[col].str.strip()
    print(f"  Step 4 — Stripped whitespace from {len(cat_cols)} categorical columns")
    # Standardise condition capitalisation
    df["condition"] = df["condition"].str.title()
    return df


# ─────────────────────────────────────────────────────────
# STEP 5: Handle missing values
# WHY: EDA showed NO missing values, so this step is a pass-through.
#      Strategy (if any appear in production): seller_rating → median
#      (numeric, skewed); categoricals → mode.
# ─────────────────────────────────────────────────────────
def step5_handle_missing(df):
    miss = df.isnull().sum().sum()
    print(f"  Step 5 — Missing values: {miss}  (no imputation needed)")
    return df


# ─────────────────────────────────────────────────────────
# STEP 6: Encode categorical variables
#
#  condition         → OrdinalEncoder (ordered quality scale: For Parts < Poor … < New)
#  listing_day_of_week → OrdinalEncoder (0–6, captures weekly temporal cycle)
#  category          → OrdinalEncoder after frequency sort (medium cardinality ~7)
#  subcategory       → Target Encoding proxy via frequency encoding (high cardinality ~40)
#  brand             → Frequency encoding (very high cardinality ~200+ brands)
#  verified_seller   → Already binary 0/1 — no change needed
#
# WHY OrdinalEncoder for condition: there is a clear quality ordering.
# WHY frequency for brand/subcategory: high cardinality makes OHE impractical;
#      frequency captures popularity signal (popular brands → higher bids).
# POSITION: must be BEFORE splitting so we can compute freq tables on full train set.
#           We save fitted encoders for test-time transform.
# ─────────────────────────────────────────────────────────
def step6_encode(df, fit=True, encoders=None):
    """
    fit=True  → compute and return encoder objects (training flow)
    fit=False → apply saved encoder objects (test / prediction flow)
    """
    if encoders is None:
        encoders = {}

    # --- condition (ordinal) ---
    cond_order = [["For Parts", "Poor", "Fair", "Good", "Very Good", "Excellent", "Like New", "New"]]
    if fit:
        enc_cond = OrdinalEncoder(categories=cond_order,
                                  handle_unknown="use_encoded_value", unknown_value=-1)
        df["condition"] = enc_cond.fit_transform(df[["condition"]])
        encoders["condition"] = enc_cond
    else:
        df["condition"] = encoders["condition"].transform(df[["condition"]])

    # --- listing_day_of_week (ordinal 0–6) ---
    if fit:
        day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
                   "Friday":4,"Saturday":5,"Sunday":6}
        encoders["day_map"] = day_map
    df["listing_day_of_week"] = df["listing_day_of_week"].map(encoders["day_map"]).fillna(0).astype(int)

    # --- category (label encoding via factorize — low cardinality) ---
    if fit:
        cat_codes = {v: i for i, v in enumerate(sorted(df["category"].unique()))}
        encoders["category_map"] = cat_codes
    df["category"] = df["category"].map(encoders["category_map"]).fillna(-1).astype(int)

    # --- subcategory (frequency encoding) ---
    if fit:
        freq_sub = df["subcategory"].value_counts().to_dict()
        encoders["freq_sub"] = freq_sub
    df["subcategory"] = df["subcategory"].map(encoders["freq_sub"]).fillna(0).astype(float)

    # --- brand (frequency encoding) ---
    if fit:
        freq_brand = df["brand"].value_counts().to_dict()
        encoders["freq_brand"] = freq_brand
    df["brand"] = df["brand"].map(encoders["freq_brand"]).fillna(0).astype(float)

    print(f"  Step 6 — Encoded: condition(ordinal), day_of_week(ordinal), "
          f"category(label), subcategory(freq), brand(freq)")
    return df, encoders


# ─────────────────────────────────────────────────────────
# STEP 7: Handle outliers
# WHY: Tree-based models (Random Forest, XGBoost, LightGBM) are largely
#      insensitive to outliers. We cap extreme values at 1st/99th percentile
#      to avoid exploding log-scale predictions for the scaler step.
# POSITION: after encoding, before splitting — we compute caps on full data
#           then save caps and apply to train/test separately.
# ─────────────────────────────────────────────────────────
def step7_handle_outliers(df, fit=True, caps=None):
    cap_cols = ["starting_price", "product_age", "seller_total_sales",
                "seller_account_age", "seller_rating"]
    if caps is None:
        caps = {}

    for col in cap_cols:
        if col not in df.columns:
            continue
        if fit:
            lo = df[col].quantile(0.01)
            hi = df[col].quantile(0.99)
            caps[col] = (lo, hi)
        lo, hi = caps[col]
        df[col] = df[col].clip(lo, hi)

    print(f"  Step 7 — Clipped outliers (1st–99th pct) for: {cap_cols}")
    return df, caps


# ─────────────────────────────────────────────────────────
# STEP 8: Handle skewed numeric features
# WHY: starting_price, seller_total_sales are heavily right-skewed.
#      log1p reduces skew and helps linear models; tree models benefit less
#      but it won't hurt them. Skew threshold: |skew| > 1.
# ─────────────────────────────────────────────────────────
LOG_COLS = ["starting_price", "seller_total_sales", "seller_account_age"]

def step8_log_transform(df, log_cols=None):
    lc = log_cols or LOG_COLS
    lc = [c for c in lc if c in df.columns]
    for col in lc:
        df[col] = np.log1p(df[col])
    print(f"  Step 8 — log1p applied to: {lc}")
    return df, lc


# ─────────────────────────────────────────────────────────
# STEP 9: Train/test split — BEFORE scaling
# WHY: splitting before scaling prevents data leakage from test set
#      statistics influencing the scaler's fit.
# ─────────────────────────────────────────────────────────
def step9_split(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"  Step 9 — Split: train={len(X_train)}  test={len(X_test)}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────
# STEP 10: Scaling — fit ONLY on train, transform both
# WHY: RobustScaler uses median + IQR → less sensitive to residual outliers
#      after capping. Fit on train ONLY — applying to test is transform-only
#      to prevent data leakage.
# ─────────────────────────────────────────────────────────
def step10_scale(X_train, X_test, fit=True, scaler=None):
    if fit:
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns, index=X_train.index)
    else:
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train),
            columns=X_train.columns, index=X_train.index)

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index)
    print("  Step 10 — RobustScaler fitted on train, transformed train+test")
    return X_train_scaled, X_test_scaled, scaler


# ═══════════════════════════════════════════════════════
#  FULL PIPELINE
# ═══════════════════════════════════════════════════════
def run_preprocessing(save=True):
    print(f"\n{'='*60}")
    print(" PHASE 2 — PREPROCESSING")
    print(f"{'='*60}")

    df = load_raw()
    df = step1_drop_duplicates(df)
    df = step2_drop_leakage(df)
    df = step3_drop_text_cols(df)
    df = step4_fix_formatting(df)
    df = step5_handle_missing(df)
    df, encoders = step6_encode(df, fit=True)
    df, caps     = step7_handle_outliers(df, fit=True)
    df, log_cols = step8_log_transform(df)

    X_train, X_test, y_train, y_test = step9_split(df)
    X_train, X_test, scaler = step10_scale(X_train, X_test, fit=True)

    if save:
        save_object(encoders, "encoders.pkl")
        save_object(caps,     "outlier_caps.pkl")
        save_object(log_cols, "log_cols.pkl")
        save_object(scaler,   "scaler.pkl")

    print(f"\n  Final feature set: {list(X_train.columns)}")
    print(f"  X_train shape: {X_train.shape}  X_test shape: {X_test.shape}")
    print(f"\n{'='*60}\n")
    return X_train, X_test, y_train, y_test, encoders, caps, log_cols, scaler


if __name__ == "__main__":
    run_preprocessing()