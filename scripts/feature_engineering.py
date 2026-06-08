"""
feature_engineering.py — Phase 4: Feature Engineering.

Applied AFTER encoding/cleaning but BEFORE train/test split (in the pipeline order
we call FE before step9_split to ensure consistent features).
All features are computed from columns available AT LISTING TIME — no leakage.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


TARGET = "final_selling_price"


def engineer_features(df):
    """
    Create new features from existing ones.
    Input df still has original columns (post-leakage-drop, post-format-fix,
    post-encode, post-outlier-cap, post-log-transform).
    """
    print(f"\n{'='*60}")
    print(" PHASE 4 — FEATURE ENGINEERING")
    print(f"{'='*60}")

    original_cols = [c for c in df.columns if c != TARGET]

    # ── 1. Price-tier relative to starting price ──────────────────────────
    # SIGNAL: how high above the floor the item is priced. Items with larger
    #         gaps between start and final tend to have strong demand.
    # (starting_price has been log-transformed → we work in log space)
    # We reconstruct: starting_price_raw via expm1 for ratio, then log again.
    # Simpler: we already have log(starting_price+1). Since target is raw,
    # we create: log_starting_price (already exists as starting_price after step 8).
    # We add starting_price_sq to capture non-linearity.
    if "starting_price" in df.columns:
        df["starting_price_sq"] = df["starting_price"] ** 2
        print("  FE1 — starting_price_sq: captures non-linear relationship between "
              "start price and final price.")

    # ── 2. Seller credibility score ───────────────────────────────────────
    # SIGNAL: combines rating and volume into a single trust proxy.
    #         High-rating + high-volume sellers get premium prices.
    if all(c in df.columns for c in ["seller_rating", "seller_total_sales"]):
        df["seller_credibility"] = df["seller_rating"] * df["seller_total_sales"]
        print("  FE2 — seller_credibility = seller_rating × log(seller_total_sales): "
              "composite trust signal.")

    # ── 3. Seller experience ratio ────────────────────────────────────────
    # SIGNAL: sales per year of account age → normalises raw sales by tenure.
    if all(c in df.columns for c in ["seller_total_sales", "seller_account_age"]):
        df["seller_activity_rate"] = (
            df["seller_total_sales"] / (df["seller_account_age"] + 1e-6)
        )
        print("  FE3 — seller_activity_rate = log_sales / log_account_age: "
              "active sellers signal demand and trustworthiness.")

    # ── 4. Auction urgency (duration bucket) ──────────────────────────────
    # SIGNAL: short auctions (<= 3 days) create urgency → may raise final price.
    #         Long auctions (> 14 days) give buyers time to forget.
    if "auction_duration" in df.columns:
        df["is_short_auction"] = (df["auction_duration"] <= 3).astype(int)
        df["is_long_auction"]  = (df["auction_duration"] >= 14).astype(int)
        print("  FE4 — is_short_auction / is_long_auction: binary flags for "
              "urgency vs. stale-listing effect.")

    # ── 5. Weekend listing flag ───────────────────────────────────────────
    # SIGNAL: items listed on Friday/Saturday/Sunday in Egypt see more
    #         weekend browsing traffic → higher bid competition.
    if "listing_day_of_week" in df.columns:
        # After ordinal encoding: Friday=4, Saturday=5, Sunday=6
        df["is_weekend_listing"] = (df["listing_day_of_week"] >= 4).astype(int)
        print("  FE5 — is_weekend_listing: captures Egyptian weekend browsing surge.")

    # ── 6. Prime-time listing flag ────────────────────────────────────────
    # SIGNAL: listings posted between 18:00–23:00 hit peak user activity.
    if "listing_hour" in df.columns:
        df["is_primetime"] = ((df["listing_hour"] >= 18) & (df["listing_hour"] <= 23)).astype(int)
        print("  FE6 — is_primetime: evening listings (18–23h) capture peak activity.")

    # ── 7. Product freshness ──────────────────────────────────────────────
    # SIGNAL: inverse of product_age — newer items fetch more. Log-scaled
    #         to handle the long tail of very old items.
    if "product_age" in df.columns:
        df["product_freshness"] = np.log1p(1 / (df["product_age"] + 1))
        print("  FE7 — product_freshness = log1p(1/(age+1)): newer = higher signal.")

    # ── 8. Category × condition interaction ───────────────────────────────
    # SIGNAL: a 'New' watch in the Fashion category is worth far more than
    #         a 'Fair' phone case. Interaction captures this joint effect.
    if all(c in df.columns for c in ["category", "condition"]):
        df["cat_x_condition"] = df["category"].astype(str) + "_" + df["condition"].astype(str)
        # Frequency encode this interaction
        freq_map = df["cat_x_condition"].value_counts().to_dict()
        df["cat_x_condition_freq"] = df["cat_x_condition"].map(freq_map)
        df = df.drop(columns=["cat_x_condition"])
        print("  FE8 — cat_x_condition_freq: frequency-encoded category×condition "
              "interaction.")

    # ── 9. Verified seller × seller rating ───────────────────────────────
    if all(c in df.columns for c in ["verified_seller", "seller_rating"]):
        df["verified_rating"] = df["verified_seller"] * df["seller_rating"]
        print("  FE9 — verified_rating: only counts rating for verified sellers; "
              "0 for unverified — filters noise from unverified ratings.")

    new_cols = [c for c in df.columns if c not in original_cols and c != TARGET]
    print(f"\n  Total new features added: {len(new_cols)}: {new_cols}")
    print(f"{'='*60}\n")
    return df


if __name__ == "__main__":
    from scripts.preprocessing import (
        load_raw, step1_drop_duplicates, step2_drop_leakage,
        step3_drop_text_cols, step4_fix_formatting, step5_handle_missing,
        step6_encode, step7_handle_outliers, step8_log_transform
    )
    df = load_raw()
    df = step1_drop_duplicates(df)
    df = step2_drop_leakage(df)
    df = step3_drop_text_cols(df)
    df = step4_fix_formatting(df)
    df = step5_handle_missing(df)
    df, _ = step6_encode(df, fit=True)
    df, _ = step7_handle_outliers(df, fit=True)
    df, _ = step8_log_transform(df)
    df = engineer_features(df)
    print(df.shape)
    print(df.columns.tolist())