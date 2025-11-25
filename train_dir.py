#!/usr/bin/env python3
"""
train_direction_classifiers.py

1) Create directions CSV: dir_10min, dir_15min, ... where dir_t = 1 if (t - prev_t) > 0, else 0.
   Saves file to ./data/directions.csv (keeps original columns + new direction columns).

2) Train per-target binary classifiers that predict direction (Up=1, Down=0) for 10,15,20,25,30 min horizons.
   Uses feature engineering, group-aware splits, stacking ensemble of base learners (XGB/LightGBM/HGB),
   per-target threshold tuning using OOF validation, and evaluation on a held-out group test set.

Outputs:
 - ./data/directions.csv
 - ./models/directions/<target>__all.joblib
 - printed metrics (test set)
"""

import os
import json
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
from joblib import dump
from tqdm import tqdm

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier

import xgboost as xgb

# optional imports
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

# Try CatBoost optionally (commented out if not wanted)
try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False

# -------- CONFIG --------
DATA_PATH = "./data/quat.csv"        # source data (your file)
DIRECTIONS_CSV = "./data/directions.csv"
OUT_DIR = "./models/directions"
METRICS_PATH = "./models/directions/metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5

# Targets in ascending horizon order
TARGETS = ["5min-ar_pct", "10min-ar_pct", "15min-ar_pct", "20min-ar_pct", "25min-ar_pct", "30min-ar_pct"]

# We will predict directions for horizons > first (10..30)
PRED_HORIZONS = ["10min-ar_pct", "15min-ar_pct", "20min-ar_pct", "25min-ar_pct", "30min-ar_pct"]

NUM_IMPUTER_STRATEGY = "median"

# Model params (these are sensible defaults)
XGB_CORE_PARAMS = {
    "objective": "binary:logistic",
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "seed": RANDOM_STATE,
    "verbosity": 0
}
XGB_NUM_ROUND = 200
HGB_PARAMS = {"max_iter": 200, "random_state": RANDOM_STATE}
LGB_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "n_estimators": 200, "random_state": RANDOM_STATE}
EPS = 1e-9

# meta model params
META_MODEL = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)

# ---------------- HELPERS ----------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def try_make_row_date(df):
    if "row_date" in df.columns:
        df["row_date"] = pd.to_datetime(df["row_date"], errors="coerce").dt.date
        if df["row_date"].notna().any():
            return df
    if "timestamp" in df.columns:
        df["row_date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
        if df["row_date"].notna().any():
            return df
    if "Period" in df.columns:
        parsed = pd.to_datetime(df["Period"].astype(str), format="%b %Y", errors="coerce")
        if parsed.isna().all():
            parsed = pd.to_datetime(df["Period"].astype(str), errors="coerce")
        df["row_date"] = parsed.dt.date
        if df["row_date"].notna().any():
            return df
    raise SystemExit("Cannot build row_date; provide 'row_date' or parseable 'timestamp'/'Period'.")

def add_direction_columns(df, targets=TARGETS):
    """
    Add direction columns to a dataframe based on difference between consecutive targets.
    For t in targets[1:], create column d_{t} (string) as 1 if t - prev_t > 0 else 0.
    Column names will be: dir_10min-ar_pct (or simplified 'dir_10min' optionally).
    We'll use names: dir_10min_ar_pct, etc. Also save to disk.
    """
    df = df.copy()
    # ensure numeric
    for t in targets:
        df[t] = pd.to_numeric(df[t], errors="coerce")
    # compute diffs
    for i in range(1, len(targets)):
        cur = targets[i]
        prev = targets[i-1]
        col_name = f"dir_{cur.replace('-','_').replace('%','pct')}"
        diff_vals = df[cur] - df[prev]
        df[col_name] = (diff_vals > 0).astype(int)
    return df

def safe_predict_booster(booster, dmat, best_it):
    # robust call for different xgboost versions
    if best_it is not None:
        try:
            return booster.predict(dmat, iteration_range=(0, best_it + 1))
        except Exception:
            pass
        try:
            return booster.predict(dmat, num_iteration=best_it + 1)
        except Exception:
            pass
        try:
            return booster.predict(dmat, ntree_limit=best_it + 1)
        except Exception:
            pass
    return booster.predict(dmat)

def safe_best_iteration(booster):
    try:
        bi = getattr(booster, "best_iteration", None)
        if bi is not None:
            return int(bi)
    except Exception:
        pass
    try:
        a = booster.attr("best_iteration")
        if a is not None:
            return int(a)
    except Exception:
        pass
    return None

# ---------------- PIPELINE ----------------
def main():
    ensure_dir(os.path.dirname(DIRECTIONS_CSV))
    ensure_dir(OUT_DIR)

    # 1) Load
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"Data not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, dtype=str)
    print("Loaded rows:", len(df))
    df = try_make_row_date(df)

    # 2) Create direction columns and write CSV
    df_dir = add_direction_columns(df, targets=TARGETS)
    df_dir.to_csv(DIRECTIONS_CSV, index=False)
    print("Saved directions CSV to:", DIRECTIONS_CSV)

    # 3) Feature engineering (like earlier): diffs, lags, company stats, interactions
    # convert diff_prev features numeric if exist (we'll include engineered ones if available)
    diff_prev_cols = [c for c in df_dir.columns if c.endswith("_diff_prev")]
    # basic features: diff_prev cols if present, else use nothing and rely on other engineered features
    # compute lags by company and company aggregates on diff_prev columns
    df_dir_sorted = df_dir.sort_values(["Company Symbol", "row_date"])
    # safe: make sure Company Symbol exists
    if "Company Symbol" not in df_dir_sorted.columns and "Company" in df_dir_sorted.columns:
        df_dir_sorted = df_dir_sorted.rename(columns={"Company": "Company Symbol"})
    if "Company Symbol" not in df_dir_sorted.columns:
        raise SystemExit("Company Symbol missing in data (required for group splits).")

    # create lag features for diff_prev columns and for the original targets as well
    feature_candidates = []
    for col in diff_prev_cols:
        df_dir_sorted[col] = pd.to_numeric(df_dir_sorted[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
        df_dir_sorted[col + "_lag1"] = df_dir_sorted.groupby("Company Symbol")[col].shift(1)
        df_dir_sorted[col + "_lag2"] = df_dir_sorted.groupby("Company Symbol")[col].shift(2)
        feature_candidates += [col, col + "_lag1", col + "_lag2"]

    # also create lags for the numeric targets themselves (these are strong predictors)
    for t in TARGETS:
        df_dir_sorted[t] = pd.to_numeric(df_dir_sorted[t], errors="coerce")
        df_dir_sorted[t + "_lag1"] = df_dir_sorted.groupby("Company Symbol")[t].shift(1)
        feature_candidates += [t + "_lag1"]

    # per-company mean & std of diff_prev and targets (computed globally here; okay because we will re-fit scalers later)
    for col in diff_prev_cols + TARGETS:
        mean_col = df_dir_sorted.groupby("Company Symbol")[col].transform("mean")
        std_col = df_dir_sorted.groupby("Company Symbol")[col].transform("std")
        df_dir_sorted[col + "_comp_mean"] = mean_col
        df_dir_sorted[col + "_comp_std"] = std_col
        feature_candidates += [col + "_comp_mean", col + "_comp_std"]

    # small set of pairwise interactions (to avoid explosion) — pick first 3 diff_prev columns
    inter_base = diff_prev_cols[:3]
    for i in range(len(inter_base)):
        for j in range(i + 1, len(inter_base)):
            a, b = inter_base[i], inter_base[j]
            name = f"{a}__{b}_mul"
            df_dir_sorted[name] = df_dir_sorted[a] * df_dir_sorted[b]
            feature_candidates.append(name)

    # final feature list: filter those present in df
    feature_cols = [c for c in feature_candidates if c in df_dir_sorted.columns]
    print("Feature columns count:", len(feature_cols), "sample:", feature_cols[:10])

    # Build X and label y for each prediction horizon
    # We'll drop rows with missing label for that horizon
    # Also drop rows with all features missing
    # Prepare group column
    groups = df_dir_sorted["Company Symbol"].astype(str).reset_index(drop=True)

    # imputer + scaler fit on training fold later; we will create train/test split first (group-aware)
    # prepare full X matrix (with features), we'll impute later per-target splits
    X_full = df_dir_sorted[feature_cols].copy()
    # Because many lag features will be NaN for first rows of each company, we let imputer handle them

    # create overall group-aware train/test split for final evaluation
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx_all, test_idx_all = next(gss.split(X_full, groups=groups))
    print("Train rows:", len(train_idx_all), "Test rows:", len(test_idx_all))

    results_summary = {}
    # Loop horizons
    for horizon in PRED_HORIZONS:
        print("\n=== Training direction classifier for:", horizon, "===")
        dir_col = f"dir_{horizon.replace('-','_').replace('%','pct')}"
        # Select rows where label exists (not NaN)
        mask_label_exists = df_dir_sorted[dir_col].notna()
        idx_valid = np.where(mask_label_exists)[0]

        # intersection with final train/test indices
        idx_train = np.intersect1d(train_idx_all, idx_valid)
        idx_test = np.intersect1d(test_idx_all, idx_valid)
        print(f"Valid train rows: {len(idx_train)}, valid test rows: {len(idx_test)}")

        if len(idx_train) < 50 or len(idx_test) < 20:
            print("Not enough data for reliable training on this horizon. Skipping.")
            results_summary[horizon] = {"note": "insufficient_data"}
            continue

        # Prepare per-target X,y
        X_train = X_full.iloc[idx_train].copy().reset_index(drop=True)
        X_test = X_full.iloc[idx_test].copy().reset_index(drop=True)
        y_train = df_dir_sorted.iloc[idx_train][dir_col].astype(int).values
        y_test = df_dir_sorted.iloc[idx_test][dir_col].astype(int).values
        groups_train = groups.iloc[idx_train].reset_index(drop=True)

        # Impute & scale features (fit on training only)
        imp = SimpleImputer(strategy=NUM_IMPUTER_STRATEGY)
        scaler = StandardScaler()
        X_train_imp = imp.fit_transform(X_train)
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_imp = imp.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imp)

        # OOF containers for stacking
        n_train = X_train_scaled.shape[0]
        oof_xgb = np.zeros(n_train, dtype=float)
        oof_lgb = np.zeros(n_train, dtype=float) if _HAS_LGB else None
        oof_hgb = np.zeros(n_train, dtype=float)

        # GroupKFold for OOF
        n_splits = min(N_FOLDS, len(np.unique(groups_train)))
        gkf = GroupKFold(n_splits=n_splits)

        fold = 0
        for tr_idx, val_idx in gkf.split(X_train_scaled, y_train, groups=groups_train):
            fold += 1
            print(f" Fold {fold}/{n_splits}")
            X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            # ---------- XGBoost base ----------
            # train xgboost classifier via core API for OOF proba predictions
            dtr = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
            booster = xgb.train(
                params=XGB_CORE_PARAMS,
                dtrain=dtr,
                num_boost_round=XGB_NUM_ROUND,
                evals=[(dtr, "train"), (dval, "val")],
                early_stopping_rounds=30,
                verbose_eval=False
            )
            best_it = safe_best_iteration(booster)
            pred_val_xgb = safe_predict_booster(booster, xgb.DMatrix(X_val), best_it)
            # ensure probabilities clipped to [0,1]
            pred_val_xgb = np.clip(pred_val_xgb, 0.0, 1.0)
            oof_xgb[val_idx] = pred_val_xgb

            # ---------- LightGBM base (if available) ----------
            if _HAS_LGB:
                lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
                from lightgbm import early_stopping, log_evaluation

                lgb_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[early_stopping(stopping_rounds=30), log_evaluation(0)]
                )


            # ---------- HGB base ----------
            hgb = HistGradientBoostingClassifier(**HGB_PARAMS)
            hgb.fit(X_tr, y_tr)
            pred_val_hgb = hgb.predict_proba(X_val)[:,1]
            oof_hgb[val_idx] = pred_val_hgb

        # Build meta-training dataset (OOF predictions)
        if _HAS_LGB:
            X_meta_train = np.vstack([oof_xgb, oof_lgb, oof_hgb]).T
        else:
            X_meta_train = np.vstack([oof_xgb, oof_hgb]).T

        # Fit meta-learner (logistic regression)
        meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        meta.fit(X_meta_train, y_train)

        # Retrain base models on full training set
        # XGB full
        dtrain_full = xgb.DMatrix(X_train_scaled, label=y_train)
        booster_full = xgb.train(
            params=XGB_CORE_PARAMS,
            dtrain=dtrain_full,
            num_boost_round=XGB_NUM_ROUND,
            verbose_eval=False
        )
        best_it_full = safe_best_iteration(booster_full)
        # XGB test preds (proba)
        pred_test_xgb = safe_predict_booster(booster_full, xgb.DMatrix(X_test_scaled), best_it_full)
        pred_test_xgb = np.clip(pred_test_xgb, 0.0, 1.0)

        # LightGBM full
        if _HAS_LGB:
            lgb_full = lgb.LGBMClassifier(**LGB_PARAMS)
            lgb_full.fit(X_train_scaled, y_train)
            pred_test_lgb = lgb_full.predict_proba(X_test_scaled)[:,1]
        else:
            lgb_full = None
            pred_test_lgb = None

        # HGB full
        hgb_full = HistGradientBoostingClassifier(**HGB_PARAMS)
        hgb_full.fit(X_train_scaled, y_train)
        pred_test_hgb = hgb_full.predict_proba(X_test_scaled)[:,1]

        # Stack test base preds
        if _HAS_LGB:
            X_meta_test = np.vstack([pred_test_xgb, pred_test_lgb, pred_test_hgb]).T
        else:
            X_meta_test = np.vstack([pred_test_xgb, pred_test_hgb]).T

        pred_test_meta_proba = meta.predict_proba(X_meta_test)[:,1]

        # Threshold tuning: pick threshold on OOF to maximize F1 (simple grid)
        # We'll generate OOF predictions for full train set to find threshold.
        # To get OOF for all train rows we already have X_meta_train (OOF predictions)
        best_thr = 0.5
        best_f1 = -1.0
        for thr in np.linspace(0.3, 0.7, 41):  # search from 0.3 to 0.7
            preds_thr = (meta.predict_proba(X_meta_train)[:,1] >= thr).astype(int)
            f1 = f1_score(y_train, preds_thr, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        print(f"Chosen threshold (meta): {best_thr:.3f} (OOF F1={best_f1:.4f})")

        # Apply threshold
        pred_test_meta_label = (pred_test_meta_proba >= best_thr).astype(int)

        # Evaluate on test set
        acc = accuracy_score(y_test, pred_test_meta_label)
        bal_acc = balanced_accuracy_score(y_test, pred_test_meta_label)
        roc = roc_auc_score(y_test, pred_test_meta_proba)
        f1 = f1_score(y_test, pred_test_meta_label, zero_division=0)

        # Also report base model metrics for comparison
        acc_xgb = accuracy_score(y_test, (pred_test_xgb >= 0.5).astype(int))
        bal_xgb = balanced_accuracy_score(y_test, (pred_test_xgb >= 0.5).astype(int))
        roc_xgb = roc_auc_score(y_test, pred_test_xgb)
        f1_xgb = f1_score(y_test, (pred_test_xgb >= 0.5).astype(int), zero_division=0)

        acc_hgb = accuracy_score(y_test, (pred_test_hgb >= 0.5).astype(int))
        bal_hgb = balanced_accuracy_score(y_test, (pred_test_hgb >= 0.5).astype(int))
        roc_hgb = roc_auc_score(y_test, pred_test_hgb)
        f1_hgb = f1_score(y_test, (pred_test_hgb >= 0.5).astype(int), zero_division=0)

        # store summary
        results_summary[horizon] = {
            "meta": {"accuracy": acc, "balanced_accuracy": bal_acc, "roc_auc": roc, "f1": f1, "threshold": best_thr},
            "xgb": {"accuracy": acc_xgb, "balanced_accuracy": bal_xgb, "roc_auc": roc_xgb, "f1": f1_xgb},
            "hgb": {"accuracy": acc_hgb, "balanced_accuracy": bal_hgb, "roc_auc": roc_hgb, "f1": f1_hgb},
            "train_rows": int(len(idx_train)),
            "test_rows": int(len(idx_test))
        }

        print(f"\nTest metrics for {horizon}:")
        print(" Meta stack -> acc: %.4f  bal_acc: %.4f  roc: %.4f  f1: %.4f" % (acc, bal_acc, roc, f1))
        print(" XGB        -> acc: %.4f  bal_acc: %.4f  roc: %.4f  f1: %.4f" % (acc_xgb, bal_xgb, roc_xgb, f1_xgb))
        print(" HGB        -> acc: %.4f  bal_acc: %.4f  roc: %.4f  f1: %.4f" % (acc_hgb, bal_hgb, roc_hgb, f1_hgb))

        # Save artifacts for this horizon
        dump({
            "feature_columns": feature_cols,
            "imputer": imp,
            "feature_scaler": scaler,
            "xgb_booster": booster_full,
            "lgb_model": lgb_full if _HAS_LGB else None,
            "hgb_model": hgb_full,
            "meta_model": meta,
            "threshold": best_thr,
            "feature_list": feature_cols
        }, os.path.join(OUT_DIR, f"{horizon.replace('-','_')}__all.joblib"))

    # Save overall results summary
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump({"results": results_summary, "timestamp": datetime.utcnow().isoformat()+"Z"}, f, indent=2)

    print("\nDone. Results summary saved to:", METRICS_PATH)
    print("Model artifacts saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
