#!/usr/bin/env python3
"""
Interactive predict.py

Prompts the user (via input) for:
 - Company Symbol
 - Row date (YYYY-MM-DD)
 - Current-quarter numeric base values required (e.g., Sales, Expenses, EPS in Rs, etc.)

If no previous historical row exists for the company, the script will interactively ask
for previous-quarter values for each required base variable so it can compute diff_prev.

Requirements:
 - ./data/directions.csv (history created by training pipeline)
 - ./models/directions/<horizon>__all.joblib artifacts (trained models)
"""

import os
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd
from joblib import load
import warnings

# sklearn helpers
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# xgboost wrapper
import xgboost as xgb

# CONFIG
DIRECTIONS_CSV = "./data/directions.csv"
MODELS_DIR = "./models/directions"
TARGETS = ["5min-ar_pct", "10min-ar_pct", "15min-ar_pct", "20min-ar_pct", "25min-ar_pct", "30min-ar_pct"]
PRED_HORIZONS = ["10min-ar_pct", "15min-ar_pct", "20min-ar_pct", "25min-ar_pct", "30min-ar_pct"]

# utilities
def try_parse_date(s):
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y", "%d/%m/%Y", "%b %Y", "%B %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    try:
        t = pd.to_datetime(s, errors="coerce")
        if pd.isna(t):
            return None
        return t.date()
    except Exception:
        return None

def load_model_artifact(horizon):
    fname = os.path.join(MODELS_DIR, f"{horizon.replace('-','_')}__all.joblib")
    if not os.path.exists(fname):
        return None, fname
    return load(fname), fname

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

def safe_predict_booster(booster, dmat, best_it):
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

# FEATURE BUILDING (mirrors training script expectations)
def build_features_for_prediction(hist_df, company, row_date, user_inputs, feature_list, prev_override=None):
    """
    Build engineered feature row matching feature_list.
    If prev_override is None the function expects a previous historical row in hist_df.
    If prev_override is provided (dict base->value) it will use those as previous-quarter values.
    Raises ValueError("NO_PREV_DATA", required_bases) when prev row missing and no override given.
    """
    df = hist_df.copy()
    if "row_date" not in df.columns:
        raise ValueError("History file must contain 'row_date' column.")
    df["row_date"] = pd.to_datetime(df["row_date"], errors="coerce").dt.date

    # canonical company column
    if "Company Symbol" not in df.columns and "Company" in df.columns:
        df = df.rename(columns={"Company": "Company Symbol"})
    if "Company Symbol" not in df.columns:
        raise ValueError("History file missing Company Symbol column.")

    # filter and sort
    comp_hist = df[df["Company Symbol"].astype(str) == str(company)].sort_values("row_date").reset_index(drop=True)

    # diff_prev columns in history (training used these)
    diff_prev_cols = [c for c in df.columns if str(c).endswith("_diff_prev")]

    # find prev row
    prev_rows = comp_hist[comp_hist["row_date"] < row_date]
    prev = prev_rows.iloc[-1] if not prev_rows.empty else None
    prev_prev = prev_rows.iloc[-2] if len(prev_rows) >= 2 else None

    if prev is None and prev_override is None:
        required_bases = [c.replace("_diff_prev", "") for c in diff_prev_cols]
        raise ValueError("NO_PREV_DATA", required_bases)

    # prepare prev values dict (either from prev row or override)
    prev_vals = {}
    if prev_override is not None:
        # use provided previous base values
        prev_vals.update(prev_override)
    else:
        for col in comp_hist.columns:
            prev_vals[col] = prev.get(col, np.nan)

    prev_prev_vals = {}
    if prev_prev is not None:
        for col in comp_hist.columns:
            prev_prev_vals[col] = prev_prev.get(col, np.nan)
    else:
        prev_prev_vals = None

    feat = {}
    # compute engineered features
    for diff_col in diff_prev_cols:
        base = diff_col.replace("_diff_prev", "")
        # prev base val
        prev_base_val = prev_vals.get(base, np.nan)
        # current base from user_inputs (case-insensitive)
        cur_base_val = None
        for k, v in user_inputs.items():
            if str(k).strip().lower() == base.strip().lower():
                cur_base_val = v
                break
        if cur_base_val is None:
            cur_base_val = np.nan

        feat[diff_col] = np.nan if pd.isna(cur_base_val) or pd.isna(prev_base_val) else float(cur_base_val) - float(prev_base_val)

        # lag1
        if prev_override is not None:
            feat[diff_col + "_lag1"] = np.nan
        else:
            prev_diff_val = pd.to_numeric(prev.get(diff_col, np.nan), errors="coerce")
            feat[diff_col + "_lag1"] = np.nan if pd.isna(prev_diff_val) else float(prev_diff_val)

        # lag2
        if prev_prev_vals is None:
            feat[diff_col + "_lag2"] = np.nan
        else:
            prev_prev_diff = pd.to_numeric(prev_prev_vals.get(diff_col, np.nan), errors="coerce")
            feat[diff_col + "_lag2"] = np.nan if pd.isna(prev_prev_diff) else float(prev_prev_diff)

        # comp mean & std
        comp_vals = pd.to_numeric(comp_hist[diff_col].astype(str).str.replace(",", "", regex=False), errors="coerce").dropna()
        feat[diff_col + "_comp_mean"] = float(comp_vals.mean()) if not comp_vals.empty else np.nan
        feat[diff_col + "_comp_std"] = float(comp_vals.std()) if not comp_vals.empty else np.nan

    # target lags
    for t in TARGETS:
        lag_col = t + "_lag1"
        if prev_override is not None:
            feat[lag_col] = np.nan
        else:
            prev_target_val = pd.to_numeric(prev.get(t, np.nan), errors="coerce")
            feat[lag_col] = np.nan if pd.isna(prev_target_val) else float(prev_target_val)

    # small pairwise interactions among first 3 diff_prev cols
    inter_base = diff_prev_cols[:3]
    for i in range(len(inter_base)):
        for j in range(i+1, len(inter_base)):
            a, b = inter_base[i], inter_base[j]
            name = f"{a}__{b}_mul"
            a_val = feat.get(a, np.nan)
            b_val = feat.get(b, np.nan)
            feat[name] = np.nan if pd.isna(a_val) or pd.isna(b_val) else float(a_val) * float(b_val)

    # assemble final features row in order of feature_list
    row = {}
    user_map = {k.strip().lower(): v for k,v in user_inputs.items()}
    for f in feature_list:
        if f in feat:
            row[f] = feat[f]
            continue
        key = f.strip().lower()
        if key in user_map:
            row[f] = user_map[key]
            continue
        row[f] = np.nan

    return pd.DataFrame([row], columns=feature_list)

# interactive prompts
def prompt_text(msg, required=True):
    while True:
        try:
            v = input(msg).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(1)
        if v == "" and required:
            print("Please enter a value.")
            continue
        return v

def prompt_float(msg, allow_skip=False):
    while True:
        v = prompt_text(msg, required=not allow_skip)
        if allow_skip and v.lower() in ("skip", "none", ""):
            return None
        try:
            return float(v.replace(",", ""))
        except Exception:
            print("Couldn't parse number. Try again (commas allowed). Type 'skip' to abort." if allow_skip else "Couldn't parse number. Try again.")

def prompt_for_prev_values(required_bases):
    print("\nNo previous historical row found for this company.")
    print("Please enter previous-quarter values for the following fields (type 'skip' to abort):")
    prev = {}
    for b in required_bases:
        val = prompt_float(f"  previous '{b}': ", allow_skip=True)
        if val is None:
            raise ValueError("User declined to provide previous data.")
        prev[b] = val
    return prev

def main():
    print("Interactive direction predictor")
    # load history
    if not os.path.exists(DIRECTIONS_CSV):
        print("History file missing:", DIRECTIONS_CSV)
        sys.exit(1)
    hist_df = pd.read_csv(DIRECTIONS_CSV, dtype=str)
    if "row_date" not in hist_df.columns:
        if "timestamp" in hist_df.columns:
            hist_df["row_date"] = pd.to_datetime(hist_df["timestamp"], errors="coerce").dt.date
        else:
            hist_df["row_date"] = pd.to_datetime(hist_df.get("Period", ""), errors="coerce").dt.date

    if "Company Symbol" not in hist_df.columns and "Company" in hist_df.columns:
        hist_df = hist_df.rename(columns={"Company": "Company Symbol"})
    if "Company Symbol" not in hist_df.columns:
        print("History file missing Company Symbol column.")
        sys.exit(1)

    # choose a model artifact to find feature_list (assume consistent across horizons)
    chosen_artifact = None
    for h in PRED_HORIZONS:
        art, fname = load_model_artifact(h)
        if art is not None:
            chosen_artifact = art
            break
    if chosen_artifact is None:
        print("No model artifact found in", MODELS_DIR)
        sys.exit(1)
    if "feature_list" in chosen_artifact:
        feature_list = chosen_artifact["feature_list"]
    elif "feature_columns" in chosen_artifact:
        feature_list = chosen_artifact["feature_columns"]
    else:
        print("Artifact missing feature list/columns.")
        sys.exit(1)

    # determine which base variables we need to ask user for current values
    # infer from diff_prev columns in history (like 'Sales_diff_prev' -> ask for 'Sales')
    diff_prev_cols = [c for c in hist_df.columns if str(c).endswith("_diff_prev")]
    required_bases = sorted({c.replace("_diff_prev", "") for c in diff_prev_cols})

    # interactive inputs: company & row_date
    company = prompt_text("Enter Company Symbol: ")
    row_date_str = prompt_text("Enter row date (YYYY-MM-DD): ")
    row_date = try_parse_date(row_date_str)
    if row_date is None:
        print("Couldn't parse date. Use YYYY-MM-DD.")
        sys.exit(1)

    # ask current-quarter base values
    print("\nEnter current-quarter values for these fields (commas allowed):")
    user_inputs = {}
    for b in required_bases:
        v = prompt_float(f"  current '{b}': ", allow_skip=False)
        user_inputs[b] = v

    # try to build features; if NO_PREV_DATA prompt for previous values
    prev_override = None
    try:
        feat_df = build_features_for_prediction(hist_df, company, row_date, user_inputs, feature_list, prev_override=None)
    except ValueError as e:
        msg = e.args[0] if len(e.args) > 0 else str(e)
        if msg == "NO_PREV_DATA":
            req_bases = e.args[1] if len(e.args) > 1 else required_bases
            try:
                prev_override = prompt_for_prev_values(req_bases)
            except ValueError:
                print("Cannot proceed without previous-quarter data. Aborting.")
                sys.exit(1)
            # rebuild
            try:
                feat_df = build_features_for_prediction(hist_df, company, row_date, user_inputs, feature_list, prev_override=prev_override)
            except Exception as e2:
                print("Failed to build features after receiving previous data:", e2)
                sys.exit(1)
        else:
            print("Error building features:", e)
            sys.exit(1)

    # run predictions for each horizon
    overall_results = {}
    for h in PRED_HORIZONS:
        art, fname = load_model_artifact(h)
        if art is None:
            print("Artifact for", h, "not found; skipping.")
            continue
        imputer = art.get("imputer")
        scaler = art.get("feature_scaler")
        booster = art.get("xgb_booster")
        lgb_model = art.get("lgb_model")
        hgb_model = art.get("hgb_model")
        meta = art.get("meta_model")
        thresh = art.get("threshold", 0.5)
        featcols = art.get("feature_list", art.get("feature_columns", feature_list))

        X_row = feat_df.copy()[featcols]
        if imputer is None or scaler is None:
            print(f"Missing imputer/scaler in artifact for {h}; skipping.")
            continue
        X_imp = imputer.transform(X_row)
        X_scaled = scaler.transform(X_imp)

        # base preds
        if booster is not None:
            try:
                best_it = safe_best_iteration(booster)
                pred_xgb = safe_predict_booster(booster, xgb.DMatrix(X_scaled), best_it)
                pred_xgb = np.clip(pred_xgb, 0.0, 1.0)
            except Exception as e:
                warnings.warn(f"XGB predict failed for {h}: {e}")
                pred_xgb = np.array([0.5])
        else:
            pred_xgb = None

        if lgb_model is not None:
            try:
                pred_lgb = lgb_model.predict_proba(X_scaled)[:,1]
            except Exception as e:
                warnings.warn(f"LGB predict failed for {h}: {e}")
                pred_lgb = None
        else:
            pred_lgb = None

        if hgb_model is not None:
            try:
                pred_hgb = hgb_model.predict_proba(X_scaled)[:,1]
            except Exception as e:
                warnings.warn(f"HGB predict failed for {h}: {e}")
                pred_hgb = None
        else:
            pred_hgb = None

        # form meta input (respect order used in training: xgb, lgb (optional), hgb)
        if art.get("lgb_model", None) is not None:
            arrs = []
            for a in (pred_xgb, pred_lgb, pred_hgb):
                if a is None:
                    arrs.append(np.array([0.5]))
                else:
                    arrs.append(a)
            X_meta = np.vstack(arrs).T
        else:
            a1 = pred_xgb if pred_xgb is not None else np.array([0.5])
            a2 = pred_hgb if pred_hgb is not None else np.array([0.5])
            X_meta = np.vstack([a1, a2]).T

        # meta proba
        if meta is not None:
            try:
                meta_proba = meta.predict_proba(X_meta)[:,1]
            except Exception as e:
                warnings.warn(f"Meta predict failed for {h}: {e}; using average.")
                meta_proba = np.mean(X_meta, axis=1)
        else:
            meta_proba = np.mean(X_meta, axis=1)

        label = (meta_proba >= thresh).astype(int)[0]
        overall_results[h] = {"proba": float(meta_proba[0]), "label": int(label), "threshold": float(thresh), "artifact": fname}

    # print summary
    print("\n=== Prediction summary ===")
    print("Company:", company)
    print("Row date:", row_date.isoformat())
    print("\nInput current-quarter values:")
    for k,v in user_inputs.items():
        print(f"  {k}: {v}")
    if prev_override is not None:
        print("\nUser-provided previous-quarter values:")
        for k,v in prev_override.items():
            print(f"  {k}: {v}")
    print("\nPredictions:")
    for h, res in overall_results.items():
        print(f" {h:12s} -> prob_up={res['proba']:.4f} (thr={res['threshold']:.3f}) => {'UP' if res['label']==1 else 'DOWN'}")

if __name__ == "__main__":
    main()
