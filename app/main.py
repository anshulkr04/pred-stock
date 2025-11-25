# app/main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Optional, Any
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from joblib import load
import warnings

# ML wrappers
import xgboost as xgb

# ---- CONFIG ----
# global for template rendering
REQUIRED_BASES = []

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root with app/ at top
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models", "directions")
DIRECTIONS_CSV = os.path.join(DATA_DIR, "directions.csv")

PRED_HORIZONS = ["10min-ar_pct", "15min-ar_pct", "20min-ar_pct", "25min-ar_pct", "30min-ar_pct"]
TARGETS = ["5min-ar_pct", "10min-ar_pct", "15min-ar_pct", "20min-ar_pct", "25min-ar_pct", "30min-ar_pct"]

# ---- FASTAPI app ----
app = FastAPI(title="Directions Predictor")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ---- Load history and artifacts at startup ----
HIST_DF = None
ARTIFACTS = {}  # horizon -> artifact dict

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

def load_artifacts():
    global ARTIFACTS
    for h in PRED_HORIZONS:
        fname = os.path.join(MODELS_DIR, f"{h.replace('-','_')}__all.joblib")
        if os.path.exists(fname):
            try:
                art = load(fname)
                ARTIFACTS[h] = art
                print(f"Loaded artifact for {h} from {fname}")
            except Exception as e:
                warnings.warn(f"Failed to load artifact {fname}: {e}")
        else:
            print(f"No artifact found for {h} at {fname}")

def load_history():
    global HIST_DF, REQUIRED_BASES
    if not os.path.exists(DIRECTIONS_CSV):
        print("Warning: history CSV not found:", DIRECTIONS_CSV)
        HIST_DF = None
        return
    df = pd.read_csv(DIRECTIONS_CSV, dtype=str)
    if "row_date" not in df.columns:
        if "timestamp" in df.columns:
            df["row_date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
        else:
            df["row_date"] = pd.to_datetime(df.get("Period", ""), errors="coerce").dt.date
    if "Company Symbol" not in df.columns and "Company" in df.columns:
        df = df.rename(columns={"Company": "Company Symbol"})
    HIST_DF = df
    print("Loaded history rows:", len(df))

    # compute required base fields from diff_prev columns
    diff_prev_cols = [c for c in df.columns if str(c).endswith("_diff_prev")]
    REQUIRED_BASES = sorted({c.replace("_diff_prev", "") for c in diff_prev_cols})
    print("Required base fields:", REQUIRED_BASES)


@app.on_event("startup")
def startup_event():
    load_history()
    load_artifacts()

# ---- Prediction helpers (same approach as your predict script) ----
def build_features_for_prediction(hist_df: pd.DataFrame, company: str, row_date: datetime.date,
                                  user_inputs: Dict[str, Any], feature_list, prev_override: Optional[Dict]=None):
    df = hist_df.copy()
    if "row_date" not in df.columns:
        raise ValueError("History file must contain 'row_date' column.")
    df["row_date"] = pd.to_datetime(df["row_date"], errors="coerce").dt.date

    if "Company Symbol" not in df.columns:
        raise ValueError("History file missing Company Symbol column.")

    comp_hist = df[df["Company Symbol"].astype(str) == str(company)].sort_values("row_date").reset_index(drop=True)
    diff_prev_cols = [c for c in df.columns if str(c).endswith("_diff_prev")]

    prev_rows = comp_hist[comp_hist["row_date"] < row_date]
    prev = prev_rows.iloc[-1] if not prev_rows.empty else None
    prev_prev = prev_rows.iloc[-2] if len(prev_rows) >=2 else None

    if prev is None and prev_override is None:
        required_bases = [c.replace("_diff_prev", "") for c in diff_prev_cols]
        # sentinel to indicate frontend should ask for them
        raise ValueError("NO_PREV_DATA", required_bases)

    prev_vals = {}
    if prev_override is not None:
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
    for diff_col in diff_prev_cols:
        base = diff_col.replace("_diff_prev", "")
        prev_base_val = prev_vals.get(base, np.nan)
        cur_base_val = None
        for k,v in user_inputs.items():
            if str(k).strip().lower() == base.strip().lower():
                cur_base_val = v
                break
        if cur_base_val is None:
            cur_base_val = np.nan

        feat[diff_col] = np.nan if pd.isna(cur_base_val) or pd.isna(prev_base_val) else float(cur_base_val) - float(prev_base_val)

        if prev_override is not None:
            feat[diff_col + "_lag1"] = np.nan
        else:
            prev_diff_val = pd.to_numeric(prev.get(diff_col, np.nan), errors="coerce")
            feat[diff_col + "_lag1"] = np.nan if pd.isna(prev_diff_val) else float(prev_diff_val)

        if prev_prev_vals is None:
            feat[diff_col + "_lag2"] = np.nan
        else:
            prev_prev_diff = pd.to_numeric(prev_prev_vals.get(diff_col, np.nan), errors="coerce")
            feat[diff_col + "_lag2"] = np.nan if pd.isna(prev_prev_diff) else float(prev_prev_diff)

        comp_vals = pd.to_numeric(comp_hist[diff_col].astype(str).str.replace(",", "", regex=False), errors="coerce").dropna()
        feat[diff_col + "_comp_mean"] = float(comp_vals.mean()) if not comp_vals.empty else np.nan
        feat[diff_col + "_comp_std"] = float(comp_vals.std()) if not comp_vals.empty else np.nan

    for t in TARGETS:
        lag_col = t + "_lag1"
        if prev_override is not None:
            feat[lag_col] = np.nan
        else:
            prev_target_val = pd.to_numeric(prev.get(t, np.nan), errors="coerce")
            feat[lag_col] = np.nan if pd.isna(prev_target_val) else float(prev_target_val)

    inter_base = diff_prev_cols[:3]
    for i in range(len(inter_base)):
        for j in range(i+1, len(inter_base)):
            a, b = inter_base[i], inter_base[j]
            name = f"{a}__{b}_mul"
            a_val = feat.get(a, np.nan)
            b_val = feat.get(b, np.nan)
            feat[name] = np.nan if pd.isna(a_val) or pd.isna(b_val) else float(a_val) * float(b_val)

    row = {}
    user_map = {k.strip().lower(): v for k, v in user_inputs.items()}
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

# ---- API schemas ----
class PredictRequest(BaseModel):
    company: str
    row_date: str
    current: Dict[str, float]
    prev: Optional[Dict[str, float]] = None   # optional previous-quarter values if user provides them

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "required_bases": REQUIRED_BASES})


@app.post("/predict")
async def predict(req: PredictRequest):
    # validate
    company = req.company
    row_date = try_parse_date(req.row_date)
    if row_date is None:
        raise HTTPException(status_code=400, detail="Invalid row_date format. Use YYYY-MM-DD.")
    current = req.current or {}
    prev_override = req.prev

    if HIST_DF is None:
        raise HTTPException(status_code=500, detail="Server history not available. Ensure directions.csv exists on server.")

    # pick a loaded artifact to get feature_list
    if not ARTIFACTS:
        raise HTTPException(status_code=500, detail="No model artifacts loaded on server.")
    # pick first artifact as schema
    any_art = next(iter(ARTIFACTS.values()))
    feature_list = any_art.get("feature_list", any_art.get("feature_columns"))
    if feature_list is None:
        raise HTTPException(status_code=500, detail="Artifacts lack feature_list/feature_columns.")

    # try to build features. if no prev data available ask for previous bases
    try:
        feat_df = build_features_for_prediction(HIST_DF, company, row_date, current, feature_list, prev_override=prev_override)
    except ValueError as e:
        msg = e.args[0] if len(e.args) > 0 else str(e)
        if msg == "NO_PREV_DATA":
            required = e.args[1] if len(e.args) > 1 else []
            return JSONResponse(status_code=200, content={"needs_prev": required})
        else:
            raise HTTPException(status_code=400, detail=str(e))

    # iterate horizons and compute stacked proba + label
    results = {}
    for h, art in ARTIFACTS.items():
        imputer = art.get("imputer")
        scaler = art.get("feature_scaler")
        booster = art.get("xgb_booster")
        lgb_model = art.get("lgb_model")
        hgb_model = art.get("hgb_model")
        meta = art.get("meta_model")
        thresh = float(art.get("threshold", 0.5))
        featcols = art.get("feature_list", art.get("feature_columns", feature_list))

        X_row = feat_df.copy()[featcols]
        try:
            X_imp = imputer.transform(X_row)
            X_scaled = scaler.transform(X_imp)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Preprocessing failed for horizon {h}: {e}")

        # xgb
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

        # lgb
        if lgb_model is not None:
            try:
                pred_lgb = lgb_model.predict_proba(X_scaled)[:,1]
            except Exception:
                pred_lgb = None
        else:
            pred_lgb = None

        # hgb
        if hgb_model is not None:
            try:
                pred_hgb = hgb_model.predict_proba(X_scaled)[:,1]
            except Exception:
                pred_hgb = None
        else:
            pred_hgb = None

        if art.get("lgb_model", None) is not None:
            arrs = []
            for a in (pred_xgb, pred_lgb, pred_hgb):
                arrs.append(a if a is not None else np.array([0.5]))
            X_meta = np.vstack(arrs).T
        else:
            a1 = pred_xgb if pred_xgb is not None else np.array([0.5])
            a2 = pred_hgb if pred_hgb is not None else np.array([0.5])
            X_meta = np.vstack([a1, a2]).T

        if meta is not None:
            try:
                meta_proba = meta.predict_proba(X_meta)[:,1]
            except Exception:
                meta_proba = np.mean(X_meta, axis=1)
        else:
            meta_proba = np.mean(X_meta, axis=1)

        label = int((meta_proba >= thresh).astype(int)[0])
        results[h] = {"proba": float(meta_proba[0]), "label": label, "threshold": float(thresh)}

    return JSONResponse(status_code=200, content={"company": company, "row_date": row_date.isoformat(), "results": results})

# Serve static index assets (this is optional but convenient)
@app.get("/favicon.ico")
def favicon():
    path = os.path.join(BASE_DIR, "static", "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404)
