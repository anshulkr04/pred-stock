#!/usr/bin/env python3
"""
viz_concise_am_only.py

Concise visualization (6-7 plots) comparing focused financial features
against targets, but only for rows where market_category == 'AM'.

Outputs saved to ./plots/
"""

import os
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from dateutil import parser
import matplotlib.pyplot as plt
import seaborn as sns

# --------------- CONFIG ---------------
INPUT_CSV = "../data/quarterly_results_fixed.csv"
PLOTS_DIR = Path("../figures")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = [
    "10min-ar_pct", "15min-ar_pct", "20min-ar_pct", "25min-ar_pct", "30min-ar_pct"
]

BASE_FEATURES = [
    "Sales", "Expenses", "Operating Profit", "OPM %", "Other Income",
    "Interest", "Depreciation", "Profit Before Tax", "Tax %", "Net Profit", "EPS in Rs"
]
DIFF_FEATURES = [f + "_diff_prev" for f in BASE_FEATURES]
FEATURE_COLS = BASE_FEATURES + DIFF_FEATURES

# --------------- HELPERS ---------------
def clean_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).replace({"nan": pd.NA, "None": pd.NA})
    s = s.str.strip().str.replace(r'^"|"$', "", regex=True)
    s = s.str.replace(",", "", regex=False)
    pct_mask = s.str.endswith("%", na=False)
    s.loc[pct_mask] = s.loc[pct_mask].str.replace("%", "", regex=False)
    s = s.replace("", pd.NA)
    return pd.to_numeric(s, errors="coerce")

def safe_savefig(fig, fname, dpi=180):
    path = PLOTS_DIR / fname
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

# --------------- LOAD ---------------
if not Path(INPUT_CSV).exists():
    print(f"ERROR: {INPUT_CSV} not found. Put the CSV in the working directory or update INPUT_CSV.")
    sys.exit(1)

df = pd.read_csv(INPUT_CSV, dtype=str)
df.columns = [c.strip() for c in df.columns]

# --------------- FILTER to AM only ---------------
if "market_category" not in df.columns:
    print("ERROR: market_category column not present. Cannot filter to AM rows. Exiting.")
    sys.exit(1)

df["market_category_norm"] = df["market_category"].astype(str).str.strip().str.upper()
df = df[df["market_category_norm"] == "AM"].copy()
df = df.drop(columns=["market_category_norm"])

if df.empty:
    print("No rows with market_category == 'AM' found. Exiting.")
    sys.exit(1)

print(f"Rows after filtering to AM: {len(df)}")

# --------------- SELECT & CLEAN ---------------
present_features = [c for c in FEATURE_COLS if c in df.columns]
present_targets = [c for c in TARGET_COLS if c in df.columns]

if not present_targets:
    print("ERROR: none of the target columns found in CSV. Exiting.")
    sys.exit(1)
if not present_features:
    print("ERROR: none of the focused features found in CSV. Exiting.")
    sys.exit(1)

print("Found targets:", present_targets)
print("Found focused features:", present_features)

keep_cols = present_features + present_targets + ["Company Symbol", "Period", "market_category", "is_weekend"]
df = df[[c for c in keep_cols if c in df.columns]]

# Clean numeric versions
num_feature_cols = []
for col in present_features:
    df[f"_num_{col}"] = clean_numeric_series(df[col])
    num_feature_cols.append(f"_num_{col}")

num_target_cols = []
for col in present_targets:
    df[f"_num_{col}"] = clean_numeric_series(df[col])
    num_target_cols.append(f"_num_{col}")

# Drop rows where all targets are NaN
df = df.dropna(axis=0, how="all", subset=num_target_cols)
if df.empty:
    print("No rows left after dropping rows with no targets. Exiting.")
    sys.exit(1)

# numeric dataframe for correlation
num_df = df[num_feature_cols + num_target_cols].copy()
num_df = num_df.loc[:, num_df.notna().any(axis=0)]
if num_df.empty:
    print("No numeric data left after cleaning. Exiting.")
    sys.exit(1)

# --------------- CORRELATIONS ---------------
corr = num_df.corr(method="pearson")
corr.to_csv(PLOTS_DIR / "concise_am_corr_matrix.csv")

# build feature-target correlation table (features rows, targets cols)
feature_num = [c for c in num_df.columns if c in num_feature_cols]
target_num = [c for c in num_df.columns if c in num_target_cols]

ft_corr = pd.DataFrame(index=[f.replace("_num_","") for f in feature_num],
                       columns=[t.replace("_num_","") for t in target_num],
                       dtype=float)
for f in feature_num:
    for t in target_num:
        if f in corr.index and t in corr.columns:
            ft_corr.loc[f.replace("_num_",""), t.replace("_num_","")] = corr.loc[f, t]
ft_corr.to_csv(PLOTS_DIR / "concise_am_feature_vs_target_correlations.csv")

# --------------- PLOTS ---------------
sns.set(style="whitegrid", context="talk")

# 1) Heatmap (features x targets)
if not ft_corr.isna().all().all():
    fig, ax = plt.subplots(figsize=(max(6, 1.2*len(ft_corr.columns)), max(4, 0.5*len(ft_corr))))
    sns.heatmap(ft_corr, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature vs Target Pearson correlations (AM only)")
    safe_savefig(fig, "heatmap_features_vs_targets_am.png")
else:
    print("No valid feature-target correlation values to plot heatmap.")

# 2) Bar plot - top 5 absolute correlations per target (grid)
n_targets = len(target_num)
fig, axes = plt.subplots(n_targets, 1, figsize=(8, 3*n_targets), squeeze=False)
for i, t in enumerate(target_num):
    vals = []
    labs = []
    for f in feature_num:
        if f in corr.index and t in corr.columns:
            v = corr.loc[f, t]
            if not np.isnan(v):
                vals.append(abs(v))
                labs.append(f.replace("_num_",""))
    if not vals:
        axes[i,0].text(0.2, 0.5, f"No data for {t.replace('_num_','')}", fontsize=12)
        axes[i,0].axis("off")
        continue
    order = np.argsort(vals)[::-1][:5]
    vals_sorted = [vals[j] for j in order]
    labs_sorted = [labs[j] for j in order]
    sns.barplot(x=vals_sorted, y=labs_sorted, ax=axes[i,0])
    axes[i,0].set_xlabel("abs(pearson corr)")
    axes[i,0].set_title(f"Top 5 features (abs corr) with {t.replace('_num_','')}")
plt.tight_layout()
safe_savefig(fig, "bar_top5_per_target_am.png")

# 3) Scatter + trend: top 1 feature per target (one file per target) — up to 5 plots
from statsmodels.nonparametric.smoothers_lowess import lowess
for t in target_num:
    best = None
    best_val = -1
    for f in feature_num:
        if f in corr.index and t in corr.columns:
            v = corr.loc[f, t]
            if not np.isnan(v) and abs(v) > best_val:
                best_val = abs(v); best = f
    if best is None:
        print(f"Skipping scatter for {t}: no best feature found.")
        continue

    x = df[best].astype(float)
    y = df[t].astype(float)
    pairs = pd.concat([x,y], axis=1).dropna()
    if pairs.shape[0] < 3:
        print(f"Skipping scatter for {t}: insufficient paired data ({pairs.shape[0]} rows).")
        continue

    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(pairs[best], pairs[t], alpha=0.6, s=30)
    # LOWESS smooth
    try:
        lo = lowess(pairs[t].values, pairs[best].values, frac=0.4)
        ax.plot(lo[:,0], lo[:,1], color="red", linewidth=2, label="LOWESS")
    except Exception:
        pass
    # linear fit
    try:
        m,b = np.polyfit(pairs[best], pairs[t], 1)
        xs = np.linspace(pairs[best].min(), pairs[best].max(), 50)
        ax.plot(xs, m*xs + b, color="black", linestyle="--", label=f"linear (r={corr.loc[best,t]:.3f})")
    except Exception:
        pass

    ax.set_xlabel(best.replace("_num_",""))
    ax.set_ylabel(t.replace("_num_",""))
    ax.set_title(f"{best.replace('_num_','')} vs {t.replace('_num_','')} (n={len(pairs)})")
    ax.legend()
    safe_savefig(fig, f"scatter_top_feature_vs_{t.replace('_num_','')}_am.png")

# --------------- SUMMARY FILE ---------------
summary = []
summary.append("Concise AM-only visualization run")
summary.append(f"Rows (after filtering to AM and cleaning): {len(df)}")
summary.append("Present focused features: " + ", ".join([f.replace("_num_","") for f in feature_num]))
summary.append("Present targets: " + ", ".join([t.replace("_num_","") for t in target_num]))
summary.append("")
summary.append("Top feature (abs corr) per target:")
for t in target_num:
    best = None; best_val=-1
    for f in feature_num:
        if f in corr.index and t in corr.columns:
            v = corr.loc[f,t]
            if not np.isnan(v) and abs(v) > best_val:
                best_val=abs(v); best=f
    if best is None:
        summary.append(f"  {t.replace('_num_','')}: None")
    else:
        summary.append(f"  {t.replace('_num_','')}: {best.replace('_num_','')} (corr={corr.loc[best,t]:.3f})")

with open(PLOTS_DIR / "concise_am_summary.txt", "w") as fh:
    fh.write("\n".join(summary))

print("Saved concise AM-only plots to:", PLOTS_DIR.resolve())
