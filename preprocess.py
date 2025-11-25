#!/usr/bin/env python3
import pandas as pd
import os

PATH = "./data/quat.csv"
PREFIX = "https://www.bseindia.com/stock-share-price/"

def is_empty(x):
    """Helper: check if a value is empty, NaN, or whitespace."""
    if pd.isna(x):
        return True
    return str(x).strip() == ""

def main():
    if not os.path.exists(PATH):
        raise SystemExit(f"File not found: {PATH}")

    df = pd.read_csv(PATH, dtype=str)
    initial = len(df)
    print("Initial rows:", initial)

    # Ensure required columns exist
    for col in ("market_category", "fetch_status", "original_url"):
        if col not in df.columns:
            df[col] = None

    # Normalize columns
    df["__mc_norm"] = df["market_category"].fillna("").astype(str).str.strip().str.upper()
    df["__fs_norm"] = df["fetch_status"].fillna("").astype(str).str.strip().str.upper()
    df["__orig"] = df["original_url"].fillna("").astype(str)

    # Filter AM + SUCCESS
    keep_mask = (df["__mc_norm"] == "AM") & (df["__fs_norm"] == "SUCCESS")
    df = df[keep_mask].copy()
    after_basic = len(df)
    print(f"After filter (AM + SUCCESS): {after_basic} rows  ({initial - after_basic} removed)")

    # Remove rows where original_url startswith PREFIX
    bad_url_mask = df["__orig"].str.startswith(PREFIX)
    removed_bad = bad_url_mask.sum()
    df = df[~bad_url_mask].copy()
    print(f"Rows removed due to original_url prefix: {removed_bad}")

    # ----------------------------
    # NEW STEP: Drop rows with ANY empty values
    # ----------------------------
    print("Dropping rows with any empty cells...")

    # Identify rows with any empty or whitespace or NaN values
    empty_row_mask = df.apply(lambda row: any(is_empty(v) for v in row), axis=1)
    removed_empty_rows = empty_row_mask.sum()

    df = df[~empty_row_mask].copy()
    print(f"Rows removed due to empty values: {removed_empty_rows}")

    # ----------------------------
    # Drop entirely empty columns
    # ----------------------------
    print("Dropping fully empty columns...")
    cols_to_drop = []
    for col in df.columns:
        if df[col].dropna().astype(str).str.strip().eq("").all():
            cols_to_drop.append(col)

    for col in cols_to_drop:
        print(f"Dropping empty column: {col}")
        df.drop(columns=[col], inplace=True)

    # Drop helper columns
    df.drop(columns=["__mc_norm", "__fs_norm", "__orig"], inplace=True, errors="ignore")

    # Save cleaned file
    df.to_csv(PATH, index=False)
    print("Final rows:", len(df))
    print("Cleaned file saved to:", PATH)

if __name__ == "__main__":
    main()
