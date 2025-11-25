# 📈 pred-stock — Quarterly → Short-Horizon Stock Direction Prediction

This repository provides a complete workflow for:

- Processing quarterly financial data
- Engineering predictive features (diff_prev, lags, company aggregates)
- Training ML models to predict short-term market direction (10–30 min)
- Deploying an interactive FastAPI-based website for predictions

The system uses boosted models (XGBoost, LightGBM, HGB) with a stacked meta-learner to predict "Up/Down" direction over multiple minute-level horizons.

## 📁 Project Structure

```
anshulkr04-pred-stock/
│
├── train.py                # Train direction models
├── predict.py              # Single-instance prediction logic
├── preprocess.py           # Data preprocessing helpers
│
├── app/
│   └── main.py             # FastAPI backend server
│
├── templates/
│   └── index.html          # Frontend UI
├── static/
│   └── styles.css          # Styling for the frontend
│
├── data/
│   └── quat.csv            # Quarterly dataset (input)
│
├── models/
│   └── directions/         # Saved models + metrics.json
│
├── figures/
│   ├── concise_am_corr_matrix.csv
│   ├── concise_am_feature_vs_target_correlations.csv
│   └── concise_am_summary.txt
│
├── src/
│   └── vis.py              # (Optional) Visualization utilities
│
├── requirements.txt
└── README.md (this file)
```

## 🔧 1. Create & activate a virtual environment

From the project root:

```bash
python3 -m venv .venv
```

Activate (macOS / Linux):

```bash
source .venv/bin/activate
```

Upgrade pip & install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 🏗️ 2. Train the model (required before running the website)

The training script:

- Loads quarterly data from `data/quat.csv`
- Computes engineered features:
  - `_diff_prev`
  - lag features (lag1, lag2)
  - per-company mean & std
  - selected interaction terms
- Computes direction labels (`dir_10min_ar_pct`, etc.)
- Performs group-aware train/test split by company
- Trains XGBoost + LightGBM + HGB
- Performs stacking via Logistic Regression
- Tunes threshold per target using OOF validation
- Saves models & metrics to `models/directions/`

Run:

```bash
python train.py
```

Expected output:

```
Loaded rows: 8110
Valid train rows: ...
Fold 1/5 ...
Chosen threshold (meta): 0.35 (OOF F1=0.62)
Test metrics for 10min-ar_pct:
 Meta stack -> acc: 0.5849   bal_acc: 0.6307  roc: 0.7060  f1: 0.6258
 ...
Done. Model artifacts saved to ./models/directions
```

After training, the predictor website can use the saved models.

## 🤖 3. How prediction works (predict.py)

`predict.py` contains:

- Loading the saved models (`models/directions/...__all.joblib`)
- Computing `diff_prev`, lag features, and company aggregates on user input
- Running the base models → stacking meta model → applying tuned threshold
- Returning:

```json
{
  "10min-ar_pct": { "proba": ..., "threshold": ..., "label": 0/1 },
  ...
}
```

If previous-quarter numbers are missing, the script asks the user for previous data before computing `diff_prev` features.

## 🌐 4. Run the website (FastAPI server)

Start the backend:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

```
http://localhost:8000
```

What the site provides:

- Clean UI for inputting fundamentals (Sales, Expenses, OPM %, EPS, etc.)
- Automatic preprocessing + feature engineering
- Short-horizon predictions displayed as:
  - **10 min**: ▲ Up
  - **Next 5 min**: ▼ Down
  - **Next 5 min**: ▲ Up
  - ...
- Probability bars and threshold reference
- Optionally links to your uploaded PDF documentation

## 📊 (Optional) Visualizations

To regenerate correlation matrices or summary statistics:

```bash
python src/vis.py
```

Outputs will appear in the `figures/` directory.

## 🛠 Troubleshooting

**Model not loading?**  
Re-run:

```bash
python train.py
```

**Missing dependencies?**  
Ensure venv is active:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**FastAPI not running?**  
Install:

```bash
pip install fastapi uvicorn
```
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
