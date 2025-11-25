
# pred-stock

Lightweight repo for stock prediction visualizations and analysis. This README explains how to set up a Python virtual environment (macOS / zsh), install dependencies, and run the visualization script `src/vis.py`.

## What this repo contains

- `data/` — CSV source data (e.g. `quarterly_results_fixed.csv`).
- `src/vis.py` — visualization script (generates plots into `figures/`).
- `figures/` — output images produced by `src/vis.py` (already contains example images).
- `requirements.txt` — pinned Python package dependencies used by the project.


## Create and activate a virtual environment (recommended)

1. From the repository root run:

```bash
python3 -m venv .venv
```

2. Activate the venv (zsh):

```bash
source .venv/bin/activate
```

3. Upgrade pip and install the project's dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- Using a named venv (here `.venv`) keeps dependencies local to the project.
- On first activation in a new shell you may need to run `source .venv/bin/activate` again.

## Run the visualization script

With the virtual environment active and dependencies installed, run:

```bash
python src/vis.py
```

What to expect:
- The script reads data from `data/` (for example `data/quarterly_results_fixed.csv`) and writes images into `figures/`.
- After running, check `figures/` for generated plots such as heatmaps, scatter plots, and summary images.

If you prefer running from VS Code, open the project folder and select the `.venv` interpreter in the bottom-right status bar or via the Command Palette (Python: Select Interpreter).

## Quick troubleshooting

- "Module not found" or import errors: ensure the virtual environment is activated (`which python` should point inside `.venv`). Reinstall deps with `pip install -r requirements.txt`.
- Permission errors when writing to `figures/`: ensure the `figures/` directory exists and is writable. Create it manually if missing:

	```bash
	mkdir -p figures
	chmod u+w figures
	```

- If `src/vis.py` expects additional environment variables or config files, check the top of `src/vis.py` for hints (or a `.env` file). This repo includes `python-dotenv` in `requirements.txt` if a `.env` is used.

## Notes on dependencies

This repo pins dependencies in `requirements.txt`. These packages were captured on a recent environment and include (abridged): numpy, pandas, matplotlib, seaborn, scikit-learn, statsmodels, etc. If you run into platform-specific build issues (rare on macOS with wheel support), ensure you have an up-to-date pip and wheel:

```bash
python -m pip install --upgrade pip wheel
```
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
