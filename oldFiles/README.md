HomeGrownNBAProjections
=======================

Purpose
-------
This repository contains a compact, experimental NBA DFS projection engine focused on DraftKings (DK). The goal is to produce repeatable, auditable projections and a small backtesting pipeline so we can iterate features, improve minutes estimation, model backups/injuries, integrate Vegas, and measure improvements via a date-driven backtest.

Inspiration / Examples
----------------------
- https://www.establishtherun.com — example of a production-grade projection pipeline and public writeups.
- https://www.stokastic.com — another example of a DFS projection product and evaluation approach.

Key scripts & structure
-----------------------
- `simple.py` — core projection engine (projects minutes and DK fantasy points). This is the main module to call from tools.
- `tools/` — helper scripts and diagnostics:
  - `backtest_grid.py` — run a grid search over minute-model hyperparameters and compare to actuals (date-mode supported).
  - `narrow_regrid_runner.py` — orchestrate narrow-grid experiments across a date window and aggregate results.
  - `rotation_test.py` — small A/B runner for the rotation-based minutes estimator.
  - `diagnostics_top_combo.py` — rerun a single tuned combo across dates and produce per-date top-N error CSVs and an aggregate offenders file.

Data sources
------------
- DraftKings salary CSVs (DK export) — used for slate players and salaries.
- NBA API (`nba_api`) — pulls recent game logs, boxscores and scoreboard data to build date-mode slates and actuals.
- Injury scraper (`injurySrape.py`) — lightweight scraper to detect injuries/outs. This is optional; code has fallbacks.
- Vegas totals/spreads scrapers (in `simple.py`) — multiple fallbacks; conservative defaults are used when scraping fails.

Requirements
------------
The project uses Python 3.10+ and these core libraries (see `requirements.txt`):
- pandas
- numpy
- requests
- beautifulsoup4
- nba_api

Quickstart — run a single projection
------------------------------------
1. Install required packages (use a virtualenv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run a single projection using your DK CSV (example):

```bash
python3 simple.py --dk DKSalaries.csv
```

Note: many of the tools call `SimpleNBAProjection` as a module (see `tools/backtest_grid.py`) and will try to infer the date from the DK CSV or run in date-mode.

Quickstart — run the narrow re-grid backtest
-------------------------------------------
This is how we reproduce the experiments used during tuning.

```bash
python3 tools/narrow_regrid_runner.py
# or run the grid directly for a date-window
python3 tools/backtest_grid.py --date 2025-12-05 --dk-file DKSalaries.csv
```

Evaluation & diagnostics
------------------------
- Primary metric: mean absolute error (MAE) between projection and DK actual points on a per-player basis.
- Secondary metrics: floor-hit rate (how often actuals >= floor), ceiling-hit rate, and number of merged players (coverage).
- Diagnostics: `tools/diagnostics/top20_errors_YYYY-MM-DD.csv` and `tools/diagnostics/top_combo_aggregate_errors*.csv` are produced by the diagnostics runner to identify recurring offenders.

Roadmap (prioritized)
----------------------
Short term (next iterations):
1. Stabilize ingestion & normalization: centralize name normalization and robustly map NBA API player IDs to DK names.
2. Improve minutes projection (role + rotation): refine the rotation estimator and blend with weighted recent minutes.
3. Backup/injury handling: persist and log backup-boost decisions and add gating logic so boost rules are auditable.
4. Salary estimator: when DK per-slate CSVs are missing, estimate salaries with a conservative heuristic.
5. Opponent defensive adjustment: apply team-level defense/pacing adjustments to FP_per_min.
6. Controlled bias correction (optional): only after stable backtests implement shrinkage/clipped biases.

Mid term:
- Add small automated tests and a CI workflow to run the projection smoke test on push.
- Add a small, documented experiment harness to reproduce grid runs and aggregate outputs.

How we will measure progress
----------------------------
- Each feature will be evaluated by re-running the same narrow re-grid and comparing MAE and the aggregated offender list before/after.
- Keep experiments reproducible: always record the parameter combo and the date window. Tools in `tools/` include helpers to save these outputs.

Contributing & development notes
--------------------------------
- Start by running `python3 -c "import simple; print('simple import OK')"` to ensure the core module imports cleanly.
- Prefer small, reversible edits and keep changes well-scoped. Add unit tests for any new data-parsing or matching logic.
- Use the `tools/` scripts for experiments; they provide date-mode support and synthetic slates when DK CSVs are missing.

Next recommended action
-----------------------
If you're ready, I can:
- Add a minimal pytest harness and CI stub so future changes are safer, or
- Implement the "stabilize ingestion & normalization" task next (centralize name matching and robust DK parsing), or
- Run a baseline narrow re-grid (2025-12-01..07) and output the baseline MAE so we have a clean baseline for the roadmap.

Tell me which next step you prefer and I'll proceed.
