"""Quick backtest grid search for minute-projection parameters.

This script runs projections for each parameter combination, fetches yesterday's
actual game logs, and computes MAE and floor-hit rates as simple calibration
metrics.

Usage: run from repo root:
    python tools/backtest_grid.py

Notes:
- This is a lightweight approximation: projections are for 'today' and actuals
  are taken from yesterday's games. It gives directional guidance for tuning.
"""
from datetime import datetime, timedelta
import itertools
import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simple import SimpleNBAProjection
from nba_api.stats.endpoints import playergamelogs, scoreboardv2, boxscoretraditionalv2
import unicodedata
import re
import difflib

# Config
DK_CSV = 'DKSalaries.csv'
N_SIMS = 300
# date for actuals: try to infer from DK salaries slate date; fallback to yesterday
target_date = None
date_mode = False

# parameter grid (small, quick)
starter_scales = [4.0, 6.0, 8.0]
last5_weights = [0.35, 0.40, 0.45]
last10_weights = [0.20, 0.25]
# ROLE_WEIGHT will be 1 - last5 - last10

results = []

parser = argparse.ArgumentParser(description='Backtest grid runner')
parser.add_argument('--date', help='YYYY-MM-DD date to run backtest for (date-driven mode)')
parser.add_argument('--dk-file', help='Path to a DK salaries CSV (optional)')
parser.add_argument('--starter-scales', help='Comma-separated starter scale values (e.g. 3.5,4.0,4.5)')
parser.add_argument('--last5-weights', help='Comma-separated last5 weight values (e.g. 0.35,0.40,0.45)')
parser.add_argument('--last10-weights', help='Comma-separated last10 weight values (e.g. 0.20,0.25)')
parser.add_argument('--n-sims', type=int, help='Number of Monte Carlo sims for run (overrides default)')
parser.add_argument('--backup-delta-threshold', type=float, help='Minimum delta value per $1k to apply backup minutes (overrides default)')
parser.add_argument('--boost-minutes-scale', type=float, help='Scale factor to multiply calculated minutes boost (default 1.0)')
args = parser.parse_args()

if args.date:
    target_date = args.date
    date_mode = True

if args.dk_file:
    DK_CSV = args.dk_file

if args.n_sims:
    N_SIMS = int(args.n_sims)

# allow overriding the small default grid from the CLI
def _parse_list_arg(s, cast=float):
    if not s:
        return None
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return [cast(p) for p in parts]

_starter = _parse_list_arg(args.starter_scales)
_l5 = _parse_list_arg(args.last5_weights)
_l10 = _parse_list_arg(args.last10_weights)

if _starter is not None:
    starter_scales = _starter
if _l5 is not None:
    last5_weights = _l5
if _l10 is not None:
    last10_weights = _l10

backup_delta_threshold = float(args.backup_delta_threshold) if getattr(args, 'backup_delta_threshold', None) is not None else None
boost_minutes_scale = float(args.boost_minutes_scale) if getattr(args, 'boost_minutes_scale', None) is not None else None

print('Initializing model...')
model = SimpleNBAProjection(dk_salaries_path=DK_CSV)
if not date_mode:
    print('Loading DK salaries...')
    if not model.load_dk_salaries():
        raise SystemExit('Could not load DK CSV')

# try to infer the slate date from the DK CSV's Game Info (format like 'PHX@MIN 12/08/2025 07:30PM ET')
if target_date is None:
    try:
        gi = model.dk_df['Game Info'].dropna().astype(str)
        if not gi.empty:
            import re as _re
            m = _re.search(r"(\d{1,2}/\d{1,2}/\d{4})", gi.iloc[0])
            if m:
                mm, dd, yyyy = m.group(1).split('/')
                target_date = f"{yyyy}-{int(mm):02d}-{int(dd):02d}"
    except Exception:
        target_date = None

# fallback to yesterday in UTC if we couldn't parse the DK file
if not target_date:
    target_date = (datetime.utcnow().date() - timedelta(days=1)).strftime('%Y-%m-%d')

# small helper to fetch actual DK points for a date
def fetch_actuals_for_date(date_str):
    # Try PlayerGameLogs with season first
    res = model._safe_api_call(playergamelogs.PlayerGameLogs, season_nullable=model.season,
                              date_from_nullable=date_str, date_to_nullable=date_str)
    if res:
        df = res.get_data_frames()[0]
        if not df.empty:
            df['PLAYER_NAME_L'] = df['PLAYER_NAME'].str.strip()
            # preserve original name casing; also create lowercase for matching
            df['PLAYER_NAME_LLOW'] = df['PLAYER_NAME'].str.lower().str.strip()
            df['DK_FP'] = df.apply(model.calculate_dk_points_from_row, axis=1)
            # try to capture team if available
            team_col = None
            for c in ('TEAM_ABBREVIATION', 'TEAM', 'TEAM_ID', 'TEAM_ABBR'):
                if c in df.columns:
                    team_col = c
                    break
            if team_col:
                out = df[['PLAYER_NAME_L', 'PLAYER_NAME_LLOW', 'DK_FP', team_col]].rename(columns={team_col: 'TEAM'})
            else:
                out = df[['PLAYER_NAME_L', 'PLAYER_NAME_LLOW', 'DK_FP']]
            return out

    # Try PlayerGameLogs without season (some endpoints behave oddly across seasons)
    res = model._safe_api_call(playergamelogs.PlayerGameLogs, date_from_nullable=date_str, date_to_nullable=date_str)
    if res:
        df = res.get_data_frames()[0]
        if not df.empty:
            df['PLAYER_NAME_L'] = df['PLAYER_NAME'].str.lower().str.strip()
            df['DK_FP'] = df.apply(model.calculate_dk_points_from_row, axis=1)
            return df[['PLAYER_NAME_L', 'DK_FP']]

    # Fallback: use scoreboard to find games on the date, then pull boxscores per game
    try:
        board = model._safe_api_call(scoreboardv2.ScoreboardV2, game_date=date_str)
        if board is None:
            return pd.DataFrame()
        games_df = board.get_data_frames()[0]
        rows = []
        for _, g in games_df.iterrows():
            game_id = g.get('GAME_ID') or g.get('GAME_ID')
            if not game_id:
                continue
            box = model._safe_api_call(boxscoretraditionalv2.BoxScoreTraditionalV2, game_id=game_id)
            if box is None:
                continue
            p_df = box.get_data_frames()[0]
            # compute DK points per row
            p_df['PLAYER_NAME_L'] = p_df['PLAYER_NAME'].str.strip()
            p_df['PLAYER_NAME_LLOW'] = p_df['PLAYER_NAME'].str.lower().str.strip()
            p_df['DK_FP'] = p_df.apply(model.calculate_dk_points_from_row, axis=1)
            # try to capture team abbreviation column
            team_col = None
            for c in ('TEAM_ABBREVIATION', 'TEAM', 'TEAM_ID', 'TEAM_ABBR'):
                if c in p_df.columns:
                    team_col = c
                    break
            if team_col:
                rows.append(p_df[['PLAYER_NAME_L', 'PLAYER_NAME_LLOW', 'DK_FP', team_col]].rename(columns={team_col: 'TEAM'}))
            else:
                rows.append(p_df[['PLAYER_NAME_L', 'PLAYER_NAME_LLOW', 'DK_FP']])
        if not rows:
            return pd.DataFrame()
        allp = pd.concat(rows, ignore_index=True)
        # If TEAM column present, keep first TEAM value per player
        if 'TEAM' in allp.columns:
            agg = allp.groupby('PLAYER_NAME_L').agg({'DK_FP': 'sum', 'PLAYER_NAME_LLOW': 'first', 'TEAM': 'first'}).reset_index()
            return agg.rename(columns={'PLAYER_NAME_L': 'PLAYER_NAME'})
        else:
            agg = allp.groupby('PLAYER_NAME_L').agg({'DK_FP': 'sum', 'PLAYER_NAME_LLOW': 'first'}).reset_index()
            return agg.rename(columns={'PLAYER_NAME_L': 'PLAYER_NAME'})
    except Exception:
        return pd.DataFrame()

print('Fetching actuals for', target_date)
# try a small neighborhood around the slate date in case NBA API returns data on an adjacent day
from datetime import datetime as _dt, timedelta as _td
date_dt = _dt.strptime(target_date, '%Y-%m-%d')
candidates = [date_dt + _td(days=d) for d in (0, -1, 1, -2, 2)]
actuals = pd.DataFrame()
used_date = None
for dt in candidates:
    ds = dt.strftime('%Y-%m-%d')
    print('  trying', ds)
    actuals = fetch_actuals_for_date(ds)
    if not actuals.empty:
        used_date = ds
        print('  found actuals for', used_date)
        break
if actuals.empty:
    print('Warning: no actuals found for any candidate around', target_date)
else:
    try:
        actuals.to_csv('tools/actuals_debug.csv', index=False)
        print('Wrote tools/actuals_debug.csv with', len(actuals), 'rows (date used:', used_date, ')')
    except Exception:
        pass

# If running in date-only mode, construct a synthetic DK slate from boxscore players
if date_mode and not actuals.empty:
    print('Date-mode active: building synthetic DK slate from actuals')
    # prefer TEAM if present
    # pick whichever name column is available
    name_col = None
    for c in ('PLAYER_NAME', 'PLAYER_NAME_L', 'PLAYER_NAME_LLOW'):
        if c in actuals.columns:
            name_col = c
            break
    if name_col is None:
        raise SystemExit('Unexpected actuals columns: ' + ','.join(actuals.columns.tolist()))
    names = actuals[name_col].astype(str)
    team_col = 'TEAM' if 'TEAM' in actuals.columns else None
    teams_list = actuals['TEAM'].astype(str) if team_col else [''] * len(names)
    default_salary = 6000
    dk_rows = []
    for n, t in zip(names.tolist(), teams_list):
        dk_rows.append({'Name': n.strip(), 'Salary': default_salary, 'Team': t.strip(), 'Position': ''})
    model.dk_df = pd.DataFrame(dk_rows)
    print(f"Built synthetic DK slate with {len(model.dk_df)} players (default salary={default_salary})")


# run grid
combos = list(itertools.product(starter_scales, last5_weights, last10_weights))
print('Running grid with', len(combos), 'combinations')
for scale, w5, w10 in combos:
    wrole = max(0.0, 1.0 - w5 - w10)
    # skip impossible weights
    if wrole <= 0:
        continue
    # configure model instance
    m = SimpleNBAProjection(dk_salaries_path=DK_CSV)
    if not date_mode:
        m.load_dk_salaries()
    else:
        # Date-mode: reuse the synthetic dk_df we built on the top-level model
        m.dk_df = model.dk_df.copy()
    m.STARTER_BONUS_SCALE = scale
    m.LAST5_WEIGHT = w5
    m.LAST10_WEIGHT = w10
    m.ROLE_WEIGHT = wrole
    # allow overriding backup/value thresholds and boost scaling from CLI
    if backup_delta_threshold is not None:
        m.BACKUP_DELTA_VALUE_THRESHOLD = backup_delta_threshold
    else:
        m.BACKUP_DELTA_VALUE_THRESHOLD = 1.0
    if boost_minutes_scale is not None:
        m.BOOST_MINUTES_SCALE = boost_minutes_scale
    else:
        m.BOOST_MINUTES_SCALE = 1.0

    print(f'Running projection: starter_scale={scale}, last5={w5}, last10={w10}, role={wrole:.2f}')
    # Configure model to treat the projection "today" as the target_date so
    # projections align with the actuals we fetched for that date.
    def set_model_date(model, date_str):
        from datetime import datetime as _dt, timedelta as _td

        def fetch_matchups_override():
            try:
                board = model._safe_api_call(scoreboardv2.ScoreboardV2, game_date=date_str)
                if board is None:
                    return False
                games_df = board.get_data_frames()[0]
                model.todays_matchups = {}
                for _, g in games_df.iterrows():
                    home_id = g.get('HOME_TEAM_ID')
                    away_id = g.get('VISITOR_TEAM_ID')
                    home_abbr = model.team_map.get(home_id)
                    away_abbr = model.team_map.get(away_id)
                    if home_abbr and away_abbr:
                        model.todays_matchups[away_abbr] = {'opponent': home_abbr}
                        model.todays_matchups[home_abbr] = {'opponent': away_abbr}
                return True
            except Exception:
                return False

        def fetch_recent_game_logs_override():
            print('📥 Fetching recent game logs (backtest window) ...')
            # use the configured days window ending on date_str
            date_to = date_str
            end_dt = _dt.strptime(date_str, '%Y-%m-%d')
            start_dt = end_dt - _td(days=model.days)
            date_from = start_dt.strftime('%Y-%m-%d')
            res = model._safe_api_call(playergamelogs.PlayerGameLogs, season_nullable=model.season,
                                      date_from_nullable=date_from, date_to_nullable=date_to)
            if res is None:
                return pd.DataFrame()
            logs_df = res.get_data_frames()[0]
            if logs_df.empty:
                return logs_df

            needed = ['PLAYER_ID', 'PLAYER_NAME', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M', 'GAME_DATE']
            for c in needed:
                if c not in logs_df.columns:
                    logs_df[c] = 0

            dk_names = set(model.dk_df['Name'].str.lower())
            logs_df['PLAYER_NAME_L'] = logs_df['PLAYER_NAME'].str.lower().str.strip()
            logs_df = logs_df[logs_df['PLAYER_NAME_L'].isin(dk_names)].copy()
            # ensure PLAYER_ID numeric
            logs_df['PLAYER_ID'] = logs_df['PLAYER_ID'].astype(int)
            return logs_df

        model.fetch_todays_matchups = fetch_matchups_override
        model.fetch_recent_game_logs = fetch_recent_game_logs_override

    try:
        # align model 'today' with the actuals date we found (used_date may be None)
        run_date = used_date or target_date
        set_model_date(m, run_date)
        df = m.run(save_csv=None, n_sims=N_SIMS)
    except Exception as e:
        print('Projection run failed for combo', (scale, w5, w10), e)
        continue

    if df is None or df.empty:
        print('No projections for combo', (scale, w5, w10))
        continue

    # merge with actuals using the model's normalize_name (consistent normalization)
    df['Name_norm'] = df['Name'].apply(model.normalize_name)
    actuals['PLAYER_NAME_norm'] = actuals['PLAYER_NAME_L'].apply(model.normalize_name)
    # Try exact normalized merge first
    merged = df.merge(actuals, left_on='Name_norm', right_on='PLAYER_NAME_norm', how='inner')
    if merged.empty:
        # Fuzzy fallback: try matching each projection name to the closest actuals name
        actual_map = dict(zip(actuals['PLAYER_NAME_norm'].tolist(), actuals['DK_FP'].tolist()))
        actual_keys = list(actual_map.keys())

        # build a DK_FP_matched column by fuzzy matching Name_norm -> best actual key
        def fuzzy_lookup(name_norm):
            if not name_norm:
                return None
            # exact containment heuristic
            for k in actual_keys:
                if name_norm == k:
                    return actual_map[k]
            # try startswith/endswith match
            for k in actual_keys:
                if name_norm.startswith(k) or name_norm.endswith(k) or k.startswith(name_norm) or k.endswith(name_norm):
                    return actual_map[k]
            # use difflib to find close matches
            close = difflib.get_close_matches(name_norm, actual_keys, n=1, cutoff=0.7)
            if close:
                return actual_map.get(close[0])
            return None

        df['DK_FP_matched'] = df['Name_norm'].apply(fuzzy_lookup)
        # keep only rows where we found a match
        matched = df[~df['DK_FP_matched'].isna()].copy()
        if matched.empty:
            mae = None
            floor_hit = None
            merged = matched
        else:
            # compute errors against matched DK_FP
            matched['abs_err'] = (matched['Projection'] - matched['DK_FP_matched']).abs()
            mae = matched['abs_err'].mean()
            # construct a pseudo-merged DataFrame for floor calculations if Floor exists
            if 'Floor' in matched.columns:
                floor_hit = (matched['DK_FP_matched'] >= matched['Floor']).mean()
            else:
                floor_hit = None
            # expose merged-like info
            merged = matched.rename(columns={'DK_FP_matched': 'DK_FP'})
    else:
        merged['abs_err'] = (merged['Projection'] - merged['DK_FP']).abs()
        mae = merged['abs_err'].mean()
        # floor hit rate: fraction where actual >= Floor (if Floor exists)
        if 'Floor' in merged.columns:
            floor_hit = (merged['DK_FP'] >= merged['Floor']).mean()
        else:
            floor_hit = None

    results.append({'starter_scale': scale, 'last5': w5, 'last10': w10, 'role': wrole, 'mae': mae, 'floor_hit_rate': floor_hit, 'num_merged': len(merged)})

# report
res_df = pd.DataFrame(results).sort_values('mae', na_position='last')
print('\nGrid search results (sorted by MAE):')
print(res_df.head(10).to_string(index=False))
out_csv = 'tools/backtest_grid_results.csv'
res_df.to_csv(out_csv, index=False)
print('\nSaved results to', out_csv)
print('Done')
