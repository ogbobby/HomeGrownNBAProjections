"""Per-player diagnostics for a single parameter combo across multiple dates.

Saves per-date top-20 absolute errors and an aggregated CSV identifying
recurring high-error players and error modes.
"""
from datetime import datetime
import os
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simple import SimpleNBAProjection
from nba_api.stats.endpoints import playergamelogs, scoreboardv2, boxscoretraditionalv2

# combo to analyze (top aggregated combo from narrow grid)
COMBO = dict(starter_scale=3.5, last5=0.50, last10=0.25)

DATES = [
    '2025-12-01','2025-12-02','2025-12-03','2025-12-04','2025-12-05','2025-12-06','2025-12-07'
]

OUTDIR = 'tools/diagnostics'
os.makedirs(OUTDIR, exist_ok=True)

def fetch_actuals_for_date(model, date_str):
    # tries playergamelogs then scoreboard->boxscores fallback
    try:
        res = model._safe_api_call(playergamelogs.PlayerGameLogs, season_nullable=model.season,
                                  date_from_nullable=date_str, date_to_nullable=date_str)
        if res:
            df = res.get_data_frames()[0]
            if not df.empty:
                df['PLAYER_NAME_L'] = df['PLAYER_NAME'].str.strip()
                df['PLAYER_NAME_LLOW'] = df['PLAYER_NAME'].str.lower().str.strip()
                df['DK_FP'] = df.apply(model.calculate_dk_points_from_row, axis=1)
                team_col = None
                for c in ('TEAM_ABBREVIATION','TEAM','TEAM_ID','TEAM_ABBR'):
                    if c in df.columns:
                        team_col = c; break
                if team_col:
                    out = df[['PLAYER_NAME_L','PLAYER_NAME_LLOW','DK_FP',team_col]].rename(columns={team_col:'TEAM'})
                else:
                    out = df[['PLAYER_NAME_L','PLAYER_NAME_LLOW','DK_FP']]
                return out.rename(columns={'PLAYER_NAME_L':'PLAYER_NAME'})
    except Exception:
        pass

    # scoreboard/boxscore fallback
    try:
        board = model._safe_api_call(scoreboardv2.ScoreboardV2, game_date=date_str)
        if board is None:
            return pd.DataFrame()
        games_df = board.get_data_frames()[0]
        rows = []
        for _, g in games_df.iterrows():
            game_id = g.get('GAME_ID')
            if not game_id:
                continue
            box = model._safe_api_call(boxscoretraditionalv2.BoxScoreTraditionalV2, game_id=game_id)
            if box is None:
                continue
            p_df = box.get_data_frames()[0]
            if p_df.empty:
                continue
            p_df['PLAYER_NAME_L'] = p_df['PLAYER_NAME'].str.strip()
            p_df['PLAYER_NAME_LLOW'] = p_df['PLAYER_NAME'].str.lower().str.strip()
            p_df['DK_FP'] = p_df.apply(model.calculate_dk_points_from_row, axis=1)
            team_col = None
            for c in ('TEAM_ABBREVIATION','TEAM','TEAM_ID','TEAM_ABBR'):
                if c in p_df.columns:
                    team_col = c; break
            if team_col:
                rows.append(p_df[['PLAYER_NAME_L','PLAYER_NAME_LLOW','DK_FP',team_col]].rename(columns={team_col:'TEAM'}))
            else:
                rows.append(p_df[['PLAYER_NAME_L','PLAYER_NAME_LLOW','DK_FP']])
        if not rows:
            return pd.DataFrame()
        allp = pd.concat(rows, ignore_index=True)
        if 'TEAM' in allp.columns:
            agg = allp.groupby('PLAYER_NAME_L').agg({'DK_FP':'sum','PLAYER_NAME_LLOW':'first','TEAM':'first'}).reset_index()
            return agg.rename(columns={'PLAYER_NAME_L':'PLAYER_NAME'})
        else:
            agg = allp.groupby('PLAYER_NAME_L').agg({'DK_FP':'sum','PLAYER_NAME_LLOW':'first'}).reset_index()
            return agg.rename(columns={'PLAYER_NAME_L':'PLAYER_NAME'})
    except Exception:
        return pd.DataFrame()

def run_for_date(d, combo):
    print('Running diagnostics for', d)
    # prepare a model helper to ensure we have a dk_df; prefer loading DK CSV, else build synthetic from actuals
    helper = SimpleNBAProjection(dk_salaries_path='DKSalaries.csv')
    loaded = False
    try:
        loaded = helper.load_dk_salaries()
    except Exception:
        loaded = False

    dk_df = None
    if loaded:
        dk_df = helper.dk_df.copy()
    else:
        # try to fetch actuals and build synthetic dk_df
        actuals_tmp = fetch_actuals_for_date(helper, d)
        if not actuals_tmp.empty:
            names = actuals_tmp['PLAYER_NAME'].astype(str)
            default_salary = 6000
            rows = [{'Name': n.strip(), 'Salary': default_salary, 'Team': (actuals_tmp.loc[i,'TEAM'] if 'TEAM' in actuals_tmp.columns else ''), 'Position': ''} for i,n in enumerate(names)]
            dk_df = pd.DataFrame(rows)

    m = SimpleNBAProjection(dk_salaries_path='DKSalaries.csv')
    if dk_df is not None:
        m.dk_df = dk_df.copy()
    # set combo
    m.STARTER_BONUS_SCALE = combo['starter_scale']
    m.LAST5_WEIGHT = combo['last5']
    m.LAST10_WEIGHT = combo['last10']
    m.ROLE_WEIGHT = max(0.0, 1.0 - combo['last5'] - combo['last10'])
    # apply tuned backup/minute-boost settings (match runner adjustments)
    # allow overrides via environment variables so we can run experiments
    m.BACKUP_DELTA_VALUE_THRESHOLD = float(os.getenv('BACKUP_DELTA_VALUE_THRESHOLD', '0.8'))
    m.BOOST_MINUTES_SCALE = float(os.getenv('BOOST_MINUTES_SCALE', '1.1'))

    # override matchups and recent logs to align to d
    from datetime import datetime as _dt, timedelta as _td
    def fetch_matchups_override():
        try:
            board = m._safe_api_call(scoreboardv2.ScoreboardV2, game_date=d)
            if board is None:
                return False
            games_df = board.get_data_frames()[0]
            m.todays_matchups = {}
            for _, g in games_df.iterrows():
                home_id = g.get('HOME_TEAM_ID')
                away_id = g.get('VISITOR_TEAM_ID')
                home_abbr = m.team_map.get(home_id)
                away_abbr = m.team_map.get(away_id)
                if home_abbr and away_abbr:
                    m.todays_matchups[away_abbr] = {'opponent': home_abbr}
                    m.todays_matchups[home_abbr] = {'opponent': away_abbr}
            return True
        except Exception:
            return False

    def fetch_recent_game_logs_override():
        # backtest window ending on date d
        date_to = d
        end_dt = _dt.strptime(d, '%Y-%m-%d')
        start_dt = end_dt - _td(days=m.days)
        date_from = start_dt.strftime('%Y-%m-%d')
        res = m._safe_api_call(playergamelogs.PlayerGameLogs, season_nullable=m.season,
                                  date_from_nullable=date_from, date_to_nullable=date_to)
        if res is None:
            return pd.DataFrame()
        logs_df = res.get_data_frames()[0]
        if logs_df.empty:
            return logs_df
        needed = ['PLAYER_ID','PLAYER_NAME','MIN','PTS','REB','AST','STL','BLK','TOV','FG3M','GAME_DATE']
        for c in needed:
            if c not in logs_df.columns:
                logs_df[c] = 0
        dk_names = set(m.dk_df['Name'].str.lower()) if hasattr(m,'dk_df') and 'Name' in m.dk_df.columns else set()
        logs_df['PLAYER_NAME_L'] = logs_df['PLAYER_NAME'].str.lower().str.strip()
        if dk_names:
            logs_df = logs_df[logs_df['PLAYER_NAME_L'].isin(dk_names)].copy()
        logs_df['PLAYER_ID'] = logs_df.get('PLAYER_ID',0).fillna(0).astype(int)
        return logs_df

    m.fetch_todays_matchups = fetch_matchups_override
    m.fetch_recent_game_logs = fetch_recent_game_logs_override

    # run projection
    try:
        df = m.run(save_csv=None, n_sims=200)
    except Exception as e:
        print('Projection failed for date', d, e)
        return None
    if df is None or df.empty:
        print('No projections for', d)
        return None

    # fetch actuals
    actuals = fetch_actuals_for_date(m, d)
    if actuals.empty:
        print('No actuals for', d)
        return None

    # normalize names
    df['Name_norm'] = df['Name'].apply(m.normalize_name)
    actuals['PLAYER_NAME_norm'] = actuals['PLAYER_NAME'].apply(m.normalize_name)

    merged = df.merge(actuals, left_on='Name_norm', right_on='PLAYER_NAME_norm', how='inner')
    if merged.empty:
        # fuzzy fallback
        import difflib
        actual_map = dict(zip(actuals['PLAYER_NAME_norm'].tolist(), actuals['DK_FP'].tolist()))
        actual_keys = list(actual_map.keys())
        def fuzzy_lookup(name_norm):
            if not name_norm:
                return None
            for k in actual_keys:
                if name_norm == k:
                    return actual_map[k]
            for k in actual_keys:
                if name_norm.startswith(k) or name_norm.endswith(k) or k.startswith(name_norm) or k.endswith(name_norm):
                    return actual_map[k]
            close = difflib.get_close_matches(name_norm, actual_keys, n=1, cutoff=0.7)
            if close:
                return actual_map.get(close[0])
            return None
        df['DK_FP_matched'] = df['Name_norm'].apply(fuzzy_lookup)
        matched = df[~df['DK_FP_matched'].isna()].copy()
        if matched.empty:
            print('No fuzzy matches for', d)
            return None
        matched['abs_err'] = (matched['Projection'] - matched['DK_FP_matched']).abs()
        out = matched.sort_values('abs_err', ascending=False).head(20)
        out_cols = ['Name','Projection','DK_FP_matched','abs_err']
        out = out[out_cols]
        out.to_csv(os.path.join(OUTDIR,f'top20_errors_{d}.csv'), index=False)
        return out
    else:
        merged['abs_err'] = (merged['Projection'] - merged['DK_FP']).abs()
        merged['minutes_proj'] = merged.get('Minutes', merged.get('Min', None))
        # try to get minutes actual from actuals if present
        if 'MIN' in actuals.columns:
            merged = merged.rename(columns={'MIN':'minutes_actual'})
        out = merged.sort_values('abs_err', ascending=False).head(20)
        out_cols = ['Name','Projection','DK_FP','abs_err','minutes_proj']
        if 'minutes_actual' in out.columns:
            out_cols.append('minutes_actual')
        out = out[out_cols]
        out.to_csv(os.path.join(OUTDIR,f'top20_errors_{d}.csv'), index=False)
        return out

if __name__ == '__main__':
    per_date_files = []
    rows = []
    for d in DATES:
        out = run_for_date(d, COMBO)
        if out is None:
            continue
        # ensure file exists
        fname = os.path.join(OUTDIR, f'top20_errors_{d}.csv')
        if os.path.exists(fname):
            per_date_files.append(fname)
            tmp = pd.read_csv(fname)
            tmp = tmp.assign(date=d)
            # unify player name col
            if 'Name' in tmp.columns:
                tmp = tmp.rename(columns={'Name':'player'})
            rows.append(tmp)

    if not rows:
        print('No per-date diagnostics produced')
        sys.exit(0)

    all_err = pd.concat(rows, ignore_index=True)
    # normalize player string
    all_err['player_norm'] = all_err['player'].astype(str).str.lower().str.strip()
    agg = all_err.groupby('player_norm').agg(
        count_dates=('date','nunique'),
        mean_abs_err=('abs_err','mean'),
        max_abs_err=('abs_err','max')
    ).reset_index().sort_values(['mean_abs_err','count_dates'], ascending=[False,False])
    agg.to_csv(os.path.join(OUTDIR,'top_combo_aggregate_errors.csv'), index=False)
    print('Wrote per-date files:', per_date_files)
    print('Wrote aggregated diagnostics to', os.path.join(OUTDIR,'top_combo_aggregate_errors.csv'))
