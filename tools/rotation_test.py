"""Quick rotation model A/B test for a single date.

Produces two projection outputs (rotation off / on) and a per-player diff CSV.
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simple import SimpleNBAProjection
from nba_api.stats.endpoints import playergamelogs, scoreboardv2, boxscoretraditionalv2

DATE = os.getenv('TEST_DATE', '2025-12-05')
OUTDIR = 'tools/diagnostics'
os.makedirs(OUTDIR, exist_ok=True)


def fetch_recent_game_logs_for_date(m, d):
    from datetime import datetime as _dt, timedelta as _td
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
    # ensure columns exist
    needed = ['PLAYER_ID','PLAYER_NAME','MIN','PTS','REB','AST','STL','BLK','TOV','FG3M','GAME_DATE']
    for c in needed:
        if c not in logs_df.columns:
            logs_df[c] = 0
    # normalize names
    logs_df['PLAYER_NAME'] = logs_df['PLAYER_NAME'].str.strip()
    logs_df['PLAYER_NAME_L'] = logs_df['PLAYER_NAME'].str.lower().str.strip()
    return logs_df


def fetch_matchups_for_date(m, d):
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


def run_ab_test(date_str):
    print('Running rotation A/B for', date_str)
    base = SimpleNBAProjection(dk_salaries_path='DKSalaries.csv')
    try:
        base.load_dk_salaries()
    except Exception:
        pass

    # build synthetic dk_df if missing
    if base.dk_df is None:
        base.dk_df = pd.DataFrame(columns=['Name','Salary','Team','Position'])

    # common overrides
    def set_overrides(m):
        m.fetch_todays_matchups = lambda: fetch_matchups_for_date(m, date_str)
        m.fetch_recent_game_logs = lambda: fetch_recent_game_logs_for_date(m, date_str)
        # use modest Monte Carlo for speed
        m.USE_ROTATION_MODEL = False
        m.ROTATION_BLEND = 0.6
        m.ROTATION_MIN_GAMES = 3
        m.ROTATION_TEAM_MINUTES = 240.0
        # match previous tuning
        m.BACKUP_DELTA_VALUE_THRESHOLD = 0.8
        m.BOOST_MINUTES_SCALE = float(os.getenv('BOOST_MINUTES_SCALE', '1.0'))

    # run without rotation
    m0 = SimpleNBAProjection(dk_salaries_path='DKSalaries.csv')
    try:
        m0.load_dk_salaries()
    except Exception:
        pass
    set_overrides(m0)
    m0.USE_ROTATION_MODEL = False
    print('-> Running without rotation')
    df0 = m0.run(save_csv=None, n_sims=200)
    out0 = os.path.join(OUTDIR, f'rotation_{date_str}_no_rotation.csv')
    if df0 is not None:
        df0.to_csv(out0, index=False)
        print('Wrote', out0)

    # run with rotation
    m1 = SimpleNBAProjection(dk_salaries_path='DKSalaries.csv')
    try:
        m1.load_dk_salaries()
    except Exception:
        pass
    set_overrides(m1)
    m1.USE_ROTATION_MODEL = True
    m1.ROTATION_BLEND = 0.6
    print('-> Running with rotation')
    df1 = m1.run(save_csv=None, n_sims=200)
    out1 = os.path.join(OUTDIR, f'rotation_{date_str}_rotation.csv')
    if df1 is not None:
        df1.to_csv(out1, index=False)
        print('Wrote', out1)

    # compare projected minutes and projection
    if df0 is None or df1 is None:
        print('One of the runs returned no data; aborting diff')
        return

    cmp = df0[['Name','ProjMin','Projection']].merge(df1[['Name','ProjMin','Projection']], on='Name', how='outer', suffixes=('_no','_rot'))
    cmp['delta_min'] = cmp['ProjMin_rot'] - cmp['ProjMin_no']
    cmp['delta_proj'] = cmp['Projection_rot'] - cmp['Projection_no']
    cmp = cmp.sort_values('delta_min', key=lambda s: s.abs(), ascending=False)
    outdiff = os.path.join(OUTDIR, f'rotation_{date_str}_diff.csv')
    cmp.to_csv(outdiff, index=False)
    print('Wrote diff to', outdiff)
    print('\nTop 10 changes by minutes:')
    print(cmp[['Name','ProjMin_no','ProjMin_rot','delta_min','Projection_no','Projection_rot','delta_proj']].head(10).to_string(index=False))


if __name__ == '__main__':
    run_ab_test(DATE)
