"""
simple_nba_projections_module.py

A compact, reliable NBA DFS projection module (Option C).

Features:
- Loads DraftKings salary file (CSV)
- Pulls recent player game logs (last 30 days) using `nba_api`
- Computes trimmed-mean fantasy points per minute (fp/min)
- Projects minutes using a weighted blend of last-5, last-10, and role expectation
- Applies a single matchup multiplier (pace + opponent defense)
- Caps projections by a realistic salary multiplier
- Outputs a pandas DataFrame and optionally saves CSV

Usage:
- install dependencies: pip install pandas numpy nba_api
- place DraftKings salaries CSV (with columns: "Name", "Salary", "TeamAbbrev", "Position")
- run: python simple_nba_projections_module.py --dk salaries.csv

This module is intentionally small, readable, and easy to extend.
"""

import argparse
import math
import time
from datetime import datetime, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd
import difflib
import re
import random
import unicodedata

# Simple headers for web requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36'
}

# pool of user agents to rotate when scraping
USER_AGENTS = [
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15'
]


def _get_request_headers():
    h = HEADERS.copy()
    h['User-Agent'] = random.choice(USER_AGENTS)
    h['Accept-Language'] = 'en-US,en;q=0.9'
    h['Referer'] = 'https://www.google.com/'
    h['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    return h

# NBA API imports (may be rate-limited; use caching if you run often)
from nba_api.stats.endpoints import playergamelogs, leaguedashteamstats, scoreboardv2
from nba_api.stats.static import players, teams
from injurySrape import get_injuries


class SimpleNBAProjection:
    LEAGUE_AVG_PACE = 100.0  # fallback; will try to read real league pace
    # tunable parameters (can be overridden on instances)
    STARTER_BONUS_SCALE = 6.0
    LAST5_WEIGHT = 0.40
    LAST10_WEIGHT = 0.25
    ROLE_WEIGHT = 0.35
    BACKUP_DELTA_VALUE_THRESHOLD = 1.0

    def __init__(self, dk_salaries_path: str, days_of_history: int = 30, season: str = None):
        self.dk_path = dk_salaries_path
        self.days = days_of_history
        self.season = season or self._guess_season()
        self.dk_df = None
        self.team_map = {t['id']: t['abbreviation'] for t in teams.get_teams()}
        self.team_stats = {}
        self.todays_matchups = {}

    def _guess_season(self) -> str:
        today = datetime.now()
        year = today.year
        if today.month >= 10:
            return f"{year}-{str(year + 1)[-2:]}"
        else:
            return f"{year - 1}-{str(year)[-2:]}"

    def load_dk_salaries(self) -> bool:
        try:
            self.dk_df = pd.read_csv(self.dk_path)
            # Normalize column names - try to handle common variants
            cols = {c.lower(): c for c in self.dk_df.columns}
            # Required: name, salary, team
            name_col = cols.get('name') or cols.get('player') or cols.get('playername')
            salary_col = cols.get('salary') or cols.get('dkpoints')
            team_col = cols.get('team') or cols.get('teamabbrev') or cols.get('team_abbrev')

            if not name_col or not salary_col or not team_col:
                print('DK CSV missing required columns. Found:', list(self.dk_df.columns))
                return False

            # Standardize
            self.dk_df = self.dk_df.rename(columns={name_col: 'Name', salary_col: 'Salary', team_col: 'Team'})
            # Optional: Position
            for c in ['position', 'pos', 'Position']:
                if c in self.dk_df.columns:
                    self.dk_df = self.dk_df.rename(columns={c: 'Position'})
                    break

            # Trim whitespace
            self.dk_df['Name'] = self.dk_df['Name'].astype(str).str.strip()
            self.dk_df['Team'] = self.dk_df['Team'].astype(str).str.strip()
            self.dk_df['Salary'] = pd.to_numeric(self.dk_df['Salary'], errors='coerce')
            self.dk_df = self.dk_df.dropna(subset=['Name', 'Salary'])

            print(f"✅ Loaded {len(self.dk_df)} players from DK salaries")
            return True
        except Exception as e:
            print('Error loading DK salaries:', e)
            return False

    def _safe_api_call(self, func, *args, **kwargs):
        # Simple retry wrapper around nba_api calls
        for attempt in range(3):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                wait = 2 ** attempt
                print(f"API call failed (attempt {attempt+1}): {e} — retrying in {wait}s")
                time.sleep(wait)
        return None

    def fetch_team_stats(self):
        print('📊 Fetching team stats...')
        try:
            res = self._safe_api_call(leaguedashteamstats.LeagueDashTeamStats, season=self.season)
            if res is None:
                return False
            df = res.get_data_frames()[0]
            # Try common column names
            pace_col = 'PACE' if 'PACE' in df.columns else None
            team_col = 'TEAM_ABBREVIATION' if 'TEAM_ABBREVIATION' in df.columns else 'TEAM_NAME' if 'TEAM_NAME' in df.columns else None
            if not team_col:
                return False
            for _, row in df.iterrows():
                abbr = row[team_col]
                pace = row[pace_col] if pace_col else self.LEAGUE_AVG_PACE
                self.team_stats[abbr] = {
                    'pace': pace,
                    # DEF_RATING or DRTG might exist; fall back to 110
                    'def_rating': row['DEF_RATING'] if 'DEF_RATING' in df.columns else row['DRTG'] if 'DRTG' in df.columns else 110.0
                }
            # set league average pace
            if pace_col:
                self.LEAGUE_AVG_PACE = df[pace_col].mean()
            print(f"✅ Loaded team stats for {len(self.team_stats)} teams; league pace {self.LEAGUE_AVG_PACE:.1f}")
            return True
        except Exception as e:
            print('Error fetching team stats:', e)
            return False

    def fetch_todays_matchups(self):
        print('📅 Fetching today\'s matchups...')
        try:
            board = self._safe_api_call(scoreboardv2.ScoreboardV2)
            if board is None:
                return False
            games_df = board.get_data_frames()[0]
            for _, g in games_df.iterrows():
                home_id = g['HOME_TEAM_ID']
                away_id = g['VISITOR_TEAM_ID']
                home_abbr = self.team_map.get(home_id)
                away_abbr = self.team_map.get(away_id)
                if home_abbr and away_abbr:
                    key = f"{away_abbr}@{home_abbr}"
                    # store paces and opponent defensive rating
                    self.todays_matchups[away_abbr] = {'opponent': home_abbr}
                    self.todays_matchups[home_abbr] = {'opponent': away_abbr}
            print(f"✅ Found {len(self.todays_matchups)} teams in today's matchups")
            return True
        except Exception as e:
            print('Error fetching scoreboard:', e)
            return False

    def fetch_recent_game_logs(self) -> pd.DataFrame:
        """Fetch recent game logs for the season and date range, then filter locally to players in DK slate."""
        print('📥 Fetching recent game logs (this may take a moment)...')
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days)
        date_from = start_date.strftime('%Y-%m-%d')
        date_to = end_date.strftime('%Y-%m-%d')

        res = self._safe_api_call(playergamelogs.PlayerGameLogs, season_nullable=self.season,
                                  date_from_nullable=date_from, date_to_nullable=date_to)
        if res is None:
            return pd.DataFrame()
        logs_df = res.get_data_frames()[0]
        if logs_df.empty:
            return logs_df

        # Ensure columns we need exist
        needed = ['PLAYER_ID', 'PLAYER_NAME', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M', 'GAME_DATE']
        for c in needed:
            if c not in logs_df.columns:
                logs_df[c] = 0

        # Filter to players in DK slate
        dk_names = set(self.dk_df['Name'].str.lower())
        logs_df['PLAYER_NAME_L'] = logs_df['PLAYER_NAME'].str.lower().str.strip()
        logs_df = logs_df[logs_df['PLAYER_NAME_L'].isin(dk_names)].copy()
        print(f"✅ Retrieved {len(logs_df)} relevant recent game logs")
        return logs_df

    @staticmethod
    def calculate_dk_points_from_row(row: pd.Series) -> float:
        # DK scoring: PTS + 1.2*REB + 1.5*AST + 3*STL + 3*BLK - 1*TOV + 0.5*FG3M
        return (
            row.get('PTS', 0) +
            1.2 * row.get('REB', 0) +
            1.5 * row.get('AST', 0) +
            3.0 * row.get('STL', 0) +
            3.0 * row.get('BLK', 0) -
            1.0 * row.get('TOV', 0) +
            0.5 * row.get('FG3M', 0)
        )

    @staticmethod
    def trimmed_mean(values: List[float], trim_fraction: float = 0.1) -> float:
        if not values:
            return 0.0
        arr = np.array([v for v in values if not (pd.isna(v) or not np.isfinite(v))])
        if len(arr) == 0:
            return 0.0
        k = int(len(arr) * trim_fraction)
        if k == 0:
            return float(arr.mean())
        arr_sorted = np.sort(arr)
        trimmed = arr_sorted[k:len(arr_sorted)-k]
        return float(trimmed.mean()) if len(trimmed) > 0 else float(arr.mean())

    def monte_carlo_floor_ceiling(self, fp_min_list: List[float], min_list: List[float], matchup_mult: float, vegas_mult: float, n_sims: int = 2000) -> Dict[str, float]:
        """Run a Monte Carlo sim to estimate floor (20th percentile) and ceiling (90th percentile).
        Uses empirical distributions when available, falls back to normal approximations.
        Returns dict with keys: 'floor', 'ceiling', 'volatility_std'
        """
        # Clean inputs
        fp_vals = np.array([v for v in fp_min_list if not (pd.isna(v) or not np.isfinite(v) or v <= 0)])
        min_vals = np.array([m for m in min_list if not (pd.isna(m) or not np.isfinite(m) or m <= 0)])

        if len(fp_vals) == 0 or len(min_vals) == 0:
            return {'floor': 0.0, 'ceiling': 0.0, 'volatility_std': 0.0}

        # Use empirical sampling if we have enough samples
        use_empirical = len(fp_vals) >= 6 and len(min_vals) >= 6

        sims = []
        rng = np.random.default_rng(seed=42)

        if use_empirical:
            for _ in range(n_sims):
                fp_per_min = rng.choice(fp_vals)
                minutes = rng.choice(min_vals)
                sim_fp = fp_per_min * minutes * matchup_mult * vegas_mult
                sims.append(sim_fp)
        else:
            # use normal approximation with truncation at zero
            fp_mu, fp_sigma = float(np.mean(fp_vals)), float(np.std(fp_vals, ddof=1) if len(fp_vals)>1 else max(0.01, float(np.mean(fp_vals))*0.2))
            min_mu, min_sigma = float(np.mean(min_vals)), float(np.std(min_vals, ddof=1) if len(min_vals)>1 else max(0.5, float(np.mean(min_vals))*0.15))

            for _ in range(n_sims):
                fp_per_min = max(0.001, rng.normal(fp_mu, fp_sigma))
                minutes = max(1.0, rng.normal(min_mu, min_sigma))
                sim_fp = fp_per_min * minutes * matchup_mult * vegas_mult
                sims.append(sim_fp)

        sims = np.array(sims)
        # Compute percentiles: floor ~ 20th, ceiling ~ 90th
        floor = float(np.percentile(sims, 20))
        ceiling = float(np.percentile(sims, 90))
        volatility_std = float(np.std(sims, ddof=1))

        return {'floor': floor, 'ceiling': ceiling, 'volatility_std': volatility_std}

    def compute_fp_per_min(self, logs_df: pd.DataFrame) -> Dict[int, Dict]:
        """Return mapping player_id -> {'fp_per_min':..., 'recent_minutes':..., 'last5_min':..., 'last10_min':..., 'games': n}"""
        out = {}
        grouped = logs_df.sort_values('GAME_DATE', ascending=False).groupby('PLAYER_ID')
        for pid, g in grouped:
            # convert MIN to floats (handle "35:21" format if present)
            mins = []
            fps = []
            for _, row in g.iterrows():
                raw_min = row['MIN']
                try:
                    if isinstance(raw_min, str) and ':' in raw_min:
                        mm, ss = raw_min.split(':')
                        m = float(mm) + float(ss) / 60.0
                    else:
                        m = float(raw_min)
                except Exception:
                    m = 0.0
                mins.append(m)
                fps.append(self.calculate_dk_points_from_row(row))
            if sum(mins) == 0:
                continue
            # compute per-game fp/min, skip games with 0 minutes
            fp_min_list = [fp / m for fp, m in zip(fps, mins) if m > 2]
            if not fp_min_list:
                continue
            fp_per_min = self.trimmed_mean(fp_min_list, trim_fraction=0.2)
            recent_minutes = float(np.median(mins[:10]))
            last5_min = float(np.mean(mins[:5])) if len(mins) >= 1 else recent_minutes
            last10_min = float(np.mean(mins[:10])) if len(mins) >= 1 else recent_minutes
            out[int(pid)] = {
                'fp_per_min': fp_per_min,
                'recent_minutes': recent_minutes,
                'last5_min': last5_min,
                'last10_min': last10_min,
                'games': len(mins),
                'fp_min_list': fp_min_list,
                'min_list': mins
            }
        print(f"✅ Computed FP/min for {len(out)} players")
        return out

    def get_player_starter_pct(self, player_id: int) -> float:
        """Estimate fraction of recent games where the player was a starter.

        Heuristic: count games in self.last_logs where minutes >= 24 OR a 'START_POSITION'
        or similar flag exists. Returns value in [0,1]. If logs are missing, returns 0.5.
        """
        try:
            logs = getattr(self, 'last_logs', None)
            if logs is None or logs.empty:
                return 0.5
            p_logs = logs[logs['PLAYER_ID'] == int(player_id)]
            if p_logs.empty:
                return 0.5
            starter_count = 0
            total = 0
            for _, r in p_logs.iterrows():
                total += 1
                # prefer explicit starter flag if present
                if 'START_POSITION' in r and pd.notna(r.get('START_POSITION')) and str(r.get('START_POSITION')).strip():
                    starter_count += 1
                    continue
                # otherwise use minutes threshold
                try:
                    raw_min = r.get('MIN')
                    if isinstance(raw_min, str) and ':' in raw_min:
                        mm, ss = raw_min.split(':')
                        m = float(mm) + float(ss) / 60.0
                    else:
                        m = float(raw_min)
                except Exception:
                    m = 0.0
                if m >= 24.0:
                    starter_count += 1
            return float(starter_count) / max(1.0, float(total))
        except Exception:
            return 0.5


    def project_minutes(self, player_stats: Dict, player_id: int = None) -> float:
        """Weighted minutes projection from last5, last10, and role expectation.

        Adds a small starter-biased adjustment using recent logs (if available).
        """
        last5 = float(player_stats.get('last5_min', 0) or 0)
        last10 = float(player_stats.get('last10_min', 0) or 0)
        recent = float(player_stats.get('recent_minutes', 0) or 0)
        # role expectation fallback = recent (could be improved with depth chart data)
        role = recent
        projected_base = last5 * self.LAST5_WEIGHT + last10 * self.LAST10_WEIGHT + role * self.ROLE_WEIGHT

        # starter adjustment: add a small bonus proportional to starter frequency
        starter_pct = 0.5
        if player_id is not None:
            try:
                starter_pct = self.get_player_starter_pct(player_id)
            except Exception:
                starter_pct = 0.5

        # translate starter_pct into minutes bonus: range ~ [-1.5, +4.0]
        starter_bonus = (starter_pct - 0.5) * float(getattr(self, 'STARTER_BONUS_SCALE', 6.0))
        projected_base += starter_bonus

        # Optional rotation-based minutes estimator
        use_rotation = bool(getattr(self, 'USE_ROTATION_MODEL', False))
        if use_rotation:
            try:
                team = None
                # attempt to find team from last_logs mapping if player_id present
                if player_id is not None and hasattr(self, 'last_logs') and self.last_logs is not None:
                    plogs = self.last_logs[self.last_logs['PLAYER_ID'] == int(player_id)]
                    if not plogs.empty and 'TEAM_ABBREVIATION' in plogs.columns:
                        team = plogs.iloc[0].get('TEAM_ABBREVIATION')
                rotation_expected = self.estimate_rotation_minutes(player_stats, player_id, team)
            except Exception:
                rotation_expected = projected_base
            blend = float(getattr(self, 'ROTATION_BLEND', 0.6))
            projected = blend * float(rotation_expected) + (1.0 - blend) * float(projected_base)
        else:
            projected = projected_base

        # bounds
        projected = max(5.0, projected)
        projected = min(48.0, projected)
        return float(projected)

    def _parse_min_str(self, minval) -> float:
        """Parse minute value which may be string like '34:12' or numeric."""
        try:
            if isinstance(minval, str) and ':' in minval:
                mm, ss = minval.split(':')
                return float(mm) + float(ss) / 60.0
            return float(minval)
        except Exception:
            return 0.0

    def estimate_rotation_minutes(self, player_stats: Dict, player_id: int = None, team_abbr: str = None) -> float:
        """Estimate minutes using a simple rotation model based on recent logs for the team.

        Approach:
        - Compute average minutes per player over recent logs for the same team.
        - Allocate team game minutes (~240) proportionally to recent averages.
        - Apply a starter_pct multiplier to slightly boost starters.
        - Return expected minutes for the player.
        """
        # parameters
        TEAM_GAME_MINUTES = float(getattr(self, 'ROTATION_TEAM_MINUTES', 240.0))
        MIN_GAMES_REQUIRED = int(getattr(self, 'ROTATION_MIN_GAMES', 3))

        # fallback to simple recent minutes if logs unavailable
        player_avg = float(player_stats.get('recent_minutes', 0) or 0)
        if not hasattr(self, 'last_logs') or self.last_logs is None or self.last_logs.empty:
            return player_avg

        logs = self.last_logs
        # try to infer team if missing
        if not team_abbr:
            if player_id is not None:
                plogs = logs[logs['PLAYER_ID'] == int(player_id)]
                if not plogs.empty and 'TEAM_ABBREVIATION' in plogs.columns:
                    team_abbr = plogs.iloc[0].get('TEAM_ABBREVIATION')

        if not team_abbr:
            return player_avg

        team_logs = logs[logs.get('TEAM_ABBREVIATION', '') == team_abbr]
        if team_logs is None or team_logs.empty:
            # some logs use 'TEAM' instead
            team_logs = logs[logs.get('TEAM', '') == team_abbr]
            if team_logs is None or team_logs.empty:
                return player_avg

        # compute average minutes per player over the window
        grp = team_logs.groupby('PLAYER_NAME').agg({'MIN': lambda s: float(s.map(self._parse_min_str)).mean() if not s.empty else 0.0})
        if grp.empty:
            return player_avg
        # require minimum games check: we use counts per player
        counts = team_logs.groupby('PLAYER_NAME').size()
        grp = grp.join(counts.rename('games'))
        # filter players with at least MIN_GAMES_REQUIRED
        grp = grp[grp['games'] >= MIN_GAMES_REQUIRED]
        if grp.empty:
            # relax filter
            grp = team_logs.groupby('PLAYER_NAME').agg({'MIN': lambda s: float(s.map(self._parse_min_str)).mean() if not s.empty else 0.0})

        # compute proportional allocation
        grp = grp.rename(columns={'MIN': 'avg_min'})
        total_avg = grp['avg_min'].sum()
        if total_avg <= 1e-6:
            return player_avg

        # find player's avg
        player_name = None
        if player_id is not None:
            plogs = team_logs[team_logs['PLAYER_ID'] == int(player_id)] if 'PLAYER_ID' in team_logs.columns else team_logs
            if not plogs.empty:
                player_name = plogs.iloc[0].get('PLAYER_NAME')
        if not player_name:
            # try to match by normalized name from player_stats if provided
            player_name = player_stats.get('name') if isinstance(player_stats.get('name'), str) else None

        player_avg_min = None
        if player_name and player_name in grp.index:
            player_avg_min = float(grp.loc[player_name, 'avg_min'])
        else:
            # try case-insensitive match
            for idx in grp.index:
                if player_name and idx.lower().startswith(player_name.lower()[:6]):
                    player_avg_min = float(grp.loc[idx, 'avg_min']); break

        if player_avg_min is None:
            # fallback to recent minutes
            player_avg_min = player_avg

        rotation_expected = TEAM_GAME_MINUTES * (player_avg_min / total_avg)

        # starter adjustment
        starter_pct = 0.5
        if player_id is not None:
            try:
                starter_pct = self.get_player_starter_pct(player_id)
            except Exception:
                starter_pct = 0.5
        # boost factor in [0.9, 1.15]
        boost = 0.9 + 0.5 * starter_pct
        rotation_expected = rotation_expected * boost

        # clamp
        rotation_expected = max(1.0, min(48.0, rotation_expected))
        return float(rotation_expected)

    def matchup_multiplier(self, team_abbr: str) -> float:
        # Use team and opponent pace and opponent def rating. Keep multiplier near 1.0 (±10%).
        opp = self.todays_matchups.get(team_abbr, {}).get('opponent')
        team_pace = self.team_stats.get(team_abbr, {}).get('pace', self.LEAGUE_AVG_PACE)
        opp_pace = self.team_stats.get(opp, {}).get('pace', self.LEAGUE_AVG_PACE) if opp else self.LEAGUE_AVG_PACE
        opp_def = self.team_stats.get(opp, {}).get('def_rating', 110.0) if opp else 110.0

        pace_adj = ((team_pace + opp_pace) / 2.0) / self.LEAGUE_AVG_PACE
        def_adj = 110.0 / float(opp_def)
        # blend with limited influence
        multiplier = 0.5 * pace_adj + 0.5 * def_adj
        # clamp to [0.85, 1.15]
        multiplier = max(0.85, min(1.15, multiplier))
        return multiplier

    def cap_projection_by_salary(self, projection: float, salary: float) -> float:
        # Realistic cap: $1k -> 6.8x baseline
        cap = salary * 0.0068
        return min(projection, cap)

    def calculate_backup_boosts(self) -> dict:
        """Calculate backup minute boosts when starters are OUT.

        Returns a mapping of player_name_lower -> minutes_boost (float).
        """
        backup_boosts = {}

        injuries = getattr(self, 'injuries', {}) or {}
        if not injuries:
            return backup_boosts

        # Find OUT players (expect injuries keyed by lowercase name -> status)
        out_players = [name for name, status in injuries.items() if str(status).upper() == 'OUT']
        if not out_players:
            return backup_boosts

        # Group DK players by (team, position)
        team_positions = {}
        for _, row in self.dk_df.iterrows():
            name = str(row.get('Name', '')).strip()
            team = str(row.get('Team', '')).strip().upper()
            pos = str(row.get('Position', '')).strip()
            if not team or not pos:
                continue
            positions = [p.strip() for p in pos.split('/')] if '/' in pos else [pos]
            for position in positions:
                key = (team, position)
                try:
                    salary = float(row.get('Salary', 0))
                except Exception:
                    salary = 0.0
                team_positions.setdefault(key, []).append({'name': name.lower(), 'salary': salary})

        # Compute boosts per OUT starter using direct-backup logic if logs are available
        for out_name in out_players:
            # identify direct backup using recent logs and DK roster
            out_row = None
            for _, r in self.dk_df.iterrows():
                if str(r.get('Name', '')).strip().lower() == out_name:
                    out_row = r
                    break
            if out_row is None:
                continue

            out_team = str(out_row.get('Team', '')).strip().upper()
            injury_status = str(out_row.get('Status', '')).strip() if 'Status' in out_row else 'OUT'

            backup = None
            try:
                backup = self.identify_direct_backup(out_team, out_name, injury_status)
            except Exception:
                backup = None

            # Require a direct backup; if none found, do not assign boosts (no fallback heuristic)
            if not backup:
                continue

            # backup may be full row dict or name string
            backup_name = backup if isinstance(backup, str) else (backup.get('name') if isinstance(backup, dict) else None)
            if not backup_name:
                continue

            minutes_boost = self.calculate_minutes_boost(backup_name, out_name, 'OUT')
            # allow external scaling of calculated minutes (set m.BOOST_MINUTES_SCALE on instance)
            scale = float(getattr(self, 'BOOST_MINUTES_SCALE', 1.0))
            minutes_boost = minutes_boost * scale

            if minutes_boost > 0:
                # compute expected fantasy point gain and value uplift to ensure this is a value play
                backup_fp_per_min = self.get_fp_per_min_by_name(backup_name)
                try:
                    # find salary from DK DF
                    sal_row = self.dk_df[self.dk_df['Name'].str.lower().str.strip() == backup_name.lower().strip()]
                    salary = float(sal_row.iloc[0]['Salary']) if not sal_row.empty else 1.0
                except Exception:
                    salary = 1.0

                # matchup and vegas multipliers
                matchup_mult = self.matchup_multiplier(out_team) if out_team else 1.0
                vegas_mult = vegas_multiplier(out_team, getattr(self, 'vegas', {})) if out_team else 1.0

                projected_fp_gain = minutes_boost * backup_fp_per_min * matchup_mult * vegas_mult
                delta_value = (projected_fp_gain / max(1.0, salary)) * 1000.0

                # threshold: only keep boosts that improve value by at least 1.0 FP per $1k
                if delta_value >= 1.0:
                    backup_boosts[backup_name] = backup_boosts.get(backup_name, 0.0) + minutes_boost
                    print(f"🔔 Backup boost: {backup_name} +{minutes_boost:.1f} min → +{projected_fp_gain:.2f} FP (Δvalue {delta_value:.2f} per $1k)")
                else:
                    print(f"✖️ Skipping small boost for {backup_name}: +{minutes_boost:.1f} min → +{projected_fp_gain:.2f} FP (Δvalue {delta_value:.2f} per $1k)")

        return backup_boosts

    def normalize_name(self, name: str) -> str:
        """Normalize player name: lower, remove punctuation, suffixes, and extra spaces."""
        if not name:
            return ""
        s = str(name)
        # Unicode normalize (decompose diacritics), then strip combining marks
        try:
            s = unicodedata.normalize('NFKD', s)
            s = ''.join(ch for ch in s if not unicodedata.combining(ch))
        except Exception:
            # fallback: leave as-is
            s = str(name)

        # standardize apostrophes/hyphens to simple ascii
        s = s.replace('’', "'").replace('`', "'").replace('–', '-').replace('—', '-')

        # remove common suffixes (JR, SR, II, III, IV, V) and ordinal dots
        s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s, flags=re.IGNORECASE)

        # replace hyphens with space and remove punctuation except alnum and space
        s = s.replace('-', ' ')
        s = re.sub(r"[^A-Za-z0-9\s]", ' ', s)

        # collapse spaces and lowercase
        s = re.sub(r"\s+", ' ', s).strip().lower()

        return s

    def build_dk_name_index(self):
        """Build indexes from DK names to support matching scraped injury names."""
        if getattr(self, '_dk_name_index', None) is not None:
            return
        self._dk_name_index = {}
        self._dk_first_last = {}
        self._dk_normalized_list = []
        for _, r in self.dk_df.iterrows():
            name = str(r.get('Name', '')).strip()
            norm = self.normalize_name(name)
            if not norm:
                continue
            self._dk_name_index[norm] = name
            self._dk_normalized_list.append(norm)
            parts = norm.split()
            if len(parts) >= 2:
                key = parts[0] + ' ' + parts[-1]
                self._dk_first_last.setdefault(key, []).append(name)

    def match_injury_name(self, inj_name: str) -> str:
        """Try to match an injury-scraper name to a DK name. Returns matched DK name or None."""
        if not inj_name:
            return None
        self.build_dk_name_index()
        inj_norm = self.normalize_name(inj_name)
        # exact normalized match
        if inj_norm in self._dk_name_index:
            return self._dk_name_index[inj_norm]
        # try first+last match
        parts = inj_norm.split()
        if len(parts) >= 2:
            key = parts[0] + ' ' + parts[-1]
            if key in self._dk_first_last:
                return self._dk_first_last[key][0]
        # fuzzy close match on normalized list
        close = difflib.get_close_matches(inj_norm, self._dk_normalized_list, n=1, cutoff=0.8)
        if close:
            return self._dk_name_index.get(close[0])
        # try substring match of last name
        last = parts[-1] if parts else inj_norm
        for norm in self._dk_normalized_list:
            if last in norm.split():
                return self._dk_name_index.get(norm)
        return None

    def get_player_positions(self, player_name: str) -> list:
        """Return list of positions for a player from DK data (normalized strings)."""
        name_l = player_name.lower().strip()
        for _, r in self.dk_df.iterrows():
            if str(r.get('Name', '')).strip().lower() == name_l:
                pos = str(r.get('Position', '')).strip()
                if not pos:
                    return []
                return [p.strip() for p in pos.split('/')] if '/' in pos else [pos]
        return []

    def get_player_typical_minutes(self, player_name: str) -> float:
        """Estimate typical minutes from recent logs or last_player_fpmin mapping."""
        try:
            # check last_player_fpmin mapping
            pf = getattr(self, 'last_player_fpmin', {})
            # pf keys are player ids; we need to find by name via logs
            logs = getattr(self, 'last_logs', pd.DataFrame())
            if not logs.empty:
                l = logs[logs['PLAYER_NAME'].str.lower().str.strip() == player_name.lower().strip()]
                if not l.empty:
                    # parse minutes column (handle mm:ss)
                    mins = []
                    for _, row in l.iterrows():
                        raw = row.get('MIN', 0)
                        try:
                            if isinstance(raw, str) and ':' in raw:
                                mm, ss = raw.split(':')
                                m = float(mm) + float(ss) / 60.0
                            else:
                                m = float(raw)
                        except Exception:
                            m = 0.0
                        mins.append(m)
                    if mins:
                        return float(sum(mins) / len(mins))

            # fallback: search in last_player_fpmin by matching names via logs index
            if hasattr(self, 'last_logs') and not self.last_logs.empty:
                # attempt to map player name to id
                df = self.last_logs
                matches = df[df['PLAYER_NAME'].str.lower().str.contains(player_name.lower().split()[0])]
                if not matches.empty:
                    mins = []
                    for _, row in matches.iterrows():
                        raw = row.get('MIN', 0)
                        try:
                            if isinstance(raw, str) and ':' in raw:
                                mm, ss = raw.split(':')
                                m = float(mm) + float(ss) / 60.0
                            else:
                                m = float(raw)
                        except Exception:
                            m = 0.0
                        mins.append(m)
                    if mins:
                        return float(sum(mins) / len(mins))

            # final fallback: conservative default
            return 20.0
        except Exception:
            return 20.0

    def get_fp_per_min_by_name(self, player_name: str) -> float:
        """Return estimated fantasy points per minute for player using recent logs mapping."""
        try:
            logs = getattr(self, 'last_logs', pd.DataFrame())
            if not logs.empty:
                l = logs[logs['PLAYER_NAME'].str.lower().str.strip() == player_name.lower().strip()]
                if not l.empty:
                    fps = []
                    mins = []
                    for _, row in l.iterrows():
                        raw = row.get('MIN', 0)
                        try:
                            if isinstance(raw, str) and ':' in raw:
                                mm, ss = raw.split(':')
                                m = float(mm) + float(ss)/60.0
                            else:
                                m = float(raw)
                        except Exception:
                            m = 0.0
                        mins.append(m)
                        fps.append(self.calculate_dk_points_from_row(row))
                    fp_per_min_list = [fp/m for fp,m in zip(fps, mins) if m > 0]
                    if fp_per_min_list:
                        return float(np.mean(fp_per_min_list))

            # fallback: try last_player_fpmin mapping by matching name via logs
            pf = getattr(self, 'last_player_fpmin', {})
            if hasattr(self, 'last_logs') and not self.last_logs.empty:
                # find player id(s) for this name
                df = self.last_logs
                matches = df[df['PLAYER_NAME'].str.lower().str.contains(player_name.lower().split()[0])]
                if not matches.empty:
                    pids = matches['PLAYER_ID'].unique().tolist()
                    for pid in pids:
                        info = pf.get(int(pid))
                        if info and info.get('fp_per_min'):
                            return float(info.get('fp_per_min'))

            # conservative fallback
            return 0.25
        except Exception:
            return 0.25
 

    def identify_direct_backup(self, team: str, injured_player: str, injury_status: str):
        """Identify the most likely direct backup using DK roster + recent logs.

        Returns backup player name (lowercase) or None.
        """
        try:
            injured_positions = self.get_player_positions(injured_player)
            if not injured_positions:
                return None

            # candidates: DK players on same team not the injured player
            candidates = []
            for _, r in self.dk_df.iterrows():
                name = str(r.get('Name', '')).strip()
                if not name or name.lower() == injured_player.lower():
                    continue
                team_r = str(r.get('Team', '')).strip().upper()
                if team_r != (team or '').upper():
                    continue
                positions = [p.strip() for p in str(r.get('Position', '')).split('/')] if '/' in str(r.get('Position', '')) else [str(r.get('Position', '')).strip()]
                # position match quality: 1 for exact, 0.7 for group match, 0 otherwise
                pos_match = 0.0
                for pos in injured_positions:
                    if pos in positions:
                        pos_match = 1.0
                        break
                if pos_match == 0.0:
                    # group match: G vs F vs C
                    def group(p):
                        if p in ['PG', 'SG']:
                            return 'G'
                        if p in ['SF', 'PF']:
                            return 'F'
                        return p
                    injured_groups = set(group(p) for p in injured_positions)
                    backup_groups = set(group(p) for p in positions)
                    if injured_groups & backup_groups:
                        pos_match = 0.7

                # estimate recent minutes and fp_per_min
                recent_min = self.get_player_typical_minutes(name)
                # opportunity score based on recent minutes and salary (higher salary tends to be primary backup)
                try:
                    sal = float(r.get('Salary', 0))
                except Exception:
                    sal = 0.0
                score = pos_match * 0.6 + min(1.0, recent_min / 30.0) * 0.3 + min(1.0, sal / 8000.0) * 0.1
                candidates.append({'name': name, 'pos_match': pos_match, 'recent_min': recent_min, 'salary': sal, 'score': score})

            if not candidates:
                return None

            candidates = [c for c in candidates if c['pos_match'] > 0]
            if not candidates:
                # pick highest recent_min as fallback
                candidates = sorted(candidates, key=lambda x: x['recent_min'], reverse=True)
                return candidates[0]['name'].lower() if candidates else None

            candidates.sort(key=lambda x: x['score'], reverse=True)
            return candidates[0]['name'].lower()
        except Exception:
            return None

    def calculate_minutes_boost(self, backup_name: str, injured_player: str, injury_status: str) -> float:
        """Calculate minutes boost for a backup given an injured starter.

        Logic mirrors the approach in real_lineup_builderV10: larger boosts for starters with high minutes.
        """
        try:
            injured_minutes = self.get_player_typical_minutes(injured_player)
            if injured_minutes < 10:
                return 0.0

            backup_minutes = self.get_player_typical_minutes(backup_name)

            if str(injury_status).upper() == 'OUT':
                # Reduced default caps to avoid over-boosting backups for large starters.
                # Allow instance override via BOOST_CAP_* attributes.
                high_cap = float(getattr(self, 'BOOST_CAP_HIGH', 8.0))
                high_ceiling = float(getattr(self, 'BOOST_CEILING_HIGH', 34.0))
                med_cap = float(getattr(self, 'BOOST_CAP_MED', 6.0))
                med_ceiling = float(getattr(self, 'BOOST_CEILING_MED', 28.0))
                low_cap = float(getattr(self, 'BOOST_CAP_LOW', 4.0))
                low_ceiling = float(getattr(self, 'BOOST_CEILING_LOW', 24.0))

                if injured_minutes >= 30:
                    minutes_boost = min(high_cap, high_ceiling - backup_minutes)
                elif injured_minutes >= 20:
                    minutes_boost = min(med_cap, med_ceiling - backup_minutes)
                else:
                    minutes_boost = min(low_cap, low_ceiling - backup_minutes)
            else:
                non_out_cap = float(getattr(self, 'BOOST_CAP_NON_OUT', 3.0))
                non_out_ceiling = float(getattr(self, 'BOOST_CEILING_NON_OUT', 28.0))
                minutes_boost = min(non_out_cap, non_out_ceiling - backup_minutes)

            return max(0.0, minutes_boost)
        except Exception:
            return 0.0

    def run(self, save_csv: str = None, injuries: dict = None, n_sims: int = 1500, vegas: dict = None) -> pd.DataFrame:
        # Preconditions
        if self.dk_df is None:
            raise RuntimeError('DK salaries not loaded')

        # If caller didn't provide injuries, attempt to scrape them automatically
        if injuries is None:
            try:
                scraped = get_injuries()
                if isinstance(scraped, dict):
                    injuries = scraped
                    print(f"🩺 Loaded {len(injuries)} injuries from scraper")
                else:
                    # defensive: if scraper returns DataFrame or other, try to convert
                    try:
                        injuries = dict(scraped)
                        print(f"🩺 Loaded {len(injuries)} injuries from scraper (converted)")
                    except Exception:
                        injuries = {}
                        print("⚠️ Injury scraper returned unexpected format; continuing without injuries")
            except Exception as e:
                print(f"⚠️ Injury scrape failed: {e}")
                injuries = {}
        else:
            injuries = injuries or {}

        # Normalize injury keys to lowercase for consistent lookup
        inj = {k.lower(): v for k, v in injuries.items()}

        # Attempt to map scraped injury names to DK names using fuzzy/normalized matching
        mapped = {}
        for orig_name, status in injuries.items():
            dk_match = self.match_injury_name(orig_name)
            if dk_match:
                mapped[dk_match.lower()] = status
            else:
                # also try matching on the lowercase scraped key directly
                if orig_name.lower() in [n.lower() for n in self.dk_df['Name'].astype(str).tolist()]:
                    mapped[orig_name.lower()] = status

        if mapped:
            print(f"🔗 Mapped {len(mapped)}/{len(injuries)} scraped injury names to DK names")
            inj = mapped

        # store normalized injuries on the object so helper methods can access them
        self.injuries = inj

        # Calculate backup boosts from current injuries (if any)
        backup_boost = self.calculate_backup_boosts()

        # Fetch team stats and today's matchups
        
        # --- ALWAYS APPLY VEGAS ---
        # Ensure we have team stats available (used for fallback vegas totals)
        self.fetch_team_stats()

        if vegas is None:
            vegas = scrape_vegas_odds()  # attempt to load

        # normalize vegas dict keys to NBA abbreviations (LAL, BOS, etc.)
        vegas = _normalize_vegas_keys(vegas)

        # If scraper returned nothing, build conservative defaults from team_stats
        if not vegas:
            print("⚠️ Scraped Vegas odds empty — building default vegas from team stats")
            vegas = {}
            for abbr, stats in self.team_stats.items():
                pace = stats.get('pace', self.LEAGUE_AVG_PACE)
                # scale default total by pace relative to league
                total = float(max(90.0, min(260.0, VEGAS_DEFAULT_TOTAL * (pace / max(1.0, self.LEAGUE_AVG_PACE)))))
                vegas[abbr] = {'total': total, 'spread': 0.0}

        print(f"🏀 Vegas odds applied to projections ({len(vegas)} teams)")

        self.fetch_todays_matchups()

        # fetch logs and compute per-minute stats before calculating backup boosts
        logs = self.fetch_recent_game_logs()
        if logs.empty:
            print('No logs available — aborting')
            return pd.DataFrame()

        # store recent logs and fp/min mapping on the object for helper methods
        self.last_logs = logs
        player_fpmin = self.compute_fp_per_min(logs)
        self.last_player_fpmin = player_fpmin

        # store vegas on the object for boost/value calculations
        self.vegas = vegas

        # calculate backup boosts now that we have recent game logs
        backup_boost = self.calculate_backup_boosts()

        results = []
        name_to_ids = {}
        for pid, g in logs.groupby('PLAYER_ID'):
            name = g.iloc[0]['PLAYER_NAME'].strip()
            name_to_ids.setdefault(name.lower(), []).append(int(pid))

        for _, row in self.dk_df.iterrows():
            name = row['Name'].strip()
            name_lower = name.lower()
            salary = float(row['Salary'])
            team = row['Team'].strip() if 'Team' in row else None
            pos = row['Position'] if 'Position' in row else None

            pid_list = name_to_ids.get(name.lower(), [])
            if not pid_list:
                key = ' '.join(name.split()[:2]).lower()
                for n, ids in name_to_ids.items():
                    if n.startswith(key):
                        pid_list = ids
                        break

            if not pid_list:
                continue

            pid = pid_list[0]
            stats = player_fpmin.get(pid)
            if not stats or stats.get('games', 0) < 3:
                continue

            fpmin = stats['fp_per_min']
            base_min = self.project_minutes(stats, pid)

            # Injury handling
            status = inj.get(name.lower())
            if status:
                status_u = status.upper()
                if status_u == 'OUT':
                    continue
                if status_u in ['DAY-TO-DAY', 'DTD']:
                    status_u = 'QUESTIONABLE'
                base_min = apply_injury_minutes_adjustment(base_min, status_u)

            # minimum 1 min for questionable players
            if base_min <= 0:
                continue

            # apply backup boosts (minutes) if calculated
            boost_min = backup_boost.get(name_lower, 0)
            if boost_min:
                base_min += boost_min
                # keep reasonable bounds
                base_min = min(base_min, 48.0)

            matchup_mult = self.matchup_multiplier(team) if team else 1.0
            vegas_mult = vegas_multiplier(team, vegas) if team else 1.0

            # combined multiplier used for point projection
            mult = matchup_mult * vegas_mult

            raw_projection = fpmin * base_min * mult
            capped = self.cap_projection_by_salary(raw_projection, salary)

            # Monte Carlo-based floor & ceiling
            fp_min_list = stats.get('fp_min_list', [])
            min_list = stats.get('min_list', [])
            # use caller-provided n_sims (passed into run) so experiments can control Monte Carlo cost
            mc = self.monte_carlo_floor_ceiling(fp_min_list, min_list, matchup_mult, vegas_mult, n_sims=n_sims)

            results.append({
                'Name': name,
                'PlayerID': pid,
                'Team': team,
                'Position': pos,
                'Salary': salary,
                'Projection': round(capped, 1),
                'RawProjection': round(raw_projection, 1),
                'FP_per_min': round(fpmin, 3),
                'ProjMin': round(base_min, 1),
                'Multiplier': round(mult, 3),
                'InjuryStatus': status or 'None',
                'Games': stats.get('games', 0),
                'BackupBoost': round(backup_boost.get(name_lower, 0), 1),
                'Floor_MC': round(mc['floor'], 1),
                'Ceiling_MC': round(mc['ceiling'], 1),
                'Volatility_STD': round(mc['volatility_std'], 2)
            })

        out_df = pd.DataFrame(results).sort_values('Projection', ascending=False)
        print(f"✅ Projected {len(out_df)} players (with injuries applied)")

        if save_csv:
            out_df.to_csv(save_csv, index=False)
            print(f"Saved projections to {save_csv}")

        # === Add Floor & Ceiling projections ===
        if not out_df.empty:
            # Floor: projection - (volatility factor)
            # Ceiling: projection + (volatility factor)
            # Volatility estimated from recent FP/min std and minutes variance
            out_df['Volatility'] = out_df['FP_per_min'] * 0.15 + (out_df['ProjMin'] * 0.03)
            out_df['Floor'] = (out_df['Projection'] - out_df['Volatility']).clip(lower=0)
            out_df['Ceiling'] = out_df['Projection'] + out_df['Volatility'] * 1.8

        return out_df


# --- Vegas Totals & Spread Integration (Automated Scraper Option C) ---
# This scraper pulls Vegas totals & spreads from Rotowire's NBA odds page.
# Method: HTML scrape (no API key needed).
# Rotowire is stable, lightweight, and fast.

import requests
from bs4 import BeautifulSoup

VEGAS_DEFAULT_TOTAL = 112.0
VEGAS_DEFAULT_SPREAD = 0.0


def scrape_vegas_odds() -> dict:
    # Wrapper: try Oddsshark first, then Action Network, then Oddschecker, then fall back to Rotowire
    vegas = {}
    try:
        vegas = scrape_vegas_odds_oddsshark()
        if vegas:
            print(f"🏀 Loaded Vegas odds from Oddsshark for {len(vegas)} teams")
            return vegas
        else:
            print("⚠️ Oddsshark scrape returned no teams — falling back to Action Network")
    except Exception as e:
        print(f"⚠️ Oddsshark scrape error: {e} — falling back to Action Network")

    try:
        vegas = scrape_vegas_odds_actionnetwork()
        if vegas:
            print(f"🏀 Loaded Vegas odds from Action Network for {len(vegas)} teams")
            return vegas
        else:
            print("⚠️ Action Network scrape returned no teams — falling back to Oddschecker")
    except Exception as e:
        print(f"⚠️ Action Network scrape error: {e} — falling back to Oddschecker")

    try:
        vegas = scrape_vegas_odds_oddschecker()
        if vegas:
            print(f"🏀 Loaded Vegas odds from Oddschecker for {len(vegas)} teams")
            return vegas
        else:
            print("⚠️ Oddschecker scrape returned no teams — falling back to Rotowire")
    except Exception as e:
        print(f"⚠️ Oddschecker scrape error: {e} — falling back to Rotowire")

    # fallback to existing Rotowire scraper
    try:
        vegas = scrape_vegas_odds_rotowire()
        if vegas:
            print(f"🏀 Loaded Vegas odds from Rotowire for {len(vegas)} teams")
        return vegas
    except Exception as e:
        print(f"❌ Rotowire fallback failed: {e}")
        return {}


def scrape_vegas_odds_oddschecker() -> dict:
    """Scrape NBA totals/spreads from Oddschecker (US NBA page).

    Returns mapping of team abbreviation -> { 'total': float, 'spread': float }
    The Oddschecker HTML varies; this function attempts a few heuristics to
    extract team names and totals. It is defensive and may return an empty dict.
    """
    url = "https://www.oddschecker.com/us/basketball/nba"
    vegas = {}

    r = requests.get(url, timeout=10, headers=_get_request_headers())
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Strategy: find game blocks and within each block look for two team names and a 'Total' market value
    # Common selectors vary; try a few known patterns
    game_selectors = ['.event', '.match', '.game', '.event__match', 'article']
    games = []
    for sel in game_selectors:
        found = soup.select(sel)
        if found:
            games = found
            break

    # If no structured game blocks, fall back to scanning text for 'Total' occurrences
    if not games:
        text = soup.get_text(separator='\n')
        # find lines containing 'Total' and a number nearby
        for line in text.splitlines():
            if 'Total' in line and any(ch.isdigit() for ch in line):
                # crude parse: try to extract a float
                m = re.search(r"Total\D*([0-9]{2,3}\.?[0-9]?)", line)
                if m:
                    # cannot map teams here; skip
                    continue
        return {}

    for g in games:
        try:
            # extract team names
            teams_elems = g.select('.event__participant, .team, .teams, .event-participant')
            team_names = [te.get_text(strip=True).upper() for te in teams_elems if te.get_text(strip=True)]
            if len(team_names) < 2:
                # try anchor tags
                a = g.find_all('a')
                team_names = [t.get_text(strip=True).upper() for t in a if t.get_text(strip=True)]
            if len(team_names) < 2:
                continue

            away = team_names[0]
            home = team_names[1]

            # find market cells labeled 'Total' within this game block
            total_val = None
            spread_val = 0.0
            # try to find a cell containing 'Total' text
            total_cells = g.find_all(string=re.compile(r'Total', re.IGNORECASE))
            if total_cells:
                # navigate to parent and search for numbers
                for tc in total_cells:
                    parent = tc.parent
                    nums = re.findall(r"([0-9]{2,3}\.?[0-9]?)", parent.get_text())
                    if nums:
                        total_val = float(nums[0])
                        break

            # fallback: search whole game block for a 3-digit-ish number that looks like a total
            if total_val is None:
                nums = re.findall(r"([0-9]{2,3}\.?[0-9]?)", g.get_text())
                # choose the most plausible (median) number
                if nums:
                    total_val = float(nums[len(nums)//2])

            if total_val is None:
                continue

            # store totals; oddschecker does not expose spread sign consistently here, so use 0 by default
            vegas[away] = {'total': float(total_val), 'spread': float(spread_val)}
            vegas[home] = {'total': float(total_val), 'spread': float(-spread_val)}
        except Exception:
            continue

def scrape_vegas_odds_actionnetwork() -> dict:
    """Scrape NBA totals/spreads from Action Network's NBA odds page.

    This is a best-effort parser: actionnetwork serves dynamic content and may
    block requests. If parsing fails or no data is found, returns {} so the
    caller can fall back to other sources.
    """
    url = "https://www.actionnetwork.com/nba/odds"
    vegas = {}

    r = requests.get(url, timeout=10, headers=_get_request_headers())
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # try identifying event blocks that contain teams and totals
    selectors = ['.odds-row', '.event', '.odds-event', '.match', 'article', '.markets']
    blocks = []
    for sel in selectors:
        found = soup.select(sel)
        if found:
            blocks = found
            break

    if not blocks:
        # fallback: scan entire page for 'Over/Under' or 'Total' text and numbers
        text = soup.get_text(separator='\n')
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            if re.search(r'Over/?Under|O\/U|Total|Over', ln, re.IGNORECASE):
                # attempt to find two nearby team names (previous lines) and a number on this line
                nums = re.findall(r"([0-9]{2,3}\.?[0-9]?)", ln)
                if not nums:
                    continue
                total = float(nums[0])
                # look back up to 3 lines for team names
                team_candidates = []
                for j in range(max(0, i-3), i):
                    tline = lines[j]
                    # crude filter: lines with letters and no punctuation-heavy content
                    if re.search(r'[A-Za-z]{2,}', tline) and len(tline) < 40:
                        team_candidates.append(tline.upper())
                if len(team_candidates) >= 2:
                    away = team_candidates[-2]
                    home = team_candidates[-1]
                    vegas[away] = {'total': total, 'spread': 0.0}
                    vegas[home] = {'total': total, 'spread': -0.0}
        return vegas

    # prepare NBA team variants for matching
    try:
        nba_team_objs = teams.get_teams()
    except Exception:
        nba_team_objs = []

    team_name_variants = []
    for t in nba_team_objs:
        abbr = (t.get('abbreviation') or '').upper()
        full = (t.get('full_name') or '').upper()
        nick = (t.get('nickname') or '').upper()
        city = ''
        if full:
            parts = full.split()
            if len(parts) > 1:
                city = ' '.join(parts[:-1])
        team_name_variants.append({'abbr': abbr, 'full': full, 'nick': nick, 'city': city})

    for b in blocks:
        try:
            block_text = b.get_text(" ", strip=True).upper()

            # find team abbreviations by first occurrence of known nick/full/city
            found = []
            for tv in team_name_variants:
                for key in (tv['full'], tv['nick'], tv['abbr'], tv['city']):
                    if not key:
                        continue
                    idx = block_text.find(key)
                    if idx != -1:
                        found.append((idx, tv['abbr']))
                        break

            # deduplicate and sort by appearance
            if found:
                found_sorted = []
                seen = set()
                for idx, abbr in sorted(found, key=lambda x: x[0]):
                    if abbr and abbr not in seen:
                        found_sorted.append(abbr)
                        seen.add(abbr)
                if len(found_sorted) >= 2:
                    away = found_sorted[0]
                    home = found_sorted[1]
                else:
                    # fallback to crude team extraction if matching failed
                    team_elems = b.select('.team, .team-name, .competitor, .event__participant, .participant')
                    team_names = [re.sub(r"\s+\(.+\)$", "", te.get_text(strip=True).upper()) for te in team_elems if te.get_text(strip=True)]
                    if len(team_names) < 2:
                        h = b.find_all(['h3', 'h4', 'span', 'a'])
                        team_names = [re.sub(r"\s+\(.+\)$", "", t.get_text(strip=True).upper()) for t in h if t.get_text(strip=True)]
                    if len(team_names) >= 2:
                        away = team_names[0]
                        home = team_names[1]
                    else:
                        continue
            else:
                # no team matches found; skip
                continue

            # find a nearby 'Total' or numeric value inside the block
            total_val = None
            candidates = b.find_all(string=re.compile(r'Over/?Under|Total|O/U|Over', re.IGNORECASE))
            for c in candidates:
                parent = c.parent
                nums = re.findall(r"([0-9]{2,3}\.?[0-9]?)", parent.get_text())
                if nums:
                    total_val = float(nums[0])
                    break

            if total_val is None:
                nums = re.findall(r"([0-9]{2,3}\.?[0-9]?)", block_text)
                if nums:
                    # prefer numbers in the latter half (totals often follow team names)
                    total_val = float(nums[len(nums)//2])

            if total_val is None:
                continue

            vegas[away] = {'total': float(total_val), 'spread': 0.0}
            vegas[home] = {'total': float(total_val), 'spread': -0.0}
        except Exception:
            continue

    # If structured parsing didn't return clean abbreviation keys, try a lenient
    # line-scan to find totals and nearby team names (useful for JS-driven pages).
    text = soup.get_text(separator='\n')
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # build variant -> abbr map for quick lookup
    variant_to_abbr = {}
    for tv in team_name_variants:
        abbr = tv['abbr']
        for key in (tv['full'], tv['nick'], tv['city'], tv['abbr']):
            if key:
                variant_to_abbr[key] = abbr

    # First, try to map any already-captured keys to known team abbreviations by substring
    # (e.g., '76ERS MONEYLINE...' -> PHI). This cleans up noisy keys produced by block parsing.
    for k in list(vegas.keys()):
        ku = k.upper()
        matched = set()
        for key, ab in variant_to_abbr.items():
            if key and key in ku:
                matched.add(ab)
        if matched:
            v = vegas.pop(k)
            for ab in matched:
                # prefer existing mapped entry if present (don't overwrite spread/total)
                if ab not in vegas:
                    vegas[ab] = v

    # find all occurrences of team names in the page text (index, abbr)
    team_positions = []
    for i, line in enumerate(lines):
        L = line.upper()
        for key, ab in variant_to_abbr.items():
            if key and key in L:
                team_positions.append((i, ab))

    # iterate nearby pairs of team occurrences and look for a total between them
    seen_pairs = set()
    for idx in range(len(team_positions)-1):
        i1, ab1 = team_positions[idx]
        i2, ab2 = team_positions[idx+1]
        # require teams to be reasonably close on the page (avoid unrelated occurrences)
        if 1 <= (i2 - i1) <= 20 and ab1 != ab2:
            # search for a numeric total in the window around the pair
            window_start = max(0, i1-2)
            window_end = min(len(lines), i2+3)
            window_text = ' '.join(lines[window_start:window_end])
            nums = re.findall(r"([0-9]{2,3}\.?[0-9]?)", window_text)
            if nums:
                total = float(nums[0])
                pair_key = (ab1, ab2, int(total))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                vegas[ab1] = {'total': total, 'spread': 0.0}
                vegas[ab2] = {'total': total, 'spread': -0.0}

    # If we have any teams with totals but missing their opponent, try to infer the
    # opponent by finding the nearest other team occurrence on the page and assign
    # the same total to them (best-effort).
    existing_abbrs = set(vegas.keys())
    # build index by abbr
    positions_by_abbr = {}
    for pos, ab in team_positions:
        positions_by_abbr.setdefault(ab, []).append(pos)

    for ab in list(existing_abbrs):
        # skip if already have a matching opponent assigned
        # (i.e., another ab with same total)
        total = vegas[ab]['total']
        # find candidates not in vegas
        if len(vegas) >= 30:
            break
        # find nearest other team occurrence
        nearest = None
        if ab in positions_by_abbr:
            for pos in positions_by_abbr[ab]:
                # search outward for nearest different abbr
                best = None
                best_dist = 9999
                for other_ab, pos_list in positions_by_abbr.items():
                    if other_ab == ab or other_ab in vegas:
                        continue
                    for p in pos_list:
                        d = abs(p - pos)
                        if d < best_dist:
                            best_dist = d
                            best = other_ab
                if best and best_dist <= 60:
                    nearest = best
                    break
        if nearest and nearest not in vegas:
            vegas[nearest] = {'total': total, 'spread': -0.0}

    return vegas


def scrape_vegas_odds_oddsshark() -> dict:
    """Scrape NBA totals/spreads from OddsShark's NBA odds page.

    Returns mapping of team abbreviation or name-like key -> { 'total': float, 'spread': float }
    This is defensive: OddsShark sometimes serves friendly static HTML, but may also
    use dynamic content. We attempt to parse common table/row structures and fall
    back to a lenient text-scan if needed.
    """
    url = "https://www.oddsshark.com/nba/odds"
    vegas = {}

    r = requests.get(url, timeout=10, headers=_get_request_headers())
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # prepare team variant lookup to validate found tokens
    try:
        nba_team_objs = teams.get_teams()
    except Exception:
        nba_team_objs = []
    variant_to_abbr = {}
    for t in nba_team_objs:
        abbr = (t.get('abbreviation') or '').upper()
        full = (t.get('full_name') or '').upper()
        nick = (t.get('nickname') or '').upper()
        city = ''
        if full:
            parts = full.split()
            if len(parts) > 1:
                city = ' '.join(parts[:-1])
        for key in (full, nick, city, abbr):
            if key:
                variant_to_abbr[key] = abbr

    # Common pattern: odds tables with rows containing two teams and markets
    # Try to find table rows first
    rows = soup.select('table tr') or soup.select('.event, .match, .odds-row')
    if rows:
        for row in rows:
            try:
                cols = [c.get_text(strip=True) for c in row.find_all(['td', 'th'])]
                if len(cols) < 2:
                    continue
                text = ' | '.join(cols).upper()

                # attempt to find two team names and a total (Over/Under)
                nums = re.findall(r"([0-9]{2,3}\.?[0-9]?)", text)
                # crude team extraction: split on '|' and validate tokens against known variants
                parts = [p.strip() for p in text.split('|') if p.strip()]
                teams_found = []
                for p in parts[:6]:
                    # skip numeric tokens or market labels
                    if re.search(r'^[0-9\-\$\(\)]+$', p):
                        continue
                    if re.search(r'MONEYLINE|OVER|UNDER|TOTAL|SPREAD|ODDS', p):
                        continue
                    # try to match token to known team variant
                    matched = None
                    for key, ab in variant_to_abbr.items():
                        if key and key in p:
                            matched = ab
                            break
                    if matched and matched not in teams_found:
                        teams_found.append(matched)
                if len(teams_found) >= 2 and nums:
                    away = teams_found[0]
                    home = teams_found[1]
                    total = float(nums[0])
                    vegas[away] = {'total': total, 'spread': 0.0}
                    vegas[home] = {'total': total, 'spread': -0.0}
            except Exception:
                continue

    # If we got few or no results, fall back to lenient line scanning similar to other scrapers
    if len(vegas) < 6:
        text = soup.get_text(separator='\n')
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            if re.search(r'OVER/?UNDER|O\/U|TOTAL|OVER', ln, re.IGNORECASE):
                nums = re.findall(r"([0-9]{2,3}\.?[0-9]?)", ln)
                if not nums:
                    look = ' '.join(lines[max(0, i-1):min(len(lines), i+2)])
                    nums = re.findall(r"([0-9]{2,3}\.?[0-9]?)", look)
                if not nums:
                    continue
                total = float(nums[0])
                # look back for two team-like lines and map them to known abbreviations
                team_candidates = []
                for j in range(max(0, i-6), i):
                    cand = lines[j]
                    if re.search(r'[A-Za-z]{2,}', cand) and len(cand) < 60 and not re.search(r'OVER|UNDER|TOTAL|MONEYLINE|SPREAD', cand, re.IGNORECASE):
                        cu = cand.upper()
                        matched_ab = None
                        for key, ab in variant_to_abbr.items():
                            if key and key in cu:
                                matched_ab = ab
                                break
                        if matched_ab and matched_ab not in team_candidates:
                            team_candidates.append(matched_ab)
                    if len(team_candidates) >= 2:
                        break
                if len(team_candidates) >= 2:
                    away = team_candidates[-2]
                    home = team_candidates[-1]
                    vegas[away] = {'total': total, 'spread': 0.0}
                    vegas[home] = {'total': total, 'spread': -0.0}

    return vegas


def scrape_vegas_odds_rotowire() -> dict:
    """Original Rotowire scraper extracted into a fallback function."""
    url = "https://www.rotowire.com/betting/nba/odds.php"
    vegas = {}

    try:
        r = requests.get(url, timeout=10, headers=_get_request_headers())
        soup = BeautifulSoup(r.text, "html.parser")

        # Rows contain matchups
        rows = soup.select("table tbody tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 7:
                continue

            # Extract team abbreviations
            away_team = cols[0].text.strip().upper()
            home_team = cols[2].text.strip().upper()

            # Extract spread + total
            # Rotowire format usually: spread in column 4, total in column 5
            try:
                spread = float(cols[4].text.strip())
            except:
                spread = VEGAS_DEFAULT_SPREAD

            try:
                total = float(cols[5].text.strip())
            except:
                total = VEGAS_DEFAULT_TOTAL

            vegas[away_team] = {"total": total, "spread": spread}
            vegas[home_team] = {"total": total, "spread": -spread}

        print(f"🏀 Loaded Vegas odds for {len(vegas)} teams")

    except Exception as e:
        print(f"❌ Vegas scrape failed: {e}")

    return vegas


def _normalize_vegas_keys(vegas: dict) -> dict:
    """Normalize keys in the vegas dict to NBA team abbreviations (e.g. LAL, BOS).

    The scraper may return full names or other formats; this tries to match keys
    to the official `abbreviation` from nba_api's teams list. If a key cannot be
    mapped, it is left as-is but reported.
    """
    try:
        nba_teams = teams.get_teams()
    except Exception:
        nba_teams = []

    # build lookup maps
    key_to_abbr = {}
    abbrs = set()
    for t in nba_teams:
        abbr = t.get('abbreviation') or t.get('abbr') or ''
        full = t.get('full_name') or ''
        nick = t.get('nickname') or ''
        if abbr:
            abbrs.add(abbr.upper())
            key_to_abbr[abbr.upper()] = abbr.upper()
        if full:
            key_to_abbr[full.strip().upper()] = abbr.upper()
        if nick:
            key_to_abbr[nick.strip().upper()] = abbr.upper()

    normalized = {}
    unmapped = []
    for k, v in vegas.items():
        if not isinstance(k, str):
            normalized[k] = v
            continue
        ku = k.strip().upper()
        if ku in key_to_abbr:
            normalized[key_to_abbr[ku]] = v
        else:
            # Try to find any known team name/variant as a substring inside the key.
            found_abbr = None
            for known_variant, abbr in key_to_abbr.items():
                if known_variant and known_variant in ku:
                    found_abbr = abbr
                    break
            if found_abbr:
                normalized[found_abbr] = v
            else:
                # As a final attempt, check if any nickname/full name appears inside the key
                for t in nba_teams:
                    ab = (t.get('abbreviation') or '').upper()
                    full = (t.get('full_name') or '').upper()
                    nick = (t.get('nickname') or '').upper()
                    if full and full in ku:
                        normalized[ab] = v
                        found_abbr = ab
                        break
                    if nick and nick in ku:
                        normalized[ab] = v
                        found_abbr = ab
                        break
                if not found_abbr:
                    unmapped.append(k)
                    normalized[ku] = v

    if unmapped:
        print(f"⚠️ Could not normalize vegas team keys for: {unmapped}")

    return normalized


def vegas_multiplier(team: str, vegas: dict) -> float:
    data = vegas.get(team, {})
    total = float(data.get('total', VEGAS_DEFAULT_TOTAL))
    spread = float(data.get('spread', VEGAS_DEFAULT_SPREAD))

    total_adj = (total / VEGAS_DEFAULT_TOTAL)

    if spread <= -10:  # heavy favorite
        blowout_adj = 0.95
    elif spread >= 10:  # heavy underdog
        blowout_adj = 0.97
    else:
        blowout_adj = 1.0

    mult = total_adj * blowout_adj
    return max(0.85, min(1.15, mult))

# --- Injury Integration ---
# Adds pace/possession adjustments based on Vegas odds
# Expected input: dict like { 'TEAM': {'total': 118.5, 'spread': -6.5}, ... }

# --- Injury Integration (ESPN Scraper Source Option A) ---
# Expected injury input format:
# injuries = {
#    'player name lower': {
#         'status': 'OUT' / 'QUESTIONABLE' / 'DOUBTFUL' / 'PROBABLE',
#         'description': str
#    }, ...
# }
# Your external ESPN scraper should return this dictionary.

INJURY_MINUTE_ADJUSTMENTS = {
    'OUT': -100,        # remove from pool
    'DOUBTFUL': -12,    # strong negative
    'QUESTIONABLE': -5, # light negative
    'PROBABLE': 0       # no change
}


def apply_injury_minutes_adjustment(base_minutes: float, injury_status: str) -> float:
    if not injury_status:
        return base_minutes
    injury_status = injury_status.upper()
    adj = INJURY_MINUTE_ADJUSTMENTS.get(injury_status, 0)
    new_min = base_minutes + adj
    # never below 0
    return max(0, new_min)


# --- Integration Helper for Existing Optimizer ---
# Your optimizer likely expects a function that returns a DataFrame or list of dicts.
# We provide a simple wrapper:

def get_projections(dk_csv_path: str, days: int = 30, season: str = None, n_sims: int = 1500) -> pd.DataFrame:
    model = SimpleNBAProjection(dk_salaries_path=dk_csv_path, days_of_history=days, season=season)
    if not model.load_dk_salaries():
        return pd.DataFrame()
    # Run projections (vegas will be scraped internally by run() if not provided)
    return model.run(save_csv=None, n_sims=n_sims)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dk', required=True, help='Path to DraftKings salaries CSV')
    parser.add_argument('--days', type=int, default=30, help='Days of history for logs (default=30)')
    parser.add_argument('--season', default=None, help='Season string e.g. 2024-25 (auto-detected by default)')
    parser.add_argument('--out', default='projections.csv', help='Output CSV path')
    parser.add_argument('--backtest', action='store_true', help='Run backtest instead of single run')
    parser.add_argument('--backtest-days', type=int, default=14, help='Number of past days to backtest')
    parser.add_argument('--n-sims', type=int, default=1500, help='Monte Carlo sims per player')
    args = parser.parse_args()

    model = SimpleNBAProjection(dk_salaries_path=args.dk, days_of_history=args.days, season=args.season)
    if not model.load_dk_salaries():
        raise SystemExit(1)

    if args.backtest:
        print('Running backtest...')
        # Backtest over the past N days (skipping today)
        end_date = datetime.now().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=args.backtest_days - 1)

        backtest_results = []
        current = start_date
        while current <= end_date:
            print(f"Backtesting date: {current}")
            try:
                df_proj = model.run(save_csv=None, injuries=None, n_sims=args.n_sims)
                # For actual results, fetch game logs for that date (actual performance)
                # We will reuse playergamelogs to fetch only that date range
                date_str = current.strftime('%Y-%m-%d')
                res = model._safe_api_call(playergamelogs.PlayerGameLogs, season_nullable=model.season, date_from_nullable=date_str, date_to_nullable=date_str)
                actual_logs = res.get_data_frames()[0] if res else pd.DataFrame()
                # Compute actual DK points
                actual_logs['DK_FP'] = actual_logs.apply(model.calculate_dk_points_from_row, axis=1)
                # Merge projections with actuals by player name
                if not df_proj.empty and not actual_logs.empty:
                    act = actual_logs[['PLAYER_NAME', 'DK_FP']].copy()
                    act['PLAYER_NAME_L'] = act['PLAYER_NAME'].str.lower().str.strip()
                    merged = df_proj.copy()
                    merged['Name_L'] = merged['Name'].str.lower().str.strip()
                    merged = merged.merge(act[['PLAYER_NAME_L', 'DK_FP']], left_on='Name_L', right_on='PLAYER_NAME_L', how='left')
                    # Evaluate calibration
                    merged['HitFloor'] = merged['DK_FP'] >= merged.get('Floor_MC', merged['Projection']*0.7)
                    merged['BeatCeiling'] = merged['DK_FP'] >= merged.get('Ceiling_MC', merged['Projection']*1.3)
                    floor_hit_rate = 100.0 * (1 - merged['HitFloor'].mean()) if not merged['HitFloor'].isna().all() else None
                    ceiling_rate = 100.0 * (merged['BeatCeiling'].mean()) if not merged['BeatCeiling'].isna().all() else None
                else:
                    floor_hit_rate = None
                    ceiling_rate = None
                backtest_results.append({'date': date_str, 'floor_hit_rate_pct': floor_hit_rate, 'ceiling_hit_rate_pct': ceiling_rate, 'num_players': len(df_proj)})
            except Exception as e:
                print('Backtest error for date', current, e)
                backtest_results.append({'date': current.strftime('%Y-%m-%d'), 'floor_hit_rate_pct': None, 'ceiling_hit_rate_pct': None, 'num_players': 0})
            current += timedelta(days=1)

        bt_df = pd.DataFrame(backtest_results)
        bt_out = args.out.replace('.csv', '_backtest.csv')
        bt_df.to_csv(bt_out, index=False)
        print(f'Backtest saved to {bt_out}')
        print(bt_df)
    else:
        df = model.run(save_csv=args.out, injuries=None, n_sims=args.n_sims)
        if not df.empty:
            df.to_csv(args.out, index=False)
            print(df.head(30).to_string(index=False))
        else:
            print('No projections available.')
