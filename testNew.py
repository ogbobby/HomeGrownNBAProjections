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

# NBA API imports (may be rate-limited; use caching if you run often)
from nba_api.stats.endpoints import playergamelogs, leaguedashteamstats, scoreboardv2
from nba_api.stats.static import players, teams
from injurySrape import get_injuries


class SimpleNBAProjection:
    LEAGUE_AVG_PACE = 100.0  # fallback; will try to read real league pace

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
                'games': len(mins)
            }
        print(f"✅ Computed FP/min for {len(out)} players")
        return out

    def project_minutes(self, player_stats: Dict) -> float:
        """Weighted minutes projection from last5, last10, and role expectation."""
        last5 = player_stats.get('last5_min', 0)
        last10 = player_stats.get('last10_min', 0)
        recent = player_stats.get('recent_minutes', 0)
        # role expectation fallback = recent (could be improved with depth chart data)
        role = recent
        projected = last5 * 0.40 + last10 * 0.25 + role * 0.35
        # bounds
        projected = max(5.0, projected)
        projected = min(38.0, projected)
        return projected

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
    
    def run(self, save_csv: str = None, injuries: dict = None) -> pd.DataFrame:
        # Preconditions
        if self.dk_df is None:
            raise RuntimeError('DK salaries not loaded')

        # Get injuries if not provided
        if injuries is None:
            injuries = get_injuries()

        # Normalize injury keys
        inj = {k.lower(): v for k, v in injuries.items()}

        print(f"✅ Loaded {len(inj)} players from injury report")

        # Fetch team stats and today's matchups
        self.fetch_team_stats()
        self.fetch_todays_matchups()

        logs = self.fetch_recent_game_logs()
        if logs.empty:
            print('No logs available — aborting')
            return pd.DataFrame()

        player_fpmin = self.compute_fp_per_min(logs)

        results = []
        name_to_ids = {}
        for pid, g in logs.groupby('PLAYER_ID'):
            name = g.iloc[0]['PLAYER_NAME'].strip()
            name_to_ids.setdefault(name.lower(), []).append(int(pid))

        # ====== NEW: IDENTIFY STARTERS WHO ARE OUT ======
        starters_out = []
        for player_name, status in injuries.items():
            status_clean = str(status).strip().upper()
            if status_clean == 'OUT':
                starters_out.append(player_name.lower())

        print(f"📋 {len(starters_out)} players marked OUT")

        # ====== NEW: CREATE BACKUP BOOST DICTIONARY ======
        backup_boost = {}

        if starters_out:
            print("🔍 Analyzing backup opportunities...")

            # Group players by team and position from DK data
            team_pos_players = {}
            for _, row in self.dk_df.iterrows():
                name = row['Name'].strip()
                name_lower = name.lower()
                team = row['Team'].strip() if 'Team' in row else None
                pos = row['Position'] if 'Position' in row else None

                if team and pos:
                    # Handle multi-position players
                    positions = [p.strip() for p in str(pos).split('/')] if '/' in str(pos) else [str(pos).strip()]

                    for position in positions:
                        key = (team, position)
                        if key not in team_pos_players:
                            team_pos_players[key] = []
                        team_pos_players[key].append({
                            'name': name,
                            'name_lower': name_lower,
                            'salary': float(row['Salary'])
                        })

            # Identify backups for OUT starters
            for out_player_lower in starters_out:
                # Find the OUT player in DK data
                out_player_row = None
                for _, row in self.dk_df.iterrows():
                    if row['Name'].strip().lower() == out_player_lower:
                        out_player_row = row
                        break
                    
                if out_player_row is None:
                    continue

                out_player_name = out_player_row['Name'].strip()
                out_player_team = out_player_row['Team'].strip() if 'Team' in out_player_row else None
                out_player_pos = out_player_row['Position'] if 'Position' in out_player_row else None

                if not out_player_team or not out_player_pos:
                    continue
                
                print(f"  {out_player_name} ({out_player_team}) is OUT - looking for backups...")

                # Get positions of the OUT player
                out_positions = [p.strip() for p in str(out_player_pos).split('/')] if '/' in str(out_player_pos) else [str(out_player_pos).strip()]

                # Find potential backups on same team
                for position in out_positions:
                    key = (out_player_team, position)
                    if key in team_pos_players:
                        candidates = team_pos_players[key]

                        # Remove the OUT player from candidates
                        candidates = [c for c in candidates if c['name_lower'] != out_player_lower]

                        if candidates:
                            # Sort by salary (higher salary often indicates more important player)
                            candidates.sort(key=lambda x: x['salary'], reverse=True)

                            # Boost the top backup (or top 2 for important positions)
                            num_backups_to_boost = 1
                            if position in ['PG', 'SG', 'SF', 'PF', 'C']:  # Main positions
                                num_backups_to_boost = min(2, len(candidates))

                            for i in range(num_backups_to_boost):
                                backup = candidates[i]
                                backup_name_lower = backup['name_lower']

                                # Determine boost amount based on position and starter's salary
                                starter_salary = float(out_player_row['Salary'])

                                if starter_salary > 8000:  # Superstar
                                    base_boost = 5.0
                                elif starter_salary > 6000:  # Star player
                                    base_boost = 4.0
                                else:  # Regular starter
                                    base_boost = 3.0

                                # Position-specific adjustments
                                if position == 'PG':
                                    base_boost += 1.0  # PGs control offense
                                elif position == 'C':
                                    base_boost += 0.5  # Centers get moderate boost

                                # Apply additional boost if backup is cheap (more minutes opportunity)
                                if backup['salary'] < 5000:
                                    base_boost += 1.0

                                # Store the boost
                                if backup_name_lower in backup_boost:
                                    backup_boost[backup_name_lower] += base_boost
                                else:
                                    backup_boost[backup_name_lower] = base_boost

                                print(f"    → {backup['name']} gets +{base_boost:.1f} min boost")

        print(f"🎯 {len(backup_boost)} backups identified for boost")
        # ====== END NEW BACKUP BOOST CODE ======

        for _, row in self.dk_df.iterrows():
            name = row['Name'].strip()
            salary = float(row['Salary'])
            team = row['Team'].strip() if 'Team' in row else None
            pos = row['Position'] if 'Position' in row else None

            # Debug for specific players
            name_lower = name.lower()
            if 'jalen' in name_lower and 'johnson' in name_lower:
                print(f"\nDEBUG Jalen Johnson:")
                print(f"  Name in DK: '{name}'")
                print(f"  Name lower: '{name_lower}'")
                print(f"  In injuries?: {name_lower in inj}")
                if name_lower in inj:
                    print(f"  Injury status: '{inj[name_lower]}'")

            pid_list = name_to_ids.get(name_lower, [])
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
            base_min = self.project_minutes(stats)

            # Injury handling
            status = inj.get(name_lower)
            injury_status_for_adjustment = None

            if status:
                status_u = status.upper()
                if status_u == 'OUT':
                    continue  # Skip OUT players entirely
                if status_u in ['DAY-TO-DAY', 'DTD']:
                    status_u = 'QUESTIONABLE'
                injury_status_for_adjustment = status_u

            # Apply injury adjustment if needed
            if injury_status_for_adjustment:
                base_min = apply_injury_minutes_adjustment(base_min, injury_status_for_adjustment)
                # Debug for specific players
                if 'jaylen brown' in name_lower or 'stephen curry' in name_lower:
                    print(f"Applied injury adjustment to {name}: {base_min} minutes")

            # ====== NEW: APPLY BACKUP BOOST ======
            if name_lower in backup_boost:
                boost_amount = backup_boost[name_lower]
                original_min = base_min
                base_min += boost_amount

                # Cap at reasonable maximum (don't exceed 40 minutes)
                base_min = min(40.0, base_min)

                # Debug output
                print(f"📈 {name} gets backup boost: {original_min:.1f} → {base_min:.1f} min (+{boost_amount:.1f})")
            # ====== END NEW BACKUP BOOST ======

            # minimum 1 min for questionable players
            if base_min <= 0:
                continue

            mult = self.matchup_multiplier(team) if team else 1.0

            raw_projection = fpmin * base_min * mult
            capped = self.cap_projection_by_salary(raw_projection, salary)

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
                'BackupBoost': round(backup_boost.get(name_lower, 0), 1)  # Track boost amount
            })

        out_df = pd.DataFrame(results).sort_values('Projection', ascending=False)
        print(f"✅ Projected {len(out_df)} players (with injuries and backup boosts applied)")

        # Print summary of backup boosts
        boosted_players = out_df[out_df['BackupBoost'] > 0]
        if not boosted_players.empty:
            print(f"\n📊 Backup Boost Summary:")
            for _, player in boosted_players.iterrows():
                print(f"  {player['Name']}: +{player['BackupBoost']} min → {player['ProjMin']} total min")

        if save_csv:
            out_df.to_csv(save_csv, index=False)
            print(f"Saved projections to {save_csv}")

        return out_df

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
    'Day-To-Day': -12,   # Strong negative
    'DTD': -12,            # Also handle abbreviation
    'QUESTIONABLE': -5, # light negative
    'PROBABLE': 0       # no change
}

injured_status = get_injuries()

def apply_injury_minutes_adjustment(base_minutes: float, injury_status: str) -> float:
    if not injury_status:
        return base_minutes
    
    # Clean the status
    injury_status = injury_status.strip().upper()
    
    # Handle variations
    if injury_status == 'DTD':
        injury_status = 'DAY-TO-DAY'
    
    # Get adjustment
    adj = INJURY_MINUTE_ADJUSTMENTS.get(injury_status, 0)
    new_min = base_minutes + adj
    
    # Never below 0
    return max(0, new_min)

# --- Integration Helper for Existing Optimizer ---
# Your optimizer likely expects a function that returns a DataFrame or list of dicts.
# We provide a simple wrapper:

def get_projections(dk_csv_path: str, days: int = 30, season: str = None) -> pd.DataFrame:
    model = SimpleNBAProjection(dk_salaries_path=dk_csv_path, days_of_history=days, season=season)
    if not model.load_dk_salaries():
        return pd.DataFrame()
    return model.run(save_csv=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dk', required=True, help='Path to DraftKings salaries CSV')
    parser.add_argument('--days', type=int, default=30, help='Days of history for logs (default=30)')
    parser.add_argument('--season', default=None, help='Season string e.g. 2024-25 (auto-detected by default)')
    parser.add_argument('--out', default='projections.csv', help='Output CSV path')
    args = parser.parse_args()

    df = get_projections(args.dk, args.days, args.season)
    if not df.empty:
        df.to_csv(args.out, index=False)
        print(df.head(30).to_string(index=False))
    else:
        print('No projections available.')
