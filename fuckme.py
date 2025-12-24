#!/usr/bin/env python3
"""
simple_nba_projections_full.py

Single-file NBA DFS projection system:
- DraftKings salary loader (auto-detects headers; supports your header)
- ESPN injury scraper (auto)
- Dynamic positional usage redistribution (PG-out, C-out, backups)
- Monte Carlo per-stat / per-minute floor & ceiling
- Vegas odds integration (Rotowire scraper)
- CLI runner
"""

import argparse
import math
import time
import re
import unicodedata
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# nba_api imports (user must have nba_api installed)
from nba_api.stats.endpoints import playergamelogs, leaguedashteamstats, scoreboardv2
from nba_api.stats.static import teams
from injurySrape import get_injuries
from getVegas import get_nba_odds

# -------------------------
# Config / constants
# -------------------------
VEGAS_DEFAULT_TOTAL = 112.0
VEGAS_DEFAULT_SPREAD = 0.0

INJURY_MINUTE_ADJUSTMENTS = {
    'OUT': -100,
    'DOUBTFUL': -12,
    'QUESTIONABLE': -5,
    'PROBABLE': 0
}
TEAM_NAME_TO_ABBR = {
    "HAWKS": "ATL",
    "CELTICS": "BOS",
    "NETS": "BKN",
    "HORNETS": "CHA",
    "BULLS": "CHI",
    "CAVALIERS": "CLE",
    "MAVERICKS": "DAL",
    "NUGGETS": "DEN",
    "PISTONS": "DET",
    "WARRIORS": "GSW",
    "ROCKETS": "HOU",
    "PACERS": "IND",
    "CLIPPERS": "LAC",
    "LAKERS": "LAL",
    "GRIZZLIES": "MEM",
    "HEAT": "MIA",
    "BUCKS": "MIL",
    "TIMBERWOLVES": "MIN",
    "PELICANS": "NOP",
    "KNICKS": "NYK",
    "THUNDER": "OKC",
    "MAGIC": "ORL",
    "76ERS": "PHI",
    "SUNS": "PHX",
    "TRAIL BLAZERS": "POR",
    "KINGS": "SAC",
    "SPURS": "SAS",
    "RAPTORS": "TOR",
    "JAZZ": "UTA",
    "WIZARDS": "WAS"
}

VOLATILITY_MULTIPLIER = {
    "HIGH": 1.25,
    "MED": 1.0,
    "LOW": 0.85
}
BIAS_CORRECTION = 1.045
# -------------------------
# Utility: CSV header mapping
# -------------------------
def detect_dk_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Return (name_col, salary_col, team_col, position_col) selected from df columns.
    Uses the header you provided as high priority.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    # Name: prefer exact "name" column; fallbacks include "name + id", "player", "playername"
    name_candidates = ['name', 'name + id', 'name+id', 'player', 'playername', 'full_name']
    name_col = None
    for cand in name_candidates:
        if cand in cols_lower:
            name_col = cols_lower[cand]
            break

    # Salary
    salary_candidates = ['salary', 'dk salary', 'dksalary']
    salary_col = None
    for cand in salary_candidates:
        if cand in cols_lower:
            salary_col = cols_lower[cand]
            break

    # Team abbreviation
    team_candidates = ['teamabbrev', 'team_abbrev', 'team', 'teamabbr', 'teamabbrv']
    team_col = None
    for cand in team_candidates:
        if cand in cols_lower:
            team_col = cols_lower[cand]
            break

    # Position
    pos_candidates = ['position', 'pos', 'roster position', 'roster_position']
    pos_col = None
    for cand in pos_candidates:
        if cand in cols_lower:
            pos_col = cols_lower[cand]
            break

    return name_col, salary_col, team_col, pos_col

def normalize_name(name: str) -> str:
    """
    Normalize player names so 'Dončić' and 'Doncic' match.
    Removes accents, trims whitespace, ignores case issues.
    """
    if not isinstance(name, str):
        return name

    # Remove accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))

    # Trim extra spaces
    name = name.replace("'", "").strip().lower()

    return name

def normalize_vegas_data(raw_games):
    """
    Convert raw Vegas API data into a dictionary keyed by TEAM ABBREVIATION
    with computed team totals and spreads.
    """
    vegas = {}

    for game in raw_games:
        t1 = game["team1"]
        t2 = game["team2"]

        # Normalize team names to abbreviations
        team1_full = t1["team_name"].strip().upper()
        team2_full = t2["team_name"].strip().upper()
        team1_abbr = TEAM_NAME_TO_ABBR.get(team1_full)
        team2_abbr = TEAM_NAME_TO_ABBR.get(team2_full)
         # 🚫 Skip postponed / NA games
        if not t1.get("total") or not t2.get("total"):
            print(f"⚠️ Skipping postponed game: {game.get('matchup')}")
            continue


        if not team1_abbr or not team2_abbr:
            print(f"⚠️ Could not normalize teams: {team1_full}, {team2_full}")
            continue

        def parse_total(val):
            """
            Parse totals like:
              'o226.5-110'
              'u230'
              '226.5'
              None
            """
            if val is None:
                return None

            # If already numeric
            if isinstance(val, (int, float)):
                return float(val)

            if not isinstance(val, str):
                return None

            val = val.strip().lower()
            if not val:
                return None

            # Strip o/u
            if val.startswith(("o", "u")):
                val = val[1:]

            # Strip juice
            if "-" in val:
                val = val.split("-")[0]

            try:
                return float(val)
            except ValueError:
                return None

        #total = parse_total(t1.get("total", 0))
        total = parse_total(t1.get("total"))

        # Extract spread
        def parse_spread(raw):
            if not raw or not isinstance(raw, str):
                return 0.0
            raw = raw.strip().lower()
            if raw in ("pk", "pick", "pick'em", "even"):
                return 0.0
            m = re.search(r'([+-]?\d+\.?\d*)', raw)
            if m:
                try:
                    return float(m.group(1))
                except:
                    return 0.0
            return 0.0

        spread1 = parse_spread(t1.get("spread", ""))
        spread2 = parse_spread(t2.get("spread", ""))

        # Compute Implied Team Totals: team_total = (total / 2) + (spread / 2)
        t1_total = (total / 2) + (spread1 / 2)
        t2_total = (total / 2) + (spread2 / 2)

        vegas[team1_abbr] = {
            "total": total,
            "spread": spread1,
            "team_total": t1_total
        }
        vegas[team2_abbr] = {
            "total": total,
            "spread": spread2,
            "team_total": t2_total
        }

    return vegas


def vegas_multiplier(team, vegas):
    team = normalize_team(team)
    if not team or team not in vegas:
        return 1.0  # default if not found

    data = vegas[team]

    total = data.get("total")
    spread = data.get("spread")

    mult = 1.0

    # High total game → slight boost
    if total:
        mult *= max(0.9, min(1.10, (total - 215) / 100 + 1))

    # Heavy favorite → more minutes, small boost
    if spread is not None:
        if spread < -5:      # favored
            mult *= 1.02
        elif spread > 5:     # underdog
            mult *= 0.98

    return mult

def normalize_team(team):
    if not team:
        return None
    t = team.strip().upper()
    return TEAM_NAME_TO_ABBR.get(t, t)

def classify_volatility(fp_min_list):  #just added 12-16
    if not fp_min_list or len(fp_min_list) < 5:
        return "MED"

    std = np.std(fp_min_list)
    if std > 0.45:
        return "HIGH"
    elif std < 0.25:
        return "LOW"
    return "MED"

def correlated_stat_draw(row, rng):   #just added 12-16
    base = rng.normal(1.0, 0.12)

    pts  = row.get('PTS', 0)  * rng.normal(base, 0.08)
    ast  = row.get('AST', 0)  * rng.normal(base, 0.10)
    reb  = row.get('REB', 0)  * rng.normal(1.0, 0.12)
    fg3  = row.get('FG3M', 0) * rng.normal(base, 0.15)
    stl  = row.get('STL', 0)  * rng.normal(1.0, 0.25)
    blk  = row.get('BLK', 0)  * rng.normal(1.0, 0.25)
    tov  = row.get('TOV', 0)  * rng.normal(1.0, 0.20)

    return {
        'PTS': max(0, pts),
        'AST': max(0, ast),
        'REB': max(0, reb),
        'FG3M': max(0, fg3),
        'STL': max(0, stl),
        'BLK': max(0, blk),
        'TOV': max(0, tov),
    }
def estimate_ownership(df):                     #Function added 12-16
    """
    Estimate ownership % using salary and projection ranks.
    Output is roughly calibrated to DK large-field GPPs.
    """
    df = df.copy()

    df['SalRank'] = df['Salary'].rank(pct=True)
    df['ProjRank'] = df['Projection'].rank(pct=True)

    # Core ownership signal
    raw = (
        0.65 * df['ProjRank'] +
        0.35 * df['SalRank']
    )

    # Nonlinear squashing
    df['Ownership'] = (
        100 * np.clip(raw ** 1.8, 0.01, 0.65)
    )

    return df

def salary_bias(salary):
    if salary < 4000:
        return -0.5
    if 4000 <= salary <= 5200:
        return 1.0
    if 5200 < salary <= 6800:
        return -0.3
    if 6800 < salary <= 8500:
        return 0.6
    return 0.3  # expensive stars

# -------------------------
# Position-based usage context and redistribution
# -------------------------
def compute_position_usage_context(team_players_df: pd.DataFrame, injuries: Dict[str, str]) -> Dict[str, object]:
    """
    For a given team roster (DataFrame subset of DK roster), compute a context of missing usage/assists/rebounds
    caused by OUT players.
    """
    ctx = {"usg_missing": 0.0, "ast_missing": 0.0, "reb_missing": 0.0, "pg_out": False, "c_out": False, "positions_out": []}
    for _, p in team_players_df.iterrows():
        name = normalize_name(p.get("Name",""))
        pos = str(p.get("Position",""))
        status = injuries.get(name)
        if status != "OUT":
            continue
        ctx["positions_out"].append(pos)
        if pos == "PG":
            ctx["pg_out"] = True
            ctx["usg_missing"] += 0.04
            ctx["ast_missing"] += 0.06
        elif pos in ("SG", "SF"):
            ctx["usg_missing"] += 0.03
        elif pos == "C":
            ctx["c_out"] = True
            ctx["reb_missing"] += 0.06
    return ctx

def dynamic_usage_redistribution(player_name: str, player_pos: str, fpmin: float, proj_min: float,
                                 injuries: Dict[str, str], dk_df: pd.DataFrame, max_boost: float = 0.25) -> float:
    """
    Main redistribution engine — returns adjusted fp/min.
    - player_name: full name (string)
    - player_pos: e.g. 'PG', 'SG', 'SF', 'PF', 'C'
    - fpmin: base fantasy points per minute
    - proj_min: projected minutes (used to skip tiny-minute players)
    - injuries: dict of lowercased player name -> status
    - dk_df: DraftKings roster DataFrame (so we can detect teammates/backups)
    """
    # skip tiny projected minutes
    if proj_min < 6:
        return fpmin

    #injuries_l = {k.lower(): v for k, v in injuries.items()}
    injuries_l = {normalize_name(k):v for k,v in injuries.items()}
    team = None
    # find player's team in dk_df
    row = dk_df[dk_df['Name'].apply(normalize_name) == normalize_name(player_name)]
    if not row.empty:
        team = row.iloc[0].get('Team')
    if team is None:
        # fallback: no team info -> return unchanged
        return fpmin

    team_players_df = dk_df[dk_df['Team'] == team]
    ctx = compute_position_usage_context(team_players_df, injuries_l)

    boost = 0.0
    name_l = normalize_name(player_name)

    # 1) Direct backup: same-position starter out (and not the player itself)
    same_pos_out = any(
        (injuries_l.get(normalize_name(p))=="out") and (pos==player_pos)
        for p, pos in zip(team_players_df['Name'], team_players_df.get('Position', pd.Series(['']*len(team_players_df))))
    )
    if same_pos_out and injuries_l.get(name_l) != 'OUT':
        # measure closeness to starter using proj_min; heavier minutes -> larger bump
        if proj_min >= 28:
            boost += 0.06
        elif proj_min >= 20:
            boost += 0.10
        else:
            boost += 0.12

    # 2) Global usage missing redistribution (distribute a fraction of missing usage to all)
    # We only distribute a portion (30-40%) of missing usage to remaining players
    if ctx.get('usg_missing', 0.0) > 0:
        boost += ctx['usg_missing'] * 0.35

    # 3) Assist redistribution when PG out: wings benefit, backup PG benefits more
    if ctx.get('pg_out'):
        if player_pos in ('SG', 'SF'):
            boost += 0.03
        elif player_pos == 'PG' and injuries_l.get(name_l) != 'OUT':
            boost += 0.05

    # 4) Rebound redistribution when center(s) out
    if ctx.get('c_out'):
        if player_pos in ('C', 'PF'):
            boost += ctx['reb_missing'] * 0.6
        elif player_pos == 'SF':
            boost += ctx['reb_missing'] * 0.15

    # 5) Small salary-based tweak (cheap backups get a bit more)
    # Attempt to find salary to decide — safe fallback if not present
    try:
        row = team_players_df[team_players_df['Name'].apply(normalize_name)==name_l]
        #row = team_players_df[team_players_df['Name'].str.strip().str.lower() == player_name.strip().lower()]
        if not row.empty:
            salary = float(row.iloc[0].get('Salary', 5000))
            if salary < 4500:
                boost += 0.02
    except Exception:
        pass

    # Cap boost to avoid unrealistic spikes
    boost = max(0.0, min(boost, max_boost))

    return fpmin * (1.0 + boost)

# -------------------------
# Monte Carlo per-stat (for better floor/ceiling & variance)
# -------------------------
def monte_carlo_per_stat(
    player_logs: pd.DataFrame,
    n_sims: int = 2000,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:

    if player_logs is None or player_logs.empty:
        return {}

    rng = np.random.default_rng(seed)
    stat_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'TOV']

    # Collect sims per stat
    sims = {s: [] for s in stat_cols}

    for _ in range(n_sims):
        # 1️⃣ Sample a real game
        row = player_logs.sample(1).iloc[0]

        # 2️⃣ Correlated draw
        base = rng.normal(1.0, 0.12)

        draw = {
            'PTS':  row.get('PTS', 0)  * rng.normal(base, 0.08),
            'AST':  row.get('AST', 0)  * rng.normal(base, 0.10),
            'REB':  row.get('REB', 0)  * rng.normal(1.0, 0.12),
            'FG3M': row.get('FG3M', 0) * rng.normal(base, 0.15),
            'STL':  row.get('STL', 0)  * rng.normal(1.0, 0.25),
            'BLK':  row.get('BLK', 0)  * rng.normal(1.0, 0.25),
            'TOV':  row.get('TOV', 0)  * rng.normal(1.0, 0.20),
        }

        # 3️⃣ Store results
        for stat in stat_cols:
            sims[stat].append(max(0.0, float(draw.get(stat, 0.0))))

    # 4️⃣ Aggregate
    mc = {}
    for stat in stat_cols:
        arr = np.array(sims[stat])
        if len(arr) == 0:
            mc[stat] = {'floor': 0.0, 'ceiling': 0.0, 'std': 0.0}
            continue

        mc[stat] = {
            'floor': float(np.percentile(arr, 20)),
            'ceiling': float(np.percentile(arr, 90)),
            'std': float(np.std(arr, ddof=1))
        }

    return mc

# -------------------------
# Core projection class
# -------------------------
class SimpleNBAProjection:
    LEAGUE_AVG_PACE = 100.0

    def __init__(self, dk_salaries_path: str, days_of_history: int = 30, season: str = None):
        self.dk_path = dk_salaries_path
        self.days = days_of_history
        self.season = season or self.guess_season()
        self.dk_df: pd.DataFrame = None
        # team map for scoreboard
        self.team_map = {t['id']: t['abbreviation'] for t in teams.get_teams()}
        self.team_stats = {}
        self.todays_matchups = {}

    def guess_season(self) -> str:
        today = datetime.now()
        y = today.year
        return f"{y}-{str(y+1)[-2:]}" if today.month >= 10 else f"{y-1}-{str(y)[-2:]}"

    def load_dk_salaries(self) -> bool:
        try:
            df = pd.read_csv(self.dk_path)
            # map columns intelligently
            name_col, salary_col, team_col, pos_col = detect_dk_columns(df)
            if not name_col or not salary_col or not team_col:
                print("❌ Could not detect required columns in DK CSV. Found:", list(df.columns))
                return False
            # Standardize columns
            df = df.rename(columns={name_col: 'Name', salary_col: 'Salary', team_col: 'Team'})
            if pos_col:
                df = df.rename(columns={pos_col: 'Position'})
            # Trim & normalize
            df['Name'] = df['Name'].astype(str).str.strip()
            df['Team'] = df['Team'].astype(str).str.strip().str.upper()
            df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
            if 'Position' in df.columns:
                df['Position'] = df['Position'].astype(str).str.strip().str.upper().replace({'': None})
            # Drop invalid rows
            df = df.dropna(subset=['Name', 'Salary'])
            self.dk_df = df.reset_index(drop=True)
            print(f"✅ Loaded {len(self.dk_df)} players from DK salaries")
            return True
        except Exception as e:
            print("❌ Error loading DK CSV:", e)
            return False

    def _safe_api_call(self, func, *args, **kwargs):
        for attempt in range(3):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                wait = 2 ** attempt
                print(f"API call failed (attempt {attempt+1}): {e} — retrying in {wait}s")
                time.sleep(wait)
        return None

    def fetch_team_stats(self) -> bool:
        res = self._safe_api_call(leaguedashteamstats.LeagueDashTeamStats, season=self.season)
        if res is None:
            return False
        try:
            df = res.get_data_frames()[0]
            pace_col = 'PACE' if 'PACE' in df.columns else None
            team_col = 'TEAM_ABBREVIATION' if 'TEAM_ABBREVIATION' in df.columns else ('TEAM_NAME' if 'TEAM_NAME' in df.columns else None)
            if not team_col:
                return False
            for _, r in df.iterrows():
                abbr = r[team_col]
                pace = r[pace_col] if pace_col else self.LEAGUE_AVG_PACE
                def_rating = r['DEF_RATING'] if 'DEF_RATING' in df.columns else (r['DRTG'] if 'DRTG' in df.columns else 110.0)
                self.team_stats[abbr] = {'pace': pace, 'def_rating': def_rating}
            if pace_col:
                self.LEAGUE_AVG_PACE = float(df[pace_col].mean())
            return True
        except Exception as e:
            print("⚠️ Error parsing team stats:", e)
            return False
        
    def fetch_todays_matchups(self) -> bool:
        res = self._safe_api_call(scoreboardv2.ScoreboardV2)
        if res is None:
            return False
        try:
            games_df = res.get_data_frames()[0]
            for _, g in games_df.iterrows():
                hid = g['HOME_TEAM_ID']; vid = g['VISITOR_TEAM_ID']
                home_abbr = self.team_map.get(hid); away_abbr = self.team_map.get(vid)
                if home_abbr and away_abbr:
                    self.todays_matchups[home_abbr] = {'opponent': away_abbr}
                    self.todays_matchups[away_abbr] = {'opponent': home_abbr}
            return True
        except Exception as e:
            print("⚠️ Error parsing scoreboard:", e)
            return False

    def fetch_recent_game_logs(self) -> pd.DataFrame:
        end = datetime.now()
        start = end - timedelta(days=self.days)
        res = self._safe_api_call(playergamelogs.PlayerGameLogs,
                                  season_nullable=self.season,
                                  date_from_nullable=start.strftime('%Y-%m-%d'),
                                  date_to_nullable=end.strftime('%Y-%m-%d'))
        if res is None:
            return pd.DataFrame()
        df = res.get_data_frames()[0]
        # Ensure required columns
        needed = ['PLAYER_ID', 'PLAYER_NAME', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M', 'GAME_DATE', 'TEAM', 'MATCHUP']
        for c in needed:
            if c not in df.columns:
                df[c] = 0
        # Lowercase player name for matching
        df['PLAYER_NAME_L'] = df['PLAYER_NAME'].apply(normalize_name)
        #df['PLAYER_NAME_L'] = df['PLAYER_NAME'].astype(str).str.lower().str.strip()
        # Filter to DK slate players
        dk_names = set(self.dk_df['Name'].apply(normalize_name))
        #dk_names = set(self.dk_df['Name'].str.lower())
        df = df[df['PLAYER_NAME_L'].isin(dk_names)].copy()
        return df

    @staticmethod
    def calculate_dk_points_from_row(row: pd.Series) -> float:
        return (
            row.get('PTS', 0) +
            1.2 * row.get('REB', 0) +
            1.5 * row.get('AST', 0) +
            3.0 * row.get('STL', 0) +
            3.0 * row.get('BLK', 0) -
            1.0 * row.get('TOV', 0) +
            0.5 * row.get('FG3M', 0)
        )

    def compute_fp_per_min(self, logs_df: pd.DataFrame) -> Dict[int, Dict]:
        out = {}
        grouped = logs_df.sort_values('GAME_DATE', ascending=False).groupby('PLAYER_ID')
        for pid, g in grouped:
            mins = []
            fps = []
            for _, row in g.iterrows():
                raw_min = row['MIN']
                try:
                    if isinstance(raw_min, str) and ':' in raw_min:
                        mm, ss = raw_min.split(':')
                        m = float(mm) + float(ss)/60.0
                    else:
                        m = float(raw_min)
                except Exception:
                    m = 0.0
                mins.append(m)
                fps.append(self.calculate_dk_points_from_row(row))
            if sum(mins) == 0:
                continue
            fp_min_list = [fp/m for fp, m in zip(fps, mins) if m > 2]
            if not fp_min_list:
                continue
            fp_per_min = float(np.mean(fp_min_list)) if len(fp_min_list) > 0 else 0.0
            recent_minutes = float(np.median(mins[:10])) if len(mins) > 0 else 0.0
            last5_min = float(np.mean(mins[:5])) if len(mins) >= 1 else recent_minutes
            last10_min = float(np.mean(mins[:10])) if len(mins) >= 1 else recent_minutes
            out[int(pid)] = {
                'fp_per_min': fp_per_min,
                'recent_minutes': recent_minutes,
                'last5_min': last5_min,
                'last10_min': last10_min,
                'games': len(mins),
                'fp_min_list': fp_min_list,
                'min_list': mins,
                'logs_df': g
            }
        return out

    def project_minutes(self, player_stats: Dict) -> float:
        last5 = player_stats.get('last5_min', 0.0)
        last10 = player_stats.get('last10_min', 0.0)
        recent = player_stats.get('recent_minutes', 0.0)
        projected = 0.4 * last5 + 0.25 * last10 + 0.35 * recent
        return float(max(0.0, min(48.0, projected)))
    
    
    def matchup_multiplier(self, team_abbr: str) -> float:
        """
        Adjust multiplier based on opponent's defense and pace.
        More extreme values produce noticeable changes in projections.
        """
        #print(f"[DEBUG] matchup_multiplier called for {team_abbr}")
        #print(f"[DEBUG] todays_matchups keys: {list(self.todays_matchups.keys())}")
        opp = self.todays_matchups.get(team_abbr, {}).get('opponent')
        #print(f"[DEBUG] No matchup found for {team_abbr}, returning 1.0")
        if not opp:
            return 1.0

        team_stats = self.team_stats.get(team_abbr, {})
        opp_stats = self.team_stats.get(opp, {})
        #print(f"[DEBUG] team_stats: {team_stats}, opp_stats: {opp_stats}")

        team_pace = team_stats.get('pace', self.LEAGUE_AVG_PACE)
        opp_pace = opp_stats.get('pace', self.LEAGUE_AVG_PACE)
        opp_def = opp_stats.get('def_rating', 110.0)

        # Pace adjustment: how fast the game is vs league average
        pace_adj = ((team_pace + opp_pace) / 2.0) / self.LEAGUE_AVG_PACE

        # Defensive adjustment: worse defense -> more points
        def_adj = 110.0 / float(opp_def)

        # Amplify the effect: instead of 50/50 blend, use weighted multiplication
        multiplier = pace_adj ** 0.6 * def_adj ** 0.4

        # Clamp to reasonable extremes
        multiplier = max(0.85, min(1.20, multiplier))
        #print(f"[DEBUG] {team_abbr} vs {opp}: pace_adj={pace_adj}, def_adj={def_adj}, multiplier={multiplier}")

        # Debug print to verify
        #print(f"[DEBUG] {team_abbr} vs {opp}: pace_adj={pace_adj:.3f}, def_adj={def_adj:.3f}, multiplier={multiplier:.3f}")

        return multiplier

    def cap_projection_by_salary(self, projection: float, salary: float) -> float:
        cap = salary * 0.0068
        return min(projection, cap)

    def run(self, save_csv: str = None, injuries: Dict[str, str] = None, n_sims: int = 1500) -> pd.DataFrame:
        """
        Main pipeline:
        - needs self.dk_df loaded
        - injuries: dict of lowercase name -> status (e.g., 'OUT'/'QUESTIONABLE')
        """
        if self.dk_df is None:
            raise RuntimeError("DK salaries not loaded")
        injuries = {k.lower(): v for k, v in (injuries or {}).items()}

        # Fetch extras
        raw_vegas = get_nba_odds()
        vegas = normalize_vegas_data(raw_vegas)
        self.fetch_team_stats()
        self.fetch_todays_matchups()
        logs = self.fetch_recent_game_logs()
        if logs.empty:
            print("⚠️ No recent logs returned — aborting")
            return pd.DataFrame()

        player_fpmin = self.compute_fp_per_min(logs)
        # map names -> player ids
        name_to_pids = {}
        for pid, g in logs.groupby('PLAYER_ID'):
            #nm = g.iloc[0]['PLAYER_NAME'].strip()
            nm = normalize_name(g.iloc[0]['PLAYER_NAME'])
            name_to_pids.setdefault(nm.lower(), []).append(int(pid))

        results = []
        for _, r in self.dk_df.iterrows():
            name = r['Name'].strip()
            name_l = normalize_name(name)
            salary = float(r['Salary'])
            team = r.get('Team', None)
            pos = r.get('Position', '')
            pids = name_to_pids.get(name_l, [])
            if not pids:
                continue
            pid = pids[0]
            stats = player_fpmin.get(pid)
            if not stats:
                continue
            # base fp/min & minutes
            fpmin = stats['fp_per_min']
            # fallback: use trimmed mean if zero-ish
            if not np.isfinite(fpmin) or fpmin <= 0:
                fpmin = float(np.mean(stats.get('fp_min_list', [0.5])))
            base_min = self.project_minutes(stats)
            # injury minutes adjustment (if player himself has injury status)
            player_status = injuries.get(name_l)
            if player_status:
                base_min = apply_injury_minutes_adjustment(base_min, player_status)
            # dynamic usage redistribution based on team injuries & position
            team = normalize_team(r['Team'])
            matchup_mult = self.matchup_multiplier(team) if team else 1.0
            vegas_mult = vegas_multiplier(team, vegas)
            fpmin = dynamic_usage_redistribution(name, pos, fpmin, base_min, injuries, self.dk_df)
            # matchup & vegas multipliers
            mult = matchup_mult * vegas_mult
            #raw_proj = fpmin * base_min * mult ###Old line THis is to change rawprojection so it show projection before adding multiplier
            raw_proj = fpmin * base_min
            proj_with_mult = raw_proj * mult
            fp_min_list = stats.get('fp_min_list', [])  #added 12-16
            vol_tier = classify_volatility(fp_min_list)     #added 12-16
            #capped_proj = self.cap_projection_by_salary(raw_proj, salary) ###old line THis is to change rawprojection so it show projection before adding multiplier
            capped_proj = self.cap_projection_by_salary(proj_with_mult, salary)
            # Monte Carlo per-minute floor/ceiling: use empirical fp_min_list and min_list
            mc_simple = self.monte_carlo(fp_min_list=stats.get('fp_min_list', []),
                                         min_list=stats.get('min_list', []),
                                         matchup_mult=matchup_mult, projected_minutes=base_min, vegas_mult=vegas_mult, n_sims=n_sims)
            # Monte Carlo per-stat for floor/ceiling derivation (optional, expensive)
            per_stat_mc = monte_carlo_per_stat(stats.get('logs_df', pd.DataFrame()), n_sims=n_sims)
            # Convert per-stat floors/ceilings to DK FP floor/ceiling with scoring weights
            stat_weights = {'PTS': 1.0, 'REB': 1.2, 'AST': 1.5, 'STL': 3.0, 'BLK': 3.0, 'FG3M': 0.5, 'TOV': -1.0}
            floor_stat = sum(per_stat_mc[s]['floor'] * stat_weights.get(s, 0.0) for s in per_stat_mc)
            ceil_stat = sum(per_stat_mc[s]['ceiling'] * stat_weights.get(s, 0.0) for s in per_stat_mc)
            floor = max(
                0.65 * mc_simple['floor'] + 0.35 * floor_stat,
                0.0
            )
            ceiling = max(
                0.35 * mc_simple['ceiling'] + 0.65 * ceil_stat,
                0.0
            )
            volatility = mc_simple.get('volatility_std', 0.0)
            mc_mean = mc_simple.get('mean', fpmin * base_min)
            mean_proj = capped_proj                 #added 12-16
            std = volatility                    #added 12-16
            vol_tier = classify_volatility(stats.get('fp_min_list', []))       #added 12-16
            sims = mc_simple.get('sims', np.array([]))    #added 12-16


            p_6x = float(np.mean(sims >= salary * 0.006)) if sims.size else 0.0     #added 12-16
            p_8x = float(np.mean(sims >= salary * 0.008)) if sims.size else 0.0     #added 12-16

            leverage_score = (ceiling - mean_proj) / max(1.0, std)          #added 12-16
            results.append({
                'Name': name,
                'PlayerID': pid,
                'Team': team,
                'Position': pos,
                'Salary': salary,
                'Projection': round(capped_proj, 1),
                'RawProjection': round(raw_proj, 1),
                'FP_per_min': round(fpmin, 3),
                'ProjMin': round(base_min, 1),
                'Multiplier': round(mult, 3),
                'InjuryStatus': player_status or 'None',
                'Games': stats.get('games', 0),
                'VolatilityTier': vol_tier,                 #just added 12-16
                'BoomScore': round(leverage_score, 2),         #just added 12-16
                'CashScore': round(mean_proj / max(1.0, std), 2),       #added 12-16
                'Floor_MC': round(floor, 1),
                'Ceiling_MC': round(ceiling, 1),
                'Volatility_STD': round(volatility, 2),
                'MC_Mean': round(mc_mean, 1),
                'P_6x': round(p_6x, 3),             #added 12-16
                'P_8x': round(p_8x, 3)              #added 12-16
            })

        out_df = pd.DataFrame(results).sort_values('Projection', ascending=False)
        out_df['BoomScore'] = out_df['BoomScore']
        out_df.fillna(0.0, inplace=True)
        proj_mean = out_df['Projection'].mean()
        salary_z = (out_df['Salary'] - out_df['Salary'].mean()) / out_df['Salary'].std()
        min_z = (out_df['ProjMin'] - out_df['ProjMin'].mean()) / out_df['ProjMin'].std()
        
        shrink = (
            0.78
            + 0.06 * salary_z.clip(-1.5, 1.5)
            + 0.04 * min_z.clip(-1.5, 1.5)
        )
        
        shrink = shrink.clip(0.70, 0.90)
        out_df['Projection'] = (
        proj_mean + 0.88 * (out_df['Projection'] - proj_mean)
        )
        out_df['Projection_bc'] = out_df['Projection'] * BIAS_CORRECTION

        # --- Volatility-tier specific z values ---
        Z_BY_TIER = {
            'LOW': 1.7,
            'MED': 2.0,
            'HIGH': 2.4
        }
        # Position ceiling adjustment
        POS_Z_ADJ = {
            'PG': 0.15,
            'SG': 0.10,
            'SF': 0.05,
            'PF': -0.05,
            'C': -0.15
        }
        out_df['z_base'] = out_df['VolatilityTier'].map(Z_BY_TIER).fillna(2.0)

        out_df['z_pos'] = (
            out_df['Position']
            .str.split('/')
            .str[0]
            .map(POS_Z_ADJ)
            .fillna(0.0)
        )

        out_df['z_vol'] = out_df['z_base'] + out_df['z_pos']
        # Map z per player
        out_df['z_vol'] = out_df['VolatilityTier'].map(Z_BY_TIER)

        # Safety fallback (should never trigger, but protects pipeline)
        out_df['z_vol'] = out_df['z_vol'].fillna(2.0)

        # Minutes volatility dampener
        out_df['MinFactor'] = np.clip(
            36 / out_df['ProjMin'],
            0.85,
            1.25
        )

        out_df['Volatility_STD_adj'] = (
            out_df['Volatility_STD'] * out_df['MinFactor']
        )

        # ------------------------
        # FINAL FLOOR / CEILING (SINGLE SOURCE OF TRUTH)
        # ------------------------

        z = 2.1  # DFS realistic

        vol = out_df['Volatility_STD'].clip(lower=1.0)

        out_df['Ceiling_MC'] = (
        out_df['Projection_bc'] + out_df['z_vol'] * out_df['Volatility_STD_adj']
        )

        out_df['Floor_MC'] = (
        out_df['Projection_bc'] - out_df['z_vol'] * out_df['Volatility_STD_adj']
        )

        #------------------------
        # salary dampener
        #------------------------
        salary_norm = out_df['Salary'] / out_df['Salary'].max()

        out_df['BoomSalaryAdj'] = (
            0.25 + 0.75 * (1 - salary_norm ** 0.85)
        )
        # ============================
        # DFS-Adjusted BoomScore
        # ============================

        raw_boom = (
            (out_df['Ceiling_MC'] - out_df['Projection_bc']) /
            out_df['Projection_bc']
        )

        # Salary pressure penalty (key fix)
        #salary_penalty = (out_df['Salary'] / 5000).clip(1.0, 3.0)
        salary_penalty = (out_df['Salary'] / 7000).clip(0.85, 1.75)

        out_df['BoomScore'] = (
            raw_boom / salary_penalty
        ).clip(0, 1.25)

        # Volatility tier scaling (keep, but softer)
        out_df['BoomScore'] *= out_df['VolatilityTier'].map({
            'LOW': 0.75,
            'MED': 1.0,
            'HIGH': 1.15
        }).fillna(1.0)

        out_df['BoomScore'] = out_df['BoomScore'] ** 0.85       #added 12-23
        
        #applying dampener to boomscore
        out_df['BoomScore'] = (
            out_df['BoomScore'] *
            out_df['BoomSalaryAdj']
        )

        # ------------------------
        # NBA REALITY CAPS
        # ------------------------

        # No one exceeds ~95–105 in realistic NBA DFS
        out_df['Ceiling_MC'] = np.minimum(
            out_df['Ceiling_MC'],
            out_df['Projection'] + 1.0 * out_df['ProjMin']
        )

        # Absolute sanity cap
        out_df['Ceiling_MC'] = np.minimum(
            out_df['Ceiling_MC'],
            out_df['Projection'] * 1.9
        )

        # Floor protection
        out_df['Floor_MC'] = np.maximum(
            out_df['Floor_MC'],
            out_df['Projection'] * 0.45
        )

        out_df['Floor_MC'] = out_df['Floor_MC'].clip(lower=0.0)

        # Ensure floors are non-negative
        out_df['Floor_MC'] = out_df['Floor_MC'].clip(lower=0.0)
    
        minute_factor = np.clip(out_df['ProjMin'] / 30, 0.4, 1.1)           #added 12-22
        out_df['Floor_MC'] *= minute_factor #added 12-22
        out_df['Floor_MC'] = out_df['Floor_MC'].clip(lower=0)   #added 12-22
        out_df['MinRisk'] = 1 / np.sqrt(out_df['ProjMin'].clip(lower=8))
        out_df['MinRisk'] *= (1 - 0.4 * out_df['BoomScore'])        #added 12-23
        out_df['MinRisk'] = out_df['MinRisk'].clip(0.05, 1.0)       #added 12-23
        out_df = estimate_ownership(out_df) #added 12-16

        # ------------------------
        # Z-score helpers (FIRST)
        # ------------------------
        def zscore(s):
            std = s.std(ddof=0)
            if std == 0 or not np.isfinite(std):
                return pd.Series(0.0, index=s.index)
            return (s - s.mean()) / std

        out_df['z_proj'] = zscore(out_df['Projection'])
        out_df['z_value'] = zscore(out_df['Projection'] / out_df['Salary'])
        out_df['z_minutes'] = zscore(out_df['ProjMin'])
        out_df['z_ceil'] = zscore(out_df['Ceiling_MC'])

        # ------------------------
        # Ownership estimation
        # ------------------------

        out_df['SalaryBias'] = out_df['Salary'].apply(salary_bias)
        out_df['OwnershipScore_Cash'] = (
            0.40 * out_df['z_value'] +
            0.30 * out_df['z_minutes'] +
            0.20 * out_df['z_proj'] +
            0.10 * (-out_df['Volatility_STD'])
        )

        out_df['OwnershipScore_GPP'] = (
            0.30 * out_df['z_proj'] +
            0.30 * out_df['z_ceil'] +
            0.15 * out_df['SalaryBias'] +
            0.15 * out_df['BoomScore'] +
            0.10 * out_df['z_minutes']
        )

        exp = np.exp(out_df['OwnershipScore_GPP'] - out_df['OwnershipScore_GPP'].max())
        out_df['OwnershipProb'] = exp / exp.sum()
        out_df['OwnershipPct'] = 100 * out_df['OwnershipProb']

        # Step 2: human-readable percentages
        out_df['OwnershipPct'] = 100 * out_df['OwnershipProb']

        # Step 3: optional: clip extremes
        out_df['OwnershipPct'] = np.clip(out_df['OwnershipPct'], 0.5, 40)

        # Step 4: renormalize so percentages sum to ~100
        out_df['OwnershipPct'] *= 100 / out_df['OwnershipPct'].sum()

        # Step 5: update probabilities to match clipped/renormalized percentages
        out_df['OwnershipProb'] = out_df['OwnershipPct'] / 100.0

        # Optional display column for human-readable output
        out_df['OwnershipPctDisplay'] = out_df['OwnershipPct'].apply(lambda x: f"{x:.1f}%")

        pos_counts = out_df['Position'].value_counts()
        
        out_df['PosScarcity'] = out_df['Position'].map(
            lambda p: 1.0 / pos_counts.get(p, 1)
        )

        out_df['OwnershipPct'] *= (1 + 0.12 * out_df['PosScarcity'])
        out_df['SalaryCluster'] = (out_df['Salary'] % 1000 == 0).astype(int)
        out_df['OwnershipPct'] *= (1 + 0.05 * out_df['SalaryCluster'])
        proj_rank = out_df['Projection'].rank(pct=True)
        out_df['CliffBoost'] = (proj_rank > 0.90).astype(int)

        out_df['OwnershipPct'] *= (1 + 0.15 * out_df['CliffBoost'])
        out_df['OwnershipPct'] *= 100.0 / out_df['OwnershipPct'].sum()
        out_df['OwnershipProb'] = out_df['OwnershipPct'] / 100.0

        # ------------------------
        # GPP ALPHA (Upside vs Ownership)
        # ------------------------
        out_df['GPP_Alpha'] = (
            out_df['BoomScore'] *
            (1 - out_df['OwnershipProb'])
        )
        out_df.sort_values('GPP_Alpha', ascending=False)
        out_df['BoomScore'].describe()

        def softmax_pct(s):
            exp = np.exp(s - s.max())
            return 100 * exp / exp.sum()

        out_df['CashOwnershipPct'] = softmax_pct(out_df['OwnershipScore_Cash'])
        out_df['GPPOwnershipPct']  = softmax_pct(out_df['OwnershipScore_GPP'])
        
        # ------------------------
        # GPP / leverage metrics
        # ------------------------
        
        out_df['Leverage'] = (
            out_df['P_8x'] / np.clip(out_df['OwnershipProb'], 1e-4, None)
        )

        out_df['ChalkRisk'] = out_df['OwnershipProb'] * (1.0 - out_df['P_6x'])
        
        # ========================================
        # GPP Tier Assignment (drop code here)
        # ========================================
        out_df['BoomPct'] = out_df['BoomScore'].rank(pct=True)
        # Assign tier
        def assign_gpp_tier(pct):
            if pct >= 0.9:
                return 'Core'
            elif pct >= 0.6:
                return 'Secondary'
            else:
                return 'Sprinkle'
        
        out_df['GPP_Tier'] = out_df['BoomPct'].apply(assign_gpp_tier).astype(str)
        
        # Optional Mini-Sprinkle for cheap players (<$8500)
        out_df.loc[out_df['Salary'] < 8500, 'GPP_Tier'] = out_df.loc[out_df['Salary'] < 8500, 'GPP_Tier'].apply(
            lambda x: 'Mini-Sprinkle' if x != 'Core' else x
        )

        # out_df['GPP_Tier'] = pd.qcut(
        #     out_df['BoomScore'], 
        #     q=[0, 0.6, 0.9, 1.0],
        #     labels=['Sprinkle', 'Secondary', 'Core'],
        #     duplicates='drop'
        # ).astype(str)  # convert to string to allow new category

        # # Then apply Mini-Sprinkle logic safely
        # out_df.loc[out_df['Salary'] < 8500, 'GPP_Tier'] = out_df.loc[out_df['Salary'] < 8500, 'GPP_Tier'].apply(
        #     lambda x: 'Mini-Sprinkle' if x != 'Core' else x
        # )

        # # Optional ownership cap for Core
        # out_df.loc[(out_df['GPP_Tier'] == 'Core') & (out_df['Ownership'] > 0.25), 'GPP_Tier'] = 'Secondary'

        # Drop temp column if you want
        out_df.drop(columns=['BoomPct'], inplace=True)
        #df = out_df.copy()  # or just use out_df if you prefer
        #df['BoomPct'] = df['BoomScore'].rank(pct=True)
#
        #def assign_gpp_tier(pct):
        #    if pct >= 0.75:
        #        return 'Core'
        #    elif pct >= 0.40:
        #        return 'Secondary'
        #    else:
        #        return 'Sprinkle'
#
        #df['GPP_Tier'] = df['BoomPct'].apply(assign_gpp_tier)
#
        ## Optional Mini-Sprinkle for cheap players (<$8500)
        #df.loc[df['Salary'] < 8500, 'GPP_Tier'] = df.loc[df['Salary'] < 8500, 'GPP_Tier'].apply(
        #    lambda x: 'Mini-Sprinkle' if x != 'Core' else x
        #)
#
        ## Optional ownership cap for Core
        #df.loc[(df['GPP_Tier'] == 'Core') & (df['Ownership'] > 0.25), 'GPP_Tier'] = 'Secondary'
#
        ## Drop temp column if you want
        #df.drop(columns=['BoomPct'], inplace=True)

        if save_csv:
             out_df.to_csv(save_csv, index=False)
             print(f"✅ Saved projections to {save_csv}")
        return out_df
    
    def monte_carlo(
        self,
        fp_min_list,
        min_list,
        matchup_mult,
        vegas_mult,
        projected_minutes,
        n_sims=2000
    ):
        """
        Monte Carlo with:
          1. Role-based noise scaling
          2. Skewed lognormal randomness (heavy upside)
        """

        fp_vals = np.array([v for v in fp_min_list if np.isfinite(v) and v > 0])
        min_vals = np.array([m for m in min_list if np.isfinite(m) and m > 0])
        vol_tier = classify_volatility(fp_min_list) #just added 12-16
        vol_mult = VOLATILITY_MULTIPLIER.get(vol_tier, 1.0) #just added 12-16

        if len(fp_vals) == 0 or len(min_vals) == 0:
            return {'floor': 0.0, 'ceiling': 0.0, 'volatility_std': 0.0}

        # --- 1. Determine noise level by role -------
        if projected_minutes > 34:
            noise_pct = 0.03   # superstars
        elif projected_minutes > 28:
            noise_pct = 0.05   # safe starters
        else:
            noise_pct = 0.10   # bench / volatile

        rng = np.random.default_rng(42)
        sims = []

        for _ in range(n_sims):
            fp_per_min = rng.choice(fp_vals)
            minutes = rng.choice(min_vals)
            fp_per_min *= rng.normal(1.0, 0.08 * vol_mult) #just added 12-16

            # --- 2. Skewed upside noise using lognormal ------
            noise = rng.lognormal(mean=0, sigma=noise_pct)

            sims.append(fp_per_min * minutes * matchup_mult * vegas_mult * noise)

        sims = np.array(sims)

        return {
            'floor': float(np.percentile(sims, 18)),
            'ceiling': float(np.percentile(sims, 88)),
            'volatility_std': float(np.std(sims, ddof=1)),
            'sims': sims     #added 12-16  
              }

# -------------------------
# Top-level helper for CLI
# -------------------------
def apply_injury_minutes_adjustment(base_minutes: float, injury_status: str) -> float:
    if not injury_status:
        return base_minutes
    s = injury_status.upper()
    adj = INJURY_MINUTE_ADJUSTMENTS.get(s, 0)
    return max(0.0, base_minutes + adj)

# -------------------------
# CLI entrypoint
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Simple NBA DFS projections with injuries & Monte Carlo")
    parser.add_argument('--salaries', required=True, help='DraftKings salaries CSV (your header supported)')
    parser.add_argument('--out', default='projections.csv', help='Optional output CSV path')
    parser.add_argument('--days', type=int, default=30, help='Days of history for logs (default 30)')
    parser.add_argument('--n_sims', type=int, default=1500, help='Monte Carlo simulations per player (default 1500)')
    parser.add_argument('--no-inj', dest='use_injuries', action='store_false', help='Disable scraping ESPN injuries')
    args = parser.parse_args()

    model = SimpleNBAProjection(dk_salaries_path=args.salaries, days_of_history=args.days)
    if not model.load_dk_salaries():
        print("❌ Failed to load DraftKings CSV — aborting")
        return

    injuries = {}
    if args.use_injuries:
        print("🔎 Scraping ESPN injuries...")
        injuries = get_injuries()
        print(f"  → got {len(injuries)} injury entries")

    print("🔎 Generating projections (this may take a minute)...")
    df = model.run(save_csv=args.out, injuries=injuries, n_sims=args.n_sims)
    if df.empty:
        print("❌ No projections created.")
    else:
        print("✅ Done — top 10 projections:")
        print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
