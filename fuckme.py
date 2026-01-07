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
import pulp
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
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

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
DK_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

def eligible(player_pos, slot):
    if slot == "UTIL":
        return True
    if slot == "G":
        return player_pos in ("PG", "SG")
    if slot == "F":
        return player_pos in ("SF", "PF")
    return player_pos == slot
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
) -> Dict[str, float]:

    if player_logs is None or player_logs.empty:
        return {
            'mean': 0.0,
            'floor': 0.0,
            'ceiling': 0.0,
            'std': 0.0,
            'sims': np.array([])
        }

    rng = np.random.default_rng(seed)

    stat_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'TOV']

    # DraftKings weights
    DK_WEIGHTS = {
        'PTS': 1.0,
        'REB': 1.25,
        'AST': 1.5,
        'STL': 2.0,
        'BLK': 2.0,
        'FG3M': 0.5,
        'TOV': -0.5
    }

    sims = []

    for _ in range(n_sims):

        # 1️⃣ Sample a real game (preserves correlations)
        row = player_logs.sample(1).iloc[0]

        # 2️⃣ Correlated volatility draw (KEEPING YOUR LOGIC)
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

        # Clamp negatives
        for k in draw:
            draw[k] = max(0.0, float(draw[k]))

        # 3️⃣ Base DK fantasy points
        dk_points = sum(
            draw[s] * DK_WEIGHTS[s]
            for s in DK_WEIGHTS
        )

        # 4️⃣ Double / Triple Double bonuses
        categories = [
            draw['PTS'] >= 10,
            draw['REB'] >= 10,
            draw['AST'] >= 10,
            draw['STL'] >= 10,
            draw['BLK'] >= 10
        ]

        count = sum(categories)

        if count >= 3:
            dk_points += 3.0
        elif count >= 2:
            dk_points += 1.5

        sims.append(dk_points)

    sims = np.array(sims)

    return {
        'mean': float(np.mean(sims)),
        'floor': float(np.percentile(sims, 20)),
        'ceiling': float(np.percentile(sims, 90)),
        'std': float(np.std(sims, ddof=1)),
        'sims': sims
    }
def evaluate_lineup_mc(lineup, n_sims=3000):
    # Stack player sims
    sims = []

    for _, p in lineup.iterrows():
        if "MC_Mean" not in p or "Volatility_STD" not in p:
            return None  # invalid lineup

        sims.append(
            np.random.normal(
                loc=p["MC_Mean"],
                scale=p["Volatility_STD"],
                size=n_sims
            )
        )

    lineup_sims = np.sum(sims, axis=0)

    return {
        "Mean": lineup_sims.mean(),
        "P90": np.percentile(lineup_sims, 90),
        "P95": np.percentile(lineup_sims, 95),
        "P99": np.percentile(lineup_sims, 99),
        "P_Cash": np.mean(lineup_sims >= 280),
        "P_Top1": np.mean(lineup_sims >= 330),
    }

#def evaluate_lineup_mc(
#    lineup: pd.DataFrame,
#    n_sims: int = 5000,
#    cash_line: float = 270.0,     # adjust per slate
#    rng: np.random.Generator | None = None
#) -> dict:
#
#    if rng is None:
#        rng = np.random.default_rng()
#
#    # Pull per-player distribution params
#    means = lineup["MC_Mean"].values
#    stds  = lineup["Volatility_STD"].values
#
#    sims = []
#
#    for _ in range(n_sims):
#        draw = rng.normal(means, stds)
#        sims.append(draw.sum())
#
#    sims = np.array(sims)
#
#    return {
#        "Lineup_Mean": float(np.mean(sims)),
#        "Lineup_P90": float(np.percentile(sims, 90)),
#        "Lineup_P95": float(np.percentile(sims, 95)),
#        "Lineup_P99": float(np.percentile(sims, 99)),
#        "P_Cash": float(np.mean(sims >= cash_line)),
#        "P_Top1": float(np.mean(sims >= np.percentile(sims, 99)))
#    }

def score_lineup(lineup: pd.DataFrame) -> dict:
    return {
        "LineupProj": lineup["Projection"].sum(),
        "LineupCeiling": lineup["Ceiling_MC"].sum(),
        "LineupFloor": lineup["Floor_MC"].sum(),
        "LineupSalary": lineup["Salary"].sum()
    }
def evaluate_lineups_mc(lineups):
    rows = []

    for i, lineup in enumerate(lineups, start=1):
        stats = evaluate_lineup_mc(lineup)

        if stats is None:
            #continue  # 🔑 THIS IS THE KEY
            print("⚠️ evaluate_lineup_mc returned None")

        row = stats.to_dict() if isinstance(stats, pd.Series) else dict(stats)
        row["Lineup"] = i
        rows.append(row)

    if not rows:
        print("❌ No lineups survived evaluate_lineup_mc()")
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("Lineup")

#def evaluate_lineups_mc(lineups):
#    rows = []
#
#    for i, lineup in enumerate(lineups, start=1):
#        stats = evaluate_lineup_mc(lineup)
#
#        row = stats.to_dict() if isinstance(stats, pd.Series) else dict(stats)
#        row["Lineup"] = i
#
#        rows.append(row)
#
#    df = pd.DataFrame(rows).set_index("Lineup")
#    return df

#def evaluate_lineups_mc(lineups: list[pd.DataFrame]) -> pd.DataFrame:
#    """
#    Evaluate multiple lineups using Monte Carlo.
#    Wrapper around evaluate_lineup_mc().
#    """
#
#    rows = []
#
#    for i, lineup in enumerate(lineups, start=1):
#        stats = evaluate_lineup_mc(lineup)
#        row = stats.to_dict() if isinstance(stats, pd.Series) else dict(stats)
#        row["Lineup"] = i
#        rows.append(row)
#
#        #stats["Lineup"] = float(i)
#        #rows.append(stats)
#
#    df = pd.DataFrame(rows).set_index("Lineup")
#    return df
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
        # Filter to DK slate players
        dk_names = set(self.dk_df['Name'].apply(normalize_name))
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
        opp = self.todays_matchups.get(team_abbr, {}).get('opponent')
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
            per_stat_mc = monte_carlo_per_stat(
                stats.get('logs_df', pd.DataFrame()),
                n_sims=n_sims
            )

            floor_stat = per_stat_mc['floor']
            ceil_stat  = per_stat_mc['ceiling']
            mc_mean    = per_stat_mc['mean']
            sims       = per_stat_mc['sims']

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

        #out_df['z_vol'] = out_df['z_base'] + out_df['z_pos']
        # Map z per player
        #out_df['z_vol'] = out_df['VolatilityTier'].map(Z_BY_TIER)

        # Safety fallback (should never trigger, but protects pipeline)
        #out_df['z_vol'] = out_df['z_vol'].fillna(2.0)

        out_df['z_vol'] = (
            out_df['VolatilityTier'].map(Z_BY_TIER).fillna(2.0)
            + out_df['Position']
                .str.split('/')
                .str[0]
                .map(POS_Z_ADJ)
                .fillna(0.0)
        )
        # Minutes volatility dampener
        out_df['MinFactor'] = np.clip(
            36 / out_df['ProjMin'],
            0.85,
            1.25
        )

        out_df['Volatility_STD_adj'] = (
            out_df['Volatility_STD'] * out_df['MinFactor']
        )
        out_df['Volatility_STD_adj'] = out_df['Volatility_STD_adj'].clip(
            lower=1.0,
            upper=0.65 * out_df['Projection']
        )


        # ------------------------
        # FINAL FLOOR / CEILING (SINGLE SOURCE OF TRUTH)
        # ------------------------

        #z = 2.1  # DFS realistic
#
        #vol = out_df['Volatility_STD'].clip(lower=1.0)
#
        #out_df['Ceiling_MC'] = (
        #out_df['Projection_bc'] + out_df['z_vol'] * out_df['Volatility_STD_adj']
        #)
#
        #out_df['Floor_MC'] = (
        #out_df['Projection_bc'] - out_df['z_vol'] * out_df['Volatility_STD_adj']
        #)

        # ------------------------
        # ASYMMETRIC VOLATILITY SIGNALS
        # ------------------------

        out_df['Min_Vol'] = np.clip(
            out_df['Volatility_STD'] / out_df['Projection'].clip(lower=5),
            0.15,
            1.0
        )

        # Usage proxy: ceiling leverage
        out_df['Usage_Vol'] = np.clip(
            (out_df['Ceiling_MC'] - out_df['Projection']) /
            out_df['Projection'].clip(lower=5),
            0.2,
            1.5
        )
        tail_boost = (
            1.0
            + 0.65 * out_df['Usage_Vol']
            + 0.35 * out_df['Min_Vol']
        )
        base_ceiling = (
        out_df['Projection_bc'] +
            1.0 * out_df['Volatility_STD_adj']
        )

        out_df['Ceiling_MC'] = base_ceiling * tail_boost
        out_df['Floor_MC'] = (
            out_df['Projection_bc'] -
            1.25 * out_df['Volatility_STD_adj']
        ).clip(lower=0.0)

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
        #out_df['Ceiling_MC'] = np.minimum(
        #    out_df['Ceiling_MC'],
        #    out_df['Projection'] + 3.0 * out_df['Volatility_STD']
        #)
        out_df['Ceiling_MC'] = (
            out_df['Projection_bc']
            + out_df['z_vol'] * out_df['Volatility_STD_adj']
        )
        soft_cap = (
            out_df['Projection'] + 1.75 * out_df['ProjMin']
        )

        out_df['Ceiling_MC'] = np.where(
            out_df['Ceiling_MC'] > soft_cap,
            soft_cap + 0.15 * (out_df['Ceiling_MC'] - soft_cap),
            out_df['Ceiling_MC']
        )
        out_df['Ceiling_MC'] = (
            0.90 * out_df['Ceiling_MC'] +
            0.10 * (
                out_df['Projection'] + 1.5 * out_df['ProjMin']
            )
        )

        # Absolute sanity cap
        #out_df['Ceiling_MC'] = np.minimum(
        #    out_df['Ceiling_MC'],
        #    out_df['Projection'] * 2.1
        #)
        #soft_cap = (
        #    out_df['Projection'] +
        #    2.6 * out_df['Volatility_STD']
        #)
#
        #out_df['Ceiling_MC'] = np.where(
        #    out_df['Ceiling_MC'] > soft_cap,
        #    soft_cap + 0.25 * (out_df['Ceiling_MC'] - soft_cap),
        #    out_df['Ceiling_MC']
        #)

        # ------------------------
        # Compute per-player boom probabilities (DraftKings scaling)
        # ------------------------

        # Typical DFS thresholds
        threshold_6x = out_df['Salary'] * 0.006
        threshold_7x = out_df['Salary'] * 0.007
        threshold_8x = out_df['Salary'] * 0.008

        # Use Monte Carlo sims if available
        out_df['P_6x'] = out_df['MC_Mean'].combine(out_df['P_6x'], lambda mc, p: p if p else 0.0)
        out_df['P_7x'] = out_df['MC_Mean'].combine(out_df['P_7x'] if 'P_7x' in out_df else 0.0,
                                                   lambda mc, p: float(np.mean(mc >= threshold_7x)) if hasattr(mc, '__iter__') else 0.0)
        out_df['P_8x'] = out_df['MC_Mean'].combine(out_df['P_8x'], lambda mc, p: float(np.mean(mc >= threshold_8x)) if hasattr(mc, '__iter__') else 0.0)

        # Sanity clamp
        for col in ['P_6x', 'P_7x', 'P_8x']:
            out_df[col] = out_df[col].clip(0.0, 1.0)

        # Dominance score (upswing / ownership)
        #out_df['Dominance'] = out_df['BoomScore'] * (1 - out_df['OwnershipProb'])

        # Optional: normalize Dominance across slate for lineup aggregation
        #out_df['DominanceNorm'] = out_df['Dominance'] / out_df['Dominance'].sum()

        # Quick stats check
        #print(out_df[['Name','Projection','Ceiling_MC','Floor_MC','P_6x','P_7x','P_8x','Dominance']].sort_values('Dominance', ascending=False).head(10))
        
        # Floor protection
        out_df['Floor_MC'] = np.maximum(
            out_df['Floor_MC'],
            out_df['Projection'] * 0.40
        )

        out_df['Floor_MC'] = out_df['Floor_MC'].clip(lower=0.0)

        # Ensure floors are non-negative
        out_df['Floor_MC'] = out_df['Floor_MC'].clip(lower=0.0)
    
        #minute_factor = np.clip(out_df['ProjMin'] / 30, 0.4, 1.1)           #added 12-22
        #out_df['Floor_MC'] *= minute_factor #added 12-22
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

        # ------------------------
        # GPP / leverage metrics
        # ------------------------
        out_df['Dominance'] = (
            out_df['BoomScore'] *
            (1 - out_df['OwnershipProb'])
        )
        # Safe normalization
        dom_sum = out_df['Dominance'].sum()
        out_df['DominanceNorm'] = (
            out_df['Dominance'] / dom_sum if dom_sum > 0 else 0.0
        )


        if 'OwnershipProb' in out_df.columns:
            out_df['Dominance'] = out_df['BoomScore'] * (1 - out_df['OwnershipProb'])
        else:
            out_df['Dominance'] = out_df['BoomScore']

        print(
            out_df[['Dominance','OwnershipProb','BoomScore']]
            .describe()
        )
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

        # Drop temp column 
        out_df.drop(columns=['BoomPct'], inplace=True)

        if save_csv:
             out_df.to_csv(save_csv, index=False)
             print(f"✅ Saved projections to {save_csv}")

             print(
            out_df.assign(
                ceil_ratio=out_df['Ceiling_MC'] / out_df['Projection']
            )['ceil_ratio'].describe()
        )
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
def optimize_dk_lineup(
    df: pd.DataFrame,
    salary_cap: int = 50000,
    prev_lineups: list[list] = None,
    max_overlap: int = 2,
    seed: int = 42
) -> pd.DataFrame:
    """
    Single lineup optimizer using linear programming (pulp)
    df: DataFrame with columns ['PlayerID', 'Position', 'Salary', 'OBJ']
    prev_lineups: list of previous lineups (each a list of PlayerIDs) for max_overlap constraints
    """

    rng = np.random.default_rng(seed)
    df = df.copy()
    playerid_to_index = dict(zip(df['PlayerID'], df.index))
    n_players = len(df)

    # Decision variables
    x = pulp.LpVariable.dicts("player", df.index, 0, 1, cat="Binary")

    prob = pulp.LpProblem("DK_Lineup", pulp.LpMaximize)

    # Objective: maximize OBJ
    prob += pulp.lpSum(df.loc[i, "OBJ"] * x[i] for i in df.index)

    # Salary cap
    prob += pulp.lpSum(df.loc[i, "Salary"] * x[i] for i in df.index) <= salary_cap

    # Exactly 8 players
    prob += pulp.lpSum(x[i] for i in df.index) == 8

    # Positional constraints: DK roster
    positions = {
        "PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1,
    }
    # Allow flex for PG/SG/SF/PF (last 3 spots)
    flex_positions = ["PG", "SG", "SF", "PF"]

    # Force at least 1 per core position, allow flex separately
    for pos, count in positions.items():
        prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "Position"] == pos) >= count

    # Max overlap with previous lineups
    if prev_lineups:
        for prev in prev_lineups:
            prev_ids = set(prev["PlayerID"])
            prob += pulp.lpSum(
                x[i] for i in df.index if df.loc[i, "PlayerID"] in prev_ids
            ) <= max_overlap
            # Only keep PlayerIDs that exist in current df
            #valid_players = [pid for pid in lineup_players if pid in playerid_to_index]
            #if valid_players:
            #    prob += pulp.lpSum(x[playerid_to_index[pid]] for pid in valid_players) <= max_overlap

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Collect selected players
    selected = [i for i in df.index if x[i].value() > 0.5]
    if not selected:
        return pd.DataFrame()  # failed to generate lineup

    return df.loc[selected].copy()

def generate_gpp_lineups(
    df: pd.DataFrame,
    n_lineups: int = 50,
    max_exposure: float = 0.4,
    salary_cap: int = 50000,
    seed: int = 42
) -> list[pd.DataFrame]:
    """
    Generate multiple GPP lineups with exposure constraints.
    df: DataFrame with ['PlayerID', 'Position', 'Salary', 'Projection', 'Ceiling_MC', 'GPP_Alpha']
    """
    rng = np.random.default_rng(seed)
    lineups = []
    exposure = defaultdict(int)
    prev_lineups = []

    max_appearances = int(n_lineups * max_exposure)

    for i in range(n_lineups):
        df_iter = df.copy()

        # -------------------------
        # Add noise for GPP variability
        # -------------------------
        for pid, count in exposure.items():
            if count >= max_appearances:
                df_iter = df_iter[df_iter["PlayerID"] != pid]

        # Safety check
        if len(df_iter) < 20:
            continue
        df_iter["OBJ"] = (
            0.50 * df_iter["Projection"] +
            0.35 * df_iter["Ceiling_MC"] +
            0.15 * df_iter["GPP_Alpha"]
        ) #* df_iter["Noise"]
        df_iter["OBJ"] *= rng.uniform(0.80, 1.25, size=len(df_iter))

        # -------------------------
        # Limit exposure based on previous lineups
        # -------------------------
        for idx in df_iter.index:
            pid = df_iter.loc[idx, "PlayerID"]
            if exposure[pid] >= max_appearances:
                df_iter.loc[idx, "OBJ"] *= 0.01  # effectively fades

        # Generate lineup
        lineup = optimize_dk_lineup(
            df=df_iter,
            #mode="gpp",
            #salary_cap=salary_cap,
            prev_lineups=lineups,
            max_overlap=5,
            #seed=int(rng.integers(1_000_000))
        )
        # -------------------------
        # Score lineup
        # -------------------------
        scores = score_lineup(lineup)

        for k, v in scores.items():
            lineup[k] = v

        lineup["LineupID"] = i + 1
        lineups.append(lineup)

        if lineup.empty:
            continue

        # Update exposure
        for pid in lineup['PlayerID']:
            exposure[pid] += 1

        prev_lineups.append(list(lineup['PlayerID']))
        mc = evaluate_lineup_mc(
            lineup,
            n_sims=3000,
            cash_line=270,   # tune per slate size
            rng=rng
        )

        lineup = lineup.assign(
            Lineup=i + 1,
            Lineup_Mean=mc["Lineup_Mean"],
            Lineup_P90=mc["Lineup_P90"],
            Lineup_P95=mc["Lineup_P95"],
            Lineup_P99=mc["Lineup_P99"],
            P_Cash=mc["P_Cash"],
            P_Top1=mc["P_Top1"]
        )

        lineups.append(lineup)

        #lineup = lineup.assign(Lineup=i + 1)
        #lineups.append(lineup)

    # Final sanity checks
    all_players = pd.concat(lineups)
    
    summary = (
    pd.concat(lineups)
      .groupby("LineupID")
      .first()[["LineupProj", "LineupCeiling", "LineupSalary"]]
      .sort_values("LineupProj", ascending=False)
    )

    print("\n📊 Lineup Summary (Top 10)")
    print(summary.head(10))
    print("\n🧪 Sanity check: exposure across lineups")
    print(all_players.groupby("PlayerID")["Lineup"].count())
    print("\n🧪 Lineup size stats:")
    print(all_players.groupby("Lineup")["PlayerID"].count())

    return lineups

def generate_candidate_lineups(
    df,
    n_lineups=100,
    salary_cap=50000,
    min_salary=47000,
    max_from_team=4,
    seed=None
):
    """
    Fast stochastic lineup generator for GPP recycling.
    No evaluation — just valid DK lineups.
    """

    if seed is not None:
        np.random.seed(seed)

    lineups = []

    # DK roster slots
    #roster = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    roster = ["PG", "SG", "G", "SF", "PF", "F", "C", "UTIL"]

    # Pre-split by position eligibility
    pos_map = {
        "PG": df[df["Position"].str.contains("PG")],
        "SG": df[df["Position"].str.contains("SG")],
        "SF": df[df["Position"].str.contains("SF")],
        "PF": df[df["Position"].str.contains("PF")],
        "C":  df[df["Position"].str.contains("C")],
        "G":  df[df["Position"].str.contains("PG|SG")],
        "F":  df[df["Position"].str.contains("SF|PF")],
        "UTIL": df,
    }

    for _ in range(n_lineups):
        lineup = []
        used = set()

        for slot in roster:
            pool = pos_map[slot]
            pool = pool[~pool["Name"].isin(used)]

            if pool.empty:
                break

            # 🔑 GPP-weighted sampling
            weights = (
                pool["OBJ"]
                if "OBJ" in pool.columns
                else pool["Projection"]
            )
            noise = np.random.lognormal(mean=0, sigma=0.30, size=len(pool))
            w = weights * noise
            
            #player = pool.sample(1, weights=w)

            #player = pool.sample(
            #    1,
            #    weights=weights,
            #    replace=False
            #).iloc[0]
            player = pool.sample(1, weights=w).iloc[0]  # <-- extract Series
            lineup.append(player)
            used.add(player["Name"])
        if len(lineup) < 8:
            # DEBUG
            print(f"❌ Broke at slot {slot}, used={len(used)}")
            continue

        if len(lineup) != 8:
            continue

        lineup_df = pd.DataFrame(lineup)

        salary = lineup_df["Salary"].sum()

        if not (min_salary <= salary <= salary_cap):
            continue

        # Optional team constraint
        if (
            lineup_df["Team"].value_counts().max()
            > max_from_team
        ):
            continue

        # Attach quick attrs
        lineup_df.attrs["Salary"] = salary
        lineup_df.attrs["Proj"] = lineup_df["Projection"].sum()

        lineups.append(lineup_df)

    return lineups

def generate_gpp_lineups_recycling(
    df: pd.DataFrame,
    n_lineups: int = 20,
    max_recycles: int = 20,
    start_cash: float = 0.48,
    cash_decay: float = 0.015,
    candidates_per_round: int = 200,
):
    """
    Robust GPP lineup recycling with safe fallbacks.
    Always returns >=1 lineup if possible.
    """

    kept_lineups = []
    cash_threshold = start_cash

    for recycle in range(1, max_recycles + 1):
        print(f"🔁 Recycling {recycle}/{max_recycles} (cash >= {cash_threshold:.3f})")

        # -----------------------------
        # 1️⃣ Generate raw candidates
        # -----------------------------
        candidates = generate_candidate_lineups(
            df=df,
            n_lineups=candidates_per_round,
        )

        if not candidates:
            print("⚠️ No candidate lineups generated")
            cash_threshold -= cash_decay
            continue

        # -----------------------------
        # 2️⃣ Evaluate candidates
        # -----------------------------
        eval_df = evaluate_lineups_mc(candidates)

        if eval_df is None or eval_df.empty:
            print("⚠️ MC evaluation returned empty")
            cash_threshold -= cash_decay
            continue

        # -----------------------------
        # 3️⃣ Debug visibility
        # -----------------------------
        print(
            "    P_Cash:",
            round(eval_df["P_Cash"].min(), 3),
            #eval_df["P_Cash"].min().round(3),
            "→",
            round(eval_df["P_Cash"].max(), 3),
            #eval_df["P_Cash"].max().round(3),
        )

        # -----------------------------
        # 4️⃣ Filter passing lineups
        # -----------------------------
        passing = eval_df[eval_df["P_Cash"] >= cash_threshold]

        if passing.empty:
            cash_threshold -= cash_decay
            continue

        # -----------------------------
        # 5️⃣ Attach attrs safely
        # -----------------------------
        for idx, row in passing.iterrows():
            lineup = candidates[int(idx) - 1]

            lineup.attrs["Lineup_Mean"] = row["Mean"]
            lineup.attrs["Lineup_P90"] = row["P90"]
            lineup.attrs["Lineup_P95"] = row["P95"]
            lineup.attrs["P_Cash"] = row["P_Cash"]
            lineup.attrs["P_Top1"] = row.get("P_Top1", 0.0)

            kept_lineups.append(lineup)

        # Stop early if enough
        if len(kept_lineups) >= n_lineups:
            break

    # -----------------------------
    # 6️⃣ HARD FALLBACK (CRITICAL)
    # -----------------------------
    if not kept_lineups:
        print("⚠️ No lineups met cash threshold — using best available")

        best = eval_df.sort_values("P_Cash", ascending=False).head(n_lineups)

        for idx, row in best.iterrows():
            lineup = candidates[int(idx) - 1]

            lineup.attrs["Lineup_Mean"] = row["Mean"]
            lineup.attrs["Lineup_P90"] = row["P90"]
            lineup.attrs["Lineup_P95"] = row["P95"]
            lineup.attrs["P_Cash"] = row["P_Cash"]
            lineup.attrs["P_Top1"] = row.get("P_Top1", 0.0)

            kept_lineups.append(lineup)

    print(f"✅ Final GPP lineups kept: {len(kept_lineups)}")
    return kept_lineups[:n_lineups]


#def generate_gpp_lineups_recycling(
#    df,
#    n_lineups=50,
#    cash_threshold=0.55,
#    max_recycles=20,
#):
#    good_lineups = []
#    all_attempts = []
#
#    for iteration in range(max_recycles):
#        dynamic_cash = max(
#            cash_threshold - 0.015 * iteration,
#            0.30  # DFS-reality floor
#        )
#
#        print(
#            f"🔁 Recycling {iteration+1}/{max_recycles} "
#            f"(cash >= {dynamic_cash:.3f})"
#        )
#
#        # 1️⃣ Generate candidates
#        candidates = generate_candidate_lineups(
#            df,
#            n_lineups=n_lineups * 2
#        )
#
#        # 2️⃣ Evaluate ONCE
#        evaluated = evaluate_lineups_mc(candidates)
#
#        # 3️⃣ Attach attrs so we don't re-sim
#        for i, lineup in enumerate(candidates):
#            for col in evaluated.columns:
#                lineup.attrs[col] = evaluated.iloc[i][col]
#
#        # 4️⃣ Save all attempts
#        all_attempts.extend(candidates)
#
#        # 5️⃣ Filter passing
#        passing = [
#            l for l in candidates
#            if l.attrs["P_Cash"] >= dynamic_cash
#        ]
#
#        good_lineups.extend(passing)
#        good_lineups = dedupe_lineups(good_lineups)
#        if passing.empty:
#            # Fallback: keep top N by GPP score
#            passing = evaluated.sort_values(
#                "GPP_OBJ", ascending=False
#            ).head(5)
#
#        if len(good_lineups) >= n_lineups:
#            break
#
#    # 🧯 FAILSAFE — never return empty
#    if len(good_lineups) == 0:
#        print("⚠️ No lineups met cash threshold — using best available")
#        good_lineups = sorted(
#            all_attempts,
#            key=lambda l: l.attrs["P_Top1"],
#            reverse=True
#        )[:n_lineups]
#
#    return good_lineups[:n_lineups]

def dedupe_lineups(lineups):
    seen = set()
    out = []
    for l in lineups:
        key = tuple(sorted(l["Name"])) + (int(l["Salary"].sum()),)
        if key not in seen:
            seen.add(key)
            out.append(l)
    return out


def lineup_sanity_checks(lineups: list[pd.DataFrame]):
    """
    Perform sanity checks and plots for multi-lineup sets.

    Args:
        lineups: list of DataFrames representing lineups
    """
    if not lineups:
        print("No lineups to check.")
        return

    # -------------------------
    # Combine all lineups
    # -------------------------
    all_lineups = pd.concat(lineups, ignore_index=True)
    
    # -------------------------
    # Exposure counts
    # -------------------------
    exposure = all_lineups.groupby("Name").size().reset_index(name="Lineup")
    print("\n🧪 Sanity check: exposure across lineups")
    print(exposure.sort_values("Lineup", ascending=False).head(15))

    # -------------------------
    # Lineup size stats
    # -------------------------
    print("\n🧪 Lineup size stats:")
    lineup_sizes = all_lineups.groupby("Lineup").size()
    print(lineup_sizes.describe())
    
    # -------------------------
    # Floor/Ceiling vs Exposure plot
    # -------------------------
    avg_stats = all_lineups.groupby("Name")[["Floor_MC", "Ceiling_MC"]].mean().reset_index()
    df_plot = exposure.merge(avg_stats, on="Name")
    df_plot = df_plot.sort_values(by="Lineup", ascending=False)

    plt.figure(figsize=(14,6))
    sns.scatterplot(data=df_plot, x="Lineup", y="Floor_MC", color="skyblue", s=100, label="Floor")
    sns.scatterplot(data=df_plot, x="Lineup", y="Ceiling_MC", color="orange", s=100, label="Ceiling")
    plt.title("Player Exposure vs. Floor and Ceiling")
    plt.xlabel("Exposure (# of lineups)")
    plt.ylabel("Points")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()

    # -------------------------
    # Optional: Exposure vs Projection
    # -------------------------
    if "Projection" in all_lineups.columns:
        avg_stats = all_lineups.groupby("Name")[["Projection"]].mean().reset_index()
        df_plot2 = exposure.merge(avg_stats, on="Name")
        df_plot2 = df_plot2.sort_values("Lineup", ascending=False)

        plt.figure(figsize=(14,6))
        sns.scatterplot(data=df_plot2, x="Lineup", y="Projection", size="Lineup",
                        hue="Projection", palette="viridis", legend="brief", sizes=(50,300))
        plt.title("Player Exposure vs Projection")
        plt.xlabel("Exposure (# of lineups)")
        plt.ylabel("Projected Points")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()
    lineup_sanity_checks(lineups)

# ============================
# LINEUP-LEVEL MONTE CARLO
# ============================

def score_lineup_mc(
    lineup: pd.DataFrame,
    n_sims: int = 3000,
    seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)

    means = lineup["Projection"].values
    stds  = lineup["Volatility_STD"].values

    sims = rng.normal(
        loc=means,
        scale=stds,
        size=(n_sims, len(lineup))
    ).sum(axis=1)

    return {
        "LineupMean": sims.mean(),
        "LineupFloor": np.percentile(sims, 20),
        "LineupCeiling": np.percentile(sims, 90),
        "LineupStd": sims.std(),
        "CashProb": (sims >= np.percentile(sims, 50)).mean(),
        "Top1Pct": (sims >= np.percentile(sims, 99)).mean()
    }


def plot_player_exposure(lineups: list[pd.DataFrame], top_n: int = 25):
    """
    Bar chart of player exposure across generated lineups
    """
    all_players = pd.concat(lineups)

    exposure = (
        all_players.groupby("Name")
        .size()
        .sort_values(ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    exposure.sort_values().plot(kind="barh")
    plt.title("Top Player Exposure Across Lineups")
    plt.xlabel("Lineups Used")
    plt.ylabel("Player")

    plt.tight_layout()
    plt.savefig("player_exposure.png", dpi=300)
    print("📊 Saved: player_exposure.png")

    try:
        plt.show()
    except:
        pass

def plot_salary_distribution(lineups: list[pd.DataFrame]):
    totals = [lu["Salary"].sum() for lu in lineups]

    plt.figure(figsize=(8, 5))
    plt.hist(totals, bins=10)
    plt.title("Salary Distribution Across Lineups")
    plt.xlabel("Total Salary")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("salary_distribution.png", dpi=300)
    print("📊 Saved: salary_distribution.png")

    try:
        plt.show()
    except:
        pass

def plot_ceiling_vs_projection(df: pd.DataFrame):
    plt.figure(figsize=(7, 6))
    plt.scatter(df["Projection"], df["Ceiling_MC"], alpha=0.5)

    plt.plot(
        [df["Projection"].min(), df["Projection"].max()],
        [df["Projection"].min(), df["Projection"].max()],
        linestyle="--"
    )

    plt.xlabel("Projection")
    plt.ylabel("Ceiling")
    plt.title("Projection vs Ceiling")

    plt.tight_layout()
    plt.savefig("ceiling_vs_projection.png", dpi=300)
    print("📊 Saved: ceiling_vs_projection.png")

    try:
        plt.show()
    except:
        pass
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
    parser = argparse.ArgumentParser(
        description="Simple NBA DFS projections with injuries & Monte Carlo"
    )
    parser.add_argument('--salaries', required=True,
                        help='DraftKings salaries CSV (your header supported)')
    parser.add_argument('--out', default='projections.csv',
                        help='Optional output CSV path')
    parser.add_argument('--days', type=int, default=30,
                        help='Days of history for logs (default 30)')
    parser.add_argument('--n_sims', type=int, default=1500,
                        help='Monte Carlo simulations per player (default 1500)')
    parser.add_argument('--no-inj', dest='use_injuries',
                        action='store_false',
                        help='Disable scraping ESPN injuries')

    parser.add_argument(
        '--optimize',
        choices=['cash', 'gpp', 'gpp-multi'],
        help='Run DraftKings lineup optimizer (cash or gpp)'
    )

    parser.add_argument(
        '--lineup-out',
        default='lineup.csv',
        help='Output CSV for optimized lineup'
    )

    args = parser.parse_args()

    model = SimpleNBAProjection(
        dk_salaries_path=args.salaries,
        days_of_history=args.days
    )

    if not model.load_dk_salaries():
        print("❌ Failed to load DraftKings CSV — aborting")
        return
    
    injuries = {}
    if args.use_injuries:
        print("🔎 Scraping ESPN injuries...")
        injuries = get_injuries()
        print(f"  → got {len(injuries)} injury entries")

    print("🔎 Generating projections (this may take a minute)...")
    
    df = model.run(
        save_csv=args.out,
        injuries=injuries,
        n_sims=args.n_sims
    )
    if df.empty:
        print("❌ No projections created.")
        return

    if args.optimize == "gpp-multi":
        print("\n🧠 Generating multiple GPP lineups...")

        # Ensure base objective exists
        if "OBJ" not in df.columns:
            df["OBJ"] = (
                0.30 * df["Projection"] +
                0.55 * df["Ceiling_MC"] +
                0.35 * df["BoomScore"] +
                0.30 * (1 - df["OwnershipProb"])
            )

        # -------------------------
        # 1️⃣ Generate lineups (WITH recycling)
        # -------------------------
        lineups = generate_gpp_lineups_recycling(
            df=df,
            n_lineups=50,
            start_cash=0.18
        )

        if not lineups:
            print("❌ No lineups generated.")
            return

        # -------------------------
        # 2️⃣ Build lineup summary FROM attrs (SINGLE SOURCE)
        # -------------------------
        #summary = pd.DataFrame(
        #    [
        #        {
        #            "Lineup": i + 1,
        #            "Salary": l.attrs["Salary"],
        #            "Proj": l.attrs["Proj"],
        #            "Lineup_Mean": l.attrs["Lineup_Mean"],
        #            "Lineup_P90": l.attrs["Lineup_P90"],
        #            "Lineup_P95": l.attrs["Lineup_P95"],
        #            "P_Cash": l.attrs["P_Cash"],
        #            "P_Top1": l.attrs["P_Top1"],
        #        }
        #        for i, l in enumerate(lineups)
        #    ]
        #).set_index("Lineup")
        summary = evaluate_lineups_mc(lineups)
        
        # -------------------------
        # 3️⃣ Filter + sort
        # -------------------------
        #summary = summary[summary["P_Cash"] >= 0.48]               
        summary = summary.sort_values("P_Top1", ascending=False)

        #if summary.empty:
        #    print("❌ No lineups passed P_Cash filter.")
        #    return
        # -------------------------
        # 🔍 DEBUG: why lineups are failing / passing
        # -------------------------
        print("\n🔎 DEBUG — Lineup outcome distribution")

        debug_rows = []
        for i, l in enumerate(lineups):
            debug_rows.append({
                "Lineup": i + 1,
                "Lineup_Mean": l.attrs.get("Lineup_Mean"),
                "P_Cash": l.attrs.get("P_Cash"),
                "P_Top1": l.attrs.get("P_Top1"),
            })

        debug_df = pd.DataFrame(debug_rows).set_index("Lineup")
        print(debug_df.describe().round(3))
        #print("\n🔎 DEBUG — Lineup outcome distribution")
        #print(
        #    summary[
        #        ["Lineup_Mean", "P_Cash", "P_Top1"]
        #    ].describe().round(4)
        #)
        # -------------------------
        # 4️⃣ Player-level output
        # -------------------------
        final_players = pd.concat(
            [
                l.assign(Lineup=i + 1)
                for i, l in enumerate(lineups)
                if (i + 1) in summary.index
            ],
            ignore_index=True
        )

        # -------------------------
        # 5️⃣ Save + display
        # -------------------------
        #summary.to_csv("lineup_summary.csv", float_format="%.4f")      #may need to add this back
        final_players.to_csv(args.lineup_out, index=False)
        if summary.empty:
            print("❌ No lineups survived MC evaluation")
            return

        summary = summary.sort_values("P_Top1", ascending=False)

        print("\n📊 TOP LINEUPS")
        print(summary.head(10).round(3))

        summary.to_csv("lineup_summary.csv", float_format="%.4f")
        print("\n📊 LINEUP EVALUATION (Top 10)")
        print(summary.head(10).round(3))

        # -------------------------
        # 6️⃣ Visuals
        # -------------------------
        plot_ceiling_vs_projection(df)
        plot_salary_distribution(lineups)
        plot_player_exposure(lineups)

        print(f"\n✅ Saved {len(summary)} filtered lineups")
        return

    print("✅ Done — top 10 projections:")
    print(df.head(10).to_string(index=False))

    # -------------------------
    # Optional lineup optimizer
    # -------------------------
    if args.optimize:
        print(f"\n🧠 Optimizing DraftKings lineup ({args.optimize.upper()})...")

        lineup = optimize_dk_lineup(
            df=df,
            mode=args.optimize,
            min_core=2 if args.optimize == 'gpp' else 0,
            max_sprinkle=2
        )

        if lineup.empty:
            print("❌ Optimizer failed to produce a lineup.")
        else:
            lineup.to_csv(args.lineup_out, index=False)

            print("\n🏀 OPTIMIZED LINEUP")
            print(
                lineup[
                    ['Name', 'Team', 'Position', 'Salary',
                     'Projection', 'Ceiling_MC', 'GPP_Tier']
                ].to_string(index=False)
            )

            print(f"\n💾 Lineup saved to: {args.lineup_out}")

if __name__ == "__main__":
    main()

