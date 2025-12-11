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

# -------------------------
# Utility: CSV header mapping
# -------------------------
def detect_dk_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols_lower = {c.lower(): c for c in df.columns}
    name_candidates = ['name', 'name + id', 'name+id', 'player', 'playername', 'full_name']
    name_col = next((cols_lower[c] for c in name_candidates if c in cols_lower), None)

    salary_candidates = ['salary', 'dk salary', 'dksalary']
    salary_col = next((cols_lower[c] for c in salary_candidates if c in cols_lower), None)

    team_candidates = ['teamabbrev', 'team_abbrev', 'team', 'teamabbr', 'teamabbrv']
    team_col = next((cols_lower[c] for c in team_candidates if c in cols_lower), None)

    pos_candidates = ['position', 'pos', 'roster position', 'roster_position']
    pos_col = next((cols_lower[c] for c in pos_candidates if c in cols_lower), None)

    return name_col, salary_col, team_col, pos_col

def normalize_name(name: str) -> str:
    """Normalize player names to remove accents, trim spaces, remove apostrophes, lowercase."""
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.replace("'", "").strip().lower()
    return name

def normalize_vegas_data(raw_games):
    vegas = {}
    for game in raw_games:
        t1 = game["team1"]
        t2 = game["team2"]
        team1_name = t1["team_name"].strip().upper()
        team2_name = t2["team_name"].strip().upper()

        def parse_total(val):
            if val.startswith(("o","u")):
                val = val[1:]
            if "-" in val:
                val = val.split("-")[0]
            try:
                return float(val)
            except:
                return VEGAS_DEFAULT_TOTAL

        def parse_spread(raw):
            if not raw or not isinstance(raw, str):
                return 0.0
            raw = raw.strip().lower()
            if raw in ("pk","pick","pick'em","even"):
                return 0.0
            m = re.search(r'([+-]?\d+\.?\d*)', raw)
            return float(m.group(1)) if m else 0.0

        total = parse_total(t1.get("total",""))
        spread1 = parse_spread(t1.get("spread",""))
        spread2 = parse_spread(t2.get("spread",""))

        t1_total = (total/2) + (spread1/2)
        t2_total = (total/2) + (spread2/2)

        vegas[team1_name] = {"total": total, "spread": spread1, "team_total": t1_total}
        vegas[team2_name] = {"total": total, "spread": spread2, "team_total": t2_total}
    return vegas

def vegas_multiplier(team_abbr: str, vegas: Dict[str, Dict[str, float]]) -> float:
    if not team_abbr:
        return 1.0
    data = vegas.get(team_abbr.upper(), {})
    total = float(data.get('total', VEGAS_DEFAULT_TOTAL))
    spread = float(data.get('spread', VEGAS_DEFAULT_SPREAD))
    total_adj = total / VEGAS_DEFAULT_TOTAL
    blowout_adj = 0.95 if spread <= -10 else 0.97 if spread >= 10 else 1.0
    mult = total_adj * blowout_adj
    return max(0.85, min(1.15, mult))

# -------------------------
# Position-based usage context
# -------------------------
def compute_position_usage_context(team_players_df: pd.DataFrame, injuries: Dict[str, str]) -> Dict[str, object]:
    ctx = {"usg_missing":0.0,"ast_missing":0.0,"reb_missing":0.0,"pg_out":False,"c_out":False,"positions_out":[]}
    for _, p in team_players_df.iterrows():
        name = normalize_name(p.get("Name",""))
        pos = str(p.get("Position",""))
        status = injuries.get(name)
        if status != "out":
            continue
        ctx["positions_out"].append(pos)
        if pos == "PG":
            ctx["pg_out"]=True
            ctx["usg_missing"]+=0.04
            ctx["ast_missing"]+=0.06
        elif pos in ("SG","SF"):
            ctx["usg_missing"]+=0.03
        elif pos == "C":
            ctx["c_out"]=True
            ctx["reb_missing"]+=0.06
    return ctx

def dynamic_usage_redistribution(player_name: str, player_pos: str, fpmin: float, proj_min: float,
                                 injuries: Dict[str,str], dk_df: pd.DataFrame, max_boost: float=0.25) -> float:
    if proj_min < 6:
        return fpmin
    injuries_l = {normalize_name(k):v for k,v in injuries.items()}
    team = None
    row = dk_df[dk_df['Name'].apply(normalize_name) == normalize_name(player_name)]
    if not row.empty:
        team = row.iloc[0].get('Team')
    if team is None:
        return fpmin
    team_players_df = dk_df[dk_df['Team']==team]
    ctx = compute_position_usage_context(team_players_df, injuries_l)

    boost = 0.0
    name_l = normalize_name(player_name)

    same_pos_out = any(
        (injuries_l.get(normalize_name(p))=="out") and (pos==player_pos)
        for p,pos in zip(team_players_df['Name'], team_players_df.get('Position', pd.Series(['']*len(team_players_df))))
    )
    if same_pos_out and injuries_l.get(name_l)!="out":
        if proj_min>=28:
            boost+=0.06
        elif proj_min>=20:
            boost+=0.10
        else:
            boost+=0.12

    if ctx.get('usg_missing',0.0)>0:
        boost+=ctx['usg_missing']*0.35

    if ctx.get('pg_out'):
        if player_pos in ('SG','SF'):
            boost+=0.03
        elif player_pos=='PG' and injuries_l.get(name_l)!="out":
            boost+=0.05

    if ctx.get('c_out'):
        if player_pos in ('C','PF'):
            boost+=ctx['reb_missing']*0.6
        elif player_pos=='SF':
            boost+=ctx['reb_missing']*0.15

    try:
        row = team_players_df[team_players_df['Name'].apply(normalize_name)==name_l]
        if not row.empty:
            salary = float(row.iloc[0].get('Salary',5000))
            if salary<4500:
                boost+=0.02
    except Exception:
        pass

    boost=max(0.0,min(boost,max_boost))
    return fpmin*(1.0+boost)

# -------------------------
# Monte Carlo per-stat (for better floor/ceiling & variance)
# -------------------------
def monte_carlo_per_stat(player_logs: pd.DataFrame, n_sims: int = 2000, seed: int = 42) -> Dict[str, Dict[str, float]]:
    """
    For a player's recent logs (DataFrame), run Monte Carlo separately per stat and return:
      { 'PTS': {'floor':..,'ceiling':..,'std':..}, ... }
    Uses empirical sampling when >=6 samples, otherwise normal approx with truncation.
    """
    rng = np.random.default_rng(seed)
    stat_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'TOV']
    mc = {}
    for stat in stat_cols:
        vals = []
        if stat in player_logs.columns:
            vals = [float(v) for v in player_logs[stat] if pd.notna(v)]
        if len(vals) == 0:
            mc[stat] = {'floor': 0.0, 'ceiling': 0.0, 'std': 0.0}
            continue
        if len(vals) >= 6:
            sims = rng.choice(vals, size=n_sims, replace=True)
        else:
            mu = float(np.mean(vals))
            sigma = float(np.std(vals, ddof=1)) if len(vals) > 1 else max(0.1, mu * 0.2)
            sims = rng.normal(mu, sigma, size=n_sims)
            sims = np.clip(sims, 0.0, None)
        mc[stat] = {
            'floor': float(np.percentile(sims, 20)),
            'ceiling': float(np.percentile(sims, 90)),
            'std': float(np.std(sims, ddof=1))
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
        opp = self.todays_matchups.get(team_abbr, {}).get('opponent')
        team_pace = self.team_stats.get(team_abbr, {}).get('pace', self.LEAGUE_AVG_PACE)
        opp_pace = self.team_stats.get(opp, {}).get('pace', self.LEAGUE_AVG_PACE) if opp else self.LEAGUE_AVG_PACE
        opp_def = self.team_stats.get(opp, {}).get('def_rating', 110.0) if opp else 110.0
        pace_adj = ((team_pace + opp_pace) / 2.0) / self.LEAGUE_AVG_PACE
        def_adj = 110.0 / float(opp_def)
        multiplier = 0.5 * pace_adj + 0.5 * def_adj
        return max(0.85, min(1.15, multiplier))

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
            #name_l = name.lower()
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
            fpmin = dynamic_usage_redistribution(name, pos, fpmin, base_min, injuries, self.dk_df)
            # matchup & vegas multipliers
            matchup_mult = self.matchup_multiplier(team) if team else 1.0
            vegas_mult = vegas_multiplier(team, vegas) if team else 1.0
            mult = matchup_mult * vegas_mult
            raw_proj = fpmin * base_min * mult
            capped_proj = self.cap_projection_by_salary(raw_proj, salary)
            # Monte Carlo per-minute floor/ceiling: use empirical fp_min_list and min_list
            mc_simple = self.monte_carlo(fp_min_list=stats.get('fp_min_list', []),
                                         min_list=stats.get('min_list', []),
                                         matchup_mult=matchup_mult, vegas_mult=vegas_mult, n_sims=n_sims)
            # Monte Carlo per-stat for floor/ceiling derivation (optional, expensive)
            per_stat_mc = monte_carlo_per_stat(stats.get('logs_df', pd.DataFrame()), n_sims=n_sims)
            # Convert per-stat floors/ceilings to DK FP floor/ceiling with scoring weights
            stat_weights = {'PTS': 1.0, 'REB': 1.2, 'AST': 1.5, 'STL': 3.0, 'BLK': 3.0, 'FG3M': 0.5, 'TOV': -1.0}
            floor_stat = sum(per_stat_mc[s]['floor'] * stat_weights.get(s, 0.0) for s in per_stat_mc)
            ceil_stat = sum(per_stat_mc[s]['ceiling'] * stat_weights.get(s, 0.0) for s in per_stat_mc)
            # Combine simple MC and stat MC: prefer stat-driven floor/ceiling if available
            floor = max(mc_simple['floor'] * 0.5 + floor_stat * 0.5, 0.0)
            ceiling = max(mc_simple['ceiling'] * 0.5 + ceil_stat * 0.5, 0.0)
            volatility = mc_simple.get('volatility_std', 0.0)

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
                'Floor_MC': round(floor, 1),
                'Ceiling_MC': round(ceiling, 1),
                'Volatility_STD': round(volatility, 2)
            })

        out_df = pd.DataFrame(results).sort_values('Projection', ascending=False)
        if save_csv:
            out_df.to_csv(save_csv, index=False)
            print(f"✅ Saved projections to {save_csv}")
        return out_df

    def monte_carlo(self, fp_min_list: List[float], min_list: List[float], matchup_mult: float, vegas_mult: float, n_sims: int = 2000) -> Dict[str, float]:
        """
        Simple Monte Carlo sampling using empirical fp/min and minutes arrays.
        """
        fp_vals = np.array([v for v in fp_min_list if np.isfinite(v) and v > 0])
        min_vals = np.array([m for m in min_list if np.isfinite(m) and m > 0])
        if len(fp_vals) == 0 or len(min_vals) == 0:
            return {'floor': 0.0, 'ceiling': 0.0, 'volatility_std': 0.0}
        rng = np.random.default_rng(42)
        sims = []
        for _ in range(n_sims):
            fp_per_min = rng.choice(fp_vals)
            minutes = rng.choice(min_vals)
            sims.append(fp_per_min * minutes * matchup_mult * vegas_mult)
        sims = np.array(sims)
        return {'floor': float(np.percentile(sims, 20)), 'ceiling': float(np.percentile(sims, 90)), 'volatility_std': float(np.std(sims, ddof=1))}

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

    injuries={}
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

if __name__=="__main__":
    main()