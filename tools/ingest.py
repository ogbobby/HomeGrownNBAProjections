import pandas as pd
import unicodedata
import re
import difflib
from typing import Optional


def normalize_name(name: str) -> str:
    """Normalize player names for robust matching across sources.

    Steps:
    - NFKD unicode normalize and strip diacritics
    - normalize common punctuation and hyphens
    - remove common suffixes (JR, SR, II, III, IV, V)
    - remove non-alphanumeric characters and collapse spaces
    - lowercase
    """
    if not name:
        return ""
    s = str(name)
    try:
        s = unicodedata.normalize('NFKD', s)
        s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    except Exception:
        s = str(name)
    s = s.replace('’', "'").replace('`', "'").replace('–', '-').replace('—', '-')
    # remove common suffixes like Jr., Sr., II, III
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", '', s, flags=re.IGNORECASE)
    s = s.replace('-', ' ')
    # keep apostrophes (O'Neal style) but normalize others to spaces
    s = re.sub(r"[^A-Za-z0-9\'\s]", ' ', s)
    # collapse multiple spaces and normalize lowercase
    s = re.sub(r"\s+", ' ', s).strip().lower()
    return s


def parse_dk_csv(path: str) -> pd.DataFrame:
    """Parse a DraftKings salaries CSV into a standardized DataFrame.

    Returns DataFrame with columns: Name, Salary (float), Team, Position (if present), Name_norm
    """
    df = pd.read_csv(path)
    # detect columns
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get('name') or cols.get('player') or cols.get('playername')
    salary_col = cols.get('salary') or cols.get('dk salary') or cols.get('dk points')
    team_col = cols.get('team') or cols.get('teamabbrev') or cols.get('team_abbrev')
    pos_col = cols.get('position') or cols.get('pos')

    if not name_col or not salary_col:
        raise ValueError(f"Could not find required columns in DK CSV: {list(df.columns)}")

    out = df.rename(columns={name_col: 'Name', salary_col: 'Salary'})
    if team_col:
        out = out.rename(columns={team_col: 'Team'})
    else:
        out['Team'] = ''
    if pos_col:
        out = out.rename(columns={pos_col: 'Position'})
    else:
        out['Position'] = ''

    out['Name'] = out['Name'].astype(str).str.strip()
    out['Salary'] = pd.to_numeric(out['Salary'], errors='coerce')
    out = out.dropna(subset=['Name', 'Salary'])
    out['Team'] = out['Team'].astype(str).str.strip()
    out['Position'] = out['Position'].astype(str).str.strip()
    out['Name_norm'] = out['Name'].apply(normalize_name)
    # keep core columns
    return out[['Name', 'Salary', 'Team', 'Position', 'Name_norm']].copy()


def build_synthetic_dk_slate_from_actuals(actuals_df: pd.DataFrame, default_salary: int = 6000) -> pd.DataFrame:
    """Given an actuals (boxscore) DataFrame with a PLAYER_NAME column, construct a synthetic DK slate.

    The resulting DataFrame will have Name, Salary, Team, Position='' and Name_norm.
    """
    # prefer PLAYER_NAME or PLAYER_NAME_L
    name_col = None
    for c in ('PLAYER_NAME', 'PLAYER_NAME_L', 'PLAYER_NAME_LLOW', 'PLAYER_NAME_LLOW'):
        if c in actuals_df.columns:
            name_col = c
            break
    if name_col is None:
        # fallback: try to find a column that looks like a name
        for c in actuals_df.columns:
            if 'name' in c.lower():
                name_col = c
                break
    if name_col is None:
        raise ValueError('Could not find player name column in actuals dataframe')

    team_col = None
    for c in ('TEAM', 'TEAM_ABBREVIATION', 'TEAM_ABBREV', 'TEAMABBREV'):
        if c in actuals_df.columns:
            team_col = c
            break

    names = actuals_df[name_col].astype(str).str.strip()
    teams = actuals_df[team_col].astype(str).str.strip() if team_col else [''] * len(names)
    rows = []
    for n, t in zip(names.tolist(), teams.tolist()):
        rows.append({'Name': n, 'Salary': float(default_salary), 'Team': t if t else '', 'Position': '', 'Name_norm': normalize_name(n)})
    return pd.DataFrame(rows)


def _try_import_rapidfuzz():
    try:
        from rapidfuzz import fuzz, process
        return True
    except Exception:
        return False


def fuzzy_match_actuals_to_dk(actuals_df: pd.DataFrame, dk_df: pd.DataFrame, actual_name_col: str = 'PLAYER_NAME', cutoff: float = 0.85) -> pd.DataFrame:
    """Return a DataFrame joining actuals to DK names using normalized exact-match then fuzzy fallback.

    Improvements over the simple approach:
    - prefer same-team candidates when a team is present
    - try first+last heuristics and last+first variants
    - use rapidfuzz token_set_ratio when available for robust token-based matching
    - fall back to difflib when rapidfuzz isn't installed

    The returned DataFrame will include columns from actuals plus these columns:
      DK_Name, DK_Name_norm, DK_Salary, DK_Team, MATCH_SCORE
    """
    act = actuals_df.copy()
    # find name column
    if actual_name_col not in act.columns:
        # try common alternatives
        for c in ('PLAYER_NAME', 'PLAYER_NAME_L', 'PLAYER_NAME_LLOW', 'PLAYER_NAME_LLOW'):
            if c in act.columns:
                actual_name_col = c
                break
    act['act_name_raw'] = act[actual_name_col].astype(str).str.strip()
    act['act_name_norm'] = act['act_name_raw'].apply(normalize_name)

    # prepare DK lookup
    dk_df = dk_df.copy()
    if 'Name_norm' not in dk_df.columns:
        dk_df['Name_norm'] = dk_df['Name'].apply(normalize_name)

    # try to use rapidfuzz if available
    has_rf = _try_import_rapidfuzz()
    if has_rf:
        from rapidfuzz import fuzz

    dk_norm_list = dk_df['Name_norm'].tolist()

    def _team_candidates(team: Optional[str]):
        """Return (candidates_df, used_team_filter:bool).
        If a team is provided and matching rows exist in dk_df, return only those rows and used_team_filter=True.
        If provided team yields no rows, return full dk_df and used_team_filter=False (so caller can tighten acceptance).
        """
        if not team:
            return dk_df, False
        # try various team column names
        for tc in ('Team', 'TEAM', 'team'):
            if tc in dk_df.columns:
                cand = dk_df[dk_df[tc].astype(str).str.upper() == str(team).upper()]
                if not cand.empty:
                    return cand, True
        return dk_df, False

    matched = []
    scores = []
    for _, r in act.iterrows():
        key = r['act_name_norm']
        team = None
        # detect team column in actuals
        for tc in ('TEAM', 'Team', 'team'):
            if tc in act.columns:
                team = r.get(tc)
                break

        dk_name = None
        best_score = 0.0

        # 1) exact normalized match
        exact = dk_df[dk_df['Name_norm'] == key]
        if len(exact) == 1:
            dk_name = exact.iloc[0]['Name']
            best_score = 100.0
        elif len(exact) > 1:
            # multiple exact matches (rare) - prefer same team
            if team is not None and 'Team' in dk_df.columns:
                same = exact[exact['Team'].astype(str).str.upper() == str(team).upper()]
                if not same.empty:
                    dk_name = same.iloc[0]['Name']
                    best_score = 100.0
                else:
                    dk_name = exact.iloc[0]['Name']
                    best_score = 100.0

        # 2) first+last / last+first heuristics
        if not dk_name:
            parts = key.split()
            if len(parts) >= 2:
                first_last = parts[0] + ' ' + parts[-1]
                last_first = parts[-1] + ' ' + parts[0]
                # check team-filtered candidates first
                candidates, used_team = _team_candidates(team)
                for dn in candidates['Name_norm'].tolist():
                    dparts = dn.split()
                    if len(dparts) >= 2 and ((dparts[0] + ' ' + dparts[-1]) == first_last or (dparts[-1] + ' ' + dparts[0]) == last_first):
                        dk_name = candidates[candidates['Name_norm'] == dn].iloc[0]['Name']
                        best_score = 95.0
                        break

        # 3) token-based fuzzy matching (rapidfuzz) or difflib
        if not dk_name:
            candidates, used_team = _team_candidates(team)
            cand_norms = candidates['Name_norm'].tolist()
            if has_rf:
                # use token_set_ratio which is order-insensitive and handles middle names well
                from rapidfuzz import process
                # process.extract returns tuples (match, score, idx)
                try:
                    res = process.extract(key, cand_norms, scorer=fuzz.token_set_ratio, limit=3)
                    if res:
                        match, score, _ = res[0]
                        # If we filtered by team and found no team candidates previously, be more strict
                        required_score = cutoff * 100
                        if team and (not used_team):
                            required_score = max(required_score, 95.0)
                        if score >= required_score:
                            # retrieve the DK name for the matched norm
                            dk_name = candidates[candidates['Name_norm'] == match].iloc[0]['Name']
                            best_score = float(score)
                except Exception:
                    # fall back to difflib in case rapidfuzz processing errors
                    has_rf = False
            if (not has_rf) and cand_norms:
                close = difflib.get_close_matches(key, cand_norms, n=1, cutoff=cutoff)
                if close:
                    # apply stricter acceptance if team was provided but no team candidates
                    required_score = cutoff
                    if team and (not used_team):
                        required_score = max(required_score, 0.95)
                    if (100.0 * required_score) <= 100.0 * cutoff:
                        # normal case
                        dk_name = candidates[candidates['Name_norm'] == close[0]].iloc[0]['Name']
                        best_score = float(100.0 * cutoff)
                    else:
                        # require higher similarity, so only accept if close actually very close
                        # compute a rough ratio by comparing the strings
                        # use difflib.SequenceMatcher ratio as fallback
                        import difflib as _dif
                        ratio = _dif.SequenceMatcher(None, key, close[0]).ratio()
                        if ratio >= 0.95:
                            dk_name = candidates[candidates['Name_norm'] == close[0]].iloc[0]['Name']
                            best_score = float(ratio * 100)

        matched.append(dk_name)
        scores.append(best_score)

    act['DK_Name'] = matched
    act['MATCH_SCORE'] = scores
    # attach DK fields
    act = act.merge(dk_df[['Name', 'Salary', 'Team', 'Name_norm']].rename(columns={'Name': 'DK_Name', 'Name_norm': 'DK_Name_norm', 'Salary': 'DK_Salary', 'Team': 'DK_Team'}), on='DK_Name', how='left')
    return act
