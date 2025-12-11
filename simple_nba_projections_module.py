"""
robust_nba_projections.py

Robust NBA DFS projection system that actually works.
"""

import argparse
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from nba_api.stats.endpoints import (playergamelogs, leaguedashteamstats, 
                                     scoreboardv2, leaguedashplayerstats)
from nba_api.stats.static import players, teams


class SimpleProjector:
    """Simple but effective projection system"""
    
    def __init__(self):
        self.team_stats = {}
        self.league_avg_pace = 100.0
        
    def parse_minutes(self, min_str):
        """Parse minutes string to float"""
        if pd.isna(min_str):
            return 0.0
        try:
            if isinstance(min_str, str) and ':' in min_str:
                parts = min_str.split(':')
                return float(parts[0]) + float(parts[1]) / 60
            return float(min_str)
        except:
            return 0.0
    
    def calculate_dk_points(self, row):
        """Calculate DK fantasy points"""
        return (
            (row.get('PTS', 0) or 0) +
            (row.get('REB', 0) or 0) * 1.2 +
            (row.get('AST', 0) or 0) * 1.5 +
            (row.get('STL', 0) or 0) * 3.0 +
            (row.get('BLK', 0) or 0) * 3.0 -
            (row.get('TOV', 0) or 0) +
            (row.get('FG3M', 0) or 0) * 0.5
        )
    
    def analyze_player(self, player_logs, team, injury_status, backup_boost=0):
        """Analyze player's recent performance"""
        if player_logs.empty or len(player_logs) < 3:
            return None
        
        # Sort by date
        player_logs = player_logs.sort_values('GAME_DATE', ascending=False)
        
        # Calculate stats for recent games (last 10 max)
        recent = player_logs.head(min(10, len(player_logs)))
        
        # Parse minutes and calculate FP
        recent['MIN_FLOAT'] = recent['MIN'].apply(self.parse_minutes)
        recent['DK_FP'] = recent.apply(self.calculate_dk_points, axis=1)
        
        # Filter games with actual minutes
        valid = recent[recent['MIN_FLOAT'] > 2]
        if valid.empty:
            return None
        
        mins = valid['MIN_FLOAT'].tolist()
        fps = valid['DK_FP'].tolist()
        
        # Calculate FP per minute
        fp_per_min = []
        for fp, minute in zip(fps, mins):
            if minute > 0:
                fp_per_min.append(fp / minute)
        
        stats = {
            'avg_min': np.mean(mins) if mins else 0,
            'last5_min': np.mean(mins[:5]) if len(mins) >= 5 else np.mean(mins) if mins else 0,
            'last3_min': np.mean(mins[:3]) if len(mins) >= 3 else np.mean(mins) if mins else 0,
            'median_min': np.median(mins) if mins else 0,
            
            'avg_fp': np.mean(fps) if fps else 0,
            'last5_fp': np.mean(fps[:5]) if len(fps) >= 5 else np.mean(fps) if fps else 0,
            'last3_fp': np.mean(fps[:3]) if len(fps) >= 3 else np.mean(fps) if fps else 0,
            
            'fp_per_min': np.mean(fp_per_min) if fp_per_min else 0,
            'fp_per_min_std': np.std(fp_per_min) if len(fp_per_min) > 1 else 0,
            
            'games': len(valid),
            'team': team,
            'injury_status': injury_status,
            'backup_boost': backup_boost
        }
        
        return stats
    
    def project(self, stats, vegas, matchup):
        """Project fantasy points for a player"""
        if not stats:
            return 0, 0
        
        # Project minutes
        base_min = (
            stats['last3_min'] * 0.4 +
            stats['last5_min'] * 0.3 +
            stats['avg_min'] * 0.2 +
            stats['median_min'] * 0.1
        )
        
        # Role adjustment
        avg_min = stats['avg_min']
        if avg_min >= 32:
            role_mult = 1.05
        elif avg_min >= 28:
            role_mult = 1.02
        elif avg_min >= 20:
            role_mult = 1.0
        elif avg_min >= 15:
            role_mult = 0.95
        else:
            role_mult = 0.9
        
        base_min *= role_mult
        
        # Injury adjustment
        injury = stats['injury_status']
        if injury:
            status = str(injury).upper()
            if status == 'QUESTIONABLE':
                base_min *= 0.8
            elif status in ['DOUBTFUL', 'DAY-TO-DAY', 'DTD']:
                base_min *= 0.5
            elif status == 'PROBABLE':
                base_min *= 0.9
        
        # Backup boost
        base_min += stats['backup_boost']
        
        # Pace adjustment
        team_pace = self.team_stats.get(stats['team'], {}).get('pace', self.league_avg_pace)
        opp_team = matchup.get('opponent') if matchup else None
        opp_pace = self.team_stats.get(opp_team, {}).get('pace', self.league_avg_pace) if opp_team else self.league_avg_pace
        
        pace_mult = (team_pace + opp_pace) / (2 * self.league_avg_pace)
        base_min *= pace_mult
        
        # Vegas adjustments
        spread = vegas.get('spread', 0)
        total = vegas.get('total', 220)
        
        # Blowout risk
        if abs(spread) >= 12:
            if spread > 0:  # Underdog
                if avg_min >= 28:
                    base_min *= 1.05
                else:
                    base_min *= 1.1
            else:  # Favorite
                if avg_min >= 28:
                    base_min *= 0.95
                else:
                    base_min *= 1.05
        
        # Bounds
        proj_min = max(0, min(40, base_min))
        
        # Project fantasy points
        base_fp = stats['fp_per_min'] * proj_min
        
        # Recent form
        recent_mult = stats['last3_fp'] / max(1, stats['avg_fp'])
        base_fp *= min(1.2, max(0.8, recent_mult))
        
        # Matchup adjustment
        if opp_team:
            opp_def = self.team_stats.get(opp_team, {}).get('def_rating', 110)
            def_mult = 110 / max(90, opp_def)
            base_fp *= def_mult
        
        # Vegas total
        total_mult = total / 220
        base_fp *= total_mult
        
        # Consistency
        if stats['fp_per_min_std'] > 0 and stats['fp_per_min'] > 0:
            volatility = stats['fp_per_min_std'] / stats['fp_per_min']
            if volatility > 0.5:
                base_fp *= 0.9
        
        proj_fp = max(0, base_fp)
        
        return round(proj_min, 1), round(proj_fp, 1)


class NBAProjectionSystem:
    """Main NBA projection system"""
    
    def __init__(self, dk_file: str, days: int = 30):
        self.dk_file = dk_file
        self.days = days
        self.dk_data = None
        self.team_stats = {}
        self.todays_games = {}
        self.vegas_odds = {}
        self.injuries = {}
        self.game_logs = pd.DataFrame()
        self.projector = SimpleProjector()
        
    def load_dk_salaries(self):
        """Load DK salaries from CSV"""
        print("💵 Loading DK salaries...")
        
        try:
            # Read the CSV
            df = pd.read_csv(self.dk_file)
            
            # Show what we have
            print(f"   Found columns: {list(df.columns)}")
            
            # Clean up the DataFrame
            df = df.copy()
            
            # Try to identify columns
            name_col = None
            salary_col = None
            team_col = None
            pos_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if 'name' in col_lower and 'id' not in col_lower:
                    name_col = col
                elif 'salary' in col_lower:
                    salary_col = col
                elif 'team' in col_lower or 'abbrev' in col_lower:
                    team_col = col
                elif 'position' in col_lower or 'pos' in col_lower:
                    pos_col = col
            
            # If we found columns, rename them for consistency
            rename_dict = {}
            if name_col:
                rename_dict[name_col] = 'Name'
                print(f"   Using '{name_col}' as Name column")
            if salary_col:
                rename_dict[salary_col] = 'Salary'
                print(f"   Using '{salary_col}' as Salary column")
            if team_col:
                rename_dict[team_col] = 'Team'
                print(f"   Using '{team_col}' as Team column")
            if pos_col:
                rename_dict[pos_col] = 'Position'
                print(f"   Using '{pos_col}' as Position column")
            
            if rename_dict:
                df = df.rename(columns=rename_dict)
            
            # Ensure we have required columns
            required = ['Name', 'Salary']
            for col in required:
                if col not in df.columns:
                    print(f"❌ Missing required column: {col}")
                    print(f"   Available columns: {list(df.columns)}")
                    return False
            
            # Clean the data
            if 'Name' in df.columns:
                df['Name'] = df['Name'].astype(str).str.strip()
            if 'Team' in df.columns:
                df['Team'] = df['Team'].astype(str).str.strip().str.upper()
            if 'Salary' in df.columns:
                df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
                df = df.dropna(subset=['Salary'])
            
            self.dk_data = df
            print(f"✅ Loaded {len(df)} players from DK")
            print(f"   Sample: {df[['Name', 'Salary', 'Team']].head(3).to_string(index=False)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading DK file: {e}")
            return False
    
    def fetch_team_stats(self):
        """Fetch team statistics from NBA API"""
        print("🏀 Fetching team stats...")
        
        try:
            stats = leaguedashteamstats.LeagueDashTeamStats(
                season='2024-25',
                per_mode_detailed='PerGame',
                timeout=30
            )
            df = stats.get_data_frames()[0]
            
            # Find team column
            team_col = None
            for col in df.columns:
                if 'TEAM' in col.upper():
                    team_col = col
                    break
            
            if not team_col:
                print("   ⚠️ Could not find team column")
                return
            
            for _, row in df.iterrows():
                team_name = str(row[team_col])
                
                # Convert to abbreviation
                team_abbr = self._team_name_to_abbr(team_name)
                
                # Get stats
                pace = row.get('PACE', 100) if 'PACE' in df.columns else 100
                def_rating = row.get('DEF_RATING', 110) if 'DEF_RATING' in df.columns else 110
                
                self.team_stats[team_abbr] = {
                    'pace': float(pace),
                    'def_rating': float(def_rating)
                }
            
            self.projector.team_stats = self.team_stats
            print(f"✅ Loaded stats for {len(self.team_stats)} teams")
            
        except Exception as e:
            print(f"❌ Error fetching team stats: {e}")
    
    def _team_name_to_abbr(self, team_name: str) -> str:
        """Convert team name to abbreviation"""
        team_name = str(team_name).strip().upper()
        
        mapping = {
            'ATLANTA HAWKS': 'ATL',
            'BOSTON CELTICS': 'BOS',
            'BROOKLYN NETS': 'BKN',
            'CHARLOTTE HORNETS': 'CHA',
            'CHICAGO BULLS': 'CHI',
            'CLEVELAND CAVALIERS': 'CLE',
            'DALLAS MAVERICKS': 'DAL',
            'DENVER NUGGETS': 'DEN',
            'DETROIT PISTONS': 'DET',
            'GOLDEN STATE WARRIORS': 'GSW',
            'HOUSTON ROCKETS': 'HOU',
            'INDIANA PACERS': 'IND',
            'LA CLIPPERS': 'LAC',
            'LOS ANGELES CLIPPERS': 'LAC',
            'LA LAKERS': 'LAL',
            'LOS ANGELES LAKERS': 'LAL',
            'MEMPHIS GRIZZLIES': 'MEM',
            'MIAMI HEAT': 'MIA',
            'MILWAUKEE BUCKS': 'MIL',
            'MINNESOTA TIMBERWOLVES': 'MIN',
            'NEW ORLEANS PELICANS': 'NOP',
            'NEW YORK KNICKS': 'NYK',
            'OKLAHOMA CITY THUNDER': 'OKC',
            'ORLANDO MAGIC': 'ORL',
            'PHILADELPHIA 76ERS': 'PHI',
            'PHOENIX SUNS': 'PHX',
            'PORTLAND TRAIL BLAZERS': 'POR',
            'SACRAMENTO KINGS': 'SAC',
            'SAN ANTONIO SPURS': 'SAS',
            'TORONTO RAPTORS': 'TOR',
            'UTAH JAZZ': 'UTA',
            'WASHINGTON WIZARDS': 'WAS'
        }
        
        # Try exact match
        if team_name in mapping:
            return mapping[team_name]
        
        # Try partial match
        for full_name, abbr in mapping.items():
            if full_name in team_name or team_name in full_name:
                return abbr
        
        # Extract from string
        words = team_name.split()
        for word in words:
            if len(word) == 3 and word.isalpha():
                return word
        
        # Default to first 3 letters
        return team_name[:3] if len(team_name) >= 3 else team_name
    
    def fetch_todays_games(self):
        """Fetch today's NBA games"""
        print("📅 Fetching today's games...")
        
        try:
            scoreboard = scoreboardv2.ScoreboardV2(timeout=30)
            games_df = scoreboard.get_data_frames()[0]
            
            if games_df.empty:
                print("   ⚠️ No games scheduled for today")
                return
            
            # Get all teams
            nba_teams = teams.get_teams()
            id_to_abbr = {t['id']: t['abbreviation'] for t in nba_teams}
            
            for _, game in games_df.iterrows():
                home_id = game['HOME_TEAM_ID']
                away_id = game['VISITOR_TEAM_ID']
                
                home_abbr = id_to_abbr.get(home_id)
                away_abbr = id_to_abbr.get(away_id)
                
                if home_abbr and away_abbr:
                    self.todays_games[away_abbr] = {'opponent': home_abbr}
                    self.todays_games[home_abbr] = {'opponent': away_abbr}
            
            print(f"✅ Found {len(self.todays_games)} teams playing today")
            if self.todays_games:
                games = []
                for team, data in list(self.todays_games.items())[:6:2]:
                    games.append(f"{team}@{data['opponent']}")
                print(f"   Games: {', '.join(games)}")
                
        except Exception as e:
            print(f"❌ Error fetching games: {e}")
    
    def load_injuries(self):
        """Load injury data"""
        print("🩹 Loading injuries...")
        
        try:
            from injurySrape import get_injuries
            self.injuries = get_injuries()
            print(f"✅ Loaded {len(self.injuries)} injuries")
        except ImportError:
            print("   ⚠️ injurySrape module not available")
            self.injuries = {}
        except Exception as e:
            print(f"   ⚠️ Error loading injuries: {e}")
            self.injuries = {}
    
    def fetch_game_logs(self):
        """Fetch game logs for all players"""
        print("📊 Fetching game logs...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days)
        date_from = start_date.strftime('%Y-%m-%d')
        date_to = end_date.strftime('%Y-%m-%d')
        
        print(f"   Date range: {date_from} to {date_to}")
        
        try:
            logs = playergamelogs.PlayerGameLogs(
                date_from_nullable=date_from,
                date_to_nullable=date_to,
                timeout=60
            )
            self.game_logs = logs.get_data_frames()[0]
            
            if not self.game_logs.empty:
                # Clean player names
                self.game_logs['PLAYER_NAME'] = self.game_logs['PLAYER_NAME'].astype(str).str.strip()
                self.game_logs['PLAYER_NAME_LOWER'] = self.game_logs['PLAYER_NAME'].str.lower()
                
                print(f"✅ Fetched {len(self.game_logs)} game logs")
                print(f"   Unique players: {self.game_logs['PLAYER_NAME'].nunique()}")
                
                # Show sample
                sample_players = self.game_logs['PLAYER_NAME'].unique()[:5]
                print(f"   Sample players: {', '.join(sample_players)}")
            else:
                print("❌ No game logs fetched")
                
        except Exception as e:
            print(f"❌ Error fetching game logs: {e}")
            raise
    
    def get_player_logs(self, player_name: str):
        """Get game logs for a specific player"""
        if self.game_logs.empty:
            return pd.DataFrame()
        
        player_lower = player_name.lower().strip()
        
        # Try exact match
        player_logs = self.game_logs[self.game_logs['PLAYER_NAME_LOWER'] == player_lower]
        
        if player_logs.empty:
            # Try fuzzy match
            for nba_name in self.game_logs['PLAYER_NAME'].unique():
                nba_lower = str(nba_name).lower()
                if (player_lower in nba_lower or nba_lower in player_lower or
                    player_lower.split()[0] == nba_lower.split()[0]):
                    player_logs = self.game_logs[self.game_logs['PLAYER_NAME'] == nba_name]
                    break
        
        return player_logs.copy()
    
    def calculate_backup_boosts(self):
        """Calculate backup boosts when starters are out"""
        print("⚡ Calculating backup boosts...")
        
        boosts = {}
        
        if not self.injuries:
            print("   No injuries loaded")
            return boosts
        
        # Find OUT players
        out_players = []
        for name, status in self.injuries.items():
            if str(status).upper() == 'OUT':
                out_players.append(name.lower())
        
        if not out_players:
            print("   No OUT players")
            return boosts
        
        print(f"   Found {len(out_players)} OUT players")
        
        # Group DK players by team and position
        team_positions = {}
        for _, row in self.dk_data.iterrows():
            name = str(row.get('Name', '')).strip()
            team = str(row.get('Team', '')).strip().upper()
            pos = str(row.get('Position', '')).strip()
            
            if not team or not pos:
                continue
            
            # Handle multi-position
            positions = [p.strip() for p in pos.split('/')] if '/' in pos else [pos]
            
            for position in positions:
                key = (team, position)
                if key not in team_positions:
                    team_positions[key] = []
                
                try:
                    salary = float(row.get('Salary', 0))
                except:
                    salary = 0
                
                team_positions[key].append({
                    'name': name.lower(),
                    'salary': salary
                })
        
        # Calculate boosts
        for out_name in out_players:
            # Find the OUT player in DK data
            out_player = None
            for _, row in self.dk_data.iterrows():
                if str(row.get('Name', '')).strip().lower() == out_name:
                    out_player = row
                    break
            
            if out_player is None:
                continue
            
            try:
                out_team = str(out_player.get('Team', '')).strip().upper()
                out_pos = str(out_player.get('Position', '')).strip()
                out_salary = float(out_player.get('Salary', 0))
            except:
                continue
            
            if not out_team or not out_pos:
                continue
            
            # Get positions
            out_positions = [p.strip() for p in out_pos.split('/')] if '/' in out_pos else [out_pos]
            
            for position in out_positions:
                key = (out_team, position)
                if key in team_positions:
                    candidates = team_positions[key]
                    
                    # Remove OUT player
                    candidates = [c for c in candidates if c['name'] != out_name]
                    
                    if candidates:
                        # Sort by salary (higher = more important)
                        candidates.sort(key=lambda x: x['salary'], reverse=True)
                        
                        # Boost top 2 max
                        num_boost = min(2, len(candidates))
                        
                        for i in range(num_boost):
                            backup = candidates[i]
                            boost = 0
                            
                            # Base boost
                            if out_salary > 8000:
                                boost = 6.0
                            elif out_salary > 6000:
                                boost = 4.5
                            else:
                                boost = 3.0
                            
                            # Position adjustments
                            if position == 'PG':
                                boost += 1.0
                            elif position == 'C':
                                boost += 0.5
                            
                            # Cheap backups
                            if backup['salary'] < 4500:
                                boost += 1.5
                            elif backup['salary'] < 5500:
                                boost += 0.5
                            
                            # Apply
                            if backup['name'] in boosts:
                                boosts[backup['name']] += boost
                            else:
                                boosts[backup['name']] = boost
        
        print(f"✅ Calculated boosts for {len(boosts)} players")
        return boosts
    
    def set_default_vegas_odds(self):
        """Set default Vegas odds"""
        print("🎲 Setting default Vegas odds...")
        
        for team in self.todays_games:
            self.vegas_odds[team] = {
                'total': 220,
                'spread': 0
            }
        
        print(f"✅ Set default odds for {len(self.vegas_odds)} teams")
    
    def run(self, output_file: str = None):
        """Run the projection system"""
        print("\n" + "="*60)
        print("NBA DFS PROJECTION SYSTEM")
        print("="*60 + "\n")
        
        # Step 1: Load DK data
        if not self.load_dk_salaries():
            print("❌ Failed to load DK data")
            return None
        
        # Step 2: Fetch game logs (most important!)
        self.fetch_game_logs()
        
        if self.game_logs.empty:
            print("❌ No game logs available")
            return None
        
        # Step 3: Fetch supporting data
        self.fetch_team_stats()
        self.fetch_todays_games()
        self.load_injuries()
        
        # Step 4: Set default Vegas odds
        self.set_default_vegas_odds()
        
        # Step 5: Calculate backup boosts
        backup_boosts = self.calculate_backup_boosts()
        
        # Step 6: Process players
        print(f"\n🎯 Processing {len(self.dk_data)} players...")
        
        results = []
        processed = 0
        skipped = 0
        
        for idx, row in self.dk_data.iterrows():
            name = str(row.get('Name', '')).strip()
            if not name:
                skipped += 1
                continue
            
            try:
                salary = float(row.get('Salary', 0))
            except:
                skipped += 1
                continue
            
            team = str(row.get('Team', '')).strip().upper()
            position = str(row.get('Position', '')).strip()
            name_lower = name.lower()
            
            # Skip OUT players
            injury_status = self.injuries.get(name, '')
            if str(injury_status).upper() == 'OUT':
                skipped += 1
                continue
            
            # Get player logs
            player_logs = self.get_player_logs(name)
            
            if player_logs.empty or len(player_logs) < 3:
                skipped += 1
                continue
            
            # Get boost
            boost = backup_boosts.get(name_lower, 0.0)
            
            # Analyze player
            stats = self.projector.analyze_player(player_logs, team, injury_status, boost)
            
            if not stats:
                skipped += 1
                continue
            
            # Get matchup and Vegas data
            matchup = self.todays_games.get(team, {})
            vegas = self.vegas_odds.get(team, {})
            
            # Project
            proj_min, proj_fp = self.projector.project(stats, vegas, matchup)
            
            if proj_fp <= 0:
                skipped += 1
                continue
            
            # Apply salary cap
            salary_cap = salary * 0.0068
            proj_fp = min(proj_fp, salary_cap)
            
            # Calculate value
            value = (proj_fp / salary * 1000) if salary > 0 else 0
            
            # Floor and ceiling
            floor = proj_fp * 0.7
            ceiling = proj_fp * 1.3
            
            # Add to results
            results.append({
                'Name': name,
                'Team': team,
                'Opponent': matchup.get('opponent', ''),
                'Position': position,
                'Salary': int(salary),
                'Projection': proj_fp,
                'Floor': round(floor, 1),
                'Ceiling': round(ceiling, 1),
                'Value': round(value, 2),
                'ProjMinutes': proj_min,
                'AvgMinutes': round(stats['avg_min'], 1),
                'FPM': round(stats['fp_per_min'], 3),
                'Games': stats['games'],
                'Injury': injury_status,
                'BackupBoost': boost
            })
            
            processed += 1
            
            if processed % 20 == 0:
                print(f"   Processed {processed} players...")
        
        print(f"\n📊 Processing complete:")
        print(f"   ✅ Processed: {processed}")
        print(f"   ⏭️  Skipped: {skipped}")
        
        if not results:
            print("❌ No projections generated!")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort by projection
        df = df.sort_values('Projection', ascending=False).reset_index(drop=True)
        
        # Display results
        print(f"\n🏆 TOP 20 PROJECTIONS:")
        display_cols = ['Name', 'Team', 'Position', 'Salary', 'Projection', 'Value', 'ProjMinutes']
        print(df[display_cols].head(20).to_string(index=False))
        
        # Show value plays
        value_df = df[df['Salary'] > 3000].copy()
        if len(value_df) > 0:
            value_df = value_df.sort_values('Value', ascending=False)
            print(f"\n💰 TOP 10 VALUE PLAYS:")
            print(value_df[['Name', 'Team', 'Position', 'Salary', 'Projection', 'Value']].head(10).to_string(index=False))
        
        # Save if requested
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\n💾 Saved to {output_file}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description='NBA DFS Projection System')
    parser.add_argument('--dk', required=True, help='Path to DraftKings salaries CSV')
    parser.add_argument('--days', type=int, default=30, help='Days of history to use')
    parser.add_argument('--out', default='nba_projections.csv', help='Output CSV path')
    
    args = parser.parse_args()
    
    print("\n🚀 NBA DFS PROJECTION SYSTEM")
    print("   Simple, robust, and effective\n")
    
    try:
        # Create system
        system = NBAProjectionSystem(args.dk, args.days)
        
        # Run projections
        df = system.run(output_file=args.out)
        
        if df is not None:
            print(f"\n✅ SUCCESS! Generated {len(df)} projections")
            print(f"   Output saved to: {args.out}")
            
            # Summary
            print(f"\n📈 SUMMARY STATS:")
            print(f"   Average projection: {df['Projection'].mean():.1f} points")
            print(f"   Top projection: {df['Projection'].max():.1f} points")
            print(f"   Best value: {df['Value'].max():.2f} pts/$1k")
            
        else:
            print("\n❌ Failed to generate projections")
            print("   Check the error messages above")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        print("   The system encountered an error")


if __name__ == '__main__':
    main()