# enhanced_projections_lineup_builder.py
import pandas as pd
import pulp
from nba_api.stats.endpoints import commonplayerinfo, playergamelogs, scoreboardv2, teamgamelogs, leaguedashteamstats
from nba_api.stats.static import teams, players
from datetime import datetime
import sys
import os
import time  # Add this import

class EnhancedProjectionsNBAData:
    def __init__(self, dk_salaries_path="DKSalaries.csv"):
        self.players_data = []
        self.todays_games = []
        self.team_id_map = {}
        self.team_stats = {}
        self.game_pace_data = {}
        self.dk_salaries_path = dk_salaries_path
        self.dk_salaries_df = None
        self.setup_team_map()
        
    def setup_team_map(self):
        """Create mapping from team IDs to abbreviations"""
        nba_teams = teams.get_teams()
        for team in nba_teams:
            self.team_id_map[team['id']] = team['abbreviation']
    
    def load_dk_salaries(self):
        """Load DraftKings salaries from CSV file"""
        print("💰 Loading DraftKings salaries...")
        
        if not os.path.exists(self.dk_salaries_path):
            print(f"❌ DraftKings salaries file not found: {self.dk_salaries_path}")
            return False
        
        try:
            self.dk_salaries_df = pd.read_csv(self.dk_salaries_path)
            print(f"✅ Loaded {len(self.dk_salaries_df)} players from DraftKings salaries")
            
            # Debug: Show column names to help with troubleshooting
            print(f"   Columns in DK file: {list(self.dk_salaries_df.columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading DraftKings salaries: {e}")
            return False

    def get_team_stats_and_pace(self):
        """Get team statistics including pace for today's games"""
        print("📊 Getting team stats and pace data...")
        
        try:
            # Get team stats for current season
            team_stats = leaguedashteamstats.LeagueDashTeamStats(season='2024-25')
            team_stats_df = team_stats.get_data_frames()[0]
            
            # Debug: Show available columns
            print(f"   Available columns in team stats: {list(team_stats_df.columns)}")
            
            # Try different possible column names for team abbreviation
            team_abbr_col = None
            for col in ['TEAM_ABBREVIATION', 'TEAM_NAME', 'TEAM_ID']:
                if col in team_stats_df.columns:
                    team_abbr_col = col
                    break
            
            if not team_abbr_col:
                print("❌ Could not find team abbreviation column in stats data")
                return False
            
            # Store team stats for easy access
            for _, team in team_stats_df.iterrows():
                team_abbr = team[team_abbr_col]
                
                # Handle different column naming possibilities
                pace_col = 'PACE' if 'PACE' in team_stats_df.columns else None
                off_rating_col = 'OFF_RATING' if 'OFF_RATING' in team_stats_df.columns else 'ORTG' if 'ORTG' in team_stats_df.columns else None
                def_rating_col = 'DEF_RATING' if 'DEF_RATING' in team_stats_df.columns else 'DRTG' if 'DRTG' in team_stats_df.columns else None
                pts_col = 'PTS' if 'PTS' in team_stats_df.columns else 'PPG' if 'PPG' in team_stats_df.columns else None
                
                self.team_stats[team_abbr] = {
                    'pace': team[pace_col] if pace_col else 100.0,  # Default if not available
                    'off_rating': team[off_rating_col] if off_rating_col else 110.0,
                    'def_rating': team[def_rating_col] if def_rating_col else 110.0,
                    'avg_points': team[pts_col] if pts_col else 110.0
                }
            
            print(f"✅ Loaded stats for {len(self.team_stats)} teams")
            
            # Calculate game pace projections for today's matchups
            self.calculate_game_pace()
            
            return True
            
        except Exception as e:
            print(f"❌ Error getting team stats: {e}")
            import traceback
            traceback.print_exc()
            return False

    def calculate_game_pace(self):
        """Calculate projected pace for today's games"""
        print("   Calculating game pace projections...")
        
        try:
            # For each game, calculate combined pace factor
            scoreboard_data = scoreboardv2.ScoreboardV2()
            games_df = scoreboard_data.get_data_frames()[0]
            
            for _, game in games_df.iterrows():
                home_team_id = game['HOME_TEAM_ID']
                away_team_id = game['VISITOR_TEAM_ID']
                
                home_team = self.team_id_map.get(home_team_id)
                away_team = self.team_id_map.get(away_team_id)
                
                if home_team and away_team:
                    home_pace = self.team_stats.get(home_team, {}).get('pace', 100.0)
                    away_pace = self.team_stats.get(away_team, {}).get('pace', 100.0)
                    avg_pace = (home_pace + away_pace) / 2
                    
                    self.game_pace_data[f"{away_team}@{home_team}"] = {
                        'pace': avg_pace,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_pace': home_pace,
                        'away_pace': away_pace
                    }
                    
                    print(f"      {away_team} @ {home_team}: Projected Pace {avg_pace:.1f}")
                    
        except Exception as e:
            print(f"❌ Error calculating game pace: {e}")

    def get_todays_games(self):
        """Get today's NBA games to filter for active players"""
        print("📅 Getting today's NBA schedule...")
        
        try:
            scoreboard_data = scoreboardv2.ScoreboardV2()
            games_df = scoreboard_data.get_data_frames()[0]
            
            if games_df.empty:
                print("❌ No games found for today in scoreboard")
                return False
            
            print(f"📊 Found {len(games_df)} games in scoreboard")
            
            home_teams = []
            away_teams = []
            
            for _, game in games_df.iterrows():
                home_team_id = game['HOME_TEAM_ID']
                away_team_id = game['VISITOR_TEAM_ID']
                
                if home_team_id in self.team_id_map:
                    home_teams.append(self.team_id_map[home_team_id])
                if away_team_id in self.team_id_map:
                    away_teams.append(self.team_id_map[away_team_id])
            
            self.todays_games = list(set(home_teams + away_teams))
            
            if not self.todays_games:
                return False
            
            print(f"✅ Today's games: {', '.join(self.todays_games)}")
            return True
            
        except Exception as e:
            print(f"❌ Error getting schedule: {e}")
            return False

    def get_real_nba_data(self):
        """Get real NBA data with enhanced projections"""
        print("📊 Getting REAL NBA data with enhanced projections...")
        print("=" * 50)
        
        if not self.load_dk_salaries():
            return False
        
        if not self.get_todays_games():
            return False
        
        # Get team stats and pace data
        if not self.get_team_stats_and_pace():
            print("⚠️  Continuing without team pace data")
        
        try:
            season = self.get_current_season()
            print(f"📅 Current season: {season}")
            
            self.players_data = self.get_enhanced_player_projections(season)
            
            if self.players_data:
                print(f"✅ Successfully loaded {len(self.players_data)} players with enhanced projections")
                self.show_data_summary()
                return True
            else:
                print("❌ No player data retrieved")
                return False
                
        except Exception as e:
            print(f"💥 Error getting NBA data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_current_season(self):
        """Get current NBA season - FIXED VERSION"""
        today = datetime.now()
        current_year = today.year
        if today.month >= 10:
            return f"{current_year}-{str(current_year + 1)[-2:]}"
        else:
            return f"{current_year - 1}-{str(current_year)[-2:]}"

    def get_all_players_with_correct_teams(self):
        """Get ALL active players with their CORRECT current teams"""
        print("   🔍 Getting all players with correct teams...")
        
        all_active_players = [p for p in players.get_players() if p['is_active']]
        players_with_teams = []
        
        batch_size = 30
        total_batches = (len(all_active_players) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(all_active_players))
            batch = all_active_players[start_idx:end_idx]
            
            print(f"      Batch {batch_num + 1}/{total_batches} ({len(batch)} players)...")
            
            for player in batch:
                try:
                    player_info = commonplayerinfo.CommonPlayerInfo(player_id=player['id'])
                    info_df = player_info.get_data_frames()[0]
                    
                    if not info_df.empty and 'TEAM_ABBREVIATION' in info_df.columns:
                        team = info_df['TEAM_ABBREVIATION'].iloc[0]
                        if team and team != '':
                            player['team'] = team
                            players_with_teams.append(player)
                            
                except Exception as e:
                    # Silently continue if we can't get team info for a player
                    continue
            
            # Add delay between batches to avoid rate limiting
            if batch_num < total_batches - 1:
                time.sleep(1)
        
        print(f"   ✅ Found {len(players_with_teams)} players with team information")
        return players_with_teams

    def get_enhanced_player_projections(self, season):
        """Get player stats with enhanced projections including minutes and pace"""
        print("🔄 Getting enhanced player projections...")
        
        player_data = []
        
        # Get all players with teams
        print("   Step 1: Getting players with teams...")
        all_players_with_teams = self.get_all_players_with_correct_teams()
        
        if not all_players_with_teams:
            print("❌ No players with teams found")
            return []
            
        # Filter to today's players
        todays_players = []
        for player in all_players_with_teams:
            if player.get('team') in self.todays_games:
                todays_players.append(player)
        
        print(f"   🎯 Found {len(todays_players)} players on today's teams")
        
        if len(todays_players) == 0:
            print("❌ No players found for today's games")
            return []
        
        # Get game logs
        print("   Step 2: Getting game logs...")
        try:
            all_game_logs = playergamelogs.PlayerGameLogs(season_nullable=season)
            all_logs_df = all_game_logs.get_data_frames()[0]
            print(f"   📈 Loaded {len(all_logs_df)} total game logs")
        except Exception as e:
            print(f"❌ Error loading game logs: {e}")
            return []
        
        # Process players with enhanced projections
        print("   Step 3: Calculating enhanced projections...")
        processed_count = 0
        
        for i, player in enumerate(todays_players):
            if processed_count % 10 == 0:
                print(f"      Processing {i+1}/{len(todays_players)}...")
            
            try:
                player_id = player['id']
                player_name = player['full_name']
                team = player['team']
                
                # Find DK salary
                dk_salary_info = self.find_player_in_dk_salaries(player_name, team)
                if not dk_salary_info:
                    continue
                
                # Get player logs
                player_logs = all_logs_df[all_logs_df['PLAYER_ID'] == player_id]
                if player_logs.empty:
                    continue
                
                recent_games = player_logs.head(10)
                if len(recent_games) < 3:
                    continue
                
                # Health check
                most_recent_game = recent_games.iloc[0]
                game_date = pd.to_datetime(most_recent_game['GAME_DATE'])
                days_since_last_game = (datetime.now() - game_date).days
                
                if days_since_last_game > 7:
                    continue
                
                recent_minutes = recent_games['MIN'].mean()
                if recent_minutes < 5:
                    continue
                
                # Get position
                position = dk_salary_info.get('position')
                if not position:
                    position = self.get_player_position(player_id, player_name)
                
                # Calculate ENHANCED projection with minutes and pace
                projection_data = self.calculate_enhanced_projection(
                    player, recent_games, position, team, dk_salary_info
                )
                
                if projection_data:
                    player_data.append(projection_data)
                    processed_count += 1
                    
            except Exception as e:
                continue
        
        print(f"   ✅ Processed {processed_count} players with enhanced projections")
        return player_data

    def calculate_enhanced_projection(self, player, recent_games, position, team, dk_salary_info):
        """Calculate enhanced fantasy projection with minutes and pace adjustments"""
        try:
            # Calculate base averages from recent games
            avg_stats = {
                'points': recent_games['PTS'].mean(),
                'rebounds': recent_games['REB'].mean(),
                'assists': recent_games['AST'].mean(),
                'steals': recent_games['STL'].mean(),
                'blocks': recent_games['BLK'].mean(),
                'turnovers': recent_games['TOV'].mean(),
                'minutes': recent_games['MIN'].mean()
            }
            
            # Skip players with very low minutes
            if avg_stats['minutes'] < 10:
                return None
            
            # Calculate projected minutes (with some basic adjustments)
            projected_minutes = self.project_minutes(avg_stats['minutes'], recent_games, team)
            
            # Calculate per-minute rates
            per_36_stats = {}
            for stat in ['points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers']:
                if avg_stats['minutes'] > 0:
                    per_36_stats[stat] = (avg_stats[stat] / avg_stats['minutes']) * 36
                else:
                    per_36_stats[stat] = 0
            
            # Apply pace adjustment
            pace_adjustment = self.get_pace_adjustment(team)
            
            # Calculate pace-adjusted per-36 stats
            pace_adjusted_stats = {}
            for stat in ['points', 'rebounds', 'assists', 'steals', 'blocks']:
                pace_adjusted_stats[stat] = per_36_stats[stat] * pace_adjustment
            pace_adjusted_stats['turnovers'] = per_36_stats['turnovers']  # Turnovers don't scale with pace as much
            
            # Calculate final projection based on projected minutes
            final_stats = {}
            for stat in pace_adjusted_stats:
                final_stats[stat] = (pace_adjusted_stats[stat] / 36) * projected_minutes
            
            # Calculate DK points from final stats
            projection = self.calculate_dk_points(final_stats)
            
            if projection < 5:
                return None
            
            # Use ACTUAL DK salary
            salary = dk_salary_info['salary']
            
            return {
                'name': player['full_name'],
                'position': position,
                'team': team,
                'salary': salary,
                'projection': round(projection, 1),
                'minutes': round(avg_stats['minutes'], 1),
                'projected_minutes': round(projected_minutes, 1),
                'pace_adjustment': round(pace_adjustment, 3),
                'per_36_points': round(per_36_stats['points'], 1),
                'games_used': len(recent_games),
                'source': 'enhanced_projections',
                'playing_today': True
            }
            
        except Exception as e:
            return None

    def project_minutes(self, recent_minutes, recent_games, team):
        """Project minutes for tonight's game"""
        # Base projection is recent average
        base_minutes = recent_minutes
        
        # Adjust for trends (last 3 games vs full average)
        if len(recent_games) >= 3:
            last_3_minutes = recent_games.head(3)['MIN'].mean()
            # If player's minutes are trending up, adjust slightly
            if last_3_minutes > recent_minutes:
                base_minutes = (recent_minutes + last_3_minutes) / 2
        
        # Adjust for blowout risk based on team strength
        # Simple adjustment - in production you'd use Vegas lines
        blowout_risk = 0.95  # Default slight reduction
        
        # Cap minutes at reasonable levels
        projected_minutes = min(38, base_minutes * blowout_risk)
        projected_minutes = max(10, projected_minutes)  # Minimum floor
        
        return projected_minutes

    def get_pace_adjustment(self, team):
        """Get pace adjustment factor for a team"""
        league_avg_pace = 100.0  # Approximate league average
        
        if team in self.team_stats:
            team_pace = self.team_stats[team]['pace']
            # Calculate adjustment relative to league average
            pace_adjustment = team_pace / league_avg_pace
            return pace_adjustment
        
        return 1.0  # Default no adjustment

    def find_player_in_dk_salaries(self, player_name, team):
        """Find player in DraftKings salaries CSV - IMPROVED VERSION"""
        if self.dk_salaries_df is None:
            return None
        
        # Debug: Show what we're looking for
        print(f"      Looking for: {player_name} ({team})")
        
        name_variations = self.get_name_variations(player_name)
        
        for name_var in name_variations:
            # Exact match
            if 'Name' in self.dk_salaries_df.columns:
                mask = self.dk_salaries_df['Name'].str.lower() == name_var.lower()
                if mask.any():
                    player_row = self.dk_salaries_df[mask].iloc[0]
                    return self.extract_salary_info(player_row)
            
            # Partial match
            if 'Name' in self.dk_salaries_df.columns:
                mask = self.dk_salaries_df['Name'].str.lower().str.contains(name_var.lower(), na=False)
                if mask.any():
                    player_row = self.dk_salaries_df[mask].iloc[0]
                    return self.extract_salary_info(player_row)
        
        return None

    def extract_salary_info(self, player_row):
        """Extract salary information from player row"""
        try:
            # Handle different possible column names
            salary_col = None
            for col in ['Salary', 'salary']:
                if col in player_row.index:
                    salary_col = col
                    break
            
            position_col = None
            for col in ['Position', 'Roster Position', 'Pos']:
                if col in player_row.index:
                    position_col = col
                    break
            
            team_col = None
            for col in ['TeamAbbrev', 'Team', 'teamAbbrev']:
                if col in player_row.index:
                    team_col = col
                    break
            
            avg_points_col = None
            for col in ['AvgPointsPerGame', 'AvgPoints', 'FPTS']:
                if col in player_row.index:
                    avg_points_col = col
                    break
            
            return {
                'salary': int(player_row[salary_col]) if salary_col else 0,
                'position': player_row.get(position_col, ''),
                'team': player_row.get(team_col, ''),
                'avg_points': float(player_row[avg_points_col]) if avg_points_col else 0
            }
        except Exception as e:
            print(f"      Error extracting salary info: {e}")
            return None

    def get_name_variations(self, full_name):
        """Generate name variations for matching"""
        parts = full_name.split()
        variations = []
        
        variations.append(full_name)
        
        if len(parts) >= 2:
            variations.append(f"{parts[0]} {parts[-1]}")
        
        if len(parts) == 3:
            variations.append(f"{parts[0]} {parts[1][0]}. {parts[2]}")
            variations.append(f"{parts[0]} {parts[1][0]}.{parts[2]}")
        
        # Common name fixes
        name_fixes = {
            "D'Angelo": "D'Angelo Russell",
            "CJ": "CJ McCollum", 
            "KJ": "KJ Martin",
            "Jabari": "Jabari Smith"
        }
        
        for wrong, right in name_fixes.items():
            if wrong in full_name:
                variations.append(right)
        
        return variations

    def get_player_position(self, player_id, player_name):
        """Get player position"""
        try:
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            info_df = player_info.get_data_frames()[0]
            
            if not info_df.empty:
                if 'POSITION' in info_df.columns and not pd.isna(info_df['POSITION'].iloc[0]):
                    pos = str(info_df['POSITION'].iloc[0]).strip()
                    if pos and pos != 'nan':
                        return self.map_to_dfs_position(pos)
        
        except Exception:
            pass
        
        return self.estimate_position_from_name(player_name)

    def calculate_dk_points(self, stats):
        """Calculate DraftKings fantasy points"""
        return (
            stats.get('points', 0) * 1.0 +
            stats.get('rebounds', 0) * 1.2 +
            stats.get('assists', 0) * 1.5 +
            stats.get('steals', 0) * 3.0 +
            stats.get('blocks', 0) * 3.0 -
            stats.get('turnovers', 0) * 1.0
        )

    def map_to_dfs_position(self, nba_position):
        """Map NBA position to DFS position"""
        if not nba_position or nba_position == 'nan':
            return 'PG'
        
        nba_position = str(nba_position).strip().upper()
        
        position_map = {
            'G': 'PG', 'PG': 'PG',
            'G-F': 'SG', 'SG': 'SG',
            'F-G': 'SF', 'SF': 'SF', 'F': 'SF',
            'F-C': 'PF', 'PF': 'PF',
            'C-F': 'C', 'C': 'C'
        }
        
        if '-' in nba_position:
            primary_pos = nba_position.split('-')[0]
            return position_map.get(primary_pos, 'PG')
        
        return position_map.get(nba_position, 'PG')

    def estimate_position_from_name(self, player_name):
        """Estimate position based on known player names"""
        known_players = {
            'PG': ['stephen curry', 'luka doncic', 'trae young', 'damian lillard'],
            'SG': ['devin booker', 'donovan mitchell', 'jaylen brown', 'anthony edwards'],
            'SF': ['lebron james', 'kevin durant', 'jayson tatum', 'kawhi leonard'],
            'PF': ['giannis antetokounmpo', 'anthony davis', 'pascal siakam', 'zion williamson'],
            'C': ['nikola jokic', 'joel embiid', 'bam adebayo', 'rudy gobert']
        }
        
        player_name_lower = player_name.lower()
        for position, player_list in known_players.items():
            for known_player in player_list:
                if known_player in player_name_lower:
                    return position
        
        positions = ['PG', 'SG', 'SF', 'PF', 'C']
        return positions[hash(player_name) % len(positions)]

    def show_data_summary(self):
        """Show summary of loaded data"""
        if not self.players_data:
            return
            
        print("\n📊 ENHANCED PROJECTIONS SUMMARY:")
        print("-" * 40)
        
        pos_count = {}
        team_count = {}
        
        for player in self.players_data:
            pos = player['position']
            team = player['team']
            pos_count[pos] = pos_count.get(pos, 0) + 1
            team_count[team] = team_count.get(team, 0) + 1
        
        print("Position Distribution:")
        for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
            count = pos_count.get(pos, 0)
            print(f"  {pos}: {count} players")
        
        print(f"\nTeams in Pool:")
        for team, count in sorted(team_count.items(), key=lambda x: x[1], reverse=True):
            print(f"  {team}: {count} players")
        
        # Show projection metrics
        avg_proj_minutes = sum(p.get('projected_minutes', 0) for p in self.players_data) / len(self.players_data)
        avg_pace_adj = sum(p.get('pace_adjustment', 1) for p in self.players_data) / len(self.players_data)
        
        print(f"\n📈 Projection Metrics:")
        print(f"  Avg Projected Minutes: {avg_proj_minutes:.1f}")
        print(f"  Avg Pace Adjustment: {avg_pace_adj:.3f}")

class EnhancedProjectionsNBAOptimizer:
    def __init__(self, dk_salaries_path="DKSalaries.csv"):
        self.data = EnhancedProjectionsNBAData(dk_salaries_path)
        
    def build_lineups(self):
        """Build lineups using enhanced projections"""
        print("\n🚀 ENHANCED PROJECTIONS NBA DFS LINEUP BUILDER")
        print("Using minutes projections and pace adjustments")
        print("=" * 60)
        
        if not self.data.get_real_nba_data():
            print("💥 Failed to get enhanced NBA data")
            return False
            
        if len(self.data.players_data) < 8:
            print(f"💥 Only {len(self.data.players_data)} players found, need at least 8")
            return False
            
        lineup = self.optimize_lineup()
        if lineup:
            self.display_lineup(lineup)
            return True
        else:
            print("💥 Optimization failed")
            return False

    def optimize_lineup(self):
        """Optimize lineup using enhanced projections"""
        print("\n🧠 Optimizing lineup with enhanced projections...")
        
        prob = pulp.LpProblem("NBA_DFS_Enhanced", pulp.LpMaximize)
        player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)
        
        # Objective
        prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] for i in range(len(self.data.players_data))])
        
        # Constraints
        prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['salary'] for i in range(len(self.data.players_data))]) <= 50000
        prob += pulp.lpSum([player_vars[i] for i in range(len(self.data.players_data))]) == 8
        
        # Position constraints
        pg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PG']
        sg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SG']
        sf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SF']
        pf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PF']
        c_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'C']
        
        print(f"   Position availability:")
        print(f"   PG: {len(pg_players)}, SG: {len(sg_players)}, SF: {len(sf_players)}, PF: {len(pf_players)}, C: {len(c_players)}")
        
        if len(pg_players) < 1 or len(sg_players) < 1 or len(sf_players) < 1 or len(pf_players) < 1 or len(c_players) < 1:
            print("❌ Not enough players for all positions")
            return None
        
        prob += pulp.lpSum([player_vars[i] for i in pg_players]) >= 1
        prob += pulp.lpSum([player_vars[i] for i in sg_players]) >= 1
        prob += pulp.lpSum([player_vars[i] for i in sf_players]) >= 1
        prob += pulp.lpSum([player_vars[i] for i in pf_players]) >= 1
        prob += pulp.lpSum([player_vars[i] for i in c_players]) == 1
        
        guard_players = pg_players + sg_players
        forward_players = sf_players + pf_players
        prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
        prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3
        
        # Solve
        print("   Solving optimization...")
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            print("   ✅ Optimization successful!")
            return self.extract_lineup(player_vars)
        else:
            print("   ❌ No optimal solution found")
            return None

    def extract_lineup(self, player_vars):
        """Extract optimal lineup from solution"""
        lineup_players = []
        total_salary = 0
        total_projection = 0
        
        for i, player in enumerate(self.data.players_data):
            if player_vars[i].value() == 1:
                lineup_players.append(player)
                total_salary += player['salary']
                total_projection += player['projection']
        
        position_order = {'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5}
        lineup_players.sort(key=lambda x: position_order.get(x['position'], 6))
        
        return {
            'players': lineup_players,
            'total_salary': total_salary,
            'total_projection': total_projection,
            'efficiency': total_projection / total_salary * 1000
        }

    def display_lineup(self, lineup):
        """Display the optimized lineup with enhanced metrics"""
        print("\n" + "=" * 70)
        print("🏆 ENHANCED PROJECTIONS NBA DFS LINEUP")
        print("=" * 70)
        print("📈 Using minutes projections and pace adjustments")
        print("=" * 70)
        
        for i, player in enumerate(lineup['players'], 1):
            value_score = (player['projection'] / player['salary']) * 1000
            pace_indicator = "⚡" if player.get('pace_adjustment', 1) > 1.02 else "🐢" if player.get('pace_adjustment', 1) < 0.98 else "➡️"
            
            print(f"{i:2d}. {player['position']:2} | {player['name']:20} | "
                  f"${player['salary']:5,} | {player['projection']:5.1f} pts | "
                  f"Min: {player.get('projected_minutes', 0):2.0f} | "
                  f"{pace_indicator} | {player['team']:3}")
        
        print("=" * 70)
        print(f"💵 Total Salary: ${lineup['total_salary']:,} / $50,000")
        print(f"📈 Total Projection: {lineup['total_projection']:.1f} fantasy points")
        print(f"⚡ Efficiency: {lineup['efficiency']:.2f} points per $1,000")
        
        # Enhanced metrics
        avg_proj_minutes = sum(p.get('projected_minutes', 0) for p in lineup['players']) / len(lineup['players'])
        avg_pace = sum(p.get('pace_adjustment', 1) for p in lineup['players']) / len(lineup['players'])
        
        print(f"📊 Enhanced Metrics:")
        print(f"  Avg Projected Minutes: {avg_proj_minutes:.1f}")
        print(f"  Avg Pace Factor: {avg_pace:.3f}")
        
        lineup_teams = set(p['team'] for p in lineup['players'])
        print(f"🏀 Teams: {', '.join(sorted(lineup_teams))}")
        print("=" * 70)

if __name__ == "__main__":
    print("🎯 ENHANCED PROJECTIONS NBA DFS LINEUP BUILDER")
    print("With minutes projections and pace adjustments")
    print()
    
    dk_file = "DKSalaries.csv"
    if not os.path.exists(dk_file):
        print(f"❌ {dk_file} not found")
        print("Please make sure your DraftKings salaries file is in the same directory")
        sys.exit(1)
    
    try:
        from nba_api.stats.endpoints import commonplayerinfo, scoreboardv2, leaguedashteamstats
        print("✅ nba_api is installed and working")
        print(f"✅ Found {dk_file}")
    except ImportError:
        print("❌ nba_api not installed")
        print("Run: pip install nba_api")
        sys.exit(1)
    
    optimizer = EnhancedProjectionsNBAOptimizer(dk_file)
    success = optimizer.build_lineups()
    
    if not success:
        print("\n💥 Failed to build lineups")
        sys.exit(1)