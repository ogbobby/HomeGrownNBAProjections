# enhanced_projections_lineup_builder.py
import pandas as pd
import pulp
from nba_api.stats.endpoints import commonplayerinfo, playergamelogs, scoreboardv2, teamgamelogs, leaguedashteamstats
from nba_api.stats.static import teams, players
from datetime import datetime, timedelta
import sys
import os
import time

class EnhancedProjectionsNBAData:
    def __init__(self, dk_salaries_path="DKSalaries.csv"):
        self.players_data = []
        self.todays_games = []
        self.team_id_map = {}
        self.team_stats = {}
        self.game_pace_data = {}
        self.matchup_data = {}
        self.team_game_schedule = {}
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
        """Calculate projected pace for today's games and store matchup data"""
        print("   Calculating game pace projections and matchups...")
        
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
                    
                    # Get defensive ratings for matchup analysis
                    home_def_rating = self.team_stats.get(home_team, {}).get('def_rating', 110.0)
                    away_def_rating = self.team_stats.get(away_team, {}).get('def_rating', 110.0)
                    
                    # Calculate matchup difficulty
                    matchup_difficulty = self.calculate_matchup_difficulty(home_team, away_team)
                    
                    self.game_pace_data[f"{away_team}@{home_team}"] = {
                        'pace': avg_pace,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_pace': home_pace,
                        'away_pace': away_pace,
                        'home_def_rating': home_def_rating,
                        'away_def_rating': away_def_rating,
                        'matchup_difficulty': matchup_difficulty
                    }
                    
                    # Store individual team matchups
                    self.matchup_data[home_team] = {
                        'opponent': away_team,
                        'location': 'home',
                        'pace': avg_pace,
                        'opp_def_rating': away_def_rating,
                        'matchup_difficulty': matchup_difficulty['home_difficulty']
                    }
                    
                    self.matchup_data[away_team] = {
                        'opponent': home_team,
                        'location': 'away',
                        'pace': avg_pace,
                        'opp_def_rating': home_def_rating,
                        'matchup_difficulty': matchup_difficulty['away_difficulty']
                    }
                    
                    print(f"      {away_team} @ {home_team}: Pace {avg_pace:.1f}, Home Def: {home_def_rating:.1f}, Away Def: {away_def_rating:.1f}")
                    
        except Exception as e:
            print(f"❌ Error calculating game pace: {e}")

    def calculate_matchup_difficulty(self, home_team, away_team):
        """Calculate matchup difficulty for both teams"""
        try:
            home_def = self.team_stats.get(home_team, {}).get('def_rating', 110.0)
            away_def = self.team_stats.get(away_team, {}).get('def_rating', 110.0)
            
            # League average defense rating (lower is better defense)
            league_avg_def = 110.0
            
            # Calculate difficulty (higher number = harder matchup)
            home_difficulty = (away_def - league_avg_def) / 10  # Positive = easier, Negative = harder
            away_difficulty = (home_def - league_avg_def) / 10  # Positive = easier, Negative = harder
            
            return {
                'home_difficulty': home_difficulty,
                'away_difficulty': away_difficulty
            }
        except:
            return {'home_difficulty': 0, 'away_difficulty': 0}

    def get_team_game_schedule(self):
        """Get recent game schedule for all teams to check for back-to-backs"""
        print("   Getting team schedule data for back-to-back checking...")
        
        try:
            # Get last 7 days of games to check for back-to-backs
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # Format dates for API
            date_from = start_date.strftime('%Y-%m-%d')
            date_to = end_date.strftime('%Y-%m-%d')
            
            team_logs = teamgamelogs.TeamGameLogs(
                season_nullable='2024-25',
                date_from_nullable=date_from,
                date_to_nullable=date_to
            )
            schedule_df = team_logs.get_data_frames()[0]
            
            # Organize games by team
            for _, game in schedule_df.iterrows():
                team_abbr = game['TEAM_ABBREVIATION']
                game_date = pd.to_datetime(game['GAME_DATE'])
                
                if team_abbr not in self.team_game_schedule:
                    self.team_game_schedule[team_abbr] = []
                
                self.team_game_schedule[team_abbr].append(game_date)
            
            print(f"✅ Loaded schedule data for {len(self.team_game_schedule)} teams")
            
        except Exception as e:
            print(f"❌ Error getting team schedule: {e}")

    def check_back_to_back(self, team):
        """Check if a team is on a back-to-back"""
        try:
            if team not in self.team_game_schedule:
                return False
            
            team_games = sorted(self.team_game_schedule[team])
            if len(team_games) < 2:
                return False
            
            # Check if last game was yesterday
            last_game_date = team_games[-1]
            today = datetime.now().date()
            
            # If last game was yesterday, it's a back-to-back
            return (last_game_date.date() == today - timedelta(days=1))
            
        except:
            return False

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
        
        # Get team schedule for back-to-back checking
        self.get_team_game_schedule()
        
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

    def debug_column_check(self, all_logs_df):
        """Debug method to check available columns"""
        print("🔍 Available columns in game logs:")
        print(list(all_logs_df.columns))
        
        # Check a sample row to see actual data
        if not all_logs_df.empty:
            sample_row = all_logs_df.iloc[0]
            print("🔍 Sample row data:")
            for col in ['PLAYER_ID', 'PLAYER_NAME', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']:
                if col in sample_row.index:
                    print(f"   {col}: {sample_row[col]}")

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
            
            # Debug: Check columns
            self.debug_column_check(all_logs_df)
            
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
                
                # Health check - ensure we have required columns
                required_columns = ['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
                missing_columns = [col for col in required_columns if col not in recent_games.columns]
                if missing_columns:
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
                # Only print errors for debugging if needed
                # print(f"      Error processing {player.get('full_name', 'Unknown')}: {e}")
                continue
        
        print(f"   ✅ Processed {processed_count} players with enhanced projections")
        return player_data

    def calculate_robust_averages(self, recent_games):
        """Calculate more robust averages using median and trimmed means"""
        try:
            # Use median for minutes to avoid outlier games skewing projections
            median_minutes = recent_games['MIN'].median()
            
            # Calculate trimmed mean (remove best and worst game) for stats
            stats = {}
            
            # Map of our stat names to NBA API column names
            stat_mapping = {
                'points': 'PTS',
                'rebounds': 'REB', 
                'assists': 'AST',
                'steals': 'STL',
                'blocks': 'BLK',
                'turnovers': 'TOV',
                'field_goals_attempted': 'FGA',
                'free_throws_attempted': 'FTA',
                'plus_minus': 'PLUS_MINUS'
            }
            
            for stat_key, nba_column in stat_mapping.items():
                if nba_column in recent_games.columns:
                    values = sorted(recent_games[nba_column].tolist())
                    if len(values) >= 5:
                        # Remove best and worst game
                        trimmed_values = values[1:-1]
                        stats[stat_key] = sum(trimmed_values) / len(trimmed_values)
                    else:
                        stats[stat_key] = recent_games[nba_column].mean()
                else:
                    stats[stat_key] = 0
            
            stats['minutes'] = median_minutes
            return stats
            
        except Exception as e:
            print(f"      Error in robust averages: {e}")
            # Fallback to simple averages
            return {
                'points': recent_games['PTS'].mean() if 'PTS' in recent_games.columns else 0,
                'rebounds': recent_games['REB'].mean() if 'REB' in recent_games.columns else 0,
                'assists': recent_games['AST'].mean() if 'AST' in recent_games.columns else 0,
                'steals': recent_games['STL'].mean() if 'STL' in recent_games.columns else 0,
                'blocks': recent_games['BLK'].mean() if 'BLK' in recent_games.columns else 0,
                'turnovers': recent_games['TOV'].mean() if 'TOV' in recent_games.columns else 0,
                'minutes': recent_games['MIN'].mean() if 'MIN' in recent_games.columns else 0,
                'field_goals_attempted': recent_games['FGA'].mean() if 'FGA' in recent_games.columns else 0,
                'free_throws_attempted': recent_games['FTA'].mean() if 'FTA' in recent_games.columns else 0,
                'plus_minus': recent_games['PLUS_MINUS'].mean() if 'PLUS_MINUS' in recent_games.columns else 0
            }

    def calculate_volatility_score(self, recent_games):
        """Calculate how volatile a player's production is"""
        try:
            if len(recent_games) < 3:
                return 1.0
            
            # Calculate fantasy points for each game
            game_scores = []
            for _, game in recent_games.iterrows():
                fp = self.calculate_dk_points({
                    'points': game['PTS'],
                    'rebounds': game['REB'],
                    'assists': game['AST'],
                    'steals': game['STL'],
                    'blocks': game['BLK'],
                    'turnovers': game['TOV']
                })
                game_scores.append(fp)
            
            # Calculate coefficient of variation (standard deviation / mean)
            mean_score = sum(game_scores) / len(game_scores)
            if mean_score == 0:
                return 1.0
                
            std_dev = (sum((x - mean_score) ** 2 for x in game_scores) / len(game_scores)) ** 0.5
            volatility = std_dev / mean_score
            
            return min(2.0, max(0.1, volatility))
            
        except:
            return 1.0

    def calculate_consistency_rating(self, recent_games):
        """Calculate consistency rating (0-100) based on performance stability"""
        try:
            if len(recent_games) < 3:
                return 50
            
            # Calculate fantasy points for each game
            game_scores = []
            for _, game in recent_games.iterrows():
                fp = self.calculate_dk_points({
                    'points': game['PTS'],
                    'rebounds': game['REB'],
                    'assists': game['AST'],
                    'steals': game['STL'],
                    'blocks': game['BLK'],
                    'turnovers': game['TOV']
                })
                game_scores.append(fp)
            
            # Calculate how often player meets baseline (75% of average)
            avg_score = sum(game_scores) / len(game_scores)
            baseline = avg_score * 0.75
            meets_baseline = sum(1 for score in game_scores if score >= baseline)
            consistency_pct = (meets_baseline / len(game_scores)) * 100
            
            return min(100, max(0, consistency_pct))
            
        except:
            return 50

    def calculate_enhanced_projection(self, player, recent_games, position, team, dk_salary_info):
        """Calculate enhanced fantasy projection with reality checks and volatility adjustments"""
        try:
            # Calculate base averages from recent games with more conservative approach
            avg_stats = self.calculate_robust_averages(recent_games)
            
            # Skip players with very low minutes
            if avg_stats['minutes'] < 10:
                return None
            
            # Calculate advanced metrics
            usage_rate = self.calculate_usage_rate(recent_games, team)
            points_per_minute = self.calculate_points_per_minute(recent_games)
            fantasy_points_per_minute = self.calculate_fantasy_points_per_minute(recent_games)
            plus_minus_rating = self.calculate_plus_minus_rating(recent_games, avg_stats['minutes'])
            
            # Calculate player volatility and consistency
            volatility_score = self.calculate_volatility_score(recent_games)
            consistency_rating = self.calculate_consistency_rating(recent_games)
            
            # Calculate home/away splits
            home_away_splits = self.calculate_home_away_splits(recent_games)
            
            # Get today's matchup data
            matchup_info = self.matchup_data.get(team, {})
            location = matchup_info.get('location', 'home')
            opponent = matchup_info.get('opponent', 'UNK')
            matchup_difficulty = matchup_info.get('matchup_difficulty', 0)
            opp_def_rating = matchup_info.get('opp_def_rating', 110.0)
            
            # Check for back-to-back
            back_to_back = self.check_back_to_back(team)
            
            # Calculate projected minutes (more conservative with volatility consideration)
            projected_minutes = self.project_minutes(
                avg_stats['minutes'], recent_games, team, usage_rate, 
                plus_minus_rating, location, back_to_back, matchup_difficulty,
                volatility_score, consistency_rating
            )
            
            # Apply reality check to minutes projection
            projected_minutes = self.apply_minutes_reality_check(projected_minutes, player['full_name'], position, usage_rate)
            
            # Calculate per-minute rates with volatility adjustment
            per_36_stats = {}
            stat_columns = {
                'points': 'PTS',
                'rebounds': 'REB', 
                'assists': 'AST',
                'steals': 'STL',
                'blocks': 'BLK',
                'turnovers': 'TOV'
            }

            for stat, nba_col in stat_columns.items():
                if nba_col in recent_games.columns and avg_stats['minutes'] > 0:
                    # Use median instead of mean for more stable projections
                    median_stat = recent_games[nba_col].median()
                    per_36_stats[stat] = (median_stat / avg_stats['minutes']) * 36
                else:
                    per_36_stats[stat] = 0
            
            # Apply various adjustments with volatility consideration
            pace_adjustment = self.get_pace_adjustment(team)
            usage_adjustment = self.get_usage_adjustment(usage_rate)
            plus_minus_adjustment = self.get_plus_minus_adjustment(plus_minus_rating)
            matchup_adjustment = self.get_matchup_adjustment(opp_def_rating, location)
            home_away_adjustment = self.get_home_away_adjustment(location, home_away_splits)
            back_to_back_adjustment = self.get_back_to_back_adjustment(back_to_back, usage_rate)
            volatility_adjustment = self.get_volatility_adjustment(volatility_score, usage_rate)
            
            # Calculate adjusted per-36 stats
            pace_adjusted_stats = {}
            scoring_stats = ['points', 'assists']
            non_scoring_stats = ['rebounds', 'steals', 'blocks']
            
            for stat in scoring_stats:
                pace_adjusted_stats[stat] = per_36_stats[stat] * pace_adjustment * usage_adjustment * plus_minus_adjustment * matchup_adjustment * home_away_adjustment * back_to_back_adjustment * volatility_adjustment
            
            for stat in non_scoring_stats:
                pace_adjusted_stats[stat] = per_36_stats[stat] * pace_adjustment * plus_minus_adjustment * matchup_adjustment * home_away_adjustment * back_to_back_adjustment * volatility_adjustment
            
            pace_adjusted_stats['turnovers'] = per_36_stats['turnovers'] * (1 + (usage_adjustment - 1) * 0.3)
            
            # Calculate final projection with additional safety factors
            final_stats = {}
            for stat in pace_adjusted_stats:
                final_stats[stat] = (pace_adjusted_stats[stat] / 36) * projected_minutes
            
            # Apply efficiency adjustment
            efficiency_adjustment = self.calculate_efficiency_adjustment(recent_games)
            final_stats['points'] *= efficiency_adjustment
            
            # Calculate DK points from final stats
            projection = self.calculate_dk_points(final_stats)
            
            # Use ACTUAL DK salary
            salary = dk_salary_info['salary']
            
            # Apply final reality checks and safety factors
            projection = self.apply_projection_reality_checks(projection, player['full_name'], position, salary, usage_rate, consistency_rating, projected_minutes)
            
            if projection < 5:
                return None
            
            value_rating = (projection / salary) * 1000
            
            # Calculate bargain rating with consistency consideration
            bargain_rating = self.calculate_bargain_rating(projection, salary, usage_rate, plus_minus_rating, fantasy_points_per_minute, consistency_rating)
            
            return {
                'name': player['full_name'],
                'position': position,
                'team': team,
                'salary': salary,
                'projection': round(projection, 1),
                'minutes': round(avg_stats['minutes'], 1),
                'projected_minutes': round(projected_minutes, 1),
                'pace_adjustment': round(pace_adjustment, 3),
                'usage_rate': round(usage_rate, 3),
                'points_per_minute': round(points_per_minute, 2),
                'fantasy_points_per_minute': round(fantasy_points_per_minute, 2),
                'plus_minus_rating': round(plus_minus_rating, 1),
                'per_36_points': round(per_36_stats['points'], 1),
                'efficiency_adjustment': round(efficiency_adjustment, 3),
                'value_rating': round(value_rating, 2),
                'bargain_rating': round(bargain_rating, 1),
                'consistency_rating': round(consistency_rating, 1),
                'volatility_score': round(volatility_score, 2),
                'location': location,
                'opponent': opponent,
                'matchup_difficulty': round(matchup_difficulty, 2),
                'back_to_back': back_to_back,
                'home_away_adjustment': round(home_away_adjustment, 3),
                'matchup_adjustment': round(matchup_adjustment, 3),
                'back_to_back_adjustment': round(back_to_back_adjustment, 3),
                'volatility_adjustment': round(volatility_adjustment, 3),
                'games_used': len(recent_games),
                'source': 'enhanced_projections',
                'playing_today': True
            }
            
        except Exception as e:
            print(f"      Error calculating projection for {player['full_name']}: {e}")
            return None

    def calculate_usage_rate(self, player_games, team):
        """Calculate player usage rate (% of team possessions used by player)"""
        try:
            # Simplified usage rate calculation
            player_usage = (
                player_games['FGA'].mean() + 
                0.44 * player_games['FTA'].mean() + 
                player_games['TOV'].mean()
            )
            
            # Normalize to per 36 minutes
            avg_minutes = player_games['MIN'].mean()
            if avg_minutes > 0:
                usage_per_36 = (player_usage / avg_minutes) * 36
            else:
                usage_per_36 = 0
            
            # Convert to a rate (0-1 scale)
            usage_rate = usage_per_36 / 100  # Rough approximation
            
            # Cap reasonable bounds
            usage_rate = max(0.10, min(0.40, usage_rate))
            
            return usage_rate
            
        except Exception as e:
            # Return league average if calculation fails
            return 0.20

    def calculate_points_per_minute(self, player_games):
        """Calculate actual points per minute"""
        try:
            total_points = player_games['PTS'].sum()
            total_minutes = player_games['MIN'].sum()
            
            if total_minutes > 0:
                return total_points / total_minutes
            else:
                return 0
        except:
            return 0

    def calculate_fantasy_points_per_minute(self, player_games):
        """Calculate fantasy points per minute"""
        try:
            total_fantasy_points = 0
            total_minutes = player_games['MIN'].sum()
            
            for _, game in player_games.iterrows():
                fantasy_points = self.calculate_dk_points({
                    'points': game['PTS'],
                    'rebounds': game['REB'],
                    'assists': game['AST'],
                    'steals': game['STL'],
                    'blocks': game['BLK'],
                    'turnovers': game['TOV']
                })
                total_fantasy_points += fantasy_points
            
            if total_minutes > 0:
                return total_fantasy_points / total_minutes
            else:
                return 0
        except:
            return 0

    def calculate_plus_minus_rating(self, player_games, avg_minutes):
        """Calculate normalized plus/minus rating"""
        try:
            if 'PLUS_MINUS' not in player_games.columns:
                return 0.0
            
            # Calculate average plus/minus per game
            avg_plus_minus = player_games['PLUS_MINUS'].mean()
            
            # Normalize by minutes to get per-minute impact
            if avg_minutes > 0:
                normalized_pm = avg_plus_minus / avg_minutes
            else:
                normalized_pm = 0
            
            # Scale to a more meaningful rating (-10 to +10 scale)
            plus_minus_rating = normalized_pm * 36  # Convert to per-36 minutes
            
            # Cap at reasonable bounds
            plus_minus_rating = max(-15, min(15, plus_minus_rating))
            
            return plus_minus_rating
            
        except Exception as e:
            return 0.0

    def calculate_home_away_splits(self, player_games):
        """Calculate player performance splits for home vs away games"""
        try:
            home_games = player_games[player_games['MATCHUP'].str.contains('vs.', na=False)]
            away_games = player_games[player_games['MATCHUP'].str.contains('@', na=False)]
            
            splits = {
                'home_ppg': home_games['PTS'].mean() if len(home_games) > 0 else 0,
                'away_ppg': away_games['PTS'].mean() if len(away_games) > 0 else 0,
                'home_minutes': home_games['MIN'].mean() if len(home_games) > 0 else 0,
                'away_minutes': away_games['MIN'].mean() if len(away_games) > 0 else 0,
                'home_games': len(home_games),
                'away_games': len(away_games)
            }
            
            return splits
            
        except:
            return {'home_ppg': 0, 'away_ppg': 0, 'home_minutes': 0, 'away_minutes': 0, 'home_games': 0, 'away_games': 0}

    def get_usage_adjustment(self, usage_rate):
        """Get adjustment factor based on usage rate"""
        # Higher usage players tend to maintain production better
        base_usage = 0.20
        adjustment = 1.0 + (usage_rate - base_usage) * 0.5
        
        # Reasonable bounds
        return max(0.8, min(1.3, adjustment))

    def get_plus_minus_adjustment(self, plus_minus_rating):
        """Get adjustment factor based on plus/minus rating"""
        # Players with positive plus/minus tend to be more reliable
        # Adjustment: 1.0 at 0, scaling up for positive ratings
        adjustment = 1.0 + (plus_minus_rating * 0.02)  # 2% boost per plus/minus point
        
        # Reasonable bounds
        return max(0.7, min(1.3, adjustment))

    def get_matchup_adjustment(self, opp_def_rating, location):
        """Get adjustment factor based on opponent defense"""
        # League average defense rating
        league_avg_def = 110.0
        
        # Calculate adjustment (lower opponent defense = harder matchup)
        # If opponent defense is better than average (lower rating), reduce projection
        def_difference = league_avg_def - opp_def_rating
        adjustment = 1.0 + (def_difference * 0.01)  # 1% change per defense point
        
        # Reasonable bounds
        return max(0.8, min(1.2, adjustment))

    def get_home_away_adjustment(self, location, home_away_splits):
        """Get adjustment factor based on home/away location and player splits"""
        try:
            # Base home court advantage
            if location == 'home':
                base_adjustment = 1.02  # 2% boost at home
            else:
                base_adjustment = 0.98  # 2% reduction on road
            
            # Apply player-specific splits if we have enough data
            if home_away_splits['home_games'] >= 3 and home_away_splits['away_games'] >= 3:
                home_ppg = home_away_splits['home_ppg']
                away_ppg = home_away_splits['away_ppg']
                
                if home_ppg > 0 and away_ppg > 0:
                    split_ratio = home_ppg / away_ppg if location == 'home' else away_ppg / home_ppg
                    # Blend base adjustment with player-specific split
                    adjustment = (base_adjustment * 0.7) + (split_ratio * 0.3)
                    return max(0.9, min(1.1, adjustment))
            
            return base_adjustment
            
        except:
            return 1.0

    def get_back_to_back_adjustment(self, back_to_back, usage_rate):
        """Get adjustment factor for back-to-back games"""
        if not back_to_back:
            return 1.0
        
        # High usage players are more affected by back-to-backs
        if usage_rate > 0.25:
            return 0.95  # 5% reduction for high usage players
        elif usage_rate > 0.18:
            return 0.97  # 3% reduction for medium usage players
        else:
            return 0.99  # 1% reduction for low usage players

    def get_volatility_adjustment(self, volatility_score, usage_rate):
        """Adjust projection based on player volatility"""
        # High volatility players get downward adjustment
        # High usage players are less volatile typically
        base_adjustment = 1.0
        
        if volatility_score > 1.2:  # High volatility
            if usage_rate < 0.15:   # Low usage + high volatility = big reduction
                base_adjustment = 0.85
            else:                   # High usage + high volatility = moderate reduction
                base_adjustment = 0.92
        elif volatility_score > 0.8:  # Medium volatility
            base_adjustment = 0.95
        
        return base_adjustment

    def apply_minutes_reality_check(self, projected_minutes, player_name, position, usage_rate):
        """Apply reality checks to minutes projections"""
        # Cap minutes based on role and usage
        max_minutes = 36
        
        if usage_rate < 0.15:  # Low usage role player
            max_minutes = 28
        elif usage_rate < 0.20:  # Medium usage
            max_minutes = 32
        elif usage_rate < 0.25:  # High usage
            max_minutes = 36
        else:  # Star player
            max_minutes = 38
        
        # Additional position-based caps
        if position == 'C':
            max_minutes = min(max_minutes, 34)  # Centers often play fewer minutes
        
        return min(projected_minutes, max_minutes)

    def apply_projection_reality_checks(self, projection, player_name, position, salary, usage_rate, consistency_rating, projected_minutes):
        """Apply final reality checks to projections"""
        # Cap projections based on salary and role
        max_projection_by_salary = salary * 0.008  # $1,000 salary = 8x max projection
        
        if usage_rate < 0.15:  # Low usage role players
            max_projection_by_salary *= 0.8  # 20% reduction
        
        # Apply consistency adjustment
        consistency_adjustment = consistency_rating / 100
        
        # Apply minutes-based cap (assume 1.5 FPTS per minute maximum for non-stars)
        if usage_rate < 0.20:
            max_by_minutes = projected_minutes * 1.3
            projection = min(projection, max_by_minutes)
        
        projection = min(projection, max_projection_by_salary)
        projection *= consistency_adjustment
        
        return projection

    def calculate_bargain_rating(self, projection, salary, usage_rate, plus_minus_rating, fantasy_points_per_minute, consistency_rating):
        """Calculate comprehensive bargain rating with consistency consideration"""
        try:
            score = 0
            
            # Base value score (30% of total)
            if salary > 0:
                value_score = (projection / salary) * 1000
                value_component = min(30, (value_score - 2) * (30 / 6))
                score += max(0, value_component)
            
            # Usage rate component (20% of total)
            usage_component = min(20, usage_rate * 100)
            score += usage_component
            
            # Plus/minus component (15% of total)
            pm_component = min(15, (plus_minus_rating + 10) * (15 / 20))
            score += max(0, pm_component)
            
            # Efficiency component (15% of total)
            if fantasy_points_per_minute > 0:
                efficiency_component = min(15, (fantasy_points_per_minute - 0.8) * (15 / 0.7))
                score += max(0, efficiency_component)
            
            # Consistency component (20% of total)
            consistency_component = consistency_rating * 0.2
            score += consistency_component
            
            return min(100, max(0, score))
            
        except Exception as e:
            if salary > 0:
                simple_value = (projection / salary) * 1000
                return min(100, (simple_value - 2) * (100 / 6))
            return 50

    def project_minutes(self, recent_minutes, recent_games, team, usage_rate=None, plus_minus_rating=None, location='home', back_to_back=False, matchup_difficulty=0, volatility_score=1.0, consistency_rating=50):
        """Project minutes for tonight's game with usage rate, plus/minus, matchup, and back-to-back consideration"""
        # Base projection is recent average
        base_minutes = recent_minutes
        
        # Adjust for trends (last 3 games vs full average)
        if len(recent_games) >= 3:
            last_3_minutes = recent_games.head(3)['MIN'].mean()
            # If player's minutes are trending up, adjust slightly
            if last_3_minutes > recent_minutes:
                base_minutes = (recent_minutes + last_3_minutes) / 2
        
        # High usage players might see slightly reduced minutes in blowouts
        # but are less likely to see random DNP-CDs
        blowout_risk = 0.95  # Default slight reduction
        
        if usage_rate and usage_rate > 0.25:  # High usage players
            blowout_risk = 0.98  # Less reduction for stars
        elif usage_rate and usage_rate < 0.15:  # Low usage players
            blowout_risk = 0.92  # More reduction for role players
        
        # Players with positive plus/minus are more likely to maintain minutes
        if plus_minus_rating and plus_minus_rating > 2:
            blowout_risk += 0.02  # Slight boost for positive impact players
        elif plus_minus_rating and plus_minus_rating < -2:
            blowout_risk -= 0.02  # Slight reduction for negative impact players
        
        # Back-to-back games might reduce minutes, especially for high usage players
        if back_to_back:
            if usage_rate and usage_rate > 0.25:
                blowout_risk -= 0.03  # Extra reduction for stars on back-to-back
            else:
                blowout_risk -= 0.01  # Small reduction for role players
        
        # Tough matchups might reduce minutes for role players
        if matchup_difficulty < -0.5:  # Hard matchup
            if usage_rate and usage_rate < 0.18:
                blowout_risk -= 0.02  # Extra reduction for role players in tough matchups
        
        # High volatility players might see more minute variability
        if volatility_score > 1.2:
            blowout_risk -= 0.02
        
        # Cap minutes at reasonable levels
        projected_minutes = min(38, base_minutes * blowout_risk)
        projected_minutes = max(10, projected_minutes)  # Minimum floor
        
        return projected_minutes

    def calculate_efficiency_adjustment(self, player_games):
        """Adjust projection based on recent shooting efficiency"""
        try:
            # Check if required columns exist
            if 'FGM' not in player_games.columns or 'FGA' not in player_games.columns or 'FG3M' not in player_games.columns:
                return 1.0
                
            total_fgm = player_games['FGM'].sum()
            total_fga = player_games['FGA'].sum()
            total_3pm = player_games['FG3M'].sum()
            
            if total_fga > 0:
                efg_percent = (total_fgm + 0.5 * total_3pm) / total_fga
                
                # Compare to league average (~0.54)
                league_avg_efg = 0.54
                efficiency_ratio = efg_percent / league_avg_efg
                
                # Apply mild regression to mean
                adjusted_ratio = 0.7 + (0.3 * efficiency_ratio)
                
                return max(0.7, min(1.3, adjusted_ratio))
            else:
                return 1.0
        except:
            return 1.0

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
        """Show summary of loaded data with new metrics"""
        if not self.players_data:
            return
            
        print("\n📊 ENHANCED PROJECTIONS SUMMARY:")
        print("-" * 70)
        
        pos_count = {}
        team_count = {}
        home_players = 0
        away_players = 0
        b2b_players = 0
        
        for player in self.players_data:
            pos = player['position']
            team = player['team']
            pos_count[pos] = pos_count.get(pos, 0) + 1
            team_count[team] = team_count.get(team, 0) + 1
            
            if player.get('location') == 'home':
                home_players += 1
            else:
                away_players += 1
                
            if player.get('back_to_back'):
                b2b_players += 1
        
        print("Position Distribution:")
        for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
            count = pos_count.get(pos, 0)
            print(f"  {pos}: {count} players")
        
        print(f"\nTeams in Pool:")
        for team, count in sorted(team_count.items(), key=lambda x: x[1], reverse=True):
            b2b = " (B2B)" if self.check_back_to_back(team) else ""
            print(f"  {team}: {count} players{b2b}")
        
        print(f"\n📍 Location Distribution:")
        print(f"  Home: {home_players} players")
        print(f"  Away: {away_players} players")
        print(f"  Back-to-Back: {b2b_players} players")
        
        # Show projection metrics
        avg_proj_minutes = sum(p.get('projected_minutes', 0) for p in self.players_data) / len(self.players_data)
        avg_pace_adj = sum(p.get('pace_adjustment', 1) for p in self.players_data) / len(self.players_data)
        avg_usage = sum(p.get('usage_rate', 0) for p in self.players_data) / len(self.players_data)
        avg_ppm = sum(p.get('points_per_minute', 0) for p in self.players_data) / len(self.players_data)
        avg_fppm = sum(p.get('fantasy_points_per_minute', 0) for p in self.players_data) / len(self.players_data)
        avg_plus_minus = sum(p.get('plus_minus_rating', 0) for p in self.players_data) / len(self.players_data)
        avg_bargain = sum(p.get('bargain_rating', 0) for p in self.players_data) / len(self.players_data)
        avg_matchup_diff = sum(p.get('matchup_difficulty', 0) for p in self.players_data) / len(self.players_data)
        avg_consistency = sum(p.get('consistency_rating', 0) for p in self.players_data) / len(self.players_data)
        avg_volatility = sum(p.get('volatility_score', 0) for p in self.players_data) / len(self.players_data)
        
        print(f"\n📈 Advanced Metrics:")
        print(f"  Avg Projected Minutes: {avg_proj_minutes:.1f}")
        print(f"  Avg Pace Adjustment: {avg_pace_adj:.3f}")
        print(f"  Avg Usage Rate: {avg_usage:.3f}")
        print(f"  Avg Points/Min: {avg_ppm:.3f}")
        print(f"  Avg Fantasy Points/Min: {avg_fppm:.3f}")
        print(f"  Avg Plus/Minus Rating: {avg_plus_minus:.1f}")
        print(f"  Avg Bargain Rating: {avg_bargain:.1f}")
        print(f"  Avg Matchup Difficulty: {avg_matchup_diff:.2f}")
        print(f"  Avg Consistency Rating: {avg_consistency:.1f}")
        print(f"  Avg Volatility Score: {avg_volatility:.2f}")
        
        # Show top players by various metrics
        if len(self.players_data) >= 5:
            print(f"\n🏆 Top 5 by Usage Rate:")
            for player in sorted(self.players_data, key=lambda x: x.get('usage_rate', 0), reverse=True)[:5]:
                print(f"  {player['name']}: {player.get('usage_rate', 0):.3f}")
            
            print(f"\n⚡ Top 5 by Fantasy Points/Min:")
            for player in sorted(self.players_data, key=lambda x: x.get('fantasy_points_per_minute', 0), reverse=True)[:5]:
                print(f"  {player['name']}: {player.get('fantasy_points_per_minute', 0):.3f}")
            
            print(f"\n💰 Top 5 by Bargain Rating:")
            for player in sorted(self.players_data, key=lambda x: x.get('bargain_rating', 0), reverse=True)[:5]:
                b2b_indicator = " (B2B)" if player.get('back_to_back') else ""
                print(f"  {player['name']}: {player.get('bargain_rating', 0):.1f} (${player['salary']:,}){b2b_indicator}")
            
            print(f"\n📊 Top 5 by Plus/Minus Rating:")
            for player in sorted(self.players_data, key=lambda x: x.get('plus_minus_rating', 0), reverse=True)[:5]:
                print(f"  {player['name']}: {player.get('plus_minus_rating', 0):.1f}")
            
            print(f"\n🎯 Easiest Matchups:")
            for player in sorted(self.players_data, key=lambda x: x.get('matchup_difficulty', 0), reverse=True)[:5]:
                loc_indicator = "H" if player.get('location') == 'home' else "A"
                print(f"  {player['name']} ({loc_indicator} vs {player.get('opponent', 'UNK')}): {player.get('matchup_difficulty', 0):.2f}")
            
            print(f"\n💎 Most Consistent Players:")
            for player in sorted(self.players_data, key=lambda x: x.get('consistency_rating', 0), reverse=True)[:5]:
                print(f"  {player['name']}: {player.get('consistency_rating', 0):.1f}")

# EnhancedProjectionsNBAOptimizer class remains the same as before
class EnhancedProjectionsNBAOptimizer:
    def __init__(self, dk_salaries_path="DKSalaries.csv"):
        self.data = EnhancedProjectionsNBAData(dk_salaries_path)
        
    def build_lineups(self):
        """Build lineups using enhanced projections"""
        print("\n🚀 ENHANCED PROJECTIONS NBA DFS LINEUP BUILDER")
        print("Using minutes projections, pace adjustments, usage rate, plus/minus, bargain ratings, matchup data, home/away splits, and consistency metrics")
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
        print("\n" + "=" * 120)
        print("🏆 ENHANCED PROJECTIONS NBA DFS LINEUP")
        print("=" * 120)
        print("📈 Using minutes projections, pace adjustments, usage rate, plus/minus, bargain ratings, matchup data, home/away splits, and consistency metrics")
        print("=" * 120)
        
        for i, player in enumerate(lineup['players'], 1):
            value_score = (player['projection'] / player['salary']) * 1000
            pace_indicator = "⚡" if player.get('pace_adjustment', 1) > 1.02 else "🐢" if player.get('pace_adjustment', 1) < 0.98 else "➡️"
            usage_indicator = "🔥" if player.get('usage_rate', 0) > 0.25 else "📊" if player.get('usage_rate', 0) > 0.18 else "💧"
            pm_indicator = "🔼" if player.get('plus_minus_rating', 0) > 2 else "🔽" if player.get('plus_minus_rating', 0) < -2 else "➡️"
            bargain_indicator = "💎" if player.get('bargain_rating', 0) > 80 else "💰" if player.get('bargain_rating', 0) > 60 else "💵"
            location_indicator = "🏠" if player.get('location') == 'home' else "✈️"
            b2b_indicator = "🔄" if player.get('back_to_back') else ""
            matchup_indicator = "🎯" if player.get('matchup_difficulty', 0) > 0.5 else "🛡️" if player.get('matchup_difficulty', 0) < -0.5 else "⚖️"
            consistency_indicator = "📈" if player.get('consistency_rating', 0) > 80 else "📊" if player.get('consistency_rating', 0) > 60 else "📉"
            
            print(f"{i:2d}. {player['position']:2} | {player['name']:20} | "
                  f"${player['salary']:5,} | {player['projection']:5.1f} pts | "
                  f"Min: {player.get('projected_minutes', 0):2.0f} | "
                  f"PPM: {player.get('points_per_minute', 0):.2f} | "
                  f"USG: {player.get('usage_rate', 0):.2f} {usage_indicator} | "
                  f"PM: {player.get('plus_minus_rating', 0):+.1f} {pm_indicator} | "
                  f"BR: {player.get('bargain_rating', 0):2.0f} {bargain_indicator} | "
                  f"CON: {player.get('consistency_rating', 0):2.0f} {consistency_indicator} | "
                  f"{location_indicator}{b2b_indicator} vs {player.get('opponent', 'UNK'):3} {matchup_indicator} | "
                  f"{pace_indicator}")
    
        print("=" * 120)
        print(f"💵 Total Salary: ${lineup['total_salary']:,} / $50,000")
        print(f"📈 Total Projection: {lineup['total_projection']:.1f} fantasy points")
        print(f"⚡ Efficiency: {lineup['efficiency']:.2f} points per $1,000")
        
        # Enhanced metrics
        avg_proj_minutes = sum(p.get('projected_minutes', 0) for p in lineup['players']) / len(lineup['players'])
        avg_pace = sum(p.get('pace_adjustment', 1) for p in lineup['players']) / len(lineup['players'])
        avg_usage = sum(p.get('usage_rate', 0) for p in lineup['players']) / len(lineup['players'])
        avg_ppm = sum(p.get('points_per_minute', 0) for p in lineup['players']) / len(lineup['players'])
        avg_fppm = sum(p.get('fantasy_points_per_minute', 0) for p in lineup['players']) / len(lineup['players'])
        avg_plus_minus = sum(p.get('plus_minus_rating', 0) for p in lineup['players']) / len(lineup['players'])
        avg_bargain = sum(p.get('bargain_rating', 0) for p in lineup['players']) / len(lineup['players'])
        avg_matchup = sum(p.get('matchup_difficulty', 0) for p in lineup['players']) / len(lineup['players'])
        avg_consistency = sum(p.get('consistency_rating', 0) for p in lineup['players']) / len(lineup['players'])
        b2b_count = sum(1 for p in lineup['players'] if p.get('back_to_back'))
        home_count = sum(1 for p in lineup['players'] if p.get('location') == 'home')
        
        print(f"📊 Enhanced Metrics:")
        print(f"  Avg Projected Minutes: {avg_proj_minutes:.1f}")
        print(f"  Avg Pace Factor: {avg_pace:.3f}")
        print(f"  Avg Usage Rate: {avg_usage:.3f}")
        print(f"  Avg Points/Min: {avg_ppm:.3f}")
        print(f"  Avg FPTS/Min: {avg_fppm:.3f}")
        print(f"  Avg Plus/Minus: {avg_plus_minus:+.1f}")
        print(f"  Avg Bargain Rating: {avg_bargain:.1f}")
        print(f"  Avg Matchup Difficulty: {avg_matchup:.2f}")
        print(f"  Avg Consistency Rating: {avg_consistency:.1f}")
        print(f"  Home Players: {home_count}/{len(lineup['players'])}")
        print(f"  Back-to-Back Players: {b2b_count}/{len(lineup['players'])}")
        
        lineup_teams = set(p['team'] for p in lineup['players'])
        print(f"🏀 Teams: {', '.join(sorted(lineup_teams))}")
        print("=" * 120)

if __name__ == "__main__":
    print("🎯 ENHANCED PROJECTIONS NBA DFS LINEUP BUILDER")
    print("With minutes projections, pace adjustments, usage rate, plus/minus, bargain ratings, matchup data, home/away splits, and consistency metrics")
    print()
    
    dk_file = "DKSalaries.csv"
    if not os.path.exists(dk_file):
        print(f"❌ {dk_file} not found")
        print("Please make sure your DraftKings salaries file is in the same directory")
        sys.exit(1)
    
    try:
        from nba_api.stats.endpoints import commonplayerinfo, scoreboardv2, leaguedashteamstats, teamgamelogs
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