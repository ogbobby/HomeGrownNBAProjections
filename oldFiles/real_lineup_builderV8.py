# real_lineup_builderV9.py
# enhanced_projections_lineup_builder.py
import pandas as pd
import pulp
from nba_api.stats.endpoints import commonplayerinfo, playergamelogs, scoreboardv2, teamgamelogs, leaguedashteamstats
from nba_api.stats.static import teams, players
from datetime import datetime, timedelta
import sys
import os
import time
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
        self.high_upside_players = []
        self.injury_report = {}
        self.team_spreads = {}
        self.player_game_logs_cache = {}
        self.contest_insights = {}
        self.setup_team_map()
        
        # Setup retry strategy for API calls
        self.session = self._create_session()
        
    def _create_session(self):
        """Create requests session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def setup_team_map(self):
        """Create mapping from team IDs to abbreviations"""
        nba_teams = teams.get_teams()
        for team in nba_teams:
            self.team_id_map[team['id']] = team['abbreviation']
    
    def safe_api_call(self, api_call, max_retries=3, delay=2):
        """Safely make API calls with retry logic and error handling"""
        for attempt in range(max_retries):
            try:
                print(f"   🔄 API attempt {attempt + 1}/{max_retries}...")
                result = api_call()
                print(f"   ✅ API call successful")
                return result
            except Exception as e:
                print(f"   ⚠️ API attempt {attempt + 1} failed: {str(e)[:100]}...")
                if attempt < max_retries - 1:
                    print(f"   💤 Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    print(f"   ❌ All API attempts failed")
                    raise e

    def load_injury_data(self):
        """Load injury data to identify potential opportunity players"""
        print("🏥 Loading injury and lineup data...")
        self.injury_report = {}
        
    def calculate_blowout_risk(self, team, opponent):
        """Calculate blowout risk based on team strength and spreads"""
        try:
            team_rating = self.team_stats.get(team, {}).get('off_rating', 110.0)
            opp_rating = self.team_stats.get(opponent, {}).get('off_rating', 110.0)
            
            expected_diff = team_rating - opp_rating
            
            if abs(expected_diff) > 10:
                blowout_risk = 0.7
            elif abs(expected_diff) > 5:
                blowout_risk = 0.4
            else:
                blowout_risk = 0.2
                
            return blowout_risk
        except:
            return 0.3

    def calculate_role_stability(self, player_games, player_name, position):
        """Calculate role stability score (0-100) based on minutes consistency"""
        try:
            if len(player_games) < 5:
                return 50
            
            minutes = player_games['MIN'].tolist()
            
            avg_minutes = np.mean(minutes)
            if avg_minutes == 0:
                return 30
                
            std_minutes = np.std(minutes)
            cv_minutes = std_minutes / avg_minutes
            
            stability_score = max(0, 100 - (cv_minutes * 100))
            
            if avg_minutes < 20:
                stability_score *= 0.8
            elif avg_minutes > 32:
                stability_score *= 1.1
                
            return min(100, stability_score)
            
        except Exception as e:
            return 50

    def calculate_recent_trend_factor(self, player_games, stat_type='PTS'):
        """Calculate if player is trending up or down in recent games"""
        try:
            if len(player_games) < 6:
                return 1.0
            
            recent_games = player_games.head(3)
            older_games = player_games.iloc[3:6]
            
            if len(older_games) == 0 or len(recent_games) == 0:
                return 1.0
                
            recent_avg = recent_games[stat_type].mean()
            older_avg = older_games[stat_type].mean()
            
            if older_avg == 0:
                return 1.0
                
            trend_ratio = recent_avg / older_avg
            
            if trend_ratio > 1.5:
                return 1.25
            elif trend_ratio < 0.7:
                return 0.8
            else:
                return trend_ratio
                
        except Exception as e:
            return 1.0

    def get_opportunity_rating(self, player_games, team, usage_rate):
        """Calculate opportunity rating based on team context and injuries"""
        try:
            rating = 50
            
            if len(player_games) >= 3:
                recent_minutes = player_games.head(3)['MIN'].mean()
                older_minutes = player_games.iloc[3:6]['MIN'].mean() if len(player_games) >= 6 else recent_minutes
                
                if recent_minutes > older_minutes * 1.15:
                    rating += 20
                elif recent_minutes > older_minutes * 1.05:
                    rating += 10
            
            if usage_rate > 0.25:
                rating += 15
            elif usage_rate < 0.15:
                rating -= 10
            
            stability = self.calculate_role_stability(player_games, "", "")
            rating = (rating + stability) / 2
            
            return min(100, max(0, rating))
            
        except Exception as e:
            return 50

    # NEW CASH GAME METHODS
    def calculate_injury_risk(self, player_games, player_name):
        """Calculate injury risk based on recent game patterns"""
        try:
            if len(player_games) < 3:
                return 0.3
                
            recent_minutes = player_games.head(3)['MIN'].tolist()
            avg_minutes = np.mean(recent_minutes)
            
            if any(minutes < 5 for minutes in recent_minutes):
                return 0.8
                
            minutes_std = np.std(recent_minutes)
            if minutes_std > 10:
                return 0.6
                
            return 0.1
            
        except:
            return 0.3

    def project_minutes_enhanced_v2(self, recent_minutes, recent_games, team, usage_rate, role_stability):
        """More conservative minutes projection for cash games"""
        base_minutes = recent_minutes * 0.8 + 25 * 0.2
        
        if usage_rate < 0.15:
            base_minutes = min(base_minutes, 28)
        elif usage_rate < 0.20:
            base_minutes = min(base_minutes, 32)
        else:
            base_minutes = min(base_minutes, 36)
            
        stability_factor = 0.8 + (role_stability / 100 * 0.4)
        projected_minutes = base_minutes * stability_factor
        
        return max(12, min(38, projected_minutes))

    def calculate_conservative_projection(self, player_games, base_projection, consistency_rating):
        """Apply conservative adjustments for cash games"""
        recent_fantasy_points = []
        for _, game in player_games.head(5).iterrows():
            fp = self.calculate_dk_points({
                'points': game['PTS'], 'rebounds': game['REB'], 
                'assists': game['AST'], 'steals': game['STL'], 
                'blocks': game['BLK'], 'turnovers': game['TOV'],
                'three_pointers_made': game.get('FG3M', 0)
            })
            recent_fantasy_points.append(fp)
        
        median_projection = np.median(recent_fantasy_points) if recent_fantasy_points else base_projection
        
        consistency_weight = consistency_rating / 100
        conservative_projection = (median_projection * consistency_weight + 
                                 base_projection * (1 - consistency_weight)) * 0.9
        
        return conservative_projection

    def filter_cash_game_players(self, players_data):
        """Filter players specifically for cash game safety"""
        cash_players = []
        
        for player in players_data:
            criteria_met = 0
            total_criteria = 5
            
            if player.get('role_stability', 0) > 70:
                criteria_met += 1
                
            if player.get('consistency_rating', 0) > 65:
                criteria_met += 1
                
            value_rating = (player['projection'] / player['salary']) * 1000
            if value_rating > 4.0:
                criteria_met += 1
                
            if not player.get('back_to_back', False):
                criteria_met += 1
                
            if player.get('injury_risk', 0) < 0.4:
                criteria_met += 1
            
            if criteria_met >= 3:
                player['cash_game_safe'] = True
                cash_players.append(player)
            else:
                player['cash_game_safe'] = False
        
        print(f"💰 Cash game safe players: {len(cash_players)}/{len(players_data)}")
        return players_data

    def validate_lineup_health(self, lineup):
        """Check if lineup has any injury risks before finalizing"""
        risky_players = []
        
        for player in lineup['players']:
            if player.get('injury_risk', 0) > 0.6:
                risky_players.append(player['name'])
        
        if risky_players:
            print(f"⚠️  LINEUP HEALTH WARNING: {risky_players}")
            return False
            
        return True

    # NEW CONTEST ANALYSIS METHODS
    def analyze_contest_results(self, results_file="cashGameContestResults.csv"):
        """Analyze contest results to improve projections"""
        print("📊 Analyzing contest results for optimization...")
        
        try:
            results_df = pd.read_csv(results_file)
            
            # Extract player performances
            player_performances = {}
            for _, row in results_df.iterrows():
                player_name = row['Player']
                fpts = row['FPTS']
                roster_pos = row['Roster Position']
                drafted_pct = row['%Drafted']
                
                if player_name not in player_performances:
                    player_performances[player_name] = []
                
                player_performances[player_name].append({
                    'fpts': fpts,
                    'position': roster_pos,
                    'ownership': drafted_pct
                })
            
            # Calculate average performance and ownership
            self.contest_insights = {}
            for player, performances in player_performances.items():
                avg_fpts = np.mean([p['fpts'] for p in performances])
                avg_ownership = np.mean([p['ownership'] for p in performances])
                
                self.contest_insights[player] = {
                    'avg_fpts': avg_fpts,
                    'avg_ownership': avg_ownership,
                    'games': len(performances)
                }
            
            print(f"✅ Analyzed {len(self.contest_insights)} players from contest results")
            
            # Show top performers
            top_players = sorted(self.contest_insights.items(), 
                               key=lambda x: x[1]['avg_fpts'], reverse=True)[:10]
            print("\n🏆 TOP CONTEST PERFORMERS:")
            for player, stats in top_players:
                print(f"   {player}: {stats['avg_fpts']:.1f} FPTS, {stats['avg_ownership']:.1f}% owned")
                
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing contest results: {e}")
            return False

    def apply_contest_insights_to_projections(self):
        """Adjust projections based on contest performance patterns"""
        if not hasattr(self, 'contest_insights') or not self.contest_insights:
            return
        
        print("🔄 Applying contest insights to projections...")
        
        adjustments_made = 0
        for player in self.players_data:
            player_name = player['name']
            
            if player_name in self.contest_insights:
                contest_stats = self.contest_insights[player_name]
                actual_fpts = contest_stats['avg_fpts']
                ownership = contest_stats['avg_ownership']
                
                # Calculate projection accuracy
                projection = player['projection']
                accuracy_ratio = actual_fpts / projection if projection > 0 else 1.0
                
                # Adjust projection based on historical accuracy
                if 0.7 <= accuracy_ratio <= 1.3:  # Reasonable range
                    # Slight adjustment toward actual performance
                    adjusted_projection = (projection * 0.7) + (actual_fpts * 0.3)
                else:
                    # Larger adjustment for significant misses
                    adjusted_projection = (projection * 0.5) + (actual_fpts * 0.5)
                
                player['projection'] = round(adjusted_projection, 1)
                player['contest_ownership'] = ownership
                player['projection_accuracy'] = accuracy_ratio
                adjustments_made += 1
        
        print(f"   Applied contest insights to {adjustments_made} players")

    def enhanced_cash_game_filters(self, players_data):
        """Enhanced cash game filtering based on contest results"""
        cash_players = []
        
        for player in players_data:
            cash_score = 0
            
            # Ownership-based scoring (high ownership = safer)
            ownership = player.get('contest_ownership', 0)
            if ownership > 80:
                cash_score += 30
            elif ownership > 60:
                cash_score += 20
            elif ownership > 40:
                cash_score += 10
            
            # Consistency scoring
            consistency = player.get('consistency_rating', 0)
            if consistency > 75:
                cash_score += 25
            elif consistency > 60:
                cash_score += 15
            
            # Role stability
            stability = player.get('role_stability', 0)
            if stability > 75:
                cash_score += 20
            elif stability > 60:
                cash_score += 10
            
            # Minutes projection
            minutes = player.get('projected_minutes', 0)
            if minutes > 30:
                cash_score += 15
            elif minutes > 25:
                cash_score += 10
            
            # Value rating
            value_rating = (player['projection'] / player['salary']) * 1000
            if value_rating > 5.0:
                cash_score += 20
            elif value_rating > 4.0:
                cash_score += 15
            elif value_rating > 3.5:
                cash_score += 10
            
            # Injury risk penalty
            injury_risk = player.get('injury_risk', 0.3)
            if injury_risk > 0.5:
                cash_score -= 25
            elif injury_risk > 0.3:
                cash_score -= 15
            
            # Volatility penalty
            volatility = player.get('volatility_score', 1.0)
            if volatility > 1.2:
                cash_score -= 15
            elif volatility > 1.0:
                cash_score -= 10
            
            player['cash_game_score'] = cash_score
            player['cash_game_safe'] = cash_score >= 60
            
            if cash_score >= 60:
                cash_players.append(player)
        
        print(f"💰 Enhanced cash game players: {len(cash_players)}/{len(players_data)}")
        
        # Show top cash game players
        top_cash = sorted(cash_players, key=lambda x: x.get('cash_game_score', 0), reverse=True)[:10]
        print("\n🏆 TOP CASH GAME PLAYS:")
        for player in top_cash:
            print(f"   {player['name']}: Score {player.get('cash_game_score', 0)}, "
                  f"Ownership {player.get('contest_ownership', 0):.1f}%, "
                  f"Value {(player['projection']/player['salary'])*1000:.2f}")
        
        return players_data

    # NEW UPSIDE METHODS
    def get_player_recent_games(self, player_name, all_logs_df):
        """Get recent games for a player"""
        if player_name in self.player_game_logs_cache:
            return self.player_game_logs_cache[player_name]
            
        player_logs = all_logs_df[all_logs_df['PLAYER_NAME'] == player_name]
        if len(player_logs) > 0:
            self.player_game_logs_cache[player_name] = player_logs.head(10)
            return player_logs.head(10)
        return pd.DataFrame()

    def calculate_realistic_upside_score(self, player_games, current_projection, salary, role_stability, injury_risk):
        """Calculate REALISTIC upside score"""
        
        if len(player_games) < 3:
            return 0
            
        recent_fantasy_points = []
        for _, game in player_games.head(8).iterrows():
            if game['MIN'] < 15:
                continue
            fp = self.calculate_dk_points({
                'points': game['PTS'], 'rebounds': game['REB'], 
                'assists': game['AST'], 'steals': game['STL'], 
                'blocks': game['BLK'], 'turnovers': game['TOV'],
                'three_pointers_made': game.get('FG3M', 0)
            })
            recent_fantasy_points.append(fp)
        
        if len(recent_fantasy_points) < 3:
            return 0
        
        realistic_ceiling = np.percentile(recent_fantasy_points, 90)
        upside_gap = realistic_ceiling - current_projection
        
        if upside_gap <= 0:
            gap_score = 0
        else:
            gap_score = min(50, (upside_gap / current_projection) * 100)
        
        risk_penalty = 0
        
        if injury_risk > 0.6:
            risk_penalty += 25
        elif injury_risk > 0.3:
            risk_penalty += 15
        
        if role_stability < 60:
            risk_penalty += 20
        elif role_stability < 75:
            risk_penalty += 10
        
        salary_penalty = max(0, (salary - 6000) / 1000) * 5
        
        final_upside_score = max(0, gap_score - risk_penalty - salary_penalty)
        
        return final_upside_score

    def validate_upside_play(self, player_data, player_games):
        """Validate if a player is truly a good upside play"""
        
        if len(player_games) < 3:
            return False
            
        validation_score = 0
        
        recent_ceiling_games = 0
        for _, game in player_games.head(5).iterrows():
            if game['MIN'] < 15:
                continue
            fp = self.calculate_dk_points({
                'points': game['PTS'], 'rebounds': game['REB'], 'assists': game['AST'],
                'steals': game['STL'], 'blocks': game['BLK'], 'turnovers': game['TOV'],
                'three_pointers_made': game.get('FG3M', 0)
            })
            if fp >= player_data['projection'] * 1.3:
                recent_ceiling_games += 1
        
        if recent_ceiling_games >= 2:
            validation_score += 40
        elif recent_ceiling_games >= 1:
            validation_score += 25
        
        if len(player_games) >= 3:
            last_3_minutes = player_games.head(3)['MIN'].mean()
            prev_3_minutes = player_games.iloc[3:6]['MIN'].mean() if len(player_games) >= 6 else last_3_minutes
            if last_3_minutes > prev_3_minutes * 1.1:
                validation_score += 20
        
        if player_data.get('opportunity_rating', 0) > 70:
            validation_score += 25
        elif player_data.get('opportunity_rating', 0) > 60:
            validation_score += 15
        
        risk_deduction = 0
        if player_data.get('injury_risk', 0) > 0.4:
            risk_deduction += 15
        if player_data.get('back_to_back', False):
            risk_deduction += 10
        if player_data.get('role_stability', 0) < 60:
            risk_deduction += 10
        
        validation_score = max(0, validation_score - risk_deduction)
        
        return validation_score >= 60

    def validate_upside_lineup(self, lineup):
        """Validate that the lineup meets upside criteria"""
        
        upside_players = 0
        total_risk_score = 0
        
        for player in lineup['players']:
            if player.get('realistic_upside_score', 0) >= 60:
                upside_players += 1
            
            risk_score = player.get('injury_risk', 0.3) * 50
            if player.get('back_to_back', False):
                risk_score += 20
            if player.get('role_stability', 0) < 60:
                risk_score += 20
            
            total_risk_score += risk_score
        
        if upside_players < 5:
            print(f"❌ Upside validation failed: Only {upside_players}/8 upside players")
            return False
        
        if total_risk_score > 300:
            print(f"❌ Upside validation failed: Risk score {total_risk_score} too high")
            return False
        
        print(f"✅ Upside validation passed: {upside_players}/8 upside players, risk score: {total_risk_score}")
        return True

    def load_dk_salaries(self):
        """Load DraftKings salaries from CSV file"""
        print("💰 Loading DraftKings salaries...")
        
        if not os.path.exists(self.dk_salaries_path):
            print(f"❌ DraftKings salaries file not found: {self.dk_salaries_path}")
            return False
        
        try:
            self.dk_salaries_df = pd.read_csv(self.dk_salaries_path)
            print(f"✅ Loaded {len(self.dk_salaries_df)} players from DraftKings salaries")
            print(f"   Columns in DK file: {list(self.dk_salaries_df.columns)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading DraftKings salaries: {e}")
            return False

    def get_team_stats_and_pace(self):
        """Get team statistics including pace for today's games"""
        print("📊 Getting team stats and pace data...")
        
        try:
            def api_call():
                team_stats = leaguedashteamstats.LeagueDashTeamStats(season='2024-25', timeout=60)
                return team_stats.get_data_frames()[0]
            
            team_stats_df = self.safe_api_call(api_call, max_retries=3, delay=3)
            
            print(f"   Available columns in team stats: {list(team_stats_df.columns)}")
            
            team_abbr_col = None
            for col in ['TEAM_ABBREVIATION', 'TEAM_NAME', 'TEAM_ID']:
                if col in team_stats_df.columns:
                    team_abbr_col = col
                    break
            
            if not team_abbr_col:
                print("❌ Could not find team abbreviation column in stats data")
                return False
            
            for _, team in team_stats_df.iterrows():
                team_abbr = team[team_abbr_col]
                
                pace_col = 'PACE' if 'PACE' in team_stats_df.columns else None
                off_rating_col = 'OFF_RATING' if 'OFF_RATING' in team_stats_df.columns else 'ORTG' if 'ORTG' in team_stats_df.columns else None
                def_rating_col = 'DEF_RATING' if 'DEF_RATING' in team_stats_df.columns else 'DRTG' if 'DRTG' in team_stats_df.columns else None
                pts_col = 'PTS' if 'PTS' in team_stats_df.columns else 'PPG' if 'PPG' in team_stats_df.columns else None
                
                self.team_stats[team_abbr] = {
                    'pace': team[pace_col] if pace_col else 100.0,
                    'off_rating': team[off_rating_col] if off_rating_col else 110.0,
                    'def_rating': team[def_rating_col] if def_rating_col else 110.0,
                    'avg_points': team[pts_col] if pts_col else 110.0
                }
            
            print(f"✅ Loaded stats for {len(self.team_stats)} teams")
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
                    
                    home_def_rating = self.team_stats.get(home_team, {}).get('def_rating', 110.0)
                    away_def_rating = self.team_stats.get(away_team, {}).get('def_rating', 110.0)
                    
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
            
            league_avg_def = 110.0
            
            home_difficulty = (away_def - league_avg_def) / 10
            away_difficulty = (home_def - league_avg_def) / 10
            
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
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            date_from = start_date.strftime('%Y-%m-%d')
            date_to = end_date.strftime('%Y-%m-%d')
            
            team_logs = teamgamelogs.TeamGameLogs(
                season_nullable='2024-25',
                date_from_nullable=date_from,
                date_to_nullable=date_to
            )
            schedule_df = team_logs.get_data_frames()[0]
            
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
            
            last_game_date = team_games[-1]
            today = datetime.now().date()
            
            return (last_game_date.date() == today - timedelta(days=1))
            
        except:
            return False

    def get_todays_games(self):
        """Get today's NBA games to filter for active players"""
        print("📅 Getting today's NBA schedule...")
        
        try:
            def api_call():
                scoreboard_data = scoreboardv2.ScoreboardV2(timeout=60)
                return scoreboard_data.get_data_frames()[0]
            
            games_df = self.safe_api_call(api_call, max_retries=3, delay=3)
            
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
                print("⚠️  No teams found for today's games")
                return False
            
            print(f"✅ Today's games: {', '.join(self.todays_games)}")
            return True
            
        except Exception as e:
            print(f"❌ Error getting schedule: {e}")
            self.todays_games = list(self.team_id_map.values())
            return True

    def get_real_nba_data(self):
        """Get real NBA data with enhanced projections - ROBUST VERSION"""
        print("📊 Getting REAL NBA data with ENHANCED PROJECTIONS v3...")
        print("=" * 50)
        
        if not self.load_dk_salaries():
            print("❌ Cannot continue without DraftKings salaries")
            return False
        
        schedule_success = self.get_todays_games()
        if not schedule_success:
            print("⚠️  Could not get today's schedule, using all teams")
            self.todays_games = list(self.team_id_map.values())
        
        stats_success = self.get_team_stats_and_pace()
        if not stats_success:
            print("⚠️  Continuing without team stats data")
        
        try:
            self.get_team_game_schedule()
        except Exception as e:
            print(f"⚠️  Could not get team schedule: {e}")
        
        self.load_injury_data()
        
        try:
            season = self.get_current_season()
            print(f"📅 Current season: {season}")
            
            self.players_data = self.get_enhanced_player_projections(season)
            
            if self.players_data:
                print(f"✅ Successfully loaded {len(self.players_data)} players with ENHANCED projections v3")
                self.identify_high_upside_players()
                self.show_data_summary_enhanced()
                return True
            else:
                print("❌ No player data retrieved")
                return self.create_fallback_data()
                
        except Exception as e:
            print(f"💥 Error getting NBA data: {e}")
            print("🆘 Attempting fallback data creation...")
            return self.create_fallback_data()

    def get_real_nba_data_enhanced(self):
        """Enhanced data collection with contest analysis"""
        print("📊 Getting REAL NBA data with CONTEST INSIGHTS...")
        
        success = self.get_real_nba_data()
        
        if success:
            # Analyze previous contest results
            self.analyze_contest_results()
            self.apply_contest_insights_to_projections()
            self.players_data = self.enhanced_cash_game_filters(self.players_data)
        
        return success

    def get_current_season(self):
        """Get current NBA season - FIXED VERSION"""
        today = datetime.now()
        current_year = today.year
        if today.month >= 10:
            return f"{current_year}-{str(current_year + 1)[-2:]}"
        else:
            return f"{current_year - 1}-{str(current_year)[-2:]}"

    def get_all_players_with_correct_teams(self):
        """Get ALL active players with their CORRECT current teams - WITH FALLBACK"""
        print("   🔍 Getting all players with correct teams...")
        
        all_active_players = [p for p in players.get_players() if p['is_active']]
        players_with_teams = []
        
        batch_size = 20
        total_batches = (len(all_active_players) + batch_size - 1) // batch_size
        
        successful_batches = 0
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(all_active_players))
            batch = all_active_players[start_idx:end_idx]
            
            print(f"      Batch {batch_num + 1}/{total_batches} ({len(batch)} players)...")
            
            batch_success = 0
            for player in batch:
                try:
                    def api_call():
                        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player['id'], timeout=30)
                        return player_info.get_data_frames()[0]
                    
                    player_info_df = self.safe_api_call(api_call, max_retries=2, delay=2)
                    
                    if not player_info_df.empty and 'TEAM_ABBREVIATION' in player_info_df.columns:
                        team = player_info_df['TEAM_ABBREVIATION'].iloc[0]
                        if team and team != '':
                            player['team'] = team
                            players_with_teams.append(player)
                            batch_success += 1
                            
                except Exception as e:
                    continue
            
            if batch_success > 0:
                successful_batches += 1
                print(f"      ✅ Got {batch_success}/{len(batch)} players in batch")
            else:
                print(f"      ⚠️  Batch {batch_num + 1} failed")
            
            if batch_num < total_batches - 1:
                time.sleep(3)
        
        print(f"   ✅ Found {len(players_with_teams)} players with team information")
        
        if len(players_with_teams) < 50:
            print("   ⚠️  Low player count, using fallback method...")
            return self.fallback_player_acquisition()
        
        return players_with_teams

    def fallback_player_acquisition(self):
        """Fallback method to get players when API fails"""
        print("   🆘 Using fallback player acquisition...")
        
        try:
            print("   Trying to get players from game logs...")
            season = self.get_current_season()
            
            def api_call():
                all_game_logs = playergamelogs.PlayerGameLogs(season_nullable=season, timeout=60)
                return all_game_logs.get_data_frames()[0]
            
            all_logs_df = self.safe_api_call(api_call, max_retries=3, delay=3)
            
            if not all_logs_df.empty:
                unique_players = all_logs_df[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION']].drop_duplicates()
                players_with_teams = []
                
                for _, row in unique_players.iterrows():
                    players_with_teams.append({
                        'id': row['PLAYER_ID'],
                        'full_name': row['PLAYER_NAME'],
                        'team': row['TEAM_ABBREVIATION']
                    })
                
                print(f"   ✅ Fallback acquired {len(players_with_teams)} players from game logs")
                return players_with_teams
                
        except Exception as e:
            print(f"   ❌ Fallback method 1 failed: {e}")
        
        print("   Using static player list with estimated teams...")
        all_active_players = [p for p in players.get_players() if p['is_active']]
        
        teams_list = list(self.team_id_map.values())
        for player in all_active_players:
            player['team'] = teams_list[hash(player['full_name']) % len(teams_list)]
        
        print(f"   ✅ Fallback assigned {len(all_active_players)} players to teams")
        return all_active_players

    def debug_column_check(self, all_logs_df):
        """Debug method to check available columns"""
        print("🔍 Available columns in game logs:")
        print(list(all_logs_df.columns))
        
        if not all_logs_df.empty:
            sample_row = all_logs_df.iloc[0]
            print("🔍 Sample row data:")
            for col in ['PLAYER_ID', 'PLAYER_NAME', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M']:
                if col in sample_row.index:
                    print(f"   {col}: {sample_row[col]}")

    def get_enhanced_player_projections(self, season):
        """Get player stats with enhanced projections including cash game safety"""
        print("🔄 Getting enhanced player projections v3...")
        
        player_data = []
        
        print("   Step 1: Getting players with teams...")
        all_players_with_teams = self.get_all_players_with_correct_teams()
        
        if not all_players_with_teams:
            print("❌ No players with teams found")
            return []
            
        todays_players = []
        for player in all_players_with_teams:
            if player.get('team') in self.todays_games:
                todays_players.append(player)
        
        print(f"   🎯 Found {len(todays_players)} players on today's teams")
        
        if len(todays_players) == 0:
            print("⚠️ No players found for today's games, using all players")
            todays_players = all_players_with_teams
        
        print("   Step 2: Getting game logs...")
        try:
            def api_call():
                all_game_logs = playergamelogs.PlayerGameLogs(season_nullable=season, timeout=90)
                return all_game_logs.get_data_frames()[0]
            
            all_logs_df = self.safe_api_call(api_call, max_retries=5, delay=5)
            print(f"   📈 Loaded {len(all_logs_df)} total game logs")
            
            self.debug_column_check(all_logs_df)
            
        except Exception as e:
            print(f"❌ Error loading game logs: {e}")
            print("⚠️  Cannot continue without game logs data")
            return []
        
        print("   Step 3: Calculating enhanced projections v3...")
        processed_count = 0
        
        for i, player in enumerate(todays_players):
            if processed_count % 10 == 0:
                print(f"      Processing {i+1}/{len(todays_players)}...")
            
            try:
                player_id = player['id']
                player_name = player['full_name']
                team = player['team']
                
                dk_salary_info = self.find_player_in_dk_salaries(player_name, team)
                if not dk_salary_info:
                    continue
                
                player_logs = all_logs_df[all_logs_df['PLAYER_ID'] == player_id]
                if player_logs.empty:
                    continue
                
                recent_games = player_logs.head(10)
                if len(recent_games) < 3:
                    continue
                
                required_columns = ['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M']
                missing_columns = [col for col in required_columns if col not in recent_games.columns]
                if missing_columns:
                    continue
                
                most_recent_game = recent_games.iloc[0]
                game_date = pd.to_datetime(most_recent_game['GAME_DATE'])
                days_since_last_game = (datetime.now() - game_date).days
                
                if days_since_last_game > 7:
                    continue
                
                recent_minutes = recent_games['MIN'].mean()
                if recent_minutes < 5:
                    continue
                
                position = dk_salary_info.get('position')
                if not position:
                    position = self.get_player_position(player_id, player_name)
                
                projection_data = self.calculate_enhanced_projection_v3(
                    player, recent_games, position, team, dk_salary_info, all_logs_df
                )
                
                if projection_data:
                    player_data.append(projection_data)
                    processed_count += 1
                    
            except Exception as e:
                continue
        
        print(f"   ✅ Processed {processed_count} players with enhanced projections v3")
        
        player_data = self.filter_cash_game_players(player_data)
        
        return player_data

    def calculate_robust_averages(self, recent_games):
        """Calculate more robust averages using median and trimmed means"""
        try:
            median_minutes = recent_games['MIN'].median()
            
            stats = {}
            
            stat_mapping = {
                'points': 'PTS',
                'rebounds': 'REB', 
                'assists': 'AST',
                'steals': 'STL',
                'blocks': 'BLK',
                'turnovers': 'TOV',
                'field_goals_attempted': 'FGA',
                'free_throws_attempted': 'FTA',
                'plus_minus': 'PLUS_MINUS',
                'three_pointers_made': 'FG3M'
            }
            
            for stat_key, nba_column in stat_mapping.items():
                if nba_column in recent_games.columns:
                    values = sorted(recent_games[nba_column].tolist())
                    if len(values) >= 5:
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
                'plus_minus': recent_games['PLUS_MINUS'].mean() if 'PLUS_MINUS' in recent_games.columns else 0,
                'three_pointers_made': recent_games['FG3M'].mean() if 'FG3M' in recent_games.columns else 0
            }

    def calculate_volatility_score(self, recent_games):
        """Calculate how volatile a player's production is"""
        try:
            if len(recent_games) < 3:
                return 1.0
            
            game_scores = []
            for _, game in recent_games.iterrows():
                fp = self.calculate_dk_points({
                    'points': game['PTS'],
                    'rebounds': game['REB'],
                    'assists': game['AST'],
                    'steals': game['STL'],
                    'blocks': game['BLK'],
                    'turnovers': game['TOV'],
                    'three_pointers_made': game.get('FG3M', 0)
                })
                game_scores.append(fp)
            
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
            
            game_scores = []
            for _, game in recent_games.iterrows():
                fp = self.calculate_dk_points({
                    'points': game['PTS'],
                    'rebounds': game['REB'],
                    'assists': game['AST'],
                    'steals': game['STL'],
                    'blocks': game['BLK'],
                    'turnovers': game['TOV'],
                    'three_pointers_made': game.get('FG3M', 0)
                })
                game_scores.append(fp)
            
            avg_score = sum(game_scores) / len(game_scores)
            baseline = avg_score * 0.75
            meets_baseline = sum(1 for score in game_scores if score >= baseline)
            consistency_pct = (meets_baseline / len(game_scores)) * 100
            
            return min(100, max(0, consistency_pct))
            
        except:
            return 50

    def calculate_enhanced_projection_v3(self, player, recent_games, position, team, dk_salary_info, all_logs_df):
        """Enhanced projection v3 with cash game safety and realistic upside"""
        try:
            avg_stats = self.calculate_robust_averages(recent_games)
            
            if avg_stats['minutes'] < 10:
                return None
            
            usage_rate = self.calculate_usage_rate(recent_games, team)
            points_per_minute = self.calculate_points_per_minute(recent_games)
            fantasy_points_per_minute = self.calculate_fantasy_points_per_minute(recent_games)
            plus_minus_rating = self.calculate_plus_minus_rating(recent_games, avg_stats['minutes'])
            
            role_stability = self.calculate_role_stability(recent_games, player['full_name'], position)
            recent_trend = self.calculate_recent_trend_factor(recent_games, 'PTS')
            opportunity_rating = self.get_opportunity_rating(recent_games, team, usage_rate)
            
            injury_risk = self.calculate_injury_risk(recent_games, player['full_name'])
            
            volatility_score = self.calculate_volatility_score(recent_games)
            consistency_rating = self.calculate_consistency_rating(recent_games)
            
            home_away_splits = self.calculate_home_away_splits(recent_games)
            
            matchup_info = self.matchup_data.get(team, {})
            location = matchup_info.get('location', 'home')
            opponent = matchup_info.get('opponent', 'UNK')
            matchup_difficulty = matchup_info.get('matchup_difficulty', 0)
            opp_def_rating = matchup_info.get('opp_def_rating', 110.0)
            
            blowout_risk = self.calculate_blowout_risk(team, opponent)
            
            back_to_back = self.check_back_to_back(team)
            
            projected_minutes = self.project_minutes_enhanced_v2(
                avg_stats['minutes'], recent_games, team, usage_rate, role_stability
            )
            
            projected_minutes = self.apply_minutes_reality_check_enhanced(
                projected_minutes, player['full_name'], position, usage_rate, role_stability
            )
            
            per_36_stats = {}
            stat_columns = {
                'points': 'PTS',
                'rebounds': 'REB', 
                'assists': 'AST',
                'steals': 'STL',
                'blocks': 'BLK',
                'turnovers': 'TOV',
                'three_pointers_made': 'FG3M'
            }

            for stat, nba_col in stat_columns.items():
                if nba_col in recent_games.columns and avg_stats['minutes'] > 0:
                    median_stat = recent_games[nba_col].median()
                    per_36_stats[stat] = (median_stat / avg_stats['minutes']) * 36
                else:
                    per_36_stats[stat] = 0
            
            pace_adjustment = self.get_pace_adjustment(team)
            usage_adjustment = self.get_usage_adjustment(usage_rate)
            plus_minus_adjustment = self.get_plus_minus_adjustment(plus_minus_rating)
            matchup_adjustment = self.get_matchup_adjustment(opp_def_rating, location)
            home_away_adjustment = self.get_home_away_adjustment(location, home_away_splits)
            back_to_back_adjustment = self.get_back_to_back_adjustment(back_to_back, usage_rate)
            volatility_adjustment = self.get_volatility_adjustment(volatility_score, usage_rate)
            
            trend_adjustment = recent_trend
            stability_adjustment = 0.8 + (role_stability / 100 * 0.4)
            blowout_adjustment = self.get_blowout_adjustment(blowout_risk, usage_rate, role_stability)
            
            pace_adjusted_stats = {}
            scoring_stats = ['points', 'assists', 'three_pointers_made']
            non_scoring_stats = ['rebounds', 'steals', 'blocks']
            
            for stat in scoring_stats:
                pace_adjusted_stats[stat] = per_36_stats[stat] * pace_adjustment * usage_adjustment * plus_minus_adjustment * matchup_adjustment * home_away_adjustment * back_to_back_adjustment * volatility_adjustment * trend_adjustment * stability_adjustment * blowout_adjustment
            
            for stat in non_scoring_stats:
                pace_adjusted_stats[stat] = per_36_stats[stat] * pace_adjustment * plus_minus_adjustment * matchup_adjustment * home_away_adjustment * back_to_back_adjustment * volatility_adjustment * stability_adjustment * blowout_adjustment
            
            pace_adjusted_stats['turnovers'] = per_36_stats['turnovers'] * (1 + (usage_adjustment - 1) * 0.3)
            
            base_projection_stats = {}
            for stat in pace_adjusted_stats:
                base_projection_stats[stat] = (pace_adjusted_stats[stat] / 36) * projected_minutes
            
            efficiency_adjustment = self.calculate_efficiency_adjustment(recent_games)
            base_projection_stats['points'] *= efficiency_adjustment
            
            base_projection = self.calculate_dk_points(base_projection_stats)
            
            conservative_projection = self.calculate_conservative_projection(recent_games, base_projection, consistency_rating)
            
            salary = dk_salary_info['salary']
            
            player_recent_games = self.get_player_recent_games(player['full_name'], all_logs_df)
            realistic_upside_score = self.calculate_realistic_upside_score(
                player_recent_games, conservative_projection, salary, role_stability, injury_risk
            )
            is_valid_upside_play = self.validate_upside_play({
                'projection': conservative_projection,
                'opportunity_rating': opportunity_rating,
                'injury_risk': injury_risk,
                'back_to_back': back_to_back,
                'role_stability': role_stability
            }, player_recent_games)
            
            ceiling_projection = self.calculate_ceiling_projection_enhanced(recent_games, projected_minutes, role_stability, volatility_score)
            
            upside_score = self.calculate_upside_score_enhanced(conservative_projection, ceiling_projection, salary, usage_rate, volatility_score, matchup_difficulty, opportunity_rating)
            
            final_projection = self.apply_projection_reality_checks_enhanced(conservative_projection, player['full_name'], position, salary, usage_rate, consistency_rating, projected_minutes, role_stability, volatility_score)
            
            if final_projection < 5:
                return None
            
            value_rating = (final_projection / salary) * 1000
            
            bargain_rating = self.calculate_bargain_rating_enhanced(final_projection, salary, usage_rate, plus_minus_rating, fantasy_points_per_minute, consistency_rating, role_stability, opportunity_rating)
            
            return {
                'name': player['full_name'],
                'position': position,
                'team': team,
                'salary': salary,
                'projection': round(final_projection, 1),
                'base_projection': round(base_projection, 1),
                'conservative_projection': round(conservative_projection, 1),
                'ceiling_projection': round(ceiling_projection, 1),
                'floor_projection': round(self.calculate_floor_projection(recent_games, projected_minutes, role_stability), 1),
                'upside_score': round(upside_score, 1),
                'realistic_upside_score': round(realistic_upside_score, 1),
                'is_valid_upside_play': is_valid_upside_play,
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
                'role_stability': round(role_stability, 1),
                'opportunity_rating': round(opportunity_rating, 1),
                'recent_trend': round(recent_trend, 3),
                'blowout_risk': round(blowout_risk, 3),
                'injury_risk': round(injury_risk, 3),
                'cash_game_safe': False,
                'location': location,
                'opponent': opponent,
                'matchup_difficulty': round(matchup_difficulty, 2),
                'back_to_back': back_to_back,
                'home_away_adjustment': round(home_away_adjustment, 3),
                'matchup_adjustment': round(matchup_adjustment, 3),
                'back_to_back_adjustment': round(back_to_back_adjustment, 3),
                'volatility_adjustment': round(volatility_adjustment, 3),
                'trend_adjustment': round(trend_adjustment, 3),
                'stability_adjustment': round(stability_adjustment, 3),
                'blowout_adjustment': round(blowout_adjustment, 3),
                'games_used': len(recent_games),
                'source': 'enhanced_projections_v3',
                'playing_today': True
            }
            
        except Exception as e:
            print(f"      Error calculating projection for {player['full_name']}: {e}")
            return None

    def project_minutes_enhanced(self, recent_minutes, recent_games, team, usage_rate=None, plus_minus_rating=None, location='home', back_to_back=False, matchup_difficulty=0, volatility_score=1.0, consistency_rating=50, role_stability=50, blowout_risk=0.3):
        """Enhanced minutes projection with role stability and blowout risk"""
        base_minutes = recent_minutes
        
        if len(recent_games) >= 3:
            last_3_minutes = recent_games.head(3)['MIN'].mean()
            if last_3_minutes > recent_minutes:
                base_minutes = (recent_minutes + last_3_minutes) / 2
        
        stability_factor = 0.9 + (role_stability / 100 * 0.2)
        base_minutes *= stability_factor
        
        blowout_risk_factor = 1.0 - (blowout_risk * 0.3)
        
        if usage_rate and usage_rate > 0.25:
            blowout_risk_factor = 1.0 - (blowout_risk * 0.15)
        elif usage_rate and usage_rate < 0.15:
            blowout_risk_factor = 1.0 - (blowout_risk * 0.5)
        
        if plus_minus_rating and plus_minus_rating > 2:
            blowout_risk_factor += 0.05
        elif plus_minus_rating and plus_minus_rating < -2:
            blowout_risk_factor -= 0.05
        
        if back_to_back:
            if usage_rate and usage_rate > 0.25:
                blowout_risk_factor -= 0.05
            else:
                blowout_risk_factor -= 0.02
        
        if matchup_difficulty < -0.5:
            if usage_rate and usage_rate < 0.18:
                blowout_risk_factor -= 0.03
        
        if volatility_score > 1.2:
            blowout_risk_factor -= 0.03
        
        projected_minutes = min(38, base_minutes * blowout_risk_factor)
        projected_minutes = max(8, projected_minutes)
        
        return projected_minutes

    def apply_minutes_reality_check_enhanced(self, projected_minutes, player_name, position, usage_rate, role_stability):
        """Enhanced minutes reality check with role stability"""
        max_minutes = 36
        
        if usage_rate < 0.15:
            max_minutes = 28
        elif usage_rate < 0.20:
            max_minutes = 32
        elif usage_rate < 0.25:
            max_minutes = 36
        else:
            max_minutes = 38
        
        if position == 'C':
            max_minutes = min(max_minutes, 34)
        
        if role_stability < 60:
            max_minutes *= 0.9
        elif role_stability > 80:
            max_minutes *= 1.05
        
        return min(projected_minutes, max_minutes)

    def calculate_ceiling_projection_enhanced(self, recent_games, projected_minutes, role_stability, volatility_score):
        """Enhanced ceiling projection with uncertainty ranges"""
        try:
            best_game_score = 0
            best_game_minutes = 0
            
            for _, game in recent_games.iterrows():
                game_score = self.calculate_dk_points({
                    'points': game['PTS'],
                    'rebounds': game['REB'],
                    'assists': game['AST'],
                    'steals': game['STL'],
                    'blocks': game['BLK'],
                    'turnovers': game['TOV'],
                    'three_pointers_made': game.get('FG3M', 0)
                })
                
                if game_score > best_game_score:
                    best_game_score = game_score
                    best_game_minutes = game['MIN']
            
            if best_game_minutes > 0 and projected_minutes > 0:
                minutes_ratio = projected_minutes / best_game_minutes
                ceiling_projection = best_game_score * minutes_ratio
                
                stability_factor = 0.9 + (role_stability / 100 * 0.2)
                volatility_factor = 1.0 + (volatility_score - 1.0) * 0.2
                
                ceiling_projection = ceiling_projection * stability_factor * volatility_factor
                
                ceiling_projection = min(ceiling_projection * 1.15, best_game_score * 1.3)
                
                return ceiling_projection
            else:
                return best_game_score * 1.1
                
        except Exception as e:
            return 0

    def calculate_floor_projection(self, recent_games, projected_minutes, role_stability):
        """Calculate floor projection based on worst recent performance"""
        try:
            worst_game_score = float('inf')
            worst_game_minutes = 0
            
            for _, game in recent_games.iterrows():
                if game['MIN'] < 10:
                    continue
                    
                game_score = self.calculate_dk_points({
                    'points': game['PTS'],
                    'rebounds': game['REB'],
                    'assists': game['AST'],
                    'steals': game['STL'],
                    'blocks': game['BLK'],
                    'turnovers': game['TOV'],
                    'three_pointers_made': game.get('FG3M', 0)
                })
                
                if game_score < worst_game_score:
                    worst_game_score = game_score
                    worst_game_minutes = game['MIN']
            
            if worst_game_score == float('inf'):
                return 0
                
            if worst_game_minutes > 0 and projected_minutes > 0:
                minutes_ratio = projected_minutes / worst_game_minutes
                floor_projection = worst_game_score * minutes_ratio
                
                stability_floor_factor = 0.8 + (role_stability / 100 * 0.4)
                floor_projection *= stability_floor_factor
                
                return floor_projection
            else:
                return worst_game_score * 0.9
                
        except Exception as e:
            return 0

    def calculate_upside_score_enhanced(self, projection, ceiling_projection, salary, usage_rate, volatility_score, matchup_difficulty, opportunity_rating):
        """Enhanced upside score with opportunity rating"""
        try:
            score = 0
            
            if projection > 0 and ceiling_projection > projection:
                ceiling_gap = (ceiling_projection - projection) / projection
                ceiling_component = min(30, ceiling_gap * 100)
                score += ceiling_component
            
            if salary > 0:
                salary_component = max(0, (10000 - salary) / 10000 * 20)
                score += salary_component
            
            usage_component = min(15, usage_rate * 100)
            score += usage_component
            
            volatility_component = min(10, (volatility_score - 0.5) * 15)
            score += max(0, volatility_component)
            
            matchup_component = min(10, (matchup_difficulty + 1) * 5)
            score += max(0, matchup_component)
            
            opportunity_component = opportunity_rating * 0.15
            score += opportunity_component
            
            return min(100, max(0, score))
            
        except:
            return 0

    def apply_projection_reality_checks_enhanced(self, projection, player_name, position, salary, usage_rate, consistency_rating, projected_minutes, role_stability, volatility_score):
        """Enhanced reality checks with role stability"""
        max_projection_by_salary = salary * 0.008
        
        if usage_rate < 0.15:
            max_projection_by_salary *= 0.8
        
        consistency_adjustment = consistency_rating / 100
        
        stability_adjustment = 0.9 + (role_stability / 100 * 0.2)
        
        if usage_rate < 0.20:
            volatility_multiplier = 1.5 - (volatility_score - 1.0) * 0.2
            max_by_minutes = projected_minutes * max(1.0, volatility_multiplier)
            projection = min(projection, max_by_minutes)
        
        projection = min(projection, max_projection_by_salary)
        projection *= consistency_adjustment
        projection *= stability_adjustment
        
        return projection

    def calculate_bargain_rating_enhanced(self, projection, salary, usage_rate, plus_minus_rating, fantasy_points_per_minute, consistency_rating, role_stability, opportunity_rating):
        """Enhanced bargain rating with stability and opportunity"""
        try:
            score = 0
            
            if salary > 0:
                value_score = (projection / salary) * 1000
                value_component = min(25, (value_score - 2) * (25 / 6))
                score += max(0, value_component)
            
            usage_component = min(15, usage_rate * 100)
            score += usage_component
            
            pm_component = min(10, (plus_minus_rating + 10) * (10 / 20))
            score += max(0, pm_component)
            
            if fantasy_points_per_minute > 0:
                efficiency_component = min(10, (fantasy_points_per_minute - 0.8) * (10 / 0.7))
                score += max(0, efficiency_component)
            
            consistency_component = consistency_rating * 0.15
            score += consistency_component
            
            stability_component = role_stability * 0.15
            score += stability_component
            
            opportunity_component = opportunity_rating * 0.10
            score += opportunity_component
            
            return min(100, max(0, score))
            
        except Exception as e:
            if salary > 0:
                simple_value = (projection / salary) * 1000
                return min(100, (simple_value - 2) * (100 / 6))
            return 50

    def get_blowout_adjustment(self, blowout_risk, usage_rate, role_stability):
        """Adjust projection based on blowout risk"""
        base_adjustment = 1.0
        
        if blowout_risk > 0.6:
            if usage_rate < 0.15 and role_stability < 60:
                base_adjustment = 0.7
            elif usage_rate < 0.15:
                base_adjustment = 0.8
            elif usage_rate > 0.25:
                base_adjustment = 0.95
        elif blowout_risk > 0.3:
            if usage_rate < 0.15:
                base_adjustment = 0.9
        
        return base_adjustment

    def calculate_usage_rate(self, player_games, team):
        """Calculate player usage rate (% of team possessions used by player)"""
        try:
            player_usage = (
                player_games['FGA'].mean() + 
                0.44 * player_games['FTA'].mean() + 
                player_games['TOV'].mean()
            )
            
            avg_minutes = player_games['MIN'].mean()
            if avg_minutes > 0:
                usage_per_36 = (player_usage / avg_minutes) * 36
            else:
                usage_per_36 = 0
            
            usage_rate = usage_per_36 / 100
            
            usage_rate = max(0.10, min(0.40, usage_rate))
            
            return usage_rate
            
        except Exception as e:
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
                    'turnovers': game['TOV'],
                    'three_pointers_made': game.get('FG3M', 0)
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
            
            avg_plus_minus = player_games['PLUS_MINUS'].mean()
            
            if avg_minutes > 0:
                normalized_pm = avg_plus_minus / avg_minutes
            else:
                normalized_pm = 0
            
            plus_minus_rating = normalized_pm * 36
            
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
        base_usage = 0.20
        adjustment = 1.0 + (usage_rate - base_usage) * 0.5
        
        return max(0.8, min(1.3, adjustment))

    def get_plus_minus_adjustment(self, plus_minus_rating):
        """Get adjustment factor based on plus/minus rating"""
        adjustment = 1.0 + (plus_minus_rating * 0.02)
        
        return max(0.7, min(1.3, adjustment))

    def get_matchup_adjustment(self, opp_def_rating, location):
        """Get adjustment factor based on opponent defense"""
        league_avg_def = 110.0
        
        def_difference = league_avg_def - opp_def_rating
        adjustment = 1.0 + (def_difference * 0.01)
        
        return max(0.8, min(1.2, adjustment))

    def get_home_away_adjustment(self, location, home_away_splits):
        """Get adjustment factor based on home/away location and player splits"""
        try:
            if location == 'home':
                base_adjustment = 1.02
            else:
                base_adjustment = 0.98
            
            if home_away_splits['home_games'] >= 3 and home_away_splits['away_games'] >= 3:
                home_ppg = home_away_splits['home_ppg']
                away_ppg = home_away_splits['away_ppg']
                
                if home_ppg > 0 and away_ppg > 0:
                    split_ratio = home_ppg / away_ppg if location == 'home' else away_ppg / home_ppg
                    adjustment = (base_adjustment * 0.7) + (split_ratio * 0.3)
                    return max(0.9, min(1.1, adjustment))
            
            return base_adjustment
            
        except:
            return 1.0

    def get_back_to_back_adjustment(self, back_to_back, usage_rate):
        """Get adjustment factor for back-to-back games"""
        if not back_to_back:
            return 1.0
        
        if usage_rate > 0.25:
            return 0.95
        elif usage_rate > 0.18:
            return 0.97
        else:
            return 0.99

    def get_volatility_adjustment(self, volatility_score, usage_rate):
        """Adjust projection based on player volatility"""
        base_adjustment = 1.0
        
        if volatility_score > 1.2:
            if usage_rate < 0.15:
                base_adjustment = 0.85
            else:
                base_adjustment = 0.92
        elif volatility_score > 0.8:
            base_adjustment = 0.95
        
        return base_adjustment

    def calculate_efficiency_adjustment(self, player_games):
        """Adjust projection based on recent shooting efficiency"""
        try:
            if 'FGM' not in player_games.columns or 'FGA' not in player_games.columns or 'FG3M' not in player_games.columns:
                return 1.0
                
            total_fgm = player_games['FGM'].sum()
            total_fga = player_games['FGA'].sum()
            total_3pm = player_games['FG3M'].sum()
            
            if total_fga > 0:
                efg_percent = (total_fgm + 0.5 * total_3pm) / total_fga
                
                league_avg_efg = 0.54
                efficiency_ratio = efg_percent / league_avg_efg
                
                adjusted_ratio = 0.7 + (0.3 * efficiency_ratio)
                
                return max(0.7, min(1.3, adjusted_ratio))
            else:
                return 1.0
        except:
            return 1.0

    def get_pace_adjustment(self, team):
        """Get pace adjustment factor for a team"""
        league_avg_pace = 100.0
        
        if team in self.team_stats:
            team_pace = self.team_stats[team]['pace']
            pace_adjustment = team_pace / league_avg_pace
            return pace_adjustment
        
        return 1.0

    def find_player_in_dk_salaries(self, player_name, team):
        """Find player in DraftKings salaries CSV - IMPROVED VERSION"""
        if self.dk_salaries_df is None:
            return None
        
        name_variations = self.get_name_variations(player_name)
        
        for name_var in name_variations:
            if 'Name' in self.dk_salaries_df.columns:
                mask = self.dk_salaries_df['Name'].str.lower() == name_var.lower()
                if mask.any():
                    player_row = self.dk_salaries_df[mask].iloc[0]
                    return self.extract_salary_info(player_row)
            
            if 'Name' in self.dk_salaries_df.columns:
                mask = self.dk_salaries_df['Name'].str.lower().str.contains(name_var.lower(), na=False)
                if mask.any():
                    player_row = self.dk_salaries_df[mask].iloc[0]
                    return self.extract_salary_info(player_row)
        
        return None

    def extract_salary_info(self, player_row):
        """Extract salary information from player row"""
        try:
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
        """Get player position with error handling"""
        try:
            def api_call():
                player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=30)
                return player_info.get_data_frames()[0]
            
            info_df = self.safe_api_call(api_call, max_retries=2, delay=2)
            
            if not info_df.empty:
                if 'POSITION' in info_df.columns and not pd.isna(info_df['POSITION'].iloc[0]):
                    pos = str(info_df['POSITION'].iloc[0]).strip()
                    if pos and pos != 'nan':
                        return self.map_to_dfs_position(pos)
        
        except Exception:
            pass
        
        return self.estimate_position_from_name(player_name)

    def calculate_dk_points(self, stats):
        """Calculate DraftKings fantasy points with double-double and triple-double bonuses"""
        points = stats.get('points', 0)
        rebounds = stats.get('rebounds', 0)
        assists = stats.get('assists', 0)
        steals = stats.get('steals', 0)
        blocks = stats.get('blocks', 0)
        turnovers = stats.get('turnovers', 0)
        three_pointers_made = stats.get('three_pointers_made', 0)
        
        fantasy_points = (
            points * 1.0 +
            three_pointers_made * 0.5 +
            rebounds * 1.25 +
            assists * 1.5 +
            steals * 2.0 +
            blocks * 2.0 -
            turnovers * 0.5
        )
        
        double_double_categories = 0
        if points >= 10:
            double_double_categories += 1
        if rebounds >= 10:
            double_double_categories += 1
        if assists >= 10:
            double_double_categories += 1
        
        if double_double_categories >= 3:
            fantasy_points += 3.0
        elif double_double_categories >= 2:
            fantasy_points += 1.5
        
        return fantasy_points

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

    def identify_high_upside_players(self):
        """Identify players with high upside potential"""
        print("\n🔍 Identifying high-upside players...")
        
        high_upside_players = []
        
        for player in self.players_data:
            upside_score = player.get('upside_score', 0)
            ceiling_projection = player.get('ceiling_projection', 0)
            projection = player.get('projection', 0)
            salary = player.get('salary', 0)
            
            if (upside_score >= 60 and 
                ceiling_projection >= projection * 1.5 and
                salary < 8000 and
                player.get('matchup_difficulty', -1) > -0.5):
                
                high_upside_players.append(player)
        
        high_upside_players.sort(key=lambda x: x.get('upside_score', 0), reverse=True)
        self.high_upside_players = high_upside_players
        
        print(f"✅ Found {len(high_upside_players)} high-upside players")
        
        if high_upside_players:
            print("\n🚀 TOP HIGH-UPSIDE PLAYS:")
            print("-" * 80)
            for i, player in enumerate(high_upside_players[:10]):
                upside_gap = player['ceiling_projection'] - player['projection']
                value_indicator = "💎" if player.get('value_rating', 0) > 5 else "💰" if player.get('value_rating', 0) > 4 else "💵"
                print(f"{i+1:2d}. {player['position']:2} {player['name']:20} | "
                      f"UPSIDE: {player['upside_score']:2.0f} | "
                      f"PROJ: {player['projection']:4.1f} | "
                      f"CEILING: {player['ceiling_projection']:4.1f} | "
                      f"GAP: +{upside_gap:3.1f} | "
                      f"${player['salary']:5,} | "
                      f"{value_indicator}")

    def create_fallback_data(self):
        """Create fallback data using only DK salaries when API fails completely"""
        print("🆘 Creating fallback data from DK salaries...")
        
        if self.dk_salaries_df is None:
            print("❌ Cannot create fallback without DK salaries")
            return False
        
        player_data = []
        
        for _, row in self.dk_salaries_df.iterrows():
            try:
                salary_info = self.extract_salary_info(row)
                if not salary_info:
                    continue
                
                name = row['Name'] if 'Name' in row else 'Unknown'
                position = salary_info['position']
                team = salary_info['team']
                salary = salary_info['salary']
                
                projection = salary * 0.005
                
                if position == 'PG':
                    projection *= 1.1
                elif position == 'C':
                    projection *= 0.9
                
                player_data.append({
                    'name': name,
                    'position': position,
                    'team': team,
                    'salary': salary,
                    'projection': round(projection, 1),
                    'ceiling_projection': round(projection * 1.3, 1),
                    'floor_projection': round(projection * 0.7, 1),
                    'upside_score': 50,
                    'minutes': 25,
                    'projected_minutes': 25,
                    'usage_rate': 0.2,
                    'points_per_minute': 0.4,
                    'fantasy_points_per_minute': 0.8,
                    'value_rating': round((projection / salary) * 1000, 2),
                    'bargain_rating': 50,
                    'consistency_rating': 50,
                    'volatility_score': 1.0,
                    'role_stability': 60,
                    'opportunity_rating': 50,
                    'location': 'home',
                    'opponent': 'UNK',
                    'back_to_back': False,
                    'source': 'fallback',
                    'playing_today': True
                })
                
            except Exception as e:
                continue
        
        if player_data:
            self.players_data = player_data
            print(f"✅ Created fallback data for {len(player_data)} players")
            self.show_data_summary_enhanced()
            return True
        else:
            print("❌ Failed to create fallback data")
            return False

    def show_data_summary_enhanced(self):
        """Enhanced data summary with new metrics"""
        if not self.players_data:
            return
            
        print("\n📊 ENHANCED PROJECTIONS v3 SUMMARY:")
        print("-" * 80)
        
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
        avg_upside = sum(p.get('upside_score', 0) for p in self.players_data) / len(self.players_data)
        
        avg_role_stability = sum(p.get('role_stability', 50) for p in self.players_data) / len(self.players_data)
        avg_opportunity = sum(p.get('opportunity_rating', 50) for p in self.players_data) / len(self.players_data)
        avg_trend = sum(p.get('recent_trend', 1) for p in self.players_data) / len(self.players_data)
        avg_blowout_risk = sum(p.get('blowout_risk', 0.3) for p in self.players_data) / len(self.players_data)
        avg_floor = sum(p.get('floor_projection', 0) for p in self.players_data) / len(self.players_data)
        avg_injury_risk = sum(p.get('injury_risk', 0.3) for p in self.players_data) / len(self.players_data)
        cash_safe_players = sum(1 for p in self.players_data if p.get('cash_game_safe', False))
        valid_upside_players = sum(1 for p in self.players_data if p.get('is_valid_upside_play', False))
        
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
        print(f"  Avg Upside Score: {avg_upside:.1f}")
        
        print(f"\n🆕 New Enhanced Metrics:")
        print(f"  Avg Role Stability: {avg_role_stability:.1f}")
        print(f"  Avg Opportunity Rating: {avg_opportunity:.1f}")
        print(f"  Avg Recent Trend: {avg_trend:.3f}")
        print(f"  Avg Blowout Risk: {avg_blowout_risk:.3f}")
        print(f"  Avg Floor Projection: {avg_floor:.1f}")
        print(f"  Avg Injury Risk: {avg_injury_risk:.3f}")
        print(f"  Cash Game Safe Players: {cash_safe_players}/{len(self.players_data)}")
        print(f"  Valid Upside Plays: {valid_upside_players}/{len(self.players_data)}")
        
        low_stability_players = [p for p in self.players_data if p.get('role_stability', 50) < 40]
        if low_stability_players:
            print(f"\n⚠️  High-Risk Players (Low Role Stability):")
            for player in sorted(low_stability_players, key=lambda x: x.get('role_stability', 50))[:5]:
                print(f"  {player['name']}: Stability {player.get('role_stability', 0):.0f}, Volatility {player.get('volatility_score', 0):.2f}")
        
        high_injury_risk_players = [p for p in self.players_data if p.get('injury_risk', 0) > 0.6]
        if high_injury_risk_players:
            print(f"\n🏥 High Injury Risk Players:")
            for player in sorted(high_injury_risk_players, key=lambda x: x.get('injury_risk', 0), reverse=True)[:5]:
                print(f"  {player['name']}: Injury Risk {player.get('injury_risk', 0):.1%}")
        
        if len(self.players_data) >= 5:
            print(f"\n💰 Top 5 Cash Game Safe Players:")
            cash_safe = [p for p in self.players_data if p.get('cash_game_safe', False)]
            for player in sorted(cash_safe, key=lambda x: x.get('bargain_rating', 0), reverse=True)[:5]:
                print(f"  {player['name']}: Bargain {player.get('bargain_rating', 0):.1f}, Stability {player.get('role_stability', 0):.0f}")
            
            print(f"\n🚀 Top 5 Valid Upside Plays:")
            valid_upside = [p for p in self.players_data if p.get('is_valid_upside_play', False)]
            for player in sorted(valid_upside, key=lambda x: x.get('realistic_upside_score', 0), reverse=True)[:5]:
                print(f"  {player['name']}: Upside {player.get('realistic_upside_score', 0):.1f}, Ceiling {player.get('ceiling_projection', 0):.1f}")

# UPDATED OPTIMIZER CLASS WITH CONTEST-ENHANCED STRATEGIES

class EnhancedProjectionsNBAOptimizer:
    def __init__(self, dk_salaries_path="DKSalaries.csv"):
        self.data = EnhancedProjectionsNBAData(dk_salaries_path)
        self.lineup_strategies = ['balanced', 'high_upside', 'stars_and_scrubs', 'high_floor', 'cash_game']
        
    def build_lineups(self, strategy='balanced', num_lineups=1):
        """Build lineups using different strategies"""
        print(f"\n🚀 ENHANCED PROJECTIONS NBA DFS LINEUP BUILDER v3 - {strategy.upper()} STRATEGY")
        print("With cash game safety and realistic upside projections")
        print("=" * 60)
        
        if not self.data.get_real_nba_data_enhanced():
            print("💥 Failed to get enhanced NBA data")
            return False
            
        if len(self.data.players_data) < 8:
            print(f"💥 Only {len(self.data.players_data)} players found, need at least 8")
            return False
        
        lineups = []
        
        if strategy == 'high_upside':
            lineups = self.build_high_upside_lineups_v2(num_lineups)
        elif strategy == 'stars_and_scrubs':
            lineups = self.build_stars_and_scrubs_lineups(num_lineups)
        elif strategy == 'high_floor':
            lineups = self.build_high_floor_lineups(num_lineups)
        elif strategy == 'cash_game':
            lineups = self.build_enhanced_cash_game_lineups(num_lineups)
        else:  # balanced
            lineup = self.optimize_lineup()
            if lineup:
                lineups = [lineup]
        
        if lineups:
            for i, lineup in enumerate(lineups):
                if lineup:
                    print(f"\n🏆 LINEUP {i+1} - {strategy.upper()} STRATEGY")
                    self.display_lineup_enhanced(lineup)
            return True
        else:
            print("💥 Optimization failed")
            return False

    def build_enhanced_cash_game_lineups(self, num_lineups=2):
        """Build cash game lineups using contest insights"""
        print(f"\n💰 Building {num_lineups} ENHANCED cash game lineups...")
        
        # Use enhanced data with contest insights
        if not self.data.get_real_nba_data_enhanced():
            print("❌ Failed to get enhanced NBA data")
            return []
        
        cash_game_players = [p for p in self.data.players_data if p.get('cash_game_safe', False)]
        print(f"   Enhanced cash game safe players: {len(cash_game_players)}")
        
        if len(cash_game_players) < 8:
            print("❌ Not enough enhanced cash game safe players")
            return []
        
        lineups = []
        
        for lineup_num in range(num_lineups):
            print(f"   Building enhanced cash game lineup {lineup_num + 1}...")
            
            prob = pulp.LpProblem(f"NBA_Enhanced_Cash_Game_{lineup_num}", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)
            
            # Optimize for cash game score instead of raw projection
            prob += pulp.lpSum([
                player_vars[i] * (
                    self.data.players_data[i].get('cash_game_score', 0) * 0.6 +
                    self.data.players_data[i]['projection'] * 0.4
                ) for i in range(len(self.data.players_data))
            ])
            
            # Standard constraints
            prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['salary'] 
                              for i in range(len(self.data.players_data))]) <= 50000
            prob += pulp.lpSum([player_vars[i] for i in range(len(self.data.players_data))]) == 8
            
            # Position constraints
            pg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PG']
            sg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SG']
            sf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SF']
            pf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PF']
            c_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'C']
            
            prob += pulp.lpSum([player_vars[i] for i in pg_players]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in sg_players]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in sf_players]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in pf_players]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in c_players]) == 1
            
            guard_players = pg_players + sg_players
            forward_players = sf_players + pf_players
            prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
            prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3
            
            # Enhanced cash game constraints
            cash_safe_indices = [i for i, p in enumerate(self.data.players_data) 
                               if p.get('cash_game_safe', False)]
            if len(cash_safe_indices) >= 6:
                prob += pulp.lpSum([player_vars[i] for i in cash_safe_indices]) >= 6
            
            # High ownership preference (based on contest results)
            high_ownership_indices = [i for i, p in enumerate(self.data.players_data) 
                                    if p.get('contest_ownership', 0) > 60]
            if len(high_ownership_indices) >= 4:
                prob += pulp.lpSum([player_vars[i] for i in high_ownership_indices]) >= 4
            
            # Limit high-risk players
            high_risk_indices = [i for i, p in enumerate(self.data.players_data)
                              if p.get('injury_risk', 0) > 0.5 or 
                                 p.get('volatility_score', 1) > 1.2]
            if high_risk_indices:
                prob += pulp.lpSum([player_vars[i] for i in high_risk_indices]) <= 1
            
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status == pulp.LpStatusOptimal:
                lineup = self.extract_lineup(player_vars)
                if lineup:
                    lineups.append(lineup)
                    print(f"   ✅ Enhanced cash game lineup {lineup_num + 1} built successfully")
                    
                    # Display lineup analysis
                    self.analyze_cash_lineup(lineup)
                else:
                    print(f"   ❌ Enhanced cash game lineup {lineup_num + 1} failed")
            else:
                print(f"   ❌ No solution for enhanced cash game lineup {lineup_num + 1}")
        
        return lineups

    def analyze_cash_lineup(self, lineup):
        """Analyze cash lineup quality"""
        cash_safe_count = sum(1 for p in lineup['players'] if p.get('cash_game_safe', False))
        avg_ownership = np.mean([p.get('contest_ownership', 0) for p in lineup['players']])
        avg_consistency = np.mean([p.get('consistency_rating', 0) for p in lineup['players']])
        avg_stability = np.mean([p.get('role_stability', 0) for p in lineup['players']])
        total_cash_score = sum(p.get('cash_game_score', 0) for p in lineup['players'])
        
        print(f"   📊 Lineup Analysis: {cash_safe_count}/8 cash safe, "
              f"Avg Ownership: {avg_ownership:.1f}%, "
              f"Avg Consistency: {avg_consistency:.1f}, "
              f"Avg Stability: {avg_stability:.1f}, "
              f"Total Cash Score: {total_cash_score:.0f}")

    def build_high_upside_lineups_v2(self, num_lineups=3):
        """Improved high-upside lineup builder with validation"""
        print(f"\n🎯 Building {num_lineups} validated high-upside lineups...")
        
        valid_upside_players = [p for p in self.data.players_data 
                              if p.get('is_valid_upside_play', False) and 
                                 p.get('realistic_upside_score', 0) >= 60]
        
        print(f"   Validated upside plays: {len(valid_upside_players)}")
        
        if len(valid_upside_players) < 8:
            print("❌ Not enough validated upside players")
            return []
        
        lineups = []
        previous_lineups_players = []
        
        for lineup_num in range(num_lineups):
            print(f"   Building upside lineup {lineup_num + 1}...")
            
            prob = pulp.LpProblem(f"NBA_Upside_v2_{lineup_num}", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)
            
            prob += pulp.lpSum([
                player_vars[i] * (
                    self.data.players_data[i]['projection'] * 0.6 +
                    self.data.players_data[i].get('realistic_upside_score', 0) * 0.4
                ) for i in range(len(self.data.players_data))
            ])
            
            prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['salary'] for i in range(len(self.data.players_data))]) <= 50000
            prob += pulp.lpSum([player_vars[i] for i in range(len(self.data.players_data))]) == 8
            
            pg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PG']
            sg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SG']
            sf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SF']
            pf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PF']
            c_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'C']
            
            prob += pulp.lpSum([player_vars[i] for i in pg_players]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in sg_players]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in sf_players]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in pf_players]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in c_players]) == 1
            
            guard_players = pg_players + sg_players
            forward_players = sf_players + pf_players
            prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
            prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3
            
            upside_indices = [i for i, p in enumerate(self.data.players_data) 
                            if p.get('is_valid_upside_play', False)]
            
            if len(upside_indices) >= 5:
                prob += pulp.lpSum([player_vars[i] for i in upside_indices]) >= 5
            
            high_risk_indices = [i for i, p in enumerate(self.data.players_data)
                              if p.get('injury_risk', 0) > 0.6 or p.get('role_stability', 0) < 50]
            
            if high_risk_indices:
                prob += pulp.lpSum([player_vars[i] for i in high_risk_indices]) <= 2
            
            if lineup_num > 0 and previous_lineups_players:
                all_prev_players = []
                for prev_lineup in previous_lineups_players:
                    all_prev_players.extend(prev_lineup)
                all_prev_players = list(set(all_prev_players))
                
                if len(all_prev_players) > 0:
                    prob += pulp.lpSum([player_vars[i] for i in all_prev_players]) <= 5
            
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status == pulp.LpStatusOptimal:
                lineup = self.extract_lineup(player_vars)
                if lineup and self.data.validate_upside_lineup(lineup):
                    lineup_indices = [i for i, p in enumerate(self.data.players_data) if player_vars[i].value() == 1]
                    previous_lineups_players.append(lineup_indices)
                    lineups.append(lineup)
                    print(f"   ✅ Upside lineup {lineup_num + 1} built successfully")
                else:
                    print(f"   ❌ Upside lineup {lineup_num + 1} failed validation")
            else:
                print(f"   ❌ No optimal solution found for upside lineup {lineup_num + 1}")
        
        return lineups

    def build_stars_and_scrubs_lineups(self, num_lineups=2):
        """Build lineups using stars and scrubs strategy"""
        print(f"\n⭐ Building {num_lineups} stars and scrubs lineups...")
        
        lineups = []
        
        stars = sorted(self.data.players_data, key=lambda x: x['projection'], reverse=True)[:20]
        scrubs = [p for p in self.data.players_data if p['salary'] <= 4500 and p['projection'] >= 20]
        
        print(f"   Top stars: {len(stars)}")
        print(f"   Value scrubs: {len(scrubs)}")
        
        previous_lineups_players = []
        
        for lineup_num in range(num_lineups):
            prob = pulp.LpProblem(f"NBA_DFS_Stars_Scrubs_{lineup_num}", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)
            
            prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] for i in range(len(self.data.players_data))])
            
            prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['salary'] for i in range(len(self.data.players_data))]) <= 50000
            prob += pulp.lpSum([player_vars[i] for i in range(len(self.data.players_data))]) == 8
            
            pg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PG']
            sg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SG']
            sf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SF']
            pf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PF']
            c_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'C']
            
            prob += pulp.lpSum([player_vars[i] for i in pg_players]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in sg_players]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in sf_players]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in pf_players]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in c_players]) == 1
            
            guard_players = pg_players + sg_players
            forward_players = sf_players + pf_players
            prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
            prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3
            
            star_indices = [i for i, p in enumerate(self.data.players_data) if p['salary'] >= 8000]
            if len(star_indices) >= 2:
                prob += pulp.lpSum([player_vars[i] for i in star_indices]) >= 2
                prob += pulp.lpSum([player_vars[i] for i in star_indices]) <= 3
            
            scrub_indices = [i for i, p in enumerate(self.data.players_data) if p['salary'] <= 4500 and p['projection'] >= 20]
            if len(scrub_indices) >= 3:
                prob += pulp.lpSum([player_vars[i] for i in scrub_indices]) >= 3
            
            if lineup_num > 0 and previous_lineups_players:
                all_prev_players = []
                for prev_lineup in previous_lineups_players:
                    all_prev_players.extend(prev_lineup)
                all_prev_players = list(set(all_prev_players))
                
                if len(all_prev_players) > 0:
                    prob += pulp.lpSum([player_vars[i] for i in all_prev_players]) <= 5
            
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status == pulp.LpStatusOptimal:
                lineup = self.extract_lineup(player_vars)
                if lineup:
                    lineup_indices = [i for i, p in enumerate(self.data.players_data) if player_vars[i].value() == 1]
                    previous_lineups_players.append(lineup_indices)
                    lineups.append(lineup)
        
        return lineups

    def build_high_floor_lineups(self, num_lineups=2):
        """Build lineups focused on consistent, high-floor players"""
        print(f"\n📊 Building {num_lineups} high-floor lineups...")
        
        lineups = []
        
        high_floor_players = [p for p in self.data.players_data if p.get('consistency_rating', 0) >= 70 and p.get('volatility_score', 1) <= 0.8]
        
        print(f"   High-floor players: {len(high_floor_players)}")
        
        previous_lineups_players = []
        
        for lineup_num in range(num_lineups):
            prob = pulp.LpProblem(f"NBA_DFS_High_Floor_{lineup_num}", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)
            
            prob += pulp.lpSum([
                player_vars[i] * (
                    self.data.players_data[i]['projection'] * 0.8 +
                    self.data.players_data[i]['consistency_rating'] * 0.2
                ) for i in range(len(self.data.players_data))
            ])
            
            prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['salary'] for i in range(len(self.data.players_data))]) <= 50000
            prob += pulp.lpSum([player_vars[i] for i in range(len(self.data.players_data))]) == 8
            
            pg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PG']
            sg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SG']
            sf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SF']
            pf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PF']
            c_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'C']
            
            prob += pulp.lpSum([player_vars[i] for i in pg_players]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in sg_players]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in sf_players]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in pf_players]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in c_players]) == 1
            
            guard_players = pg_players + sg_players
            forward_players = sf_players + pf_players
            prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
            prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3
            
            high_floor_indices = [i for i, p in enumerate(self.data.players_data) if p.get('consistency_rating', 0) >= 70]
            if len(high_floor_indices) >= 5:
                prob += pulp.lpSum([player_vars[i] for i in high_floor_indices]) >= 5
            
            low_vol_indices = [i for i, p in enumerate(self.data.players_data) if p.get('volatility_score', 1) <= 0.8]
            if len(low_vol_indices) >= 4:
                prob += pulp.lpSum([player_vars[i] for i in low_vol_indices]) >= 4
            
            if lineup_num > 0 and previous_lineups_players:
                all_prev_players = []
                for prev_lineup in previous_lineups_players:
                    all_prev_players.extend(prev_lineup)
                all_prev_players = list(set(all_prev_players))
                
                if len(all_prev_players) > 0:
                    prob += pulp.lpSum([player_vars[i] for i in all_prev_players]) <= 5
            
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status == pulp.LpStatusOptimal:
                lineup = self.extract_lineup(player_vars)
                if lineup:
                    lineup_indices = [i for i, p in enumerate(self.data.players_data) if player_vars[i].value() == 1]
                    previous_lineups_players.append(lineup_indices)
                    lineups.append(lineup)
        
        return lineups

    def optimize_max_points_lineup(self):
        """Build a lineup focused purely on maximum projected points"""
        print("\n💥 Building MAXIMUM POINTS lineup...")
        
        prob = pulp.LpProblem("NBA_DFS_Max_Points", pulp.LpMaximize)
        player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)
        
        prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] for i in range(len(self.data.players_data))])
        
        prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['salary'] for i in range(len(self.data.players_data))]) <= 50000
        prob += pulp.lpSum([player_vars[i] for i in range(len(self.data.players_data))]) == 8
        
        pg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PG']
        sg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SG']
        sf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SF']
        pf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PF']
        c_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'C']
        
        prob += pulp.lpSum([player_vars[i] for i in pg_players]) >= 1
        prob += pulp.lpSum([player_vars[i] for i in sg_players]) >= 1
        prob += pulp.lpSum([player_vars[i] for i in sf_players]) >= 1
        prob += pulp.lpSum([player_vars[i] for i in pf_players]) >= 1
        prob += pulp.lpSum([player_vars[i] for i in c_players]) == 1
        
        guard_players = pg_players + sg_players
        forward_players = sf_players + pf_players
        prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
        prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            return self.extract_lineup(player_vars)
        else:
            print("   ❌ No optimal solution found for max points lineup")
            return None

    def optimize_lineup(self):
        """Optimize lineup using enhanced projections"""
        print("\n🧠 Optimizing lineup with enhanced projections...")
        
        prob = pulp.LpProblem("NBA_DFS_Enhanced", pulp.LpMaximize)
        player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)
        
        prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] for i in range(len(self.data.players_data))])
        
        prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['salary'] for i in range(len(self.data.players_data))]) <= 50000
        prob += pulp.lpSum([player_vars[i] for i in range(len(self.data.players_data))]) == 8
        
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
        total_ceiling = 0
        total_upside = 0
        
        for i, player in enumerate(self.data.players_data):
            if player_vars[i].value() == 1:
                lineup_players.append(player)
                total_salary += player['salary']
                total_projection += player['projection']
                total_ceiling += player.get('ceiling_projection', player['projection'])
                total_upside += player.get('upside_score', 0)
        
        position_order = {'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5}
        lineup_players.sort(key=lambda x: position_order.get(x['position'], 6))
        
        return {
            'players': lineup_players,
            'total_salary': total_salary,
            'total_projection': total_projection,
            'total_ceiling': total_ceiling,
            'avg_upside_score': total_upside / len(lineup_players) if lineup_players else 0,
            'efficiency': total_projection / total_salary * 1000 if total_salary > 0 else 0
        }

    def display_lineup_enhanced(self, lineup):
        """Display the optimized lineup with enhanced metrics"""
        print("\n" + "=" * 140)
        print("🏆 ENHANCED PROJECTIONS NBA DFS LINEUP v3")
        print("With Cash Game Safety & Realistic Upside")
        print("=" * 140)
        
        source = lineup['players'][0].get('source', 'enhanced') if lineup['players'] else 'enhanced'
        if source == 'fallback':
            print("⚠️  USING FALLBACK DATA - NBA API unavailable")
            print("=" * 140)
        
        for i, player in enumerate(lineup['players'], 1):
            value_score = (player['projection'] / player['salary']) * 1000
            
            cash_indicator = "💰" if player.get('cash_game_safe', False) else "🎯"
            upside_indicator = "🚀" if player.get('realistic_upside_score', 0) > 70 else "⬆️" if player.get('realistic_upside_score', 0) > 50 else "➡️"
            injury_indicator = "🩹" if player.get('injury_risk', 0) > 0.5 else "✅"
            stability_indicator = "🛡️" if player.get('role_stability', 0) > 80 else "📊" if player.get('role_stability', 0) > 60 else "⚠️"
            
            ownership_info = f"Own: {player.get('contest_ownership', 0):.1f}%" if player.get('contest_ownership') else ""
            
            print(f"{i:2d}. {player['position']:2} | {player['name']:20} | "
                  f"${player['salary']:5,} | {player['projection']:5.1f} pts | "
                  f"CASH: {cash_indicator} | UPSIDE: {player.get('realistic_upside_score', 0):2.0f} {upside_indicator} | "
                  f"INJ: {injury_indicator} | STAB: {stability_indicator} | "
                  f"Min: {player.get('projected_minutes', 0):2.0f} | USG: {player.get('usage_rate', 0):.2f} | {ownership_info}")
    
        print("=" * 140)
        print(f"💵 Total Salary: ${lineup['total_salary']:,} / $50,000")
        print(f"📈 Total Projection: {lineup['total_projection']:.1f} fantasy points")
        print(f"🛡️  Cash Game Safe Players: {sum(1 for p in lineup['players'] if p.get('cash_game_safe', False))}/8")
        print(f"🚀 Valid Upside Plays: {sum(1 for p in lineup['players'] if p.get('is_valid_upside_play', False))}/8")
        print(f"📉 Total Floor: {sum(p.get('floor_projection', 0) for p in lineup['players']):.1f} fantasy points")
        
        if source == 'fallback':
            print("🔴 NOTE: Using fallback data - projections are salary-based estimates only")
        else:
            print("🟢 Using enhanced projections v3 with NBA API data")
        
        print("=" * 140)

if __name__ == "__main__":
    print("🎯 ENHANCED PROJECTIONS NBA DFS LINEUP BUILDER v3 - CASH GAME & UPSIDE FIXES")
    print("With realistic upside projections and cash game safety")
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
    
    print("\n🎯 Available Lineup Strategies:")
    print("1. Balanced (default)")
    print("2. High Upside (validated tournament plays)")
    print("3. Stars & Scrubs (premium players + value picks)")
    print("4. High Floor (cash game focus)")
    print("5. Cash Game (maximum safety)")
    print("6. Maximum Points (pure projection optimization)")
    print("7. Build ALL strategies")
    
    choice = input("\nSelect strategy (1-7, default 1): ").strip()
    
    if choice == "2":
        num_lineups = input("How many validated upside lineups? (default 3): ").strip()
        num_lineups = int(num_lineups) if num_lineups.isdigit() else 3
        success = optimizer.build_lineups(strategy='high_upside', num_lineups=num_lineups)
        
    elif choice == "3":
        num_lineups = input("How many stars & scrubs lineups? (default 2): ").strip()
        num_lineups = int(num_lineups) if num_lineups.isdigit() else 2
        success = optimizer.build_lineups(strategy='stars_and_scrubs', num_lineups=num_lineups)
        
    elif choice == "4":
        num_lineups = input("How many high-floor lineups? (default 2): ").strip()
        num_lineups = int(num_lineups) if num_lineups.isdigit() else 2
        success = optimizer.build_lineups(strategy='high_floor', num_lineups=num_lineups)
        
    elif choice == "5":
        num_lineups = input("How many cash game lineups? (default 2): ").strip()
        num_lineups = int(num_lineups) if num_lineups.isdigit() else 2
        success = optimizer.build_lineups(strategy='cash_game', num_lineups=num_lineups)
        
    elif choice == "6":
        max_points_lineup = optimizer.optimize_max_points_lineup()
        if max_points_lineup:
            print("\n💥 MAXIMUM POINTS LINEUP (Pure Projection Optimization)")
            optimizer.display_lineup_enhanced(max_points_lineup)
            success = True
        else:
            success = False
            
    elif choice == "7":
        print("\n🏗️  Building ALL lineup strategies...")
        
        print("\n" + "="*60)
        print("💰 CASH GAME STRATEGY")
        print("="*60)
        cash_success = optimizer.build_lineups(strategy='cash_game', num_lineups=1)
        
        print("\n" + "="*60)
        print("⚖️  BALANCED STRATEGY")
        print("="*60)
        balanced_success = optimizer.build_lineups(strategy='balanced', num_lineups=1)
        
        print("\n" + "="*60)
        print("🚀 VALIDATED HIGH UPSIDE STRATEGY")
        print("="*60)
        upside_success = optimizer.build_lineups(strategy='high_upside', num_lineups=2)
        
        success = cash_success or balanced_success or upside_success
        
    else:
        success = optimizer.build_lineups(strategy='balanced', num_lineups=1)
    
    if not success:
        print("\n💥 Failed to build lineups")
        sys.exit(1)