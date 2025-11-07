# real_lineup_builderV9_fast.py
# enhanced_projections_lineup_builder.py - OPTIMIZED VERSION
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
import pickle
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

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
        
        # Caching system
        self._cache = {}
        self._cache_enabled = True
        self._cache_file = "nba_data_cache.pkl"
        
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

    # CACHE METHODS
    def cache_get(self, key):
        """Get item from cache"""
        if not self._cache_enabled:
            return None
        return self._cache.get(key)
    
    def cache_set(self, key, value):
        """Set item in cache"""
        if self._cache_enabled:
            self._cache[key] = value
    
    def load_cache(self):
        """Load cache from file"""
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                print(f"✅ Loaded cache with {len(self._cache)} items")
                return True
        except:
            self._cache = {}
        return False
    
    def save_cache(self):
        """Save cache to file"""
        try:
            with open(self._cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
            print(f"💾 Saved cache with {len(self._cache)} items")
            return True
        except:
            return False

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
    def analyze_contest_results_fast(self, results_file="cashGameContestResults.csv"):
        """Fast contest analysis - FIXED VERSION"""
        print("📊 Fast contest analysis...")

        cache_key = f"contest_analysis_{os.path.getmtime(results_file) if os.path.exists(results_file) else 'none'}"
        cached_insights = self.cache_get(cache_key)
        if cached_insights:
            self.contest_insights = cached_insights
            print(f"✅ Loaded contest insights from cache ({len(self.contest_insights)} players)")
            return True

        try:
            # Handle encoding and ensure proper numeric conversion
            results_df = pd.read_csv(results_file, encoding='utf-8-sig')  # Handle BOM

            # Extract player performances - FIXED: Convert to numeric properly
            player_performances = {}
            for _, row in results_df.iterrows():
                player_name = str(row['Player']).strip()

                # Safely convert FPTS to numeric
                try:
                    fpts = float(str(row['FPTS']).replace(',', '')) if pd.notna(row['FPTS']) else 0
                except:
                    fpts = 0

                # Safely convert %Drafted to numeric  
                try:
                    drafted_pct = float(str(row['%Drafted']).replace('%', '').replace(',', '')) if pd.notna(row['%Drafted']) else 0
                except:
                    drafted_pct = 0

                roster_pos = str(row['Roster Position']).strip() if pd.notna(row['Roster Position']) else ''

                if player_name and player_name != 'nan':
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
                if performances:  # Only process if we have data
                    avg_fpts = np.mean([p['fpts'] for p in performances if p['fpts'] > 0])
                    avg_ownership = np.mean([p['ownership'] for p in performances if p['ownership'] > 0])

                    self.contest_insights[player] = {
                        'avg_fpts': avg_fpts,
                        'avg_ownership': avg_ownership,
                        'games': len(performances)
                    }

            print(f"✅ Analyzed {len(self.contest_insights)} players from contest results")

            # Cache the results
            self.cache_set(cache_key, self.contest_insights)
            return True

        except Exception as e:
            print(f"❌ Error analyzing contest results: {e}")
            import traceback
            traceback.print_exc()
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
        top_cash = sorted(cash_players, key=lambda x: x.get('cash_game_score', 0), reverse=True)[:5]
        print("\n🏆 TOP CASH GAME PLAYS:")
        for player in top_cash:
            print(f"   {player['name']}: Score {player.get('cash_game_score', 0)}, "
                  f"Ownership {player.get('contest_ownership', 0):.1f}%, "
                  f"Value {(player['projection']/player['salary'])*1000:.2f}")
        
        return players_data

    # FAST DATA LOADING METHODS
    def load_dk_salaries_fast(self):
        """Fast DraftKings salaries loading"""
        print("💰 Loading DraftKings salaries...")
        
        if not os.path.exists(self.dk_salaries_path):
            print(f"❌ DraftKings salaries file not found: {self.dk_salaries_path}")
            return False
        
        try:
            self.dk_salaries_df = pd.read_csv(self.dk_salaries_path)
            print(f"✅ Loaded {len(self.dk_salaries_df)} players from DraftKings salaries")
            return True
            
        except Exception as e:
            print(f"❌ Error loading DraftKings salaries: {e}")
            return False

    def get_todays_games_fast(self):
        """Fast version of today's games lookup"""
        cache_key = "todays_games"
        cached_games = self.cache_get(cache_key)
        if cached_games:
            self.todays_games = cached_games
            print(f"✅ Today's games from cache: {', '.join(self.todays_games)}")
            return True
        
        # Fallback to static list for speed
        self.todays_games = list(self.team_id_map.values())
        print(f"⚠️  Using all teams for speed: {len(self.todays_games)} teams")
        
        # Cache the result
        self.cache_set(cache_key, self.todays_games)
        return True

    def get_team_stats_and_pace_fast(self):
        """Fast version of team stats collection"""
        print("📊 Getting team stats and pace data (FAST MODE)...")
        
        cache_key = "team_stats"
        cached_stats = self.cache_get(cache_key)
        if cached_stats:
            self.team_stats = cached_stats
            print("   ✅ Loaded team stats from cache")
            self.calculate_game_pace_fast()
            return True
        
        try:
            def api_call():
                team_stats = leaguedashteamstats.LeagueDashTeamStats(season='2024-25', timeout=30)
                return team_stats.get_data_frames()[0]
            
            team_stats_df = self.safe_api_call(api_call, max_retries=2, delay=2)
            
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
                
                self.team_stats[team_abbr] = {
                    'pace': team[pace_col] if pace_col else 100.0,
                    'off_rating': team[off_rating_col] if off_rating_col else 110.0,
                    'def_rating': team[def_rating_col] if def_rating_col else 110.0,
                    'avg_points': 110.0
                }
            
            print(f"✅ Loaded stats for {len(self.team_stats)} teams")
            
            # Cache the results
            self.cache_set(cache_key, self.team_stats)
            self.calculate_game_pace_fast()
            return True
            
        except Exception as e:
            print(f"❌ Error getting team stats: {e}")
            return False

    def calculate_game_pace_fast(self):
        """Fast game pace calculation"""
        print("   Calculating game pace projections...")
        
        try:
            # Simplified pace calculation
            for home_team in self.todays_games[:4]:  # Limit to first 4 teams for speed
                for away_team in self.todays_games[:4]:
                    if home_team != away_team:
                        home_pace = self.team_stats.get(home_team, {}).get('pace', 100.0)
                        away_pace = self.team_stats.get(away_team, {}).get('pace', 100.0)
                        avg_pace = (home_pace + away_pace) / 2
                        
                        self.game_pace_data[f"{away_team}@{home_team}"] = {
                            'pace': avg_pace,
                            'home_team': home_team,
                            'away_team': away_team
                        }
                        
        except Exception as e:
            print(f"❌ Error calculating game pace: {e}")

    # PARALLEL PROCESSING METHODS
    def process_single_player_fast(self, task):
        """Process a single player (for parallel execution)"""
        player = task['player']
        all_logs_df = task['all_logs_df']
        
        try:
            player_id = player['id']
            player_name = player['full_name']
            team = player['team']
            
            # Check cache first
            cache_key = f"player_{player_id}_{team}"
            cached_result = self.cache_get(cache_key)
            if cached_result:
                return cached_result
            
            dk_salary_info = self.find_player_in_dk_salaries(player_name, team)
            if not dk_salary_info:
                return None
            
            player_logs = all_logs_df[all_logs_df['PLAYER_ID'] == player_id]
            if player_logs.empty:
                return None
            
            recent_games = player_logs.head(10)
            if len(recent_games) < 3:
                return None
            
            # Skip players with insufficient data quickly
            required_columns = ['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M']
            missing_columns = [col for col in required_columns if col not in recent_games.columns]
            if missing_columns:
                return None
            
            most_recent_game = recent_games.iloc[0]
            game_date = pd.to_datetime(most_recent_game['GAME_DATE'])
            days_since_last_game = (datetime.now() - game_date).days
            
            if days_since_last_game > 7:
                return None
            
            recent_minutes = recent_games['MIN'].mean()
            if recent_minutes < 5:
                return None
            
            position = dk_salary_info.get('position')
            if not position:
                position = self.get_player_position_fast(player_id, player_name)
            
            projection_data = self.calculate_enhanced_projection_v3_fast(
                player, recent_games, position, team, dk_salary_info, all_logs_df
            )
            
            if projection_data:
                # Cache the result
                self.cache_set(cache_key, projection_data)
                return projection_data
                
        except Exception as e:
            return None
        
        return None

    def get_player_position_fast(self, player_id, player_name):
        """Fast player position lookup"""
        cache_key = f"position_{player_id}"
        cached_pos = self.cache_get(cache_key)
        if cached_pos:
            return cached_pos
        
        # Use estimation for speed
        position = self.estimate_position_from_name(player_name)
        self.cache_set(cache_key, position)
        return position

    def calculate_enhanced_projection_v3_fast(self, player, recent_games, position, team, dk_salary_info, all_logs_df):
        """Fast version of projection calculation"""
        try:
            # Simplified projection calculation
            avg_stats = self.calculate_robust_averages(recent_games)
            
            if avg_stats['minutes'] < 10:
                return None
            
            usage_rate = self.calculate_usage_rate(recent_games, team)
            consistency_rating = self.calculate_consistency_rating(recent_games)
            role_stability = self.calculate_role_stability(recent_games, player['full_name'], position)
            
            # Simplified projection
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
            
            projected_minutes = self.project_minutes_enhanced_v2(
                avg_stats['minutes'], recent_games, team, usage_rate, role_stability
            )
            
            base_projection_stats = {}
            for stat in per_36_stats:
                base_projection_stats[stat] = (per_36_stats[stat] / 36) * projected_minutes
            
            base_projection = self.calculate_dk_points(base_projection_stats)
            conservative_projection = self.calculate_conservative_projection(recent_games, base_projection, consistency_rating)
            
            salary = dk_salary_info['salary']
            
            if conservative_projection < 5:
                return None
            
            value_rating = (conservative_projection / salary) * 1000
            
            return {
                'name': player['full_name'],
                'position': position,
                'team': team,
                'salary': salary,
                'projection': round(conservative_projection, 1),
                'ceiling_projection': round(conservative_projection * 1.3, 1),
                'floor_projection': round(conservative_projection * 0.7, 1),
                'minutes': round(avg_stats['minutes'], 1),
                'projected_minutes': round(projected_minutes, 1),
                'usage_rate': round(usage_rate, 3),
                'value_rating': round(value_rating, 2),
                'consistency_rating': round(consistency_rating, 1),
                'role_stability': round(role_stability, 1),
                'cash_game_safe': False,
                'source': 'enhanced_projections_fast',
                'playing_today': True
            }
            
        except Exception as e:
            return None

    def get_enhanced_player_projections_fast(self, season):
        """Fast parallel version of player projections - FIXED to get more players"""
        print("🔄 Getting enhanced player projections (FAST MODE)...")

        # Get players and logs
        all_players_with_teams = self.get_all_players_with_correct_teams_fast()
        if not all_players_with_teams:
            return []

        todays_players = [p for p in all_players_with_teams if p.get('team') in self.todays_games]
        if not todays_players:
            todays_players = all_players_with_teams

        print(f"   🎯 Found {len(todays_players)} players on today's teams")

        # Get game logs
        try:
            cache_key = f"game_logs_{season}"
            all_logs_df = self.cache_get(cache_key)

            if all_logs_df is None:
                def api_call():
                    all_game_logs = playergamelogs.PlayerGameLogs(season_nullable=season, timeout=60)
                    return all_game_logs.get_data_frames()[0]

                all_logs_df = self.safe_api_call(api_call, max_retries=3, delay=3)
                self.cache_set(cache_key, all_logs_df)

            print(f"   📈 Loaded {len(all_logs_df)} total game logs")
        except Exception as e:
            print(f"❌ Error loading game logs: {e}")
            return []

        # Prepare data for parallel processing - INCREASE player limit
        player_data = []
        tasks = []

        # Process more players - increased from 100 to 200
        for player in todays_players[:200]:  
            tasks.append({
                'player': player,
                'all_logs_df': all_logs_df,
                'season': season
            })

        # Process in parallel
        print(f"   🚀 Processing {len(tasks)} players in parallel...")

        # Use ThreadPoolExecutor for I/O bound tasks
        with ThreadPoolExecutor(max_workers=min(8, mp.cpu_count())) as executor:
            futures = []
            for task in tasks:
                future = executor.submit(self.process_single_player_fast, task)
                futures.append(future)

            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=15)  # 15 second timeout per player
                    if result:
                        player_data.append(result)
                except Exception as e:
                    continue
                
        print(f"   ✅ Processed {len(player_data)} players with enhanced projections")

        # If we still have too few players, use fallback
        if len(player_data) < 20:
            print("   ⚠️  Too few players, using fallback method...")
            player_data = self.get_fallback_player_data()

        player_data = self.filter_cash_game_players(player_data)

        return player_data
    
    def get_fallback_player_data(self):
        """Fallback to ensure we have enough players for optimization"""
        print("   🆘 Using fallback player data...")

        if self.dk_salaries_df is None:
            return []

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

                # Better projection calculation for fallback
                if position == 'PG':
                    projection = salary * 0.0055
                elif position == 'SG':
                    projection = salary * 0.0052
                elif position == 'SF':
                    projection = salary * 0.0050
                elif position == 'PF':
                    projection = salary * 0.0048
                else:  # C
                    projection = salary * 0.0045

                # Add some variance based on name hash for differentiation
                variance = (hash(name) % 100 - 50) * 0.1  # -5% to +5% variance
                projection = projection * (1 + variance)

                player_data.append({
                    'name': name,
                    'position': position,
                    'team': team,
                    'salary': salary,
                    'projection': round(projection, 1),
                    'ceiling_projection': round(projection * 1.4, 1),
                    'floor_projection': round(projection * 0.6, 1),
                    'value_rating': round((projection / salary) * 1000, 2),
                    'cash_game_safe': salary < 7000,  # Cheaper players are safer in fallback
                    'source': 'fallback',
                    'playing_today': True
                })

            except Exception as e:
                continue
            
        print(f"   ✅ Fallback created {len(player_data)} players")
        return player_data

    def get_all_players_with_correct_teams_fast(self):
        """Fast player acquisition"""
        print("   🔍 Getting players with teams (FAST MODE)...")
        
        cache_key = "all_players"
        cached_players = self.cache_get(cache_key)
        if cached_players:
            print(f"   ✅ Loaded {len(cached_players)} players from cache")
            return cached_players
        
        # Use fallback for speed
        try:
            season = self.get_current_season()
            
            def api_call():
                all_game_logs = playergamelogs.PlayerGameLogs(season_nullable=season, timeout=60)
                return all_game_logs.get_data_frames()[0]
            
            all_logs_df = self.safe_api_call(api_call, max_retries=2, delay=2)
            
            if not all_logs_df.empty:
                unique_players = all_logs_df[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION']].drop_duplicates()
                players_with_teams = []
                
                for _, row in unique_players.iterrows():
                    players_with_teams.append({
                        'id': row['PLAYER_ID'],
                        'full_name': row['PLAYER_NAME'],
                        'team': row['TEAM_ABBREVIATION']
                    })
                
                print(f"   ✅ Acquired {len(players_with_teams)} players from game logs")
                self.cache_set(cache_key, players_with_teams)
                return players_with_teams
                
        except Exception as e:
            print(f"   ❌ Fast player acquisition failed: {e}")
        
        # Final fallback
        all_active_players = [p for p in players.get_players() if p['is_active']]
        teams_list = list(self.team_id_map.values())
        for player in all_active_players:
            player['team'] = teams_list[hash(player['full_name']) % len(teams_list)]
        
        print(f"   ✅ Fallback assigned {len(all_active_players)} players to teams")
        self.cache_set(cache_key, all_active_players)
        return all_active_players

    # FAST DATA ENTRY POINT
    def get_real_nba_data_fast(self, use_cache=True):
        """Ultra-fast data loading with caching and parallel processing"""
        print("🚀 FAST MODE: Loading NBA data with optimizations...")
        
        # Load cache if enabled
        if use_cache:
            self.load_cache()
        
        # Check for complete cached dataset
        cache_key = "complete_player_data"
        cached_players = self.cache_get(cache_key)
        if cached_players and use_cache:
            self.players_data = cached_players
            print(f"✅ Loaded {len(self.players_data)} players from cache")
            self.identify_high_upside_players()
            self.show_data_summary_fast()
            return True
        
        # Fast path - essential data only
        if not self.load_dk_salaries_fast():
            return False
        
        # Get today's games quickly
        self.get_todays_games_fast()
        
        # Get team stats quickly
        self.get_team_stats_and_pace_fast()
        
        # Use parallel processing for players
        season = self.get_current_season()
        self.players_data = self.get_enhanced_player_projections_fast(season)
        
        if self.players_data:
            print(f"✅ Successfully loaded {len(self.players_data)} players (FAST MODE)")
            
            # Apply contest insights
            self.analyze_contest_results_fast()
            self.apply_contest_insights_to_projections()
            self.players_data = self.enhanced_cash_game_filters(self.players_data)
            
            # Cache complete dataset
            if use_cache:
                self.cache_set(cache_key, self.players_data)
                self.save_cache()
            
            self.identify_high_upside_players()
            self.show_data_summary_fast()
            return True
        else:
            return self.create_fallback_data()

    def show_data_summary_fast(self):
        """Fast data summary"""
        if not self.players_data:
            return
            
        print("\n📊 FAST MODE SUMMARY:")
        print("-" * 60)
        
        pos_count = {}
        for player in self.players_data:
            pos = player['position']
            pos_count[pos] = pos_count.get(pos, 0) + 1
        
        print("Position Distribution:")
        for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
            count = pos_count.get(pos, 0)
            print(f"  {pos}: {count} players")
        
        avg_proj = sum(p['projection'] for p in self.players_data) / len(self.players_data)
        avg_salary = sum(p['salary'] for p in self.players_data) / len(self.players_data)
        cash_safe = sum(1 for p in self.players_data if p.get('cash_game_safe', False))
        
        print(f"\n📈 Averages: Projection {avg_proj:.1f}, Salary ${avg_salary:,.0f}")
        print(f"💰 Cash Game Safe: {cash_safe}/{len(self.players_data)}")

    # EXISTING METHODS (minimal changes)
    def get_current_season(self):
        today = datetime.now()
        current_year = today.year
        if today.month >= 10:
            return f"{current_year}-{str(current_year + 1)[-2:]}"
        else:
            return f"{current_year - 1}-{str(current_year)[-2:]}"

    def calculate_robust_averages(self, recent_games):
        try:
            median_minutes = recent_games['MIN'].median()
            stats = {}
            
            stat_mapping = {
                'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST',
                'steals': 'STL', 'blocks': 'BLK', 'turnovers': 'TOV',
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
            return {
                'points': recent_games['PTS'].mean() if 'PTS' in recent_games.columns else 0,
                'rebounds': recent_games['REB'].mean() if 'REB' in recent_games.columns else 0,
                'assists': recent_games['AST'].mean() if 'AST' in recent_games.columns else 0,
                'steals': recent_games['STL'].mean() if 'STL' in recent_games.columns else 0,
                'blocks': recent_games['BLK'].mean() if 'BLK' in recent_games.columns else 0,
                'turnovers': recent_games['TOV'].mean() if 'TOV' in recent_games.columns else 0,
                'minutes': recent_games['MIN'].mean() if 'MIN' in recent_games.columns else 0,
                'three_pointers_made': recent_games['FG3M'].mean() if 'FG3M' in recent_games.columns else 0
            }

    def calculate_volatility_score(self, recent_games):
        try:
            if len(recent_games) < 3:
                return 1.0
            
            game_scores = []
            for _, game in recent_games.iterrows():
                fp = self.calculate_dk_points({
                    'points': game['PTS'], 'rebounds': game['REB'], 'assists': game['AST'],
                    'steals': game['STL'], 'blocks': game['BLK'], 'turnovers': game['TOV'],
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
        try:
            if len(recent_games) < 3:
                return 50
            
            game_scores = []
            for _, game in recent_games.iterrows():
                fp = self.calculate_dk_points({
                    'points': game['PTS'], 'rebounds': game['REB'], 'assists': game['AST'],
                    'steals': game['STL'], 'blocks': game['BLK'], 'turnovers': game['TOV'],
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

    def calculate_usage_rate(self, player_games, team):
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

    def find_player_in_dk_salaries(self, player_name, team):
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
            
            return {
                'salary': int(player_row[salary_col]) if salary_col else 0,
                'position': player_row.get(position_col, ''),
                'team': player_row.get(team_col, ''),
                'avg_points': 0
            }
        except Exception as e:
            return None

    def get_name_variations(self, full_name):
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

    def calculate_dk_points(self, stats):
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

    def estimate_position_from_name(self, player_name):
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
        high_upside_players = []
        
        for player in self.players_data:
            upside_score = player.get('upside_score', 0)
            ceiling_projection = player.get('ceiling_projection', 0)
            projection = player.get('projection', 0)
            salary = player.get('salary', 0)
            
            if (upside_score >= 60 and 
                ceiling_projection >= projection * 1.5 and
                salary < 8000):
                
                high_upside_players.append(player)
        
        high_upside_players.sort(key=lambda x: x.get('upside_score', 0), reverse=True)
        self.high_upside_players = high_upside_players
        
        print(f"✅ Found {len(high_upside_players)} high-upside players")

    def create_fallback_data(self):
        print("🆘 Creating fallback data from DK salaries...")
        
        if self.dk_salaries_df is None:
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
                
                player_data.append({
                    'name': name,
                    'position': position,
                    'team': team,
                    'salary': salary,
                    'projection': round(projection, 1),
                    'ceiling_projection': round(projection * 1.3, 1),
                    'floor_projection': round(projection * 0.7, 1),
                    'value_rating': round((projection / salary) * 1000, 2),
                    'cash_game_safe': False,
                    'source': 'fallback',
                    'playing_today': True
                })
                
            except Exception as e:
                continue
        
        if player_data:
            self.players_data = player_data
            print(f"✅ Created fallback data for {len(player_data)} players")
            return True
        else:
            return False

# FAST OPTIMIZER CLASS
class EnhancedProjectionsNBAOptimizer:
    def __init__(self, dk_salaries_path="DKSalaries.csv", fast_mode=True):
        self.data = EnhancedProjectionsNBAData(dk_salaries_path)
        self.lineup_strategies = ['balanced', 'high_upside', 'stars_and_scrubs', 'high_floor', 'cash_game']
        self.fast_mode = fast_mode
    
    def build_lineups_fast(self, strategy='balanced', num_lineups=1):
        """Fast lineup building"""
        print(f"🚀 FAST MODE: Building {strategy.upper()} lineups...")
        
        if self.fast_mode:
            success = self.data.get_real_nba_data_fast(use_cache=True)
        else:
            success = self.data.get_real_nba_data()
            
        if not success:
            print("💥 Failed to get NBA data")
            return False
            
        if len(self.data.players_data) < 8:
            print(f"💥 Only {len(self.data.players_data)} players found")
            return False
        
        # Use simplified lineup building for speed
        if strategy == 'cash_game':
            lineups = self.build_enhanced_cash_game_lineups_fast(num_lineups)
        elif strategy == 'high_upside':
            lineups = self.build_high_upside_lineups_fast(num_lineups)
        else:
            lineups = [self.optimize_lineup_fast()]
        
        if lineups:
            for i, lineup in enumerate(lineups):
                if lineup:
                    print(f"\n🏆 LINEUP {i+1} - {strategy.upper()} (FAST MODE)")
                    self.display_lineup_enhanced_fast(lineup)
            return True
        return False

    def optimize_lineup_fast(self):
        """Fast lineup optimization with simplified constraints"""
        print("🧠 Fast lineup optimization...")
        
        prob = pulp.LpProblem("NBA_DFS_Fast", pulp.LpMaximize)
        player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)
        
        # Simple objective - just maximize projections
        prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] for i in range(len(self.data.players_data))])
        
        # Basic constraints only
        prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['salary'] for i in range(len(self.data.players_data))]) <= 50000
        prob += pulp.lpSum([player_vars[i] for i in range(len(self.data.players_data))]) == 8
        
        # Simplified position constraints
        pg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PG']
        sg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SG'] 
        sf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SF']
        pf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PF']
        c_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'C']
        
        prob += pulp.lpSum([player_vars[i] for i in pg_players]) >= 1
        prob += pulp.lpSum([player_vars[i] for i in sg_players]) >= 1
        prob += pulp.lpSum([player_vars[i] for i in sf_players]) >= 1
        prob += pulp.lpSum([player_vars[i] for i in pf_players]) >= 1
        prob += pulp.lpSum([player_vars[i] for i in c_players]) >= 1
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))  # 30 second time limit
        
        if prob.status == pulp.LpStatusOptimal:
            lineup = self.extract_lineup(player_vars)
            #if lineup and lineup['total_projection'] >= 240:
            return lineup
        return None

    def build_enhanced_cash_game_lineups_fast(self, num_lineups=2):
        """Fast cash game lineup building - MORE FLEXIBLE"""
        print(f"\n💰 Building {num_lineups} cash game lineups (FAST MODE)...")

        cash_game_players = [p for p in self.data.players_data if p.get('cash_game_safe', False)]
        print(f"   Cash game safe players: {len(cash_game_players)}")

        # If we don't have enough cash game players, relax the criteria
        if len(cash_game_players) < 8:
            print("   ⚠️  Not enough cash game players, relaxing criteria...")
            # Include players with decent value ratings
            cash_game_players = [p for p in self.data.players_data 
                               if p.get('value_rating', 0) > 3.5 or p.get('cash_game_safe', False)]
            print(f"   📊 Now using {len(cash_game_players)} players with value > 3.5")

        if len(cash_game_players) < 8:
            print("❌ Still not enough players for lineup construction")
            return []

        lineups = []
        previous_lineup_players = []  # Track previous lineups to avoid duplicates

        for lineup_num in range(num_lineups):
            prob = pulp.LpProblem(f"NBA_Cash_Fast_{lineup_num}", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)

            # Optimize for cash game safety AND value
            prob += pulp.lpSum([
                player_vars[i] * (
                    self.data.players_data[i].get('cash_game_score', 0) * 0.7 +
                    self.data.players_data[i].get('value_rating', 0) * 0.3
                ) 
                for i in range(len(self.data.players_data))
            ])

            # Standard constraints
            prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['salary'] for i in range(len(self.data.players_data))]) <= 50000
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
            prob += pulp.lpSum([player_vars[i] for i in c_players]) >= 1

            # Guard/forward constraints
            guard_players = pg_players + sg_players
            forward_players = sf_players + pf_players
            prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
            prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3

            # Cash game constraint
            cash_safe_indices = [i for i, p in enumerate(self.data.players_data) if p.get('cash_game_safe', False)]
            if len(cash_safe_indices) >= 5:
                prob += pulp.lpSum([player_vars[i] for i in cash_safe_indices]) >= 4

            # ANTI-DUPLICATE CONSTRAINT: Ensure this lineup is different from previous ones
            if lineup_num > 0 and previous_lineup_players:
                # Don't allow more than 6 of the same players from any previous lineup
                for prev_lineup_indices in previous_lineup_players:
                    prob += pulp.lpSum([player_vars[i] for i in prev_lineup_indices]) <= 6

            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

            if prob.status == pulp.LpStatusOptimal:
                lineup = self.extract_lineup(player_vars)
                if lineup:
                    # Track which players were used in this lineup
                    lineup_indices = [i for i, p in enumerate(self.data.players_data) if player_vars[i].value() == 1]
                    previous_lineup_players.append(lineup_indices)

                    lineups.append(lineup)
                    print(f"   ✅ Cash game lineup {lineup_num + 1} built - Projection: {lineup['total_projection']:.1f}")
                else:
                    #projection = lineup['total_projection'] if lineup else 0
                    #print(f"   ❌ Cash game lineup {lineup_num + 1} failed projection check: {projection:.1f}")
                    print(f"   ❌ Cash game lineup {lineup_num + 1} failed to extract")
            else:
                print(f"   ❌ No solution for cash game lineup {lineup_num + 1}")

    def build_high_upside_lineups_fast(self, num_lineups=2):
        """Fast upside lineup building"""
        print(f"\n🎯 Building {num_lineups} upside lineups (FAST MODE)...")
        
        lineups = []
        previous_lineup_players = []  # Track previous lineups
        
        for lineup_num in range(num_lineups):
            prob = pulp.LpProblem(f"NBA_Upside_Fast_{lineup_num}", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)
            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=20))
        
        if prob.status == pulp.LpStatusOptimal:
            lineup = self.extract_lineup(player_vars)
            if lineup:
                # Track which players were used
                lineup_indices = [i for i, p in enumerate(self.data.players_data) if player_vars[i].value() == 1]
                previous_lineup_players.append(lineup_indices)
                
                lineups.append(lineup)
                print(f"   ✅ Upside lineup {lineup_num + 1} built - Projection: {lineup['total_projection']:.1f}")
            else:
                projection = lineup['total_projection'] if lineup else 0
                print(f"   ❌ Upside lineup {lineup_num + 1} failed projection: {projection:.1f}")
            
            # Optimize for ceiling projections
            prob += pulp.lpSum([
                player_vars[i] * self.data.players_data[i].get('ceiling_projection', self.data.players_data[i]['projection'])
                for i in range(len(self.data.players_data))
            ])
            
            # Standard constraints
            prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['salary'] for i in range(len(self.data.players_data))]) <= 50000
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
            
            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=20))
            
            if prob.status == pulp.LpStatusOptimal:
                lineup = self.extract_lineup(player_vars)
                if lineup:
                    lineups.append(lineup)
                    print(f"   ✅ Upside lineup {lineup_num + 1} built")
        
        return lineups

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
            'efficiency': total_projection / total_salary * 1000 if total_salary > 0 else 0
        }

    def display_lineup_enhanced_fast(self, lineup):
        """Fast lineup display"""
        print("\n" + "=" * 100)
        print("🏆 FAST MODE LINEUP")
        print("=" * 100)
        
        for i, player in enumerate(lineup['players'], 1):
            value_score = (player['projection'] / player['salary']) * 1000
            cash_indicator = "💰" if player.get('cash_game_safe', False) else "  "
            
            print(f"{i:2d}. {player['position']:2} | {player['name']:20} | "
                  f"${player['salary']:5,} | {player['projection']:5.1f} pts | "
                  f"Value: {value_score:4.1f} {cash_indicator}")
    
        print("=" * 100)
        print(f"💵 Total Salary: ${lineup['total_salary']:,} / $50,000")
        print(f"📈 Total Projection: {lineup['total_projection']:.1f} fantasy points")
        print(f"📊 Efficiency: {lineup['efficiency']:.1f} points per $1,000")
        print(f"🛡️  Cash Game Safe: {sum(1 for p in lineup['players'] if p.get('cash_game_safe', False))}/8")
        print("=" * 100)

# MAIN EXECUTION
if __name__ == "__main__":
    print("🚀 ENHANCED NBA DFS OPTIMIZER - FAST MODE")
    print("=" * 50)
        # Add cache clearing option
    clear_cache = input("Clear cache and reload fresh data? (y/n, default n): ").strip().lower() == 'y'
    if clear_cache:
        if os.path.exists("nba_data_cache.pkl"):
            os.remove("nba_data_cache.pkl")
            print("🗑️  Cache cleared")
    
    # Add fast mode option
    use_fast_mode = input("Enable FAST mode? (y/n, default y): ").strip().lower() != 'n'
    
    dk_file = "DKSalaries.csv"
    if not os.path.exists(dk_file):
        print(f"❌ {dk_file} not found")
        sys.exit(1)
    
    try:
        from nba_api.stats.endpoints import commonplayerinfo, scoreboardv2, leaguedashteamstats, teamgamelogs
        print("✅ nba_api is installed and working")
    except ImportError:
        print("❌ nba_api not installed")
        print("Run: pip install nba_api")
        sys.exit(1)
    
    optimizer = EnhancedProjectionsNBAOptimizer(dk_file, fast_mode=use_fast_mode)
    
    if use_fast_mode:
        print("⚡ FAST MODE ENABLED - Using caching and parallel processing")
        print("   First run: 2-3 minutes, Subsequent runs: 30-60 seconds")
        success = optimizer.build_lineups_fast('cash_game', 2)
    else:
        print("🐢 FULL MODE - Complete analysis with all features") 
        success = optimizer.build_lineups('cash_game', 2)
    
    if not success:
        print("\n💥 Failed to build lineups")
        sys.exit(1)