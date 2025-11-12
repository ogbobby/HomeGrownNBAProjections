# enhanced_projections_lineup_builderV7.py
import pandas as pd
import pulp
from nba_api.stats.endpoints import commonplayerinfo, playergamelogs, scoreboardv2, teamgamelogs, leaguedashteamstats
from nba_api.stats.static import teams, players
from datetime import datetime, timedelta
import sys
import os
import time
import numpy as np
import pickle
import hashlib
from datetime import datetime
import concurrent.futures
import threading
from threading import Lock
from injurySrape import get_injuries

class EnhancedProjectionsNBAData:
    #def __init__(self, dk_salaries_path="DKSalaries.csv"):
    def __init__(self, dk_salaries_path="DKSalaries.csv", target_date=None):
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
        self.target_date = target_date  # NEW: Allow custom date
        self.player_info_cache = {}
        self.cache_file = "player_cache.pkl"
        self.load_player_cache()
        self.setup_team_map()
        self.processing_lock = Lock()

    def make_api_call_with_retry(self, api_call_func, max_retries=3, initial_delay=1):
        """Make API call with retry logic and exponential backoff"""
        for attempt in range(max_retries):
            try:
                return api_call_func()
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e
                delay = initial_delay * (2 ** attempt)  # Exponential backoff
                print(f"   ⚠️ API call failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                time.sleep(delay)
        return None

    def filter_to_dk_slate_games(self):
        """Filter to only include games that are in the DraftKings slate"""
        if self.dk_salaries_df is None:
            print("❌ No DK salaries loaded")
            return False

        # Get unique teams from DK salaries
        dk_teams = set()
        team_col = None

        # Find the team column in DK file
        for col in ['TeamAbbrev', 'Team', 'teamAbbrev', 'TEAM_ABBREVIATION']:
            if col in self.dk_salaries_df.columns:
                team_col = col
                break
            
        if not team_col:
            print("❌ Could not find team column in DK salaries")
            return False

        # Get all unique teams from DK salaries
        dk_teams = set(self.dk_salaries_df[team_col].dropna().unique())
        print(f"📋 DK Slate Teams ({len(dk_teams)}): {', '.join(sorted(dk_teams))}")

        # Filter todays_games to only include teams in DK slate
        original_games = self.todays_games.copy()
        self.todays_games = [team for team in self.todays_games if team in dk_teams]

        removed_teams = set(original_games) - set(self.todays_games)
        if removed_teams:
            print(f"🚫 Removed {len(removed_teams)} teams not in DK slate: {', '.join(sorted(removed_teams))}")

        print(f"✅ Final {len(self.todays_games)} teams for optimization: {', '.join(sorted(self.todays_games))}")

        # Also verify we have enough players
        dk_player_count = len(self.dk_salaries_df)
        print(f"📊 DK Slate has {dk_player_count} players")

        return True
    
    def get_all_players_with_correct_teams_parallel(self):
        """Get ALL active players with their CORRECT current teams - SLOWER BUT RELIABLE VERSION"""
        print("   🔍 Getting all players with correct teams (reliable parallel)...")
        
        all_active_players = [p for p in players.get_players() if p['is_active']]
        players_with_teams = []
        
        # First, check cache for as many players as possible
        cached_players = []
        uncached_players = []
        
        for player in all_active_players:
            cache_key = f"{player['id']}_{player['full_name']}"
            if cache_key in self.player_info_cache:
                cached_data, cache_time = self.player_info_cache[cache_key]
                if (datetime.now() - cache_time).days < 3:  # 3-day cache
                    team = cached_data['TEAM_ABBREVIATION'].iloc[0] if 'TEAM_ABBREVIATION' in cached_data.columns else None
                    if team and team != '':
                        player['team'] = team
                        cached_players.append(player)
                        continue
            uncached_players.append(player)
        
        print(f"   📊 Cache hit: {len(cached_players)} players, Miss: {len(uncached_players)} players")
        
        # Only process uncached players in parallel with SLOWER processing
        def process_player_batch(batch):
            batch_results = []
            for player in batch:
                try:
                    player_info = self.get_cached_player_info(player['id'], player['full_name'])
                    if player_info is not None and 'TEAM_ABBREVIATION' in player_info.columns:
                        team = player_info['TEAM_ABBREVIATION'].iloc[0]
                        if team and team != '':
                            player['team'] = team
                            batch_results.append(player)
                except Exception as e:
                    continue
            return batch_results
        
        # Process uncached players in parallel with MUCH smaller batches and delays
        if uncached_players:
            batch_size = 5  # Even smaller batches
            batches = [uncached_players[i:i + batch_size] for i in range(0, len(uncached_players), batch_size)]
            
            print(f"   🔄 Processing {len(batches)} batches of {batch_size} players each...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:  # Only 2 workers!
                for i, batch in enumerate(batches):
                    print(f"      Batch {i+1}/{len(batches)}...")
                    batch_result = process_player_batch(batch)
                    players_with_teams.extend(batch_result)
                    
                    # Add delay between batches
                    if i < len(batches) - 1:  # Don't delay after last batch
                        time.sleep(3)  # 3 second delay between batches
        
        # Add cached players to results
        players_with_teams.extend(cached_players)
        
        # Save cache at the end
        self.save_player_cache()
        
        print(f"   ✅ Found {len(players_with_teams)} players with team information")
        return players_with_teams
    
    def load_player_cache(self):
        """Load cached player info to avoid API calls"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.player_info_cache = pickle.load(f)
                print(f"✅ Loaded {len(self.player_info_cache)} cached players")

                # Clean old cache entries (older than 30 days)
                original_size = len(self.player_info_cache)
                current_time = datetime.now()
                self.player_info_cache = {k: v for k, v in self.player_info_cache.items() 
                                        if (current_time - v[1]).days < 30}
                if len(self.player_info_cache) < original_size:
                    print(f"🧹 Cleaned {original_size - len(self.player_info_cache)} expired cache entries")
        except Exception as e:
            print(f"⚠️ Could not load cache: {e}")
            self.player_info_cache = {}

    def save_player_cache(self):
        """Save player cache to file"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.player_info_cache, f)
            print(f"💾 Saved {len(self.player_info_cache)} players to cache")
        except Exception as e:
            print(f"⚠️ Could not save cache: {e}")

    def get_cached_player_info(self, player_id, player_name):
        """Get player info from cache or API with retry logic"""
        cache_key = f"{player_id}_{player_name}"

        # Check if we have recent cache (within 3 days for faster updates)
        if cache_key in self.player_info_cache:
            cached_data, cache_time = self.player_info_cache[cache_key]
            if (datetime.now() - cache_time).days < 3:  # Reduced from 7 to 3 days
                return cached_data

        # If not in cache or expired, get from API with retry
        def fetch_player_info():
            try:
                player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=60)  # Increased timeout
                info_df = player_info.get_data_frames()[0]
                return info_df
            except Exception as e:
                print(f"      API Error for {player_name}: {e}")
                raise e

        try:
            info_df = self.make_api_call_with_retry(fetch_player_info, max_retries=2, initial_delay=2)

            if info_df is not None and not info_df.empty:
                # Cache the result
                self.player_info_cache[cache_key] = (info_df, datetime.now())
                return info_df
            else:
                print(f"      ❌ Failed to get player info for {player_name} after retries")
                return None

        except Exception as e:
            print(f"      ❌ Final failure for {player_name}: {e}")
            return None
    
    
    def check_player_availability(self, player_name, team):
        """Check if player is expected to play - CRITICAL FIX"""
        # Known injured players (update this daily based on injury reports)
        injured_players = get_injuries()
        rest_players = {player: status for player, status in injured_players.items() if status.lower() in ("day-to-day", "gtd")}
        out_players = {player: status for player, status in injured_players.items() if status.lower() == "out"}
        print("rest_players repr:", [repr(p) for p in rest_players])
        print("player_name repr:", repr(player_name))
        # Players who are resting or load management
        #rest_players = {
            #'LeBron James': 'GTD',
            #'Kevin Durant': 'GTD'
        #}
         # Check exact match first
        if player_name in rest_players:
            print(f"⚠️ {player_name} is GTD (rest)")
            return 'GTD'
        elif player_name in out_players:
            print(f"🚫 {player_name} is OUT (injured)")
            return 'OUT'
        # if player_name in injured_players:
        #     print(f"      🚫 {player_name} is OUT (injured)")
        #     return 'OUT'
        # elif player_name in rest_players:
        #     print(f"      ⚠️ {player_name} is GTD (rest)")
        #     return 'GTD'

        # Check partial matches (in case of name variations)
        for rest_player in rest_players:
            if rest_player.lower() in player_name.lower():
                print(f"⚠️ {player_name} is GTD (partial match: {rest_player})")
                return 'GTD'

        for out_player in out_players:
            if out_player.lower() in player_name.lower():
                print(f"🚫 {player_name} is OUT (partial match: {out_player})")
                return 'OUT'
        # for injured_player, status in injured_players.items():
        #     if injured_player.lower() in player_name.lower():
        #         print(f"      🚫 {player_name} is OUT (partial match: {injured_player})")
        #         return 'OUT'

        # for rest_player, status in rest_players.items():
        #     if rest_player.lower() in player_name.lower():
        #         print(f"      ⚠️ {player_name} is GTD (partial match: {rest_player})")
        #         return 'GTD'

        return 'PROBABLE'

    
    def filter_available_players(self, players_data):
        """Filter out injured players from the dataset"""
        available_players = []
        injured_count = 0
        gtd_count = 0

        for player in players_data:
            status = self.check_player_availability(player['name'], player['team'])
            if status == 'OUT':
                injured_count += 1
                continue
            elif status == 'GTD':
                gtd_count += 1
                # Include GTD players but mark them
                player['injury_status'] = 'GTD'
            else:
                player['injury_status'] = 'PROBABLE'

            available_players.append(player)

        if injured_count > 0 or gtd_count > 0:
            print(f"   ⚠️ Injury Report: {injured_count} OUT, {gtd_count} GTD")

        return available_players
        
    def setup_team_map(self):
        """Create mapping from team IDs to abbreviations"""
        nba_teams = teams.get_teams()
        for team in nba_teams:
            self.team_id_map[team['id']] = team['abbreviation']

    def get_game_total(self, team):
        """Get projected game total for a team's game - FIXED VERSION"""
        try:
            if team in self.matchup_data:
                matchup = self.matchup_data[team]
                opponent = matchup.get('opponent')

                # Use self directly, not self.data
                if team in self.team_stats and opponent in self.team_stats:
                    team_off = self.team_stats[team].get('off_rating', 110)
                    opp_def = self.team_stats[opponent].get('def_rating', 110)

                    # Simple projection
                    pace_factor = matchup.get('pace', 100) / 100
                    projected_total = ((team_off + (110 - opp_def + 110)) / 2) * pace_factor
                    return projected_total
        except Exception as e:
            # Silently fail - game total isn't critical
            pass
        
        return 220  # Default fallback
    
    def load_injury_data(self):
        """Load injury data to identify potential opportunity players"""
        print("🏥 Loading injury and lineup data...")
        # This would integrate with injury APIs in a real implementation
        # For now, we'll create a placeholder structure
        self.injury_report = {}
        # In practice, you'd populate this from an injury API or news source

    def project_ownership_by_salary(self, player):
        """Project ownership based on salary tiers"""
        salary = player['salary']

        if salary >= 10000:
            return 15  # Stars get lower ownership in cash
        elif salary >= 9000:
            return 20
        elif salary >= 8000:
            return 25
        elif salary >= 7000:
            return 30
        elif salary >= 6000:
            return 35
        elif salary >= 5000:
            return 40  # Sweet spot for value
        elif salary >= 4000:
            return 25  # Punt plays
        else:
            return 15  # Minimum salary

    def project_ownership_by_value(self, player):
        """Project ownership based on value score"""
        value_score = (player['projection'] / player['salary']) * 1000

        if value_score >= 6.0:
            return 60  # Elite value = high ownership
        elif value_score >= 5.5:
            return 50
        elif value_score >= 5.0:
            return 40
        elif value_score >= 4.5:
            return 30
        elif value_score >= 4.0:
            return 20
        else:
            return 10

    def project_ownership_by_matchup(self, player):
        """Project ownership based on matchup quality"""
        matchup_diff = player.get('matchup_difficulty', 0)

        # Easy matchup = higher ownership
        if matchup_diff > 1.0:
            return 40
        elif matchup_diff > 0.5:
            return 30
        elif matchup_diff > 0:
            return 25
        elif matchup_diff > -0.5:
            return 20
        else:
            return 15  # Tough matchup

    def project_ownership_by_game_environment(self, player):
        """Project ownership based on game pace and total"""
        pace_adj = player.get('pace_adjustment', 1.0)
        game_total = self.get_game_total(player['team'])

        ownership = 20  # Base

        # High pace boost
        if pace_adj > 1.05:
            ownership += 10
        elif pace_adj > 1.02:
            ownership += 5

        # High game total boost
        if game_total > 230:
            ownership += 15
        elif game_total > 225:
            ownership += 10
        elif game_total > 220:
            ownership += 5

        return min(60, ownership)

    def project_ownership_by_role(self, player):
        """Project ownership based on role security and usage"""
        ownership = 25  # Base

        # Minutes security boost
        projected_minutes = player.get('projected_minutes', 0)
        if projected_minutes >= 35:
            ownership += 15
        elif projected_minutes >= 30:
            ownership += 10
        elif projected_minutes >= 25:
            ownership += 5

        # Usage rate boost
        usage_rate = player.get('usage_rate', 0)
        if usage_rate >= 0.25:
            ownership += 10
        elif usage_rate >= 0.20:
            ownership += 5

        # Role stability boost
        role_stability = player.get('role_stability', 50)
        if role_stability >= 80:
            ownership += 10
        elif role_stability >= 60:
            ownership += 5

        return min(60, ownership)

    def project_composite_ownership(self, player):
        """Combine all factors for final ownership projection"""
        salary_own = self.project_ownership_by_salary(player)
        value_own = self.project_ownership_by_value(player)
        matchup_own = self.project_ownership_by_matchup(player)
        game_env_own = self.project_ownership_by_game_environment(player)
        role_own = self.project_ownership_by_role(player)

        # Weight the factors - value and role security are most important
        composite = (
            salary_own * 0.20 +
            value_own * 0.30 +    # Value is very important
            matchup_own * 0.15 +
            game_env_own * 0.15 +
            role_own * 0.20       # Role security matters a lot
        )

        # Dynamic popularity adjustment based on actual player attributes
        popularity_boost = 0

        # High-projection players get a boost
        if player.get('projection', 0) >= 50:
            popularity_boost += 10
        elif player.get('projection', 0) >= 40:
            popularity_boost += 5

        # Recent strong performers get a boost
        if player.get('recent_trend', 1) > 1.2:
            popularity_boost += 8
        elif player.get('recent_trend', 1) > 1.1:
            popularity_boost += 4

        composite += popularity_boost

        return min(80, max(5, composite))

    def get_ownership_reasons(self, player):
        """Explain why a player will be highly owned"""
        reasons = []

        value_score = (player['projection'] / player['salary']) * 1000
        if value_score >= 5.0:
            reasons.append(f"Elite value ({value_score:.1f})")
        elif value_score >= 4.0:
            reasons.append(f"Good value ({value_score:.1f})")

        if player['salary'] <= 5000 and player['projection'] >= 25:
            reasons.append("Strong value play")

        if player.get('pace_adjustment', 1) > 1.05:
            reasons.append("High pace game")

        if player.get('matchup_difficulty', 0) > 0.5:
            reasons.append("Great matchup")

        # Add minutes security
        projected_minutes = player.get('projected_minutes', 0)
        if projected_minutes >= 30:
            reasons.append(f"Secure minutes ({projected_minutes:.0f})")

        # Add role stability
        role_stability = player.get('role_stability', 0)
        if role_stability >= 80:
            reasons.append("Very stable role")
        elif role_stability >= 60:
            reasons.append("Stable role")

        # Add recent trend
        recent_trend = player.get('recent_trend', 1)
        if recent_trend > 1.1:
            reasons.append(f"Hot streak ({recent_trend:.2f}x)")

        return reasons

    def identify_chalk_plays(self, min_ownership=40):
        """Identify players likely to be highly owned"""
        chalk_plays = []

        for player in self.players_data:
            projected_own = self.project_composite_ownership(player)

            if projected_own >= min_ownership:
                chalk_plays.append({
                    'player': player,
                    'projected_ownership': projected_own,
                    'reasons': self.get_ownership_reasons(player)
                })

        # Sort by projected ownership
        chalk_plays.sort(key=lambda x: x['projected_ownership'], reverse=True)
        return chalk_plays

    def get_player_recent_games(self, player_name):
        """Get recent games for a player - helper method for other functions"""
        try:
            # Try to find player ID first
            player_id = None
            for player in self.players.get_players():
                if player['full_name'].lower() == player_name.lower():
                    player_id = player['id']
                    break
            
            if not player_id:
                return None
                
            # Use cached game logs if available
            if player_id in self.player_game_logs_cache:
                return self.player_game_logs_cache[player_id]
                
            # Get recent game logs (last 30 days)
            end_date = self.target_date if self.target_date else datetime.now()
            start_date = end_date - timedelta(days=30)
            
            date_from = start_date.strftime('%Y-%m-%d')
            date_to = end_date.strftime('%Y-%m-%d')
            
            game_logs = playergamelogs.PlayerGameLogs(
                player_id_nullable=player_id,
                season_nullable=self.get_current_season(),
                date_from_nullable=date_from,
                date_to_nullable=date_to
            )
            logs_df = game_logs.get_data_frames()[0]
            
            # Cache the results
            self.player_game_logs_cache[player_id] = logs_df
            return logs_df
            
        except Exception as e:
            return None

    def get_recent_minutes(self, player_name):
        """Get recent actual minutes for a player from game logs"""
        try:
            player_games = self.get_player_recent_games(player_name)
            if player_games is not None and len(player_games) > 0:
                recent_minutes = []
                for _, game in player_games.head(10).iterrows():  # Last 10 games
                    if 'MIN' in game and game['MIN'] > 0:
                        # Convert minutes string "35:12" to float 35.2
                        minutes_str = str(game['MIN'])
                        if ':' in minutes_str:
                            parts = minutes_str.split(':')
                            minutes_float = float(parts[0]) + float(parts[1]) / 60.0
                        else:
                            minutes_float = float(minutes_str)
                        recent_minutes.append(minutes_float)
                return recent_minutes if recent_minutes else None
        except Exception as e:
            print(f"      ⚠️ Error getting minutes for {player_name}: {e}")
        return None

    def get_recent_fantasy_production(self, player_name):
        """Get recent fantasy point production for a player"""
        try:
            player_games = self.get_player_recent_games(player_name)
            if player_games is not None and len(player_games) > 0:
                recent_fp = []
                for _, game in player_games.head(10).iterrows():  # Last 10 games
                    fp = self.calculate_dk_points({
                        'points': game.get('PTS', 0),
                        'rebounds': game.get('REB', 0),
                        'assists': game.get('AST', 0),
                        'steals': game.get('STL', 0),
                        'blocks': game.get('BLK', 0),
                        'turnovers': game.get('TOV', 0),
                        'three_pointers_made': game.get('FG3M', 0)
                    })
                    recent_fp.append(fp)
                return recent_fp if recent_fp else None
        except Exception as e:
            print(f"      ⚠️ Error getting fantasy production for {player_name}: {e}")
        return None
        
    def calculate_blowout_risk(self, team, opponent):
        """Calculate blowout risk based on team strength and spreads"""
        try:
            # Get team ratings
            team_rating = self.team_stats.get(team, {}).get('off_rating', 110.0)
            opp_rating = self.team_stats.get(opponent, {}).get('off_rating', 110.0)
            
            # Calculate expected point differential
            expected_diff = team_rating - opp_rating
            
            # Convert to blowout probability (0-1 scale)
            if abs(expected_diff) > 10:
                blowout_risk = 0.7
            elif abs(expected_diff) > 5:
                blowout_risk = 0.4
            else:
                blowout_risk = 0.2
                
            return blowout_risk
        except:
            return 0.3  # Default moderate risk
    
    def calculate_role_stability(self, player_games, player_name, position):
        """Calculate role stability score (0-100) based on minutes consistency"""
        try:
            if len(player_games) < 5:
                return 50  # Default for limited data
            
            minutes = player_games['MIN'].tolist()
            
            # Calculate coefficient of variation for minutes
            avg_minutes = np.mean(minutes)
            if avg_minutes == 0:
                return 30
                
            std_minutes = np.std(minutes)
            cv_minutes = std_minutes / avg_minutes
            
            # Convert to stability score (lower CV = higher stability)
            stability_score = max(0, 100 - (cv_minutes * 100))
            
            # Adjust for position and role
            if avg_minutes < 20:
                stability_score *= 0.8  # Bench players less stable
            elif avg_minutes > 32:
                stability_score *= 1.1  # Starters more stable
                
            return min(100, stability_score)
            
        except Exception as e:
            return 50

    def calculate_recent_trend_factor(self, player_games, stat_type='PTS'):
        """Calculate if player is trending up or down in recent games"""
        try:
            if len(player_games) < 6:
                return 1.0  # Neutral with insufficient data
            
            # Split into two halves: recent vs older
            recent_games = player_games.head(3)
            older_games = player_games.iloc[3:6]
            
            if len(older_games) == 0 or len(recent_games) == 0:
                return 1.0
                
            recent_avg = recent_games[stat_type].mean()
            older_avg = older_games[stat_type].mean()
            
            if older_avg == 0:
                return 1.0
                
            trend_ratio = recent_avg / older_avg
            
            # Apply regression to mean for extreme values
            if trend_ratio > 1.5:
                return 1.25  # Cap upward trends
            elif trend_ratio < 0.7:
                return 0.8   # Cap downward trends
            else:
                return trend_ratio
                
        except Exception as e:
            return 1.0

    def get_opportunity_rating(self, player_games, team, usage_rate):
        """Calculate opportunity rating based on team context and injuries"""
        try:
            rating = 50  # Base rating
            
            # Recent minutes trend
            if len(player_games) >= 3:
                recent_minutes = player_games.head(3)['MIN'].mean()
                older_minutes = player_games.iloc[3:6]['MIN'].mean() if len(player_games) >= 6 else recent_minutes
                
                if recent_minutes > older_minutes * 1.15:
                    rating += 20  # Significant minutes increase
                elif recent_minutes > older_minutes * 1.05:
                    rating += 10  # Moderate minutes increase
            
            # Usage rate adjustment
            if usage_rate > 0.25:
                rating += 15  # High usage players have more stable opportunity
            elif usage_rate < 0.15:
                rating -= 10  # Low usage players more volatile
            
            # Role stability adjustment
            stability = self.calculate_role_stability(player_games, "", "")
            rating = (rating + stability) / 2  # Blend with stability
            
            return min(100, max(0, rating))
            
        except Exception as e:
            return 50

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
        print("📊 Getting REAL NBA data with ENHANCED PROJECTIONS v2...")
        print("=" * 50)

        if not self.load_dk_salaries():
            return False

        if not self.get_todays_games():
            return False

        # NEW: FILTER TO DK SLATE ONLY
        print("\n🎯 Filtering to DraftKings slate games...")
        if not self.filter_to_dk_slate_games():
            print("⚠️  Using all games (DK filter failed)")

        # Check if we have enough games after filtering
        if len(self.todays_games) < 2:
            print("❌ Not enough games in DK slate after filtering")
            return False

        # Get team stats and pace data
        if not self.get_team_stats_and_pace():
            print("⚠️  Continuing without team pace data")

        # Get team schedule for back-to-back checking
        self.get_team_game_schedule()

        # Load injury data
        self.load_injury_data()

        try:
            season = self.get_current_season()
            print(f"📅 Current season: {season}")

            self.players_data = self.get_enhanced_player_projections(season)

            # Filter out injured players
            if self.players_data:
                print(f"🩹 Filtering out injured players...")
                original_count = len(self.players_data)
                self.players_data = self.filter_available_players(self.players_data)
                filtered_count = len(self.players_data)
                print(f"✅ Available players: {filtered_count}/{original_count}")

            if self.players_data:
                print(f"✅ Successfully loaded {len(self.players_data)} players with ENHANCED projections v2")
                # Identify high upside players
                self.identify_high_upside_players()
                self.show_data_summary_enhanced()
                return True
            else:
                print("❌ No player data retrieved")
                return False

        except Exception as e:
            print(f"💥 Error getting NBA data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_real_nba_data_faster(self):
        """Faster version of data retrieval"""
        print("📊 Getting REAL NBA data with ENHANCED PROJECTIONS v2 (FAST)...")
        print("=" * 50)

        if not self.load_dk_salaries():
            return False

        if not self.get_todays_games():
            return False

        # EARLY EXIT: If very few games, reduce processing
        if len(self.todays_games) <= 2:
            print("⚠️  Small slate detected - using optimized processing")
            # Use simpler methods for small slates

        # Get team stats and pace data in background
        print("   Getting team stats (non-blocking)...")
        stats_thread = threading.Thread(target=self.get_team_stats_and_pace)
        stats_thread.start()

        # Get team schedule for back-to-back checking
        self.get_team_game_schedule()

        # Load injury data
        self.load_injury_data()

        try:
                season = self.get_current_season()
                print(f"📅 Current season: {season}")

                # Try the main method first
                print("   Attempting main player data collection...")
                self.players_data = self.get_enhanced_player_projections_faster(season)

                # If that fails, try fallback
                if not self.players_data or len(self.players_data) < 20:
                    print("   ⚠️ Main method failed, trying fallback...")
                    # You might need to implement a fallback projection method

                if self.players_data:
                    print(f"✅ Successfully loaded {len(self.players_data)} players")
                    self.identify_high_upside_players()
                    self.show_data_summary_enhanced()
                    return True
                else:
                    print("❌ No player data retrieved")
                    return False

        except Exception as e:
            print(f"💥 Error getting NBA data: {e}")
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
            for col in ['PLAYER_ID', 'PLAYER_NAME', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M']:
                if col in sample_row.index:
                    print(f"   {col}: {sample_row[col]}")

    def get_enhanced_player_projections_faster(self, season):
        """MUCH FASTER version of player projections - FIXED"""
        print("🔄 Getting enhanced player projections (OPTIMIZED)...")

        player_data = []

        # Get all players with teams (uses cache)
        print("   Step 1: Getting players with teams...")
        all_players_with_teams = self.get_all_players_with_correct_teams_parallel()

        if not all_players_with_teams:
            print("❌ No players with teams found")
            return []

        # Filter to today's players (DK slate only)
        todays_players = [p for p in all_players_with_teams if p.get('team') in self.todays_games]
        print(f"   🎯 Found {len(todays_players)} players on today's DK slate teams")

        if len(todays_players) == 0:
            print("❌ No players found for today's DK slate games")
            return []

        # Get game logs - FIXED: Don't filter by player_id in the API call (it's causing issues)
        print("   Step 2: Getting game logs (optimized batch)...")
        try:
            # Use date range to limit data but DON'T filter by player_id in the API call
            end_date = self.target_date if self.target_date else datetime.now()
            start_date = end_date - timedelta(days=30)  # 30 days of data

            date_from = start_date.strftime('%Y-%m-%d')
            date_to = end_date.strftime('%Y-%m-%d')

            print(f"   📅 Getting logs from {date_from} to {date_to}")

            # Single API call for recent games (without player_id filter)
            all_game_logs = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                date_from_nullable=date_from,
                date_to_nullable=date_to
                # REMOVED: player_id_nullable=today_player_ids - this was causing the issue
            )
            all_logs_df = all_game_logs.get_data_frames()[0]
            print(f"   📈 Loaded {len(all_logs_df)} total game logs")

            if all_logs_df.empty:
                print("❌ No game logs found")
                return []

        except Exception as e:
            print(f"❌ Error loading game logs: {e}")
            import traceback
            traceback.print_exc()
            return []

        # Get player IDs for today's players for filtering
        today_player_ids = [p['id'] for p in todays_players]

        # Filter the DataFrame locally (much more reliable)
        filtered_logs = all_logs_df[all_logs_df['PLAYER_ID'].isin(today_player_ids)]
        print(f"   🔍 Filtered to {len(filtered_logs)} logs for today's players")

        # Group logs by player for faster lookup
        player_logs_dict = {}
        for player_id in today_player_ids:
            player_logs = filtered_logs[filtered_logs['PLAYER_ID'] == player_id]
            if not player_logs.empty:
                # Sort by date and take most recent 10 games
                player_logs = player_logs.sort_values('GAME_DATE', ascending=False).head(10)
                player_logs_dict[player_id] = player_logs

        print(f"   📊 Found logs for {len(player_logs_dict)} relevant players")

        # Process players
        print("   Step 3: Calculating projections...")
        processed_count = 0
        skipped_no_logs = 0
        skipped_low_minutes = 0
        skipped_no_salary = 0

        for player in todays_players:
            try:
                player_id = player['id']
                player_name = player['full_name']
                team = player['team']

                # Skip if no logs available
                if player_id not in player_logs_dict:
                    skipped_no_logs += 1
                    continue

                player_logs = player_logs_dict[player_id]
                recent_games = player_logs.head(10)

                if len(recent_games) < 3:
                    skipped_no_logs += 1
                    continue

                # Quick health checks
                required_columns = ['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M']
                missing_columns = [col for col in required_columns if col not in recent_games.columns]
                if missing_columns:
                    continue

                # Skip players with very low minutes
                recent_minutes = recent_games['MIN'].mean()
                if recent_minutes < 8:
                    skipped_low_minutes += 1
                    continue

                # Find DK salary
                dk_salary_info = self.find_player_in_dk_salaries(player_name, team)
                if not dk_salary_info:
                    skipped_no_salary += 1
                    continue

                # Get position
                position = dk_salary_info.get('position')
                if not position:
                    position = self.get_player_position(player_id, player_name)

                # Calculate projection
                projection_data = self.calculate_enhanced_projection(
                    player, recent_games, position, team, dk_salary_info
                )

                if projection_data:
                    player_data.append(projection_data)
                    processed_count += 1

                    if processed_count % 10 == 0:
                        print(f"      Processed {processed_count}/{len(todays_players)} players...")

            except Exception as e:
                continue
        # NEW: Apply automatic filters
        player_data = self.apply_automatic_player_filters(player_data)

        print(f"   ✅ Processed {processed_count} players with enhanced projections")
        print(f"   📊 Skipped: {skipped_no_logs} no logs, {skipped_low_minutes} low minutes, {skipped_no_salary} no salary")
        return player_data

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
                required_columns = ['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M']
                missing_columns = [col for col in required_columns if col not in recent_games.columns]
                if missing_columns:
                    continue

                availability = self.check_player_availability(player_name, team)
                if availability == 'OUT':
                    print(f"      ⚠️ Skipping {player_name} - INJURED")
                    continue
                elif availability == 'GTD':
                    print(f"      ⚠️ {player_name} - Game Time Decision")
                
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
                'plus_minus': 'PLUS_MINUS',
                'three_pointers_made': 'FG3M'
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
                'plus_minus': recent_games['PLUS_MINUS'].mean() if 'PLUS_MINUS' in recent_games.columns else 0,
                'three_pointers_made': recent_games['FG3M'].mean() if 'FG3M' in recent_games.columns else 0
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
                    'turnovers': game['TOV'],
                    'three_pointers_made': game.get('FG3M', 0)
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
                    'turnovers': game['TOV'],
                    'three_pointers_made': game.get('FG3M', 0)
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
        
    def project_minutes_aggressive(self, recent_minutes, recent_games, team, usage_rate, matchup_info):
        """More aggressive but realistic minutes projection"""
        # Base is recent average
        base_minutes = recent_minutes

        # If trending up, be more aggressive
        if len(recent_games) >= 3:
            last_3_minutes = recent_games.head(3)['MIN'].mean()
            if last_3_minutes > recent_minutes * 1.1:  # Significant uptrend
                base_minutes = last_3_minutes

        # Aggressive adjustments for good situations
        matchup_boost = 1.0
        if matchup_info.get('matchup_difficulty', 0) > 0.5:
            matchup_boost = 1.05  # Good matchup = more minutes

        usage_boost = 1.0
        if usage_rate > 0.25:
            usage_boost = 1.08  # High usage players get more run

        projected = base_minutes * matchup_boost * usage_boost

        # Realistic caps based on role
        if usage_rate > 0.25:
            return min(38, projected)  # Stars can play 38
        elif usage_rate > 0.18:
            return min(34, projected)  # Starters ~34
        else:
            return min(28, projected)  # Role players ~28
        
    def apply_aggressive_reality_checks(self, projection, player_name, position, salary, usage_rate, projected_minutes):
        """Less conservative reality checks"""
        # Salary-based cap - MORE AGGRESSIVE
        max_by_salary = salary * 0.0075  # $1,000 = 7.5x (was 6x)
        projection = min(projection, max_by_salary)

        # Minutes-based cap - MORE AGGRESSIVE
        if projected_minutes >= 35:
            projection = min(projection, 75)  # Stars can hit 75
        elif projected_minutes >= 30:
            projection = min(projection, 55)
        elif projected_minutes >= 25:
            projection = min(projection, 40)
        elif projected_minutes >= 20:
            projection = min(projection, 30)
        else:
            projection = min(projection, 20)

        # Usage-based adjustments
        if usage_rate > 0.25:
            projection *= 1.1  # High usage boost
        elif usage_rate < 0.15:
            projection *= 0.9  # Low usage penalty

        return projection
    
    def get_matchup_adjustment_aggressive(self, matchup_info):
        """More aggressive matchup adjustments"""
        base_adj = 1.0

        matchup_diff = matchup_info.get('matchup_difficulty', 0)
        opp_def_rating = matchup_info.get('opp_def_rating', 110)

        # More aggressive adjustments
        if matchup_diff > 1.0:
            base_adj = 1.15  # Elite matchup = 15% boost
        elif matchup_diff > 0.5:
            base_adj = 1.08
        elif matchup_diff > 0:
            base_adj = 1.03
        elif matchup_diff < -1.0:
            base_adj = 0.85  # Terrible matchup = 15% reduction

        # Defensive rating adjustments
        if opp_def_rating < 105:  # Elite defense
            base_adj *= 0.92
        elif opp_def_rating > 115:  # Poor defense
            base_adj *= 1.08

        return base_adj
    
    def calculate_ceiling_projection_aggressive(self, recent_games, projected_minutes):
        """More aggressive ceiling projection for tournaments"""
        try:
            # Find the best fantasy performance in recent games
            best_game_score = 0
            best_game_minutes = 0

            for _, game in recent_games.iterrows():
                game_score = self.calculate_dk_points({
                    'points': game['PTS'], 'rebounds': game['REB'], 
                    'assists': game['AST'], 'steals': game['STL'], 
                    'blocks': game['BLK'], 'turnovers': game['TOV'], 
                    'three_pointers_made': game.get('FG3M', 0)
                })

                if game_score > best_game_score:
                    best_game_score = game_score
                    best_game_minutes = game['MIN']

            if best_game_minutes > 0 and projected_minutes > 0:
                # MORE AGGRESSIVE scaling for tournaments
                minutes_ratio = projected_minutes / best_game_minutes
                ceiling_projection = best_game_score * minutes_ratio * 1.25  # 25% upside (was 1.1)
                return min(ceiling_projection, best_game_score * 1.5)  # Cap at 50% above best (was 1.3)
            else:
                return best_game_score * 1.35  # 35% upside fallback (was 1.15)

        except Exception as e:
            return 0
    
    # def calculate_ceiling_projection_aggressive(self, recent_games, projected_minutes):
    #     """More aggressive ceiling projection"""
    #     try:
    #         # Find the best fantasy performance in recent games
    #         best_game_score = 0
    #         best_game_minutes = 0

    #         for _, game in recent_games.iterrows():
    #             game_score = self.calculate_dk_points({
    #                 'points': game['PTS'], 'rebounds': game['REB'], 
    #                 'assists': game['AST'], 'steals': game['STL'], 
    #                 'blocks': game['BLK'], 'turnovers': game['TOV'], 
    #                 'three_pointers_made': game.get('FG3M', 0)
    #             })

    #             if game_score > best_game_score:
    #                 best_game_score = game_score
    #                 best_game_minutes = game['MIN']

    #         if best_game_minutes > 0 and projected_minutes > 0:
    #             # Scale to projected minutes aggressively
    #             minutes_ratio = projected_minutes / best_game_minutes
    #             ceiling_projection = best_game_score * minutes_ratio * 1.1  # 10% upside
    #             #return min(ceiling_projection, best_game_score * 1.3)  # Cap at 30% above best
    #             return ceiling_projection
    #         else:
    #             return best_game_score * 1.15  # 15% upside fallback

    #     except Exception as e:
    #         return 0
        
    def project_minutes_conservative(self, recent_minutes, recent_games, team, usage_rate, matchup_info):
        """Conservative minutes projection for cash games"""
        # Base is recent average, no aggressive boosting
        base_minutes = recent_minutes

        # Small adjustments only
        matchup_boost = 1.0
        if matchup_info.get('matchup_difficulty', 0) > 0.5:
            matchup_boost = 1.02  # Tiny boost for good matchups

        usage_boost = 1.0
        if usage_rate > 0.25:
            usage_boost = 1.03  # Small boost for high usage

        projected = base_minutes * matchup_boost * usage_boost

        # Conservative caps
        if usage_rate > 0.25:
            return min(36, projected)  # Stars max at 36
        elif usage_rate > 0.18:
            return min(32, projected)  # Starters ~32
        else:
            return min(28, projected)  # Role players ~28

    def get_matchup_adjustment_conservative(self, matchup_info):
        """Conservative matchup adjustments"""
        base_adj = 1.0
        
        matchup_diff = matchup_info.get('matchup_difficulty', 0)
        
        # Conservative adjustments
        if matchup_diff > 1.0:
            base_adj = 1.08  # Good matchup = 8% boost
        elif matchup_diff > 0.5:
            base_adj = 1.04
        elif matchup_diff > 0:
            base_adj = 1.02
        elif matchup_diff < -1.0:
            base_adj = 0.92  # Bad matchup = 8% reduction
        
        return base_adj
    
    def apply_realistic_reality_checks(self, projection, player_name, position, salary, usage_rate, projected_minutes):
        """Realistic reality checks for cash games"""
        # Salary-based cap - REALISTIC
        max_by_salary = salary * 0.006  # $1,000 = 6x (more realistic)
        projection = min(projection, max_by_salary)
        
        # Position-based realistic caps
        position_caps = {
            'PG': 55, 'SG': 50, 'SF': 48, 'PF': 52, 'C': 60
        }
        max_by_position = position_caps.get(position, 50)
        projection = min(projection, max_by_position)
        
        # Minutes-based cap - REALISTIC
        if projected_minutes >= 35:
            projection = min(projection, 60)  # Stars can hit 60
        elif projected_minutes >= 30:
            projection = min(projection, 45)
        elif projected_minutes >= 25:
            projection = min(projection, 35)
        elif projected_minutes >= 20:
            projection = min(projection, 25)
        else:
            projection = min(projection, 20)
        
        # Usage-based adjustments (small)
        if usage_rate > 0.25:
            projection *= 1.05  # Small high usage boost
        elif usage_rate < 0.15:
            projection *= 0.95  # Small low usage penalty
        
        return max(10, projection)  # Ensure minimum
        
    def calculate_enhanced_projection(self, player, recent_games, position, team, dk_salary_info):
        """REALISTIC projection - not overly aggressive"""
        try:
            # Use ALL recent games, not just best ones for cash games
            avg_stats = self.calculate_robust_averages(recent_games)

            if avg_stats['minutes'] < 12:
                return None

            # Calculate advanced metrics
            usage_rate = self.calculate_usage_rate(recent_games, team)

            # Get matchup info
            matchup_info = self.matchup_data.get(team, {})

            # MORE CONSERVATIVE minutes projection
            projected_minutes = self.project_minutes_conservative(
                avg_stats['minutes'], recent_games, team, usage_rate, matchup_info
            )

            # Calculate per-minute stats using MEDIAN (more realistic than best games)
            per_min_stats = {}
            stat_columns = {
                'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST',
                'steals': 'STL', 'blocks': 'BLK', 'turnovers': 'TOV',
                'three_pointers_made': 'FG3M'
            }

            for stat, nba_col in stat_columns.items():
                if nba_col in recent_games.columns and avg_stats['minutes'] > 0:
                    # Use MEDIAN for cash games (more stable than mean of best games)
                    median_stat = recent_games[nba_col].median()
                    per_min_stats[stat] = (median_stat / avg_stats['minutes']) * projected_minutes
                else:
                    per_min_stats[stat] = 0

            # CONSERVATIVE adjustments
            pace_adjustment = self.get_pace_adjustment(team)
            usage_adjustment = 1.0 + (usage_rate - 0.20) * 0.3  # Much smaller usage boost
            matchup_adjustment = self.get_matchup_adjustment_conservative(matchup_info)

            # Apply adjustments conservatively
            final_stats = {}
            for stat in per_min_stats:
                if stat in ['points', 'assists', 'three_pointers_made']:
                    final_stats[stat] = per_min_stats[stat] * pace_adjustment * usage_adjustment * matchup_adjustment
                else:
                    final_stats[stat] = per_min_stats[stat] * pace_adjustment * matchup_adjustment

            # Calculate DK points
            projection = self.calculate_dk_points(final_stats)
            salary = dk_salary_info['salary']

            # REALISTIC reality checks
            projection = self.apply_realistic_reality_checks(
                projection, player['full_name'], position, salary, usage_rate, projected_minutes
            )

            if projection < 15:
                return None

            # Calculate derived metrics
            value_rating = (projection / salary) * 1000

            # REALISTIC ceiling projection (25-30% above base)
            ceiling_projection = projection * 1.25

            # Calculate other metrics
            points_per_minute = self.calculate_points_per_minute(recent_games)
            fantasy_points_per_minute = self.calculate_fantasy_points_per_minute(recent_games)
            plus_minus_rating = self.calculate_plus_minus_rating(recent_games, avg_stats['minutes'])
            role_stability = self.calculate_role_stability(recent_games, player['full_name'], position)
            consistency_rating = self.calculate_consistency_rating(recent_games)
            volatility_score = self.calculate_volatility_score(recent_games)

            # Project ownership
            player_data_for_ownership = {
                'name': player['full_name'],
                'salary': salary,
                'projection': projection,
                'team': team,
                'matchup_difficulty': matchup_info.get('matchup_difficulty', 0),
                'pace_adjustment': pace_adjustment,
                'projected_minutes': projected_minutes,
                'usage_rate': usage_rate,
                'role_stability': role_stability,
            }
            projected_ownership = self.project_composite_ownership(player_data_for_ownership)

            return {
                'name': player['full_name'],
                'position': position,
                'team': team,
                'salary': salary,
                'projection': round(projection, 1),
                'ceiling_projection': round(ceiling_projection, 1),
                'floor_projection': round(projection * 0.7, 1),
                'value_rating': round(value_rating, 2),
                'projected_ownership': round(projected_ownership, 1),
                'minutes': round(avg_stats['minutes'], 1),
                'projected_minutes': round(projected_minutes, 1),
                'pace_adjustment': round(pace_adjustment, 3),
                'usage_rate': round(usage_rate, 3),
                'points_per_minute': round(points_per_minute, 2),
                'fantasy_points_per_minute': round(fantasy_points_per_minute, 2),
                'plus_minus_rating': round(plus_minus_rating, 1),
                'per_36_points': round(per_min_stats['points'] * (36 / projected_minutes), 1) if projected_minutes > 0 else 0,
                'consistency_rating': round(consistency_rating, 1),
                'volatility_score': round(volatility_score, 2),
                'role_stability': round(role_stability, 1),
                'location': matchup_info.get('location', 'home'),
                'opponent': matchup_info.get('opponent', 'UNK'),
                'matchup_difficulty': round(matchup_info.get('matchup_difficulty', 0), 2),
                'back_to_back': self.check_back_to_back(team),
                'playing_today': True
            }

        except Exception as e:
            print(f"      Error calculating projection for {player['full_name']}: {e}")
            return None

    # def calculate_enhanced_projection(self, player, recent_games, position, team, dk_salary_info):
    #     """Enhanced projection with new factors"""
    #     try:
    #         # Calculate base averages - use BEST recent games
    #         if len(recent_games) >= 5:
    #             recent_games = recent_games.nlargest(int(len(recent_games) * 0.75), 'PTS')
    #         avg_stats = self.calculate_robust_averages(recent_games)

    #         if avg_stats['minutes'] < 12:
    #             return None

    #         # Calculate advanced metrics
    #         usage_rate = self.calculate_usage_rate(recent_games, team)
    #         points_per_minute = self.calculate_points_per_minute(recent_games)
    #         fantasy_points_per_minute = self.calculate_fantasy_points_per_minute(recent_games)
    #         plus_minus_rating = self.calculate_plus_minus_rating(recent_games, avg_stats['minutes'])

    #         # Get matchup info and project minutes
    #         matchup_info = self.matchup_data.get(team, {})
    #         location = matchup_info.get('location', 'home')
    #         opponent = matchup_info.get('opponent', 'UNK')
    #         matchup_difficulty = matchup_info.get('matchup_difficulty', 0)

    #         # Project minutes using aggressive method
    #         projected_minutes = self.project_minutes_aggressive(
    #             avg_stats['minutes'], recent_games, team, usage_rate, matchup_info
    #         )

    #         # Calculate additional metrics
    #         role_stability = self.calculate_role_stability(recent_games, player['full_name'], position)
    #         recent_trend = self.calculate_recent_trend_factor(recent_games, 'PTS')
    #         opportunity_rating = self.get_opportunity_rating(recent_games, team, usage_rate)
    #         volatility_score = self.calculate_volatility_score(recent_games)
    #         consistency_rating = self.calculate_consistency_rating(recent_games)
    #         blowout_risk = self.calculate_blowout_risk(team, opponent)
    #         back_to_back = self.check_back_to_back(team)

    #         # Calculate per-minute stats aggressively
    #         per_min_stats = {}
    #         stat_columns = {
    #             'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST',
    #             'steals': 'STL', 'blocks': 'BLK', 'turnovers': 'TOV',
    #             'three_pointers_made': 'FG3M'
    #         }

    #         for stat, nba_col in stat_columns.items():
    #             if nba_col in recent_games.columns and avg_stats['minutes'] > 0:
    #                 best_games = recent_games.nlargest(3, nba_col)
    #                 best_avg = best_games[nba_col].mean()
    #                 per_min_stats[stat] = (best_avg / avg_stats['minutes']) * projected_minutes
    #             else:
    #                 per_min_stats[stat] = 0

    #         # Apply adjustments
    #         pace_adjustment = self.get_pace_adjustment(team)
    #         usage_adjustment = 1.0 + (usage_rate - 0.20) * 0.8
    #         matchup_adjustment = self.get_matchup_adjustment_aggressive(matchup_info)

    #         final_stats = {}
    #         for stat in per_min_stats:
    #             if stat in ['points', 'assists', 'three_pointers_made']:
    #                 final_stats[stat] = per_min_stats[stat] * pace_adjustment * usage_adjustment * matchup_adjustment
    #             else:
    #                 final_stats[stat] = per_min_stats[stat] * pace_adjustment * matchup_adjustment

    #         # Calculate DK points
    #         projection = self.calculate_dk_points(final_stats)
    #         salary = dk_salary_info['salary']

    #         # Apply reality checks
    #         projection = self.apply_aggressive_reality_checks(
    #             projection, player['full_name'], position, salary, usage_rate, projected_minutes
    #         )

    #         if projection < 15:
    #             return None

    #         # Calculate derived metrics
    #         value_rating = (projection / salary) * 1000
    #         ceiling_projection = self.calculate_ceiling_projection_aggressive(recent_games, projected_minutes)
    #         upside_score = 50  # placeholder
    #         projected_ownership = 25  # placeholder

    #         return {
    #             'name': player['full_name'],
    #             'position': position,
    #             'team': team,
    #             'salary': salary,
    #             'projection': round(projection, 1),
    #             'ceiling_projection': round(ceiling_projection, 1),
    #             'floor_projection': round(projection * 0.7, 1),  # simplified
    #             'upside_score': round(upside_score, 1),
    #             'projected_ownership': round(projected_ownership, 1),
    #             'minutes': round(avg_stats['minutes'], 1),
    #             'projected_minutes': round(projected_minutes, 1),
    #             'pace_adjustment': round(pace_adjustment, 3),
    #             'usage_rate': round(usage_rate, 3),
    #             'points_per_minute': round(points_per_minute, 2),
    #             'fantasy_points_per_minute': round(fantasy_points_per_minute, 2),
    #             'plus_minus_rating': round(plus_minus_rating, 1),
    #             'per_36_points': round(per_min_stats['points'] * (36 / projected_minutes), 1) if projected_minutes > 0 else 0,
    #             'value_rating': round(value_rating, 2),
    #             'consistency_rating': round(consistency_rating, 1),
    #             'volatility_score': round(volatility_score, 2),
    #             'role_stability': round(role_stability, 1),
    #             'opportunity_rating': round(opportunity_rating, 1),
    #             'recent_trend': round(recent_trend, 3),
    #             'location': location,
    #             'opponent': opponent,
    #             'matchup_difficulty': round(matchup_difficulty, 2),
    #             'back_to_back': back_to_back,
    #             'playing_today': True
    #         }

    #     except Exception as e:
    #         print(f"      Error calculating projection for {player['full_name']}: {e}")
    #         return None

    def project_minutes_enhanced(self, recent_minutes, recent_games, team, usage_rate=None, plus_minus_rating=None, location='home', back_to_back=False, matchup_difficulty=0, volatility_score=1.0, consistency_rating=50, role_stability=50, blowout_risk=0.3):
        """Enhanced minutes projection with role stability and blowout risk"""
        # Base projection is recent average
        base_minutes = recent_minutes
        
        # Adjust for trends (last 3 games vs full average)
        if len(recent_games) >= 3:
            last_3_minutes = recent_games.head(3)['MIN'].mean()
            # If player's minutes are trending up, adjust slightly
            if last_3_minutes > recent_minutes:
                base_minutes = (recent_minutes + last_3_minutes) / 2
        
        # NEW: Role stability adjustment
        stability_factor = 0.9 + (role_stability / 100 * 0.2)  # 0.9-1.1 range
        base_minutes *= stability_factor
        
        # High usage players might see slightly reduced minutes in blowouts
        # but are less likely to see random DNP-CDs
        blowout_risk_factor = 1.0 - (blowout_risk * 0.3)  # Up to 30% reduction in blowouts
        
        if usage_rate and usage_rate > 0.25:  # High usage players
            blowout_risk_factor = 1.0 - (blowout_risk * 0.15)  # Less reduction for stars
        elif usage_rate and usage_rate < 0.15:  # Low usage players
            blowout_risk_factor = 1.0 - (blowout_risk * 0.5)  # More reduction for role players
        
        # Players with positive plus/minus are more likely to maintain minutes
        if plus_minus_rating and plus_minus_rating > 2:
            blowout_risk_factor += 0.05  # Slight boost for positive impact players
        elif plus_minus_rating and plus_minus_rating < -2:
            blowout_risk_factor -= 0.05  # Slight reduction for negative impact players
        
        # Back-to-back games might reduce minutes, especially for high usage players
        if back_to_back:
            if usage_rate and usage_rate > 0.25:
                blowout_risk_factor -= 0.05  # Extra reduction for stars on back-to-back
            else:
                blowout_risk_factor -= 0.02  # Small reduction for role players
        
        # Tough matchups might reduce minutes for role players
        if matchup_difficulty < -0.5:  # Hard matchup
            if usage_rate and usage_rate < 0.18:
                blowout_risk_factor -= 0.03  # Extra reduction for role players in tough matchups
        
        # High volatility players might see more minute variability
        if volatility_score > 1.2:
            blowout_risk_factor -= 0.03
        
        # Cap minutes at reasonable levels
        projected_minutes = min(38, base_minutes * blowout_risk_factor)
        projected_minutes = max(8, projected_minutes)  # Slightly lower minimum floor
        
        return projected_minutes
    
    def apply_minutes_reality_check_enhanced(self, projected_minutes, player_name, position, usage_rate, role_stability):
        """Enhanced minutes reality check - MORE AGGRESSIVE"""
        # Get recent actual minutes data to inform our caps
        recent_actual_minutes = self.get_recent_minutes(player_name)

        # If we have recent data, use it to cap projections
        if recent_actual_minutes and len(recent_actual_minutes) >= 3:
            avg_recent_minutes = sum(recent_actual_minutes) / len(recent_actual_minutes)
            max_recent_minutes = max(recent_actual_minutes)

            # CRITICAL: If player hasn't played meaningful minutes recently, cap aggressively
            if avg_recent_minutes < 5:
                # Player is emergency/deep bench - cap VERY low
                max_allowed = min(8, max_recent_minutes * 1.5)  # No more than 50% above recent max
                return min(projected_minutes, max_allowed)
            elif avg_recent_minutes < 12:
                # Player is occasional bench - cap conservatively
                max_allowed = min(20, max_recent_minutes * 1.3)  # No more than 30% above recent max
                return min(projected_minutes, max_allowed)
            elif avg_recent_minutes < 20:
                # Player is rotation but not starter
                max_allowed = min(30, max_recent_minutes * 1.2)  # No more than 20% above recent max
                return min(projected_minutes, max_allowed)

        # Base caps on minutes based on role and usage (your existing logic)
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
            max_minutes = min(max_minutes, 34)

        # Role stability adjustment
        if role_stability < 60:  # Low stability players
            max_minutes *= 0.9   # 10% reduction
        elif role_stability > 80:  # High stability players
            max_minutes *= 1.05  # 5% increase

        return min(projected_minutes, max_minutes)

    # def apply_minutes_reality_check_enhanced(self, projected_minutes, player_name, position, usage_rate, role_stability):
    #     """Enhanced minutes reality check with role stability"""
    #     # Base caps on minutes based on role and usage
    #     max_minutes = 36
        
    #     if usage_rate < 0.15:  # Low usage role player
    #         max_minutes = 28
    #     elif usage_rate < 0.20:  # Medium usage
    #         max_minutes = 32
    #     elif usage_rate < 0.25:  # High usage
    #         max_minutes = 36
    #     else:  # Star player
    #         max_minutes = 38
        
    #     # Additional position-based caps
    #     if position == 'C':
    #         max_minutes = min(max_minutes, 34)  # Centers often play fewer minutes
        
    #     # NEW: Role stability adjustment to max minutes
    #     if role_stability < 60:  # Low stability players
    #         max_minutes *= 0.9   # 10% reduction
    #     elif role_stability > 80:  # High stability players
    #         max_minutes *= 1.05  # 5% increase
        
    #     return min(projected_minutes, max_minutes)

    def calculate_ceiling_projection_enhanced(self, recent_games, projected_minutes, role_stability, volatility_score):
        """Enhanced ceiling projection with uncertainty ranges"""
        try:
            # Find the best fantasy performance in recent games
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
                # Scale to projected minutes
                minutes_ratio = projected_minutes / best_game_minutes
                ceiling_projection = best_game_score * minutes_ratio
                
                # NEW: Adjust ceiling based on role stability and volatility
                stability_factor = 0.9 + (role_stability / 100 * 0.2)  # 0.9-1.1
                volatility_factor = 1.0 + (volatility_score - 1.0) * 0.2  # More volatile = higher ceiling potential
                
                ceiling_projection = ceiling_projection * stability_factor * volatility_factor
                
                # Apply a reasonable cap
                ceiling_projection = min(ceiling_projection * 1.15, best_game_score * 1.3)
                
                return ceiling_projection
            else:
                return best_game_score * 1.1
                
        except Exception as e:
            return 0

    def calculate_floor_projection(self, recent_games, projected_minutes, role_stability):
        """Calculate floor projection based on worst recent performance"""
        try:
            # Find the worst fantasy performance in recent games (where minutes > 10)
            worst_game_score = float('inf')
            worst_game_minutes = 0
            
            for _, game in recent_games.iterrows():
                if game['MIN'] < 10:  # Skip games with very low minutes
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
            
            if worst_game_score == float('inf'):  # No valid games found
                return 0
                
            if worst_game_minutes > 0 and projected_minutes > 0:
                # Scale to projected minutes
                minutes_ratio = projected_minutes / worst_game_minutes
                floor_projection = worst_game_score * minutes_ratio
                
                # Adjust based on role stability
                stability_floor_factor = 0.8 + (role_stability / 100 * 0.4)  # 0.8-1.2
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
            
            # Ceiling vs Projection gap (30% of score)
            if projection > 0 and ceiling_projection > projection:
                ceiling_gap = (ceiling_projection - projection) / projection
                ceiling_component = min(30, ceiling_gap * 100)
                score += ceiling_component
            
            # Salary-based value (20% of score) - cheaper players have more upside
            if salary > 0:
                salary_component = max(0, (10000 - salary) / 10000 * 20)
                score += salary_component
            
            # Usage rate (15% of score) - high usage players have more upside
            usage_component = min(15, usage_rate * 100)
            score += usage_component
            
            # Volatility (10% of score) - volatile players have more upside potential
            volatility_component = min(10, (volatility_score - 0.5) * 15)
            score += max(0, volatility_component)
            
            # Matchup (10% of score) - good matchups increase upside
            matchup_component = min(10, (matchup_difficulty + 1) * 5)
            score += max(0, matchup_component)
            
            # NEW: Opportunity rating (15% of score)
            opportunity_component = opportunity_rating * 0.15
            score += opportunity_component
            
            return min(100, max(0, score))
            
        except:
            return 0
        
    def apply_projection_reality_checks_enhanced(self, projection, player_name, position, salary, usage_rate, consistency_rating, projected_minutes, role_stability, volatility_score):
        """Enhanced reality checks - MORE AGGRESSIVE"""

        # Get recent fantasy point production
        recent_fantasy_points = self.get_recent_fantasy_production(player_name)

        # If we have recent production data, use it to cap projections
        if recent_fantasy_points and len(recent_fantasy_points) >= 3:
            avg_recent_fp = sum(recent_fantasy_points) / len(recent_fantasy_points)
            max_recent_fp = max(recent_fantasy_points)

            # CRITICAL: Cap projections based on actual recent production
            if avg_recent_fp < 10:
                # Player produces very little - cap aggressively
                max_allowed_fp = min(15, max_recent_fp * 1.5)
                projection = min(projection, max_allowed_fp)
            elif avg_recent_fp < 20:
                # Moderate producer - cap conservatively
                max_allowed_fp = min(30, max_recent_fp * 1.3)
                projection = min(projection, max_allowed_fp)
            elif avg_recent_fp < 30:
                # Good producer - reasonable cap
                max_allowed_fp = min(45, max_recent_fp * 1.2)
                projection = min(projection, max_allowed_fp)

        # Cap projections based on salary and role
        max_projection_by_salary = salary * 0.008  # $1,000 salary = 8x max projection

        if usage_rate < 0.15:  # Low usage role players
            max_projection_by_salary *= 0.8  # 20% reduction

        # Apply consistency adjustment
        consistency_adjustment = consistency_rating / 100

        # Role stability adjustment
        stability_adjustment = 0.9 + (role_stability / 100 * 0.2)  # 0.9-1.1

        # Apply minutes-based cap (adjust based on volatility)
        if usage_rate < 0.20:
            volatility_multiplier = 1.5 - (volatility_score - 1.0) * 0.2  # More volatile = lower multiplier
            max_by_minutes = projected_minutes * max(1.0, volatility_multiplier)
            projection = min(projection, max_by_minutes)

        projection = min(projection, max_projection_by_salary)
        projection *= consistency_adjustment
        projection *= stability_adjustment

        # FINAL AGGRESSIVE CHECK: If projected minutes < 10, cap projection aggressively
        if projected_minutes < 10:
            projection = min(projection, 15)  # Can't score much in <10 minutes

        return projection

    # def apply_projection_reality_checks_enhanced(self, projection, player_name, position, salary, usage_rate, consistency_rating, projected_minutes, role_stability, volatility_score):
    #     """Enhanced reality checks with role stability"""
    #     # Cap projections based on salary and role
    #     max_projection_by_salary = salary * 0.008  # $1,000 salary = 8x max projection
        
    #     if usage_rate < 0.15:  # Low usage role players
    #         max_projection_by_salary *= 0.8  # 20% reduction
        
    #     # Apply consistency adjustment
    #     consistency_adjustment = consistency_rating / 100
        
    #     # NEW: Role stability adjustment
    #     stability_adjustment = 0.9 + (role_stability / 100 * 0.2)  # 0.9-1.1
        
    #     # Apply minutes-based cap (adjust based on volatility)
    #     if usage_rate < 0.20:
    #         volatility_multiplier = 1.5 - (volatility_score - 1.0) * 0.2  # More volatile = lower multiplier
    #         max_by_minutes = projected_minutes * max(1.0, volatility_multiplier)
    #         projection = min(projection, max_by_minutes)
        
    #     projection = min(projection, max_projection_by_salary)
    #     projection *= consistency_adjustment
    #     projection *= stability_adjustment
        
    #     return projection

    def calculate_bargain_rating_enhanced(self, projection, salary, usage_rate, plus_minus_rating, fantasy_points_per_minute, consistency_rating, role_stability, opportunity_rating):
        """Enhanced bargain rating with stability and opportunity"""
        try:
            score = 0
            
            # Base value score (25% of total)
            if salary > 0:
                value_score = (projection / salary) * 1000
                value_component = min(25, (value_score - 2) * (25 / 6))
                score += max(0, value_component)
            
            # Usage rate component (15% of total)
            usage_component = min(15, usage_rate * 100)
            score += usage_component
            
            # Plus/minus component (10% of total)
            pm_component = min(10, (plus_minus_rating + 10) * (10 / 20))
            score += max(0, pm_component)
            
            # Efficiency component (10% of total)
            if fantasy_points_per_minute > 0:
                efficiency_component = min(10, (fantasy_points_per_minute - 0.8) * (10 / 0.7))
                score += max(0, efficiency_component)
            
            # Consistency component (15% of total)
            consistency_component = consistency_rating * 0.15
            score += consistency_component
            
            # NEW: Role stability component (15% of total)
            stability_component = role_stability * 0.15
            score += stability_component
            
            # NEW: Opportunity component (10% of total)
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
        
        if blowout_risk > 0.6:  # High blowout risk
            if usage_rate < 0.15 and role_stability < 60:  # Low usage, unstable role
                base_adjustment = 0.7  # 30% reduction
            elif usage_rate < 0.15:
                base_adjustment = 0.8  # 20% reduction
            elif usage_rate > 0.25:
                base_adjustment = 0.95  # Only 5% reduction for stars
        elif blowout_risk > 0.3:  # Medium blowout risk
            if usage_rate < 0.15:
                base_adjustment = 0.9  # 10% reduction
        
        return base_adjustment

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
        """Calculate DraftKings fantasy points with double-double and triple-double bonuses"""
        # Basic stats
        points = stats.get('points', 0)
        rebounds = stats.get('rebounds', 0)
        assists = stats.get('assists', 0)
        steals = stats.get('steals', 0)
        blocks = stats.get('blocks', 0)
        turnovers = stats.get('turnovers', 0)
        three_pointers_made = stats.get('three_pointers_made', 0)
        
        # Calculate basic fantasy points
        fantasy_points = (
            points * 1.0 +  # 1 point per point
            three_pointers_made * 0.5 +  # 0.5 bonus points per made 3-pointer
            rebounds * 1.25 +  # 1.25 points per rebound
            assists * 1.5 +  # 1.5 points per assist
            steals * 2.0 +  # 2 points per steal
            blocks * 2.0 -  # 2 points per block
            turnovers * 0.5  # -0.5 points per turnover
        )
        
        # Check for double-double and triple-double
        double_double_categories = 0
        if points >= 10:
            double_double_categories += 1
        if rebounds >= 10:
            double_double_categories += 1
        if assists >= 10:
            double_double_categories += 1
        
        # Apply bonuses (only one bonus per player - either double-double OR triple-double)
        if double_double_categories >= 3:  # Triple-double
            fantasy_points += 3.0
        elif double_double_categories >= 2:  # Double-double
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
            
            # Criteria for high upside:
            # 1. Upside score > 60
            # 2. Ceiling projection at least 1.5x projection
            # 3. Reasonable salary (< $8000 for stars, < $6000 for others)
            # 4. Good matchup (matchup_difficulty > -0.5)
            
            if (upside_score >= 60 and 
                ceiling_projection >= projection * 1.5 and
                salary < 8000 and
                player.get('matchup_difficulty', -1) > -0.5):
                
                high_upside_players.append(player)
        
        # Sort by upside score
        high_upside_players.sort(key=lambda x: x.get('upside_score', 0), reverse=True)
        self.high_upside_players = high_upside_players
        
        print(f"✅ Found {len(high_upside_players)} high-upside players")
        
        # Show top high-upside players
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
                
    def get_target_date_string(self):
        """Get the target date string for API calls"""
        if self.target_date:
            return self.target_date.strftime('%Y-%m-%d')
        else:
            return datetime.now().strftime('%Y-%m-%d')
    
    def get_todays_games(self):
        """Get NBA games for the target date (today or custom date)"""
        print("📅 Getting NBA schedule...")
        
        try:
            # Use target date or today
            game_date = self.get_target_date_string()
            print(f"   Target date: {game_date}")
            
            scoreboard_data = scoreboardv2.ScoreboardV2(game_date=game_date)
            games_df = scoreboard_data.get_data_frames()[0]
            
            if games_df.empty:
                print(f"❌ No games found for {game_date}")
                return False
            
            print(f"📊 Found {len(games_df)} games for {game_date}")
            
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
            
            print(f"✅ Games for {game_date}: {', '.join(self.todays_games)}")
            return True
            
        except Exception as e:
            print(f"❌ Error getting schedule for {game_date}: {e}")
            return False

    def get_team_game_schedule(self):
        """Get recent game schedule for all teams to check for back-to-backs"""
        print("   Getting team schedule data for back-to-back checking...")
        
        try:
            # Use target date as end date
            end_date = self.target_date if self.target_date else datetime.now()
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
        """Check if a team is on a back-to-back for the target date"""
        try:
            if team not in self.team_game_schedule:
                return False
            
            team_games = sorted(self.team_game_schedule[team])
            if len(team_games) < 2:
                return False
            
            # Check if last game was the day before target date
            last_game_date = team_games[-1]
            target_date = self.target_date if self.target_date else datetime.now().date()
            
            # If last game was yesterday relative to target date, it's a back-to-back
            return (last_game_date.date() == target_date - timedelta(days=1))
            
        except:
            return False

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
        
        # Filter to today's players (now filtered to DK slate)
        todays_players = []
        for player in all_players_with_teams:
            if player.get('team') in self.todays_games:
                todays_players.append(player)

        print(f"   🎯 Found {len(todays_players)} players on today's DK slate teams")

        if len(todays_players) == 0:
            print("❌ No players found for today's DK slate games")
            return []
            
        # Filter to target date's players
        target_date_players = []
        for player in all_players_with_teams:
            if player.get('team') in self.todays_games:
                target_date_players.append(player)
        
        print(f"   🎯 Found {len(target_date_players)} players on {self.get_target_date_string()}'s teams")
        
        if len(target_date_players) == 0:
            print("❌ No players found for target date's games")
            return []
        
        # Get game logs - extend lookback period for more data
        print("   Step 2: Getting game logs...")
        try:
            # Extend date range to get more historical data
            end_date = self.target_date if self.target_date else datetime.now()
            start_date = end_date - timedelta(days=30)  # 30 days lookback
            
            date_from = start_date.strftime('%Y-%m-%d')
            date_to = end_date.strftime('%Y-%m-%d')
            
            all_game_logs = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                date_from_nullable=date_from,
                date_to_nullable=date_to
            )
            all_logs_df = all_game_logs.get_data_frames()[0]
            print(f"   📈 Loaded {len(all_logs_df)} game logs from {date_from} to {date_to}")
            
            # Debug: Check columns
            self.debug_column_check(all_logs_df)
            
        except Exception as e:
            print(f"❌ Error loading game logs: {e}")
            return []
        
        # Process players with enhanced projections
        print("   Step 3: Calculating enhanced projections...")
        processed_count = 0
        
        for i, player in enumerate(target_date_players):
            if processed_count % 10 == 0:
                print(f"      Processing {i+1}/{len(target_date_players)}...")
            
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
                
                recent_games = player_logs.head(10)  # Use last 10 games
                if len(recent_games) < 3:
                    continue
                
                # Health check - ensure we have required columns
                required_columns = ['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M']
                missing_columns = [col for col in required_columns if col not in recent_games.columns]
                if missing_columns:
                    continue
                
                # Health check - use target date for recency calculation
                most_recent_game = recent_games.iloc[0]
                game_date = pd.to_datetime(most_recent_game['GAME_DATE'])
                target_date = self.target_date if self.target_date else datetime.now()
                days_since_last_game = (target_date - game_date).days
                
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
        
        print(f"   ✅ Processed {processed_count} players with enhanced projections for {self.get_target_date_string()}")
        return player_data
    
    def apply_automatic_player_filters(self, players_data):
        """Automatically filter out players who don't meet minimum thresholds"""
        print("   🔍 Applying automatic player filters...")

        filtered_players = []
        excluded_count = 0

        for player in players_data:
            should_include = True

            # Filter 1: Minimum recent minutes threshold
            recent_minutes = player.get('minutes', 0)
            projected_minutes = player.get('projected_minutes', 0)

            # If player averages <5 minutes and projected for <12, likely won't play
            if recent_minutes < 5 and projected_minutes < 12:
                should_include = False
                excluded_count += 1

            # Filter 2: Minimum projection threshold for salary
            value_score = (player['projection'] / player['salary']) * 1000
            if value_score < 2.0:  # Extremely poor value
                should_include = False
                excluded_count += 1

            # Filter 3: Role stability too low with high projection
            if (player.get('role_stability', 50) < 40 and 
                player['projection'] > 25):
                should_include = False
                excluded_count += 1

            if should_include:
                filtered_players.append(player)

        print(f"   🚫 Automatically excluded {excluded_count} players")
        print(f"   ✅ {len(filtered_players)} players passed automatic filters")
        return filtered_players

    def show_data_summary_enhanced(self):
        """Enhanced data summary with new metrics"""
        if not self.players_data:
            return
            
        print("\n📊 ENHANCED PROJECTIONS v2 SUMMARY:")
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
        avg_upside = sum(p.get('upside_score', 0) for p in self.players_data) / len(self.players_data)
        
        # NEW: Enhanced metrics
        avg_role_stability = sum(p.get('role_stability', 50) for p in self.players_data) / len(self.players_data)
        avg_opportunity = sum(p.get('opportunity_rating', 50) for p in self.players_data) / len(self.players_data)
        avg_trend = sum(p.get('recent_trend', 1) for p in self.players_data) / len(self.players_data)
        avg_blowout_risk = sum(p.get('blowout_risk', 0.3) for p in self.players_data) / len(self.players_data)
        avg_floor = sum(p.get('floor_projection', 0) for p in self.players_data) / len(self.players_data)
        
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
        
        # Show players with low stability (potential risks)
        low_stability_players = [p for p in self.players_data if p.get('role_stability', 50) < 40]
        if low_stability_players:
            print(f"\n⚠️  High-Risk Players (Low Role Stability):")
            for player in sorted(low_stability_players, key=lambda x: x.get('role_stability', 50))[:5]:
                print(f"  {player['name']}: Stability {player.get('role_stability', 0):.0f}, Volatility {player.get('volatility_score', 0):.2f}")
        
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
            
            print(f"\n🚀 Top 5 by Upside Score:")
            for player in sorted(self.players_data, key=lambda x: x.get('upside_score', 0), reverse=True)[:5]:
                upside_gap = player['ceiling_projection'] - player['projection']
                print(f"  {player['name']}: {player.get('upside_score', 0):.1f} (Proj: {player['projection']:.1f}, Ceiling: {player['ceiling_projection']:.1f}, Gap: +{upside_gap:.1f})")
            
            print(f"\n🛡️  Most Stable Roles:")
            for player in sorted(self.players_data, key=lambda x: x.get('role_stability', 0), reverse=True)[:5]:
                print(f"  {player['name']}: {player.get('role_stability', 0):.1f}")
            
            print(f"\n📈 Best Recent Trends:")
            for player in sorted(self.players_data, key=lambda x: x.get('recent_trend', 1), reverse=True)[:5]:
                print(f"  {player['name']}: {player.get('recent_trend', 1):.3f}")

class EnhancedProjectionsNBAOptimizer:
    def __init__(self, dk_salaries_path="DKSalaries.csv", target_date=None):
        self.data = EnhancedProjectionsNBAData(dk_salaries_path, target_date)
        self.lineup_strategies = ['balanced', 'high_upside', 'stars_and_scrubs', 'high_floor', 'tournament']

    def identify_tournament_hammers(self, players_data):
     """Debug version to see what's happening"""
     hammers = []

     print(f"   🔍 DEBUG: Checking {len(players_data)} players for hammer criteria...")

     for player in players_data:
         ceiling = player.get('ceiling_projection', 0)
         projection = player['projection']
         ownership = self.estimate_tournament_ownership(player)
         salary = player['salary']
         usage = player.get('usage_rate', 0)
         minutes = player.get('projected_minutes', 0)

         # Check each criteria
         ceiling_ok = ceiling >= 55
         projection_ok = projection >= 40
         ownership_ok = ownership < 30
         salary_ok = salary >= 7000
         usage_ok = usage >= 0.25
         minutes_ok = minutes >= 30

         if (ceiling_ok and projection_ok and ownership_ok and 
             salary_ok and usage_ok and minutes_ok):
             hammers.append(player)
             print(f"      ✅ HAMMER: {player['name']} - Ceil:{ceiling:.1f}, Proj:{projection:.1f}, Own:{ownership:.1f}%, ${salary:,}")

     # If no hammers found, show the closest candidates
     if len(hammers) == 0:
         print("   🔍 DEBUG - Top candidates that failed hammer criteria:")
         candidates = []
         for player in players_data:
             ceiling = player.get('ceiling_projection', 0)
             projection = player['projection']
             if ceiling >= 45 and projection >= 30:  # Relaxed criteria for debugging
                 ownership = self.estimate_tournament_ownership(player)
                 salary = player['salary']
                 usage = player.get('usage_rate', 0)
                 minutes = player.get('projected_minutes', 0)

                 candidates.append({
                     'player': player,
                     'ceiling': ceiling,
                     'projection': projection,
                     'ownership': ownership,
                     'salary': salary,
                     'usage': usage,
                     'minutes': minutes,
                     'failed_criteria': self.get_failed_criteria(player, ceiling, projection, ownership, salary, usage, minutes)
                 })

         # Show top 10 closest candidates
         candidates.sort(key=lambda x: x['ceiling'], reverse=True)
         for cand in candidates[:10]:
             print(f"      ❌ {cand['player']['name']}: Ceil:{cand['ceiling']:.1f}, Proj:{cand['projection']:.1f}, "
                   f"Own:{cand['ownership']:.1f}%, ${cand['salary']:,}, Usage:{cand['usage']:.3f}, Min:{cand['minutes']:.1f}")
             print(f"         Failed: {cand['failed_criteria']}")

     hammers.sort(key=lambda x: x.get('ceiling_projection', 0), reverse=True)
     print(f"   🔨 Found {len(hammers)} tournament hammers")
     return hammers

    def get_failed_criteria(self, player, ceiling, projection, ownership, salary, usage, minutes):
        """Helper to identify which criteria failed"""
        failed = []
        if ceiling < 55: failed.append(f"ceiling({ceiling:.1f}<55)")
        if projection < 40: failed.append(f"projection({projection:.1f}<40)")
        if ownership >= 30: failed.append(f"ownership({ownership:.1f}>=30)")
        if salary < 7000: failed.append(f"salary(${salary:,}<7000)")
        if usage < 0.25: failed.append(f"usage({usage:.3f}<0.25)")
        if minutes < 30: failed.append(f"minutes({minutes:.1f}<30)")
        return ", ".join(failed) if failed else "ALL PASSED"

    def identify_tournament_value_smashes(self, players_data):
        """Debug version for value smashes"""
        value_smashes = []

        print(f"   🔍 DEBUG: Checking {len(players_data)} players for value criteria...")

        for player in players_data:
            salary = player['salary']
            ceiling = player.get('ceiling_projection', 0)
            projection = player['projection']
            minutes = player.get('projected_minutes', 0)
            ownership = self.estimate_tournament_ownership(player)
            volatility = player.get('volatility_score', 1)

            # Basic criteria
            salary_ok = salary <= 5500
            ceiling_ok = ceiling >= 35
            projection_ok = projection >= 25
            minutes_ok = minutes >= 26
            ownership_ok = ownership < 25

            if (salary_ok and ceiling_ok and projection_ok and minutes_ok and ownership_ok):
                value_smashes.append(player)
                print(f"      ✅ VALUE: {player['name']} - Ceil:{ceiling:.1f}, Proj:{projection:.1f}, ${salary:,}, Own:{ownership:.1f}%")

        # If no value smashes found, show closest candidates
        if len(value_smashes) == 0:
            print("   🔍 DEBUG - Top value candidates that failed criteria:")
            candidates = []
            for player in players_data:
                salary = player['salary']
                if salary <= 6000:  # Slightly relaxed for debugging
                    ceiling = player.get('ceiling_projection', 0)
                    projection = player['projection']
                    if ceiling >= 25 or projection >= 20:  # Relaxed criteria
                        minutes = player.get('projected_minutes', 0)
                        ownership = self.estimate_tournament_ownership(player)

                        candidates.append({
                            'player': player,
                            'ceiling': ceiling,
                            'projection': projection,
                            'salary': salary,
                            'minutes': minutes,
                            'ownership': ownership,
                            'failed_criteria': self.get_failed_value_criteria(player, ceiling, projection, salary, minutes, ownership)
                        })

            # Show top 10 closest candidates
            candidates.sort(key=lambda x: x['ceiling'], reverse=True)
            for cand in candidates[:10]:
                print(f"      ❌ {cand['player']['name']}: Ceil:{cand['ceiling']:.1f}, Proj:{cand['projection']:.1f}, "
                      f"${cand['salary']:,}, Min:{cand['minutes']:.1f}, Own:{cand['ownership']:.1f}%")
                print(f"         Failed: {cand['failed_criteria']}")

        value_smashes.sort(key=lambda x: x.get('ceiling_projection', 0), reverse=True)
        print(f"   💥 Found {len(value_smashes)} tournament value smashes")
        return value_smashes

    def get_failed_value_criteria(self, player, ceiling, projection, salary, minutes, ownership):
        """Helper to identify which value criteria failed"""
        failed = []
        if salary > 5500: failed.append(f"salary(${salary:,}>5500)")
        if ceiling < 35: failed.append(f"ceiling({ceiling:.1f}<35)")
        if projection < 25: failed.append(f"projection({projection:.1f}<25)")
        if minutes < 26: failed.append(f"minutes({minutes:.1f}<26)")
        if ownership >= 25: failed.append(f"ownership({ownership:.1f}>=25)")
        return ", ".join(failed) if failed else "ALL PASSED"

    # def identify_tournament_hammers(self, players_data):
    #     """Identify the players that can actually WIN tournaments (high ceiling + low ownership)"""
    #     hammers = []

    #     for player in players_data:
    #         # Use methods that actually exist in your code
    #         ceiling_ok = player.get('ceiling_projection', 0) >= 55  # Must have 55+ ceiling
    #         projection_ok = player['projection'] >= 40  # Must have 40+ base projection
    #         ownership_est = self.estimate_tournament_ownership(player)
    #         low_ownership = ownership_est < 35  # Slightly higher ownership allowed for true hammers
    #         has_smash_spot = self.has_tournament_smash_spot(player)
    #         salary_viable = player['salary'] >= 6000  # Lowered from 7000 to 6000

    #         # Use existing metrics instead of non-existent method
    #         high_usage = player.get('usage_rate', 0) >= 0.22
    #         good_minutes = player.get('projected_minutes', 0) >= 30

    #         # Check if player has demonstrated upside using existing data
    #         has_demonstrated_upside = (player.get('ceiling_projection', 0) >= 
    #                                   player['projection'] * 1.4)  # 40%+ upside gap

    #         if (ceiling_ok and projection_ok and low_ownership and has_smash_spot and 
    #             salary_viable and high_usage and good_minutes and has_demonstrated_upside):
    #             hammers.append(player)

    #     # Sort by ceiling projection
    #     hammers.sort(key=lambda x: x.get('ceiling_projection', 0), reverse=True)
    #     print(f"   🔨 Found {len(hammers)} tournament hammers")
    #     return hammers

    # def identify_tournament_hammers(self, players_data):
    #     """Identify the players that can actually WIN tournaments (high ceiling + low ownership)"""
    #     hammers = []

    #     for player in players_data:
    #         # Must meet tournament criteria:
    #         ceiling_ok = player.get('ceiling_projection', 0) >= 50  # Must have 50+ ceiling
    #         projection_ok = player['projection'] >= 35  # Must have 35+ base projection
    #         ownership_ok = self.estimate_tournament_ownership(player) < 35  # Higher threshold
    #         #ownership_est = self.estimate_tournament_ownership(player)
    #         #low_ownership = ownership_est < 35  # Slightly higher ownership allowed for true hammers
    #         has_smash_spot = self.has_tournament_smash_spot(player)
    #         #salary_viable = player['salary'] >= 7000  # True hammers are usually expensive
    #         salary_ok = player['salary'] >= 6000  # Lower salary requirement
    #         recent_high = self.get_highest_recent_fantasy_score(player['name'])
    #         has_demonstrated_upside = recent_high >= 45
    #         high_usage = player.get('usage_rate', 0) >= 0.22

    #         #ceiling_ok = player.get('ceiling_projection', 0) >= player['projection'] * 1.4
    #         #ownership_est = self.estimate_tournament_ownership(player)
    #         #low_ownership = ownership_est < 25  # Target <25% projected ownership
    #         #has_smash_spot = self.has_tournament_smash_spot(player)
    #         #salary_viable = player['salary'] >= 6000  # Not a pure value play
    #         if (ceiling_ok and projection_ok and ownership_ok and salary_ok and 
    #             has_demonstrated_upside and high_usage):
    #             hammers.append(player)
    #     hammers.sort(key=lambda x: x.get('ceiling_projection', 0), reverse=True)
    #     print(f"   🔨 Found {len(hammers)} tournament hammers")
    #     return hammers

        #     if ceiling_ok and low_ownership and has_smash_spot and salary_viable:
        #         hammers.append(player)

        # # Sort by ceiling projection
        # hammers.sort(key=lambda x: x.get('ceiling_projection', 0), reverse=True)
        # print(f"   🔨 Found {len(hammers)} tournament hammers")
        # return hammers
    # def identify_tournament_value_smashes(self, players_data):
    #     """Find the value plays that can actually SMASH in tournaments"""
    #     value_smashes = []

    #     for player in players_data:
    #         # MORE AGGRESSIVE tournament value criteria:
    #         salary_ok = player['salary'] <= 5500  # Slightly higher salary cap for better players
    #         ceiling_high = player.get('ceiling_projection', 0) >= 35  # Must have 35+ ceiling
    #         projection_ok = player['projection'] >= 25  # Must have 25+ base projection
    #         minutes_path = player.get('projected_minutes', 0) >= 26  # More minutes required
    #         low_ownership = self.estimate_tournament_ownership(player) < 25

    #         # FIXED: Define has_volatility properly
    #         has_volatility = player.get('volatility_score', 1) >= 1.2  # We WANT volatility in tournaments

    #         # Known tournament value indicators
    #         is_opportunity_play = self.is_opportunity_play(player)
    #         has_recent_trend = player.get('recent_trend', 1) > 1.1

    #         # FIXED: Use the properly defined has_volatility variable
    #         if (salary_ok and ceiling_high and projection_ok and minutes_path and low_ownership and 
    #             (has_volatility or is_opportunity_play or has_recent_trend)):
    #             value_smashes.append(player)

    #     value_smashes.sort(key=lambda x: x.get('ceiling_projection', 0), reverse=True)
    #     print(f"   💥 Found {len(value_smashes)} tournament value smashes (35+ ceiling)")
    #     return value_smashes

    def identify_tournament_value_smashes(self, players_data):
        """FIXED: More realistic value play identification"""
        value_smashes = []

        print(f"   🔍 DEBUG: Checking {len(players_data)} players for value criteria...")

        for player in players_data:
            salary = player['salary']
            ceiling = player.get('ceiling_projection', 0)
            projection = player['projection']
            minutes = player.get('projected_minutes', 0)
            ownership = self.estimate_tournament_ownership(player)

            # RELAXED CRITERIA - focus on finding viable cheap players
            salary_ok = salary <= 6000  # Increased from 5500
            ceiling_ok = ceiling >= 30  # Lowered from 35
            projection_ok = projection >= 22  # Lowered from 25
            minutes_ok = minutes >= 22  # Lowered from 26
            ownership_ok = ownership < 30  # Increased from 25

            # Additional opportunity factors
            is_opportunity_play = self.is_opportunity_play(player)
            has_recent_trend = player.get('recent_trend', 1) > 1.05
            great_matchup = player.get('matchup_difficulty', 0) > 0.5

            if (salary_ok and ceiling_ok and projection_ok and minutes_ok and ownership_ok and
                (is_opportunity_play or has_recent_trend or great_matchup)):
                value_smashes.append(player)
                print(f"      ✅ VALUE: {player['name']} - Ceil:{ceiling:.1f}, Proj:{projection:.1f}, ${salary:,}, Own:{ownership:.1f}%")

        # If still no value smashes, use emergency fallback
        if len(value_smashes) == 0:
            print("   ⚠️ No value smashes found, using emergency fallback...")
            # Emergency: any player under $6000 with reasonable projection
            for player in players_data:
                if player['salary'] <= 6000 and player['projection'] >= 20:
                    value_smashes.append(player)
                    print(f"      🆘 EMERGENCY VALUE: {player['name']} - Proj:{player['projection']:.1f}, ${player['salary']:,}")

        value_smashes.sort(key=lambda x: x.get('ceiling_projection', 0), reverse=True)
        print(f"   💥 Found {len(value_smashes)} tournament value smashes")
        return value_smashes
    
    def estimate_tournament_ownership(self, player):
        """More accurate tournament ownership estimation"""
        base_ownership = 0

        # Salary-based ownership (different from cash)
        if player['salary'] >= 9000:
            base_ownership += 15  # Stars get lower ownership in tournaments
        elif player['salary'] <= 4500:
            base_ownership += 8   # Value plays get less ownership

        # Ceiling projection boost (tournaments care about ceiling)
        if player.get('ceiling_projection', 0) > player['projection'] * 1.5:
            base_ownership += 10

        # Popular player penalty (we want lower ownership)
        popular_players = ['Nikola Jokic', 'Luka Doncic', 'Stephen Curry', 'LeBron James']
        if player['name'] in popular_players:
            base_ownership += 20

        # Value score ownership (high value = higher ownership)
        value_score = (player['projection'] / player['salary']) * 1000
        if value_score > 6:
            base_ownership += 15
        elif value_score > 5:
            base_ownership += 10

        # Recent performance boost (coming off big game = chalk)
        if player.get('recent_trend', 1) > 1.3:
            base_ownership += 10

        return min(50, base_ownership)  # Tournament ownership rarely exceeds 50%

    def has_tournament_smash_spot(self, player):
        """Check if player has legitimate tournament smash spot"""
        # High pace game
        if player.get('pace_adjustment', 1) > 1.05:
            return True

        # Great matchup (opponent poor defense)
        if player.get('matchup_difficulty', 0) > 0.8:
            return True

        # Injury to teammate creating opportunity
        if self.has_injury_opportunity(player):
            return True

        # Projected high-scoring game
        game_total = self.get_game_total(player['team'])
        if game_total > 230:  # High total game
            return True

        return False

    def is_opportunity_play(self, player):
        """Check if this is a true opportunity play (injuries, role change, etc.)"""
        # Known opportunity situations from your contest results
        opportunity_players = {
            'Tre Mann': 'Starting PG role',
            'Kon Knueppel': 'Increased role, hot hand',
            'Noah Clowney': 'Injury to starter',
            'Bub Carrington': 'Backup getting starters minutes',
            'Kel\'el Ware': 'Clear path to 30+ minutes',
            'Jalen Duren': 'Favorable center matchup'
        }
    
        return player['name'] in opportunity_players

    def estimate_ownership(self, player):
        """Estimate player ownership based on salary, projection, and matchup"""
        base_ownership = 0

        # Salary-based ownership
        if player['salary'] >= 9000:
            base_ownership += 25  # Stars get higher ownership
        elif player['salary'] <= 4500:
            base_ownership += 15  # Value plays get attention

        # Projection-based ownership
        value_score = (player['projection'] / player['salary']) * 1000
        if value_score > 6:
            base_ownership += 20
        elif value_score > 5:
            base_ownership += 15
        elif value_score > 4:
            base_ownership += 10

        # Ceiling projection boost
        if player.get('ceiling_projection', 0) > player['projection'] * 1.4:
            base_ownership += 10

        # Matchup boost for easy matchups
        if player.get('matchup_difficulty', 0) > 0.5:
            base_ownership += 5

        # Popular player boost
        popular_players = ['Nikola Jokic', 'Luka Doncic', 'Giannis Antetokounmpo', 'Joel Embiid', 'Stephen Curry']
        if player['name'] in popular_players:
            base_ownership += 15

        # High usage players get more ownership
        if player.get('usage_rate', 0) > 0.25:
            base_ownership += 10

        return min(80, base_ownership)

    def get_game_total(self, team):
        """Get projected game total for a team's game"""
        try:
            if team in self.data.matchup_data:
                matchup = self.data.matchup_data[team]
                opponent = matchup.get('opponent')
                
                if team in self.data.team_stats and opponent in self.data.team_stats:
                    team_off = self.data.team_stats[team].get('off_rating', 110)
                    opp_def = self.data.team_stats[opponent].get('def_rating', 110)
                    
                    # Simple projection: (team offense + opponent defense) / 2 * pace factor
                    pace_factor = matchup.get('pace', 100) / 100
                    projected_total = ((team_off + (110 - opp_def + 110)) / 2) * pace_factor
                    return projected_total
        except Exception as e:
            print(f"   ⚠️ Error calculating game total for {team}: {e}")
        
        return 220  # Default fallback

    def get_high_total_game_players(self):
        """Get players from games with projected high totals"""
        high_total_players = []
        for i, player in enumerate(self.data.players_data):
            game_total = self.get_game_total(player['team'])
            if game_total > 225:  # High total game threshold
                high_total_players.append(i)
        return high_total_players

    def get_viable_team_stacks(self):
        """Get teams that have multiple viable tournament plays"""
        team_stacks = {}
        for i, player in enumerate(self.data.players_data):
            team = player['team']
            if team not in team_stacks:
                team_stacks[team] = []
            team_stacks[team].append(i)
        
        # Only return teams with at least 3 viable players
        return {team: players for team, players in team_stacks.items() if len(players) >= 3}

    def has_injury_opportunity(self, player):
        """Check if player benefits from teammate injury"""
        # This would integrate with injury news
        # For now, manual list based on research
        injury_opportunities = {
            'Tre Mann': 'Starting PG with injuries to other guards',
            'Noah Clowney': 'Benefiting from frontcourt injuries',
            'Kel\'el Ware': 'Clear path due to team context'
        }
        return player['name'] in injury_opportunities

    def add_injury_constraints(self, prob, player_vars):
        """Add constraints to exclude injured players"""
        injured_players = []
        for i, player in enumerate(self.data.players_data):
            status = self.data.check_player_availability(player['name'], player['team'])
            if status == 'OUT':
                injured_players.append(i)
        
        # Force injured players to 0
        for injured_idx in injured_players:
            prob += player_vars[injured_idx] == 0

    def build_lineups(self, strategy='balanced', num_lineups=1):
        """Build lineups using different strategies"""
        print(f"\n🚀 ENHANCED PROJECTIONS NBA DFS LINEUP BUILDER v2 - {strategy.upper()} STRATEGY")
        print("Using role stability, blowout risk, opportunity ratings, and trend analysis")
        print("=" * 60)
        
        if not self.data.get_real_nba_data_faster():
            print("💥 Failed to get enhanced NBA data")
            return False
        
        if len(self.data.players_data) < 8:
            print(f"💥 Only {len(self.data.players_data)} players found, need at least 8")
            return False
        
        lineups = []
        
        if strategy == 'high_upside':
            lineups = self.build_high_upside_lineups(num_lineups)
        elif strategy == 'stars_and_scrubs':
            lineups = self.build_stars_and_scrubs_lineups(num_lineups)
        elif strategy == 'high_floor':
            lineups = self.build_high_floor_lineups(num_lineups)
        elif strategy == 'tournament':
            lineups = self.build_tournament_lineups_v2(num_lineups)
        #elif strategy == 'CASH GAME':
        #    lineups = self.build_cash_lineups(num_lineups)
        else:  # balanced
            lineup = self.optimize_lineup_faster()
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
        
    def test_lineup_feasibility(self):
     """Test if any valid lineup can be built with current player pool"""
     print("   🔍 Testing lineup feasibility...")

     prob = pulp.LpProblem("Feasibility_Test", pulp.LpMaximize)
     player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)

     # Only basic constraints
     prob += pulp.lpSum([player_vars[i] for i in range(len(self.data.players_data))])  # Dummy objective
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

     prob.solve(pulp.PULP_CBC_CMD(msg=0))

     if prob.status == pulp.LpStatusOptimal:
         print("   ✅ Basic lineup construction is feasible")
         return True
     else:
         print("   ❌ BASIC LINEUP CONSTRUCTION FAILED - Check position availability")
         print(f"      Available: PG:{len(pg_players)}, SG:{len(sg_players)}, SF:{len(sf_players)}, PF:{len(pf_players)}, C:{len(c_players)}")
         return False
        
    def build_tournament_lineups_v2(self, num_lineups=20):
        """FIXED: Tournament builder with lineup diversity"""
        print(f"\n🏆 Building {num_lineups} TOURNAMENT lineups (v2)...")

        # Get players
        hammers = self.identify_tournament_hammers(self.data.players_data)
        value_smashes = self.identify_tournament_value_smashes(self.data.players_data)

        print(f"   Tournament assets: {len(hammers)} hammers, {len(value_smashes)} value smashes")

        lineups = []
        previous_lineups = []  # Store previous lineups for diversity
        lineup_count = 0
        max_attempts = num_lineups * 5

        for lineup_num in range(max_attempts):
            if lineup_count >= num_lineups:
                break

            prob = pulp.LpProblem(f"NBA_Tournament_v2_{lineup_num}", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)

            # CRITICAL: ADD INJURY CONSTRAINTS
            self.add_injury_constraints(prob, player_vars)

            # TOURNAMENT OBJECTIVE - Add some randomness for diversity
            import random
            random_factor = random.uniform(0.9, 1.1)  # Small random variation

            prob += pulp.lpSum([
                player_vars[i] * (
                    self.data.players_data[i].get('ceiling_projection', 0) * 0.7 * random_factor +
                    (100 - self.estimate_tournament_ownership(self.data.players_data[i])) * 0.3
                ) for i in range(len(self.data.players_data))
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

            guard_players = pg_players + sg_players
            forward_players = sf_players + pf_players
            prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
            prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3

            # DIVERSITY CONSTRAINTS - PREVENT IDENTICAL LINEUPS
            if previous_lineups:
                for prev_lineup in previous_lineups:
                    # Force at least 3 different players from each previous lineup
                    prob += pulp.lpSum([player_vars[i] for i in prev_lineup]) <= 5
                    print(f"   🔄 Diversity: Max 5 same players as previous lineup")

            # TOURNAMENT CONSTRAINTS WITH VARIATION

            # 1. Hammer constraint - vary the number required
            hammer_indices = [i for i, p in enumerate(self.data.players_data) if p in hammers]
            if len(hammer_indices) >= 1:
                if lineup_count == 0:
                    # First lineup: require 2 hammers
                    prob += pulp.lpSum([player_vars[i] for i in hammer_indices]) >= 2
                    print(f"   🔨 Lineup 1: Requiring 2+ hammers")
                elif lineup_count == 1:
                    # Second lineup: require 1 hammer
                    prob += pulp.lpSum([player_vars[i] for i in hammer_indices]) >= 1
                    print(f"   🔨 Lineup 2: Requiring 1+ hammer")  
                else:
                    # Later lineups: vary between 1-2 hammers
                    required_hammers = 1 if lineup_count % 2 == 0 else 2
                    prob += pulp.lpSum([player_vars[i] for i in hammer_indices]) >= required_hammers
                    print(f"   🔨 Lineup {lineup_count + 1}: Requiring {required_hammers}+ hammers")

            # 2. Value constraint - vary the salary threshold
            if lineup_count == 0:
                cheap_threshold = 5500
            elif lineup_count == 1:
                cheap_threshold = 6000  
            else:
                cheap_threshold = 5000 + (lineup_count * 200)  # Varying threshold

            cheap_players = [i for i, p in enumerate(self.data.players_data) if p['salary'] <= cheap_threshold]
            if len(cheap_players) >= 3:
                prob += pulp.lpSum([player_vars[i] for i in cheap_players]) >= 3
                print(f"   💰 Requiring 3+ players under ${cheap_threshold:,}")

            # 3. Team stacking variation
            team_stacks = self.get_viable_team_stacks()
            if team_stacks:
                # Rotate through different teams for stacking
                team_list = list(team_stacks.keys())
                if lineup_count < len(team_list):
                    target_team = team_list[lineup_count]
                    team_players = [i for i, p in enumerate(self.data.players_data) if p['team'] == target_team]
                    if len(team_players) >= 2:
                        prob += pulp.lpSum([player_vars[i] for i in team_players]) >= 2
                        print(f"   🏀 Stacking {target_team} (2+ players)")

            # 4. Position distribution variation
            if lineup_count % 3 == 0:
                # More guards lineup
                prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 4
                print(f"   🎯 Lineup {lineup_count + 1}: Heavy on guards (4+)")
            elif lineup_count % 3 == 1:
                # More forwards lineup  
                prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 4
                print(f"   🎯 Lineup {lineup_count + 1}: Heavy on forwards (4+)")

            # 5. Salary construction variation
            if lineup_count == 0:
                # Stars and scrubs
                expensive_players = [i for i, p in enumerate(self.data.players_data) if p['salary'] >= 8000]
                if len(expensive_players) >= 2:
                    prob += pulp.lpSum([player_vars[i] for i in expensive_players]) >= 2
                    print(f"   ⭐ Lineup 1: Stars & scrubs construction")
            elif lineup_count == 1:
                # Balanced construction
                mid_players = [i for i, p in enumerate(self.data.players_data) if 6000 <= p['salary'] <= 7500]
                if len(mid_players) >= 3:
                    prob += pulp.lpSum([player_vars[i] for i in mid_players]) >= 3
                    print(f"   ⚖️ Lineup 2: Balanced construction")

            # 6. MINIMUM projection
            min_projection = 240 + (lineup_count * 5)  # Slightly increasing minimum
            prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] for i in range(len(self.data.players_data))]) >= min_projection

            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=0))

            if prob.status == pulp.LpStatusOptimal:
                lineup = self.extract_lineup(player_vars)
                if lineup:
                    lineup_indices = [i for i, p in enumerate(self.data.players_data) if player_vars[i].value() == 1]

                    # Check if this lineup is significantly different
                    is_diverse = True
                    if previous_lineups:
                        for prev_lineup in previous_lineups:
                            common_players = set(lineup_indices) & set(prev_lineup)
                            if len(common_players) >= 6:  # Too similar
                                is_diverse = False
                                print(f"   🔄 Skipping similar lineup ({len(common_players)} same players)")
                                break
                            
                    if is_diverse:
                        previous_lineups.append(lineup_indices)
                        lineups.append(lineup)
                        lineup_count += 1

                        # Show lineup summary
                        total_salary = sum(p['salary'] for p in lineup['players'])
                        total_projection = sum(p['projection'] for p in lineup['players'])
                        hammer_count = sum(1 for p in lineup['players'] if p in hammers)
                        teams = set(p['team'] for p in lineup['players'])

                        print(f"   ✅ Built unique lineup {lineup_count}: ${total_salary:,}, {total_projection:.1f} pts")
                        print(f"      Hammers: {hammer_count}, Teams: {len(teams)} ({', '.join(teams)})")
                    else:
                        print(f"   🔄 Lineup too similar to previous, trying again...")
                else:
                    print(f"   ❌ Failed to extract lineup")
            else:
                print(f"   ❌ No solution found for lineup {lineup_num + 1}")

        print(f"   🎯 Built {lineup_count}/{num_lineups} unique tournament lineups")
        return lineups

    # def build_tournament_lineups_v2(self, num_lineups=20):
    #     """TOURNAMENT-SPECIFIC lineup building based on actual winning patterns"""
    #     """First check if we have viable data"""
    #     print(f"\n🏆 Building {num_lineups} TOURNAMENT lineups (v2)...")

    #     # DEBUG: Check what players we actually have
    #     if not self.data.players_data:
    #         print("   ❌ CRITICAL: No player data available!")
    #         return []

    #     print(f"   📊 Total players available: {len(self.data.players_data)}")

    #     # Show top players by ceiling to see if projections are reasonable
    #     top_ceiling_players = sorted(self.data.players_data, 
    #                                key=lambda x: x.get('ceiling_projection', 0), 
    #                                reverse=True)[:10]
    #     print("   🔝 Top 10 players by ceiling projection:")
    #     for p in top_ceiling_players:
    #         print(f"      {p['name']}: Ceil={p.get('ceiling_projection', 0):.1f}, Proj={p['projection']:.1f}, "
    #               f"Salary=${p['salary']:,}, Usage={p.get('usage_rate', 0):.3f}")
    #     print(f"\n🏆 Building {num_lineups} TOURNAMENT lineups (v2)...")

    #     hammers = self.identify_tournament_hammers(self.data.players_data)

    #     min_tournament_projection = self.calculate_minimum_tournament_projection()

    #     # NEW: Tournament-specific player identification
    #     hammers = self.identify_tournament_hammers(self.data.players_data)
    #     value_smashes = self.identify_tournament_value_smashes(self.data.players_data)

    #     print(f"   Tournament assets: {len(hammers)} hammers, {len(value_smashes)} value smashes")

    #     lineups = []
    #     previous_lineups_players = []
    #     lineup_count = 0
    #     max_attempts = num_lineups * 3  # Prevent infinite loops

    #     for lineup_num in range(num_lineups):
    #         prob = pulp.LpProblem(f"NBA_Tournament_v2_{lineup_num}", pulp.LpMaximize)
    #         player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)

    #         # CRITICAL: ADD INJURY CONSTRAINTS TO EXCLUDE INJURED PLAYERS
    #         self.add_injury_constraints(prob, player_vars)

    #         # TOURNAMENT OBJECTIVE: Maximize CEILING with ownership leverage
    #         prob += pulp.lpSum([
    #             player_vars[i] * (
    #                 self.data.players_data[i].get('ceiling_projection', 0) * 0.7 +  # 70% ceiling focus
    #                 (100 - self.estimate_tournament_ownership(self.data.players_data[i])) * 0.3  # 30% ownership leverage
    #             ) for i in range(len(self.data.players_data))
    #         ])

    #         # Standard constraints
    #         prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['salary'] for i in range(len(self.data.players_data))]) <= 50000
    #         prob += pulp.lpSum([player_vars[i] for i in range(len(self.data.players_data))]) == 8

    #         # Position constraints
    #         pg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PG']
    #         sg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SG']
    #         sf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SF']
    #         pf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PF']
    #         c_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'C']

    #         prob += pulp.lpSum([player_vars[i] for i in pg_players]) >= 1
    #         prob += pulp.lpSum([player_vars[i] for i in sg_players]) >= 1
    #         prob += pulp.lpSum([player_vars[i] for i in sf_players]) >= 1
    #         prob += pulp.lpSum([player_vars[i] for i in pf_players]) >= 1
    #         prob += pulp.lpSum([player_vars[i] for i in c_players]) == 1

    #         guard_players = pg_players + sg_players
    #         forward_players = sf_players + pf_players
    #         prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
    #         prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3

    #         # TOURNAMENT-SPECIFIC CONSTRAINTS
    #         # CRITICAL: MINIMUM PROJECTION CONSTRAINT - Tournament lineups need high projections!
    #         min_tournament_projection = 270  # Adjust based on slate size and scoring environment
    #         prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] 
    #                       for i in range(len(self.data.players_data))]) >= min_tournament_projection

    #         # 1. REQUIRE at least 1 tournament hammer
    #         hammer_indices = [i for i, p in enumerate(self.data.players_data) if p in hammers]
    #         if len(hammer_indices) >= 1:
    #             prob += pulp.lpSum([player_vars[i] for i in hammer_indices]) >= 1

    #         # 2. REQUIRE at least 3 value smashes
    #         value_smash_indices = [i for i, p in enumerate(self.data.players_data) if p in value_smashes]
    #         if len(value_smash_indices) >= 3:
    #             prob += pulp.lpSum([player_vars[i] for i in value_smash_indices]) >= 3

    #         # 3. MAXIMUM ownership constraints (avoid chalk)
    #         chalky_players = [i for i, p in enumerate(self.data.players_data) 
    #                          if self.estimate_tournament_ownership(p) > 30]
    #         if len(chalky_players) > 0:
    #             prob += pulp.lpSum([player_vars[i] for i in chalky_players]) <= 2

    #         # 4. STACK high-total games (like winning lineups did)
    #         high_total_players = self.get_high_total_game_players()
    #         if len(high_total_players) >= 4:
    #             prob += pulp.lpSum([player_vars[i] for i in high_total_players]) >= 4

    #         # 5. CORRELATION: Require at least 2 players from same team (like Giannis + value)
    #         team_stacks = self.get_viable_team_stacks()
    #         stack_constraint_added = False
    #         for team, player_indices in team_stacks.items():
    #             if len(player_indices) >= 3 and not stack_constraint_added:
    #                 prob += pulp.lpSum([player_vars[i] for i in player_indices]) >= 2
    #                 stack_constraint_added = True
    #                 print(f"   Added team stack constraint for {team}")
    #                 break
                
    #         # 6. CEILING requirement: At least 4 players with 40+ ceiling
    #         high_ceiling_indices = [i for i, p in enumerate(self.data.players_data) 
    #                               if p.get('ceiling_projection', 0) >= 40]
    #         if len(high_ceiling_indices) >= 4:
    #             prob += pulp.lpSum([player_vars[i] for i in high_ceiling_indices]) >= 4

    #         # 7. Player diversity across lineups
    #         if lineup_num > 0 and previous_lineups_players:
    #             all_prev_players = []
    #             for prev_lineup in previous_lineups_players:
    #                 all_prev_players.extend(prev_lineup)
    #             all_prev_players = list(set(all_prev_players))

    #             if len(all_prev_players) > 0:
    #                 prob += pulp.lpSum([player_vars[i] for i in all_prev_players]) <= 4

    #         # Solve
    #         prob.solve(pulp.PULP_CBC_CMD(msg=0))

    #         if prob.status == pulp.LpStatusOptimal:
    #             lineup = self.extract_lineup(player_vars)
    #             if lineup:
    #                 # NEW: Check if lineup meets minimum projection threshold
    #                 if lineup['total_projection'] >= min_tournament_projection:
    #                     lineup_indices = [i for i, p in enumerate(self.data.players_data) if player_vars[i].value() == 1]
    #                     previous_lineups_players.append(lineup_indices)
    #                     lineups.append(lineup)
    #                     lineup_count += 1
    #                 #lineup_indices = [i for i, p in enumerate(self.data.players_data) if player_vars[i].value() == 1]
    #                 #previous_lineups_players.append(lineup_indices)
    #                 #lineups.append(lineup)

    #                 # Print tournament-specific lineup analysis
    #                 self.analyze_tournament_lineup(lineup, lineup_num)
    #             else:
    #                 print(f"   ❌ Failed to extract tournament lineup {lineup_num + 1}")
    #         else:
    #             print(f"   ❌ No optimal solution found for tournament lineup {lineup_num + 1}")
    #         # If we're struggling to find lineups, relax the minimum projection
    #         if max_attempts < num_lineups * 2 and lineup_count < num_lineups / 2:
    #             min_tournament_projection = 220  # Slightly lower threshold
    #             print(f"   🔄 Relaxing minimum projection to {min_tournament_projection} pts") 
    #     print(f"   🎯 Built {lineup_count}/{num_lineups} tournament lineups")
    #     return lineups
    
    def calculate_minimum_tournament_projection(self):
        """Calculate minimum projection based on slate size and scoring environment"""
        slate_size = len(self.data.todays_games) / 2  # Number of games

        if slate_size <= 3:  # Small slate
            return 280
        elif slate_size <= 6:  # Medium slate
            return 270  
        elif slate_size <= 9:  # Large slate
            return 260
        else:  # Massive slate
            return 250

    def analyze_tournament_lineup(self, lineup, lineup_num):
        """Analyze lineup for tournament viability"""
        total_ceiling = sum(p.get('ceiling_projection', p['projection']) for p in lineup['players'])
        avg_ownership = sum(self.estimate_tournament_ownership(p) for p in lineup['players']) / 8
        value_smashes = [p for p in lineup['players'] if p['salary'] <= 5000 and p.get('ceiling_projection', 0) >= 30]
        hammers = [p for p in lineup['players'] if p['salary'] >= 7000 and p.get('ceiling_projection', 0) >= 55]
        
        print(f"   ✅ Tournament Lineup {lineup_num + 1}:")
        print(f"      Ceiling: {total_ceiling:.1f} | Avg Ownership: {avg_ownership:.1f}%")
        print(f"      Hammers: {len(hammers)} | Value Smashes: {len(value_smashes)}")

    def build_high_upside_lineups(self, num_lineups=3):
        """Build lineups focused on high upside players"""
        print(f"\n🎯 Building {num_lineups} high-upside lineups...")
        
        lineups = []
        
        # Filter for high upside players
        high_upside_players = [p for p in self.data.players_data if p.get('upside_score', 0) >= 60]
        high_ceiling_players = [p for p in self.data.players_data if p.get('ceiling_projection', 0) >= p.get('projection', 0) * 1.3]
        
        print(f"   High upside players: {len(high_upside_players)}")
        print(f"   High ceiling players: {len(high_ceiling_players)}")
        
        # Track previously selected players to ensure diversity
        previous_lineups_players = []
        
        for lineup_num in range(num_lineups):
            print(f"   Building lineup {lineup_num + 1}...")
            
            prob = pulp.LpProblem(f"NBA_DFS_High_Upside_{lineup_num}", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)
            self.add_injury_constraints(prob, player_vars)
            
            # Objective: Maximize both projection and upside score
            prob += pulp.lpSum([
                player_vars[i] * (
                    self.data.players_data[i]['projection'] * 0.7 +  # 70% weight on projection
                    self.data.players_data[i]['upside_score'] * 0.3   # 30% weight on upside
                ) for i in range(len(self.data.players_data))
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
            
            guard_players = pg_players + sg_players
            forward_players = sf_players + pf_players
            prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
            prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3
            
            # High upside constraints (require at least 4 high upside players)
            high_upside_indices = [i for i, p in enumerate(self.data.players_data) if p.get('upside_score', 0) >= 60]
            if len(high_upside_indices) >= 4:
                prob += pulp.lpSum([player_vars[i] for i in high_upside_indices]) >= 4
            
            # High ceiling constraints (require at least 2 players with ceiling 30%+ above projection)
            high_ceiling_indices = [i for i, p in enumerate(self.data.players_data) if p.get('ceiling_projection', 0) >= p.get('projection', 0) * 1.3]
            if len(high_ceiling_indices) >= 2:
                prob += pulp.lpSum([player_vars[i] for i in high_ceiling_indices]) >= 2
            
            # Prevent same players from previous lineups
            if lineup_num > 0 and previous_lineups_players:
                all_prev_players = []
                for prev_lineup in previous_lineups_players:
                    all_prev_players.extend(prev_lineup)
                
                all_prev_players = list(set(all_prev_players))
                
                if len(all_prev_players) > 0:
                    prob += pulp.lpSum([player_vars[i] for i in all_prev_players]) <= 5
            
            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status == pulp.LpStatusOptimal:
                lineup = self.extract_lineup(player_vars)
                if lineup:
                    lineup_indices = [i for i, p in enumerate(self.data.players_data) if player_vars[i].value() == 1]
                    previous_lineups_players.append(lineup_indices)
                    lineups.append(lineup)
                    print(f"   ✅ Lineup {lineup_num + 1} built successfully")
                else:
                    print(f"   ❌ Failed to extract lineup {lineup_num + 1}")
            else:
                print(f"   ❌ No optimal solution found for lineup {lineup_num + 1}")
    
        return lineups

    def build_stars_and_scrubs_lineups(self, num_lineups=2):
        """Build lineups using stars and scrubs strategy"""
        print(f"\n⭐ Building {num_lineups} stars and scrubs lineups...")
        
        lineups = []
        
        # Identify stars (high projection, high salary) and scrubs (good value, low salary)
        stars = sorted(self.data.players_data, key=lambda x: x['projection'], reverse=True)[:20]
        scrubs = [p for p in self.data.players_data if p['salary'] <= 4500 and p['projection'] >= 20]
        
        print(f"   Top stars: {len(stars)}")
        print(f"   Value scrubs: {len(scrubs)}")
        
        # Track previously selected players to ensure diversity
        previous_lineups_players = []
        
        for lineup_num in range(num_lineups):
            prob = pulp.LpProblem(f"NBA_DFS_Stars_Scrubs_{lineup_num}", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)
            
            # Objective: Maximize projection
            prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] for i in range(len(self.data.players_data))])
            
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
            
            guard_players = pg_players + sg_players
            forward_players = sf_players + pf_players
            prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
            prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3
            
            # Stars constraint (2-3 high-priced players)
            star_indices = [i for i, p in enumerate(self.data.players_data) if p['salary'] >= 8000]
            if len(star_indices) >= 2:
                prob += pulp.lpSum([player_vars[i] for i in star_indices]) >= 2
                prob += pulp.lpSum([player_vars[i] for i in star_indices]) <= 3
            
            # Scrubs constraint (3-4 value players)
            scrub_indices = [i for i, p in enumerate(self.data.players_data) if p['salary'] <= 4500 and p['projection'] >= 20]
            if len(scrub_indices) >= 3:
                prob += pulp.lpSum([player_vars[i] for i in scrub_indices]) >= 3
            
            # Prevent same players from previous lineups
            if lineup_num > 0 and previous_lineups_players:
                all_prev_players = []
                for prev_lineup in previous_lineups_players:
                    all_prev_players.extend(prev_lineup)
                all_prev_players = list(set(all_prev_players))
                
                if len(all_prev_players) > 0:
                    prob += pulp.lpSum([player_vars[i] for i in all_prev_players]) <= 5
            
            # Solve
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
        
        # Identify high-floor players (high consistency, low volatility)
        high_floor_players = [p for p in self.data.players_data if p.get('consistency_rating', 0) >= 70 and p.get('volatility_score', 1) <= 0.8]
        
        print(f"   High-floor players: {len(high_floor_players)}")
        
        # Track previously selected players to ensure diversity
        previous_lineups_players = []
        
        for lineup_num in range(num_lineups):
            prob = pulp.LpProblem(f"NBA_DFS_High_Floor_{lineup_num}", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)
            
            # Objective: Maximize projection with consistency bonus
            prob += pulp.lpSum([
                player_vars[i] * (
                    self.data.players_data[i]['projection'] * 0.8 +  # 80% weight on projection
                    self.data.players_data[i]['consistency_rating'] * 0.2  # 20% weight on consistency
                ) for i in range(len(self.data.players_data))
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
            
            guard_players = pg_players + sg_players
            forward_players = sf_players + pf_players
            prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
            prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3
            
            # High floor constraints (require at least 5 high-consistency players)
            high_floor_indices = [i for i, p in enumerate(self.data.players_data) if p.get('consistency_rating', 0) >= 70]
            if len(high_floor_indices) >= 5:
                prob += pulp.lpSum([player_vars[i] for i in high_floor_indices]) >= 5
            
            # Low volatility constraint
            low_vol_indices = [i for i, p in enumerate(self.data.players_data) if p.get('volatility_score', 1) <= 0.8]
            if len(low_vol_indices) >= 4:
                prob += pulp.lpSum([player_vars[i] for i in low_vol_indices]) >= 4
            
            # Prevent same players from previous lineups
            if lineup_num > 0 and previous_lineups_players:
                all_prev_players = []
                for prev_lineup in previous_lineups_players:
                    all_prev_players.extend(prev_lineup)
                all_prev_players = list(set(all_prev_players))
                
                if len(all_prev_players) > 0:
                    prob += pulp.lpSum([player_vars[i] for i in all_prev_players]) <= 5
            
            # Solve
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
        
        # Objective: Pure points maximization
        prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] for i in range(len(self.data.players_data))])
        
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
        
        guard_players = pg_players + sg_players
        forward_players = sf_players + pf_players
        prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
        prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3
        
        # Solve
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
    
    def optimize_lineup_faster(self):
        """Faster lineup optimization"""
        print("\n🧠 Optimizing lineup with enhanced projections (FAST)...")
        
        # Pre-calculate position indices for faster access
        pg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PG']
        sg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SG']
        sf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SF']
        pf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PF']
        c_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'C']
        
        # Quick check for position viability
        if (len(pg_players) < 2 or len(sg_players) < 2 or len(sf_players) < 2 or 
            len(pf_players) < 2 or len(c_players) < 2):
            print("❌ Not enough players for viable lineup construction")
            return None
        
        prob = pulp.LpProblem("NBA_DFS_Enhanced_Fast", pulp.LpMaximize)
        player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)

        # Objective
        prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] for i in range(len(self.data.players_data))])

        # Constraints
        prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['salary'] for i in range(len(self.data.players_data))]) <= 50000
        prob += pulp.lpSum([player_vars[i] for i in range(len(self.data.players_data))]) == 8

        # Use pre-calculated position indices
        prob += pulp.lpSum([player_vars[i] for i in pg_players]) >= 1
        prob += pulp.lpSum([player_vars[i] for i in sg_players]) >= 1
        prob += pulp.lpSum([player_vars[i] for i in sf_players]) >= 1
        prob += pulp.lpSum([player_vars[i] for i in pf_players]) >= 1
        prob += pulp.lpSum([player_vars[i] for i in c_players]) == 1

        guard_players = pg_players + sg_players
        forward_players = sf_players + pf_players
        prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
        prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3

        # Solve with timeout
        print("   Solving optimization (fast mode)...")
        try:
            # Use faster solver with timeout
            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))  # 30 second timeout
        except:
            # Fallback to regular solve
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
        print("🏆 ENHANCED PROJECTIONS NBA DFS LINEUP v2")
        print("=" * 140)
        print("📈 Using role stability, blowout risk, opportunity ratings, and trend analysis")
        print("=" * 140)
        
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
            upside_indicator = "🚀" if player.get('upside_score', 0) > 70 else "⬆️" if player.get('upside_score', 0) > 50 else "➡️"
            stability_indicator = "🛡️" if player.get('role_stability', 0) > 80 else "📊" if player.get('role_stability', 0) > 60 else "⚠️"
            trend_indicator = "📈" if player.get('recent_trend', 1) > 1.1 else "📉" if player.get('recent_trend', 1) < 0.9 else "➡️"
            opportunity_indicator = "🎯" if player.get('opportunity_rating', 0) > 70 else "📊" if player.get('opportunity_rating', 0) > 50 else "💤"
            
            print(f"{i:2d}. {player['position']:2} | {player['name']:20} | "
                  f"${player['salary']:5,} | {player['projection']:5.1f} pts | "
                  f"UPSIDE: {player.get('upside_score', 0):2.0f} {upside_indicator} | "
                  f"Min: {player.get('projected_minutes', 0):2.0f} | "
                  f"USG: {player.get('usage_rate', 0):.2f} {usage_indicator} | "
                  f"STAB: {player.get('role_stability', 0):2.0f} {stability_indicator} | "
                  f"TREND: {player.get('recent_trend', 1):.2f} {trend_indicator} | "
                  f"OPP: {player.get('opportunity_rating', 0):2.0f} {opportunity_indicator} | "
                  f"{location_indicator}{b2b_indicator} vs {player.get('opponent', 'UNK'):3} {matchup_indicator}")

        print("=" * 140)
        print(f"💵 Total Salary: ${lineup['total_salary']:,} / $50,000")
        print(f"📈 Total Projection: {lineup['total_projection']:.1f} fantasy points")
        print(f"📊 Total Ceiling: {lineup.get('total_ceiling', lineup['total_projection']):.1f} fantasy points")
        print(f"📉 Total Floor: {sum(p.get('floor_projection', 0) for p in lineup['players']):.1f} fantasy points")
        print(f"⚡ Efficiency: {lineup['efficiency']:.2f} points per $1,000")
        print(f"🚀 Avg Upside Score: {lineup.get('avg_upside_score', 0):.1f}")
        
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
        avg_upside = sum(p.get('upside_score', 0) for p in lineup['players']) / len(lineup['players'])
        avg_stability = sum(p.get('role_stability', 0) for p in lineup['players']) / len(lineup['players'])
        avg_opportunity = sum(p.get('opportunity_rating', 0) for p in lineup['players']) / len(lineup['players'])
        avg_trend = sum(p.get('recent_trend', 1) for p in lineup['players']) / len(lineup['players'])
        avg_blowout_risk = sum(p.get('blowout_risk', 0.3) for p in lineup['players']) / len(lineup['players'])
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
        print(f"  Avg Role Stability: {avg_stability:.1f}")
        print(f"  Avg Opportunity Rating: {avg_opportunity:.1f}")
        print(f"  Avg Recent Trend: {avg_trend:.3f}")
        print(f"  Avg Blowout Risk: {avg_blowout_risk:.3f}")
        print(f"  Home Players: {home_count}/{len(lineup['players'])}")
        print(f"  Back-to-Back Players: {b2b_count}/{len(lineup['players'])}")
        
        lineup_teams = set(p['team'] for p in lineup['players'])
        print(f"🏀 Teams: {', '.join(sorted(lineup_teams))}")
        print("=" * 140)

class CashGameNBAOptimizer(EnhancedProjectionsNBAOptimizer):
    # def __init__(self, dk_salaries_path="DKSalaries.csv"):
    #     super().__init__(dk_salaries_path)
    #     self.cash_strategies = ['cash_balanced', 'cash_elite_anchor', 'cash_value_focus']

    def __init__(self, dk_salaries_path="DKSalaries.csv", target_date=None):
        super().__init__(dk_salaries_path, target_date)
        self.cash_strategies = ['cash_balanced', 'cash_elite_anchor', 'cash_value_focus']
    
    def build_cash_lineups(self, strategy='cash_balanced', num_lineups=3):
        """Build lineups optimized for cash games (Double Ups, 50/50s)"""
        print(f"\n💰 Building {num_lineups} CASH GAME lineups ({strategy})...")
        
        #if not self.data.get_real_nba_data():
        if not self.data.get_real_nba_data_faster():
            print("💥 Failed to get NBA data")
            return []
            
        lineups = []
        
        if strategy == 'cash_elite_anchor':
            lineups = self.build_elite_anchor_cash_lineups(num_lineups)
        elif strategy == 'cash_value_focus':
            lineups = self.build_value_focus_cash_lineups(num_lineups)
        elif strategy == 'cash_high_scoring':  # NEW OPTION
            lineups = self.build_realistic_cash_lineups(num_lineups)
        else:  # cash_balanced
            lineups = self.build_balanced_cash_lineups(num_lineups)
            
        return lineups
    
    def build_balanced_cash_lineups(self, num_lineups=3):
        """Balanced cash approach with chalk identification"""
        lineups = []
        previous_lineups_players = []

        print("   📊 Projecting ownership for chalk identification...")

        # Identify chalk plays before building lineups
        chalk_plays = self.data.identify_chalk_plays(min_ownership=40)
        chalk_indices = []

        for chalk_data in chalk_plays:
            for i, player in enumerate(self.data.players_data):
                if player['name'] == chalk_data['player']['name']:
                    chalk_indices.append(i)
                    break
                
        print(f"   🎯 Identified {len(chalk_plays)} potential chalk plays")
        for chalk in chalk_plays[:5]:  # Show top 5
            player = chalk['player']
            print(f"      {player['name']}: {chalk['projected_ownership']:.0f}% - {', '.join(chalk['reasons'])}")

        for lineup_num in range(num_lineups):
            prob = pulp.LpProblem(f"NBA_Cash_Balanced_{lineup_num}", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)

            # ADD INJURY CONSTRAINTS
            self.add_injury_constraints(prob, player_vars)

            # CASH GAME OBJECTIVE: Maximize floor projection with consistency bonus
            prob += pulp.lpSum([
                player_vars[i] * (
                    self.data.players_data[i]['projection'] * 0.6 +           # 60% projection
                    self.data.players_data[i].get('floor_projection', 0) * 0.3 +  # 30% floor
                    self.data.players_data[i]['consistency_rating'] * 0.1     # 10% consistency
                ) for i in range(len(self.data.players_data))
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

            guard_players = pg_players + sg_players
            forward_players = sf_players + pf_players
            prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
            prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3

            # ENHANCED CASH GAME CONSTRAINTS

            # 1. CHALK CONSTRAINT: Require 3-4 high-ownership players
            if len(chalk_indices) >= 3:
                prob += pulp.lpSum([player_vars[i] for i in chalk_indices]) >= 3
                prob += pulp.lpSum([player_vars[i] for i in chalk_indices]) <= 5
                print(f"   🔒 Added chalk constraint: 3-5 of {len(chalk_indices)} chalk plays")

            # 2. HIGH CONSISTENCY REQUIREMENT
            consistent_players = [i for i, p in enumerate(self.data.players_data) 
                                if p.get('consistency_rating', 0) >= 60]
            if len(consistent_players) >= 6:
                prob += pulp.lpSum([player_vars[i] for i in consistent_players]) >= 6

            # 3. ROLE STABILITY REQUIREMENT
            stable_players = [i for i, p in enumerate(self.data.players_data) 
                            if p.get('role_stability', 50) >= 60]
            if len(stable_players) >= 5:
                prob += pulp.lpSum([player_vars[i] for i in stable_players]) >= 5

            # 4. LIMIT HIGH-RISK PLAYERS
            high_risk_players = [i for i, p in enumerate(self.data.players_data) 
                               if p.get('volatility_score', 1) > 1.2]
            if len(high_risk_players) > 0:
                prob += pulp.lpSum([player_vars[i] for i in high_risk_players]) <= 2

            # 5. VALUE REQUIREMENT
            value_plays = [i for i, p in enumerate(self.data.players_data) 
                         if p['salary'] <= 5000 and p['projection'] >= 20]
            if len(value_plays) >= 2:
                prob += pulp.lpSum([player_vars[i] for i in value_plays]) >= 2

            # 6. MINUTES SECURITY
            secure_minutes = [i for i, p in enumerate(self.data.players_data) 
                            if p.get('projected_minutes', 0) >= 25]
            if len(secure_minutes) >= 6:
                prob += pulp.lpSum([player_vars[i] for i in secure_minutes]) >= 6

            # 7. AVOID TOO MANY BACK-TO-BACK PLAYERS
            b2b_players = [i for i, p in enumerate(self.data.players_data) 
                          if p.get('back_to_back', False)]
            if len(b2b_players) > 0:
                prob += pulp.lpSum([player_vars[i] for i in b2b_players]) <= 2

            # Player diversity across lineups
            if lineup_num > 0 and previous_lineups_players:
                all_prev_players = []
                for prev_lineup in previous_lineups_players:
                    all_prev_players.extend(prev_lineup)
                all_prev_players = list(set(all_prev_players))

                if len(all_prev_players) > 0:
                    prob += pulp.lpSum([player_vars[i] for i in all_prev_players]) <= 4

            # Solve
            try:
                prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
            except:
                prob.solve(pulp.PULP_CBC_CMD(msg=0))

            if prob.status == pulp.LpStatusOptimal:
                lineup = self.extract_lineup(player_vars)
                if lineup:
                    lineup_indices = [i for i, p in enumerate(self.data.players_data) if player_vars[i].value() == 1]
                    previous_lineups_players.append(lineup_indices)
                    lineups.append(lineup)

                    # Analyze chalk usage in this lineup
                    chalk_count = sum(1 for i in lineup_indices if i in chalk_indices)
                    print(f"   ✅ Cash Lineup {lineup_num + 1} built successfully ({chalk_count}/8 chalk plays)")
                else:
                    print(f"   ❌ Failed to extract cash lineup {lineup_num + 1}")
            else:
                print(f"   ❌ No optimal solution found for cash lineup {lineup_num + 1}")

        return lineups
    
    def build_high_scoring_cash_lineups(self, num_lineups=3):
        """Build lineups that actually target 330+ points with diversity"""
        print(f"   🎯 Building {num_lineups} HIGH-SCORING cash lineups (330+ target)...")
        
        lineups = []
        previous_lineups_players = []  # Track players from previous lineups
        max_attempts = num_lineups * 10  # More attempts for diversity
        
        for lineup_num in range(max_attempts):
            if len(lineups) >= num_lineups:
                break
                
            prob = pulp.LpProblem(f"NBA_Cash_High_Score_{lineup_num}", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)
            
            # ADD INJURY CONSTRAINTS
            self.add_injury_constraints(prob, player_vars)
            
            # OBJECTIVE: Pure points maximization
            prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] for i in range(len(self.data.players_data))])
            
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
            
            guard_players = pg_players + sg_players
            forward_players = sf_players + pf_players
            prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
            prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3
            
            # CRITICAL: MINIMUM PROJECTION CONSTRAINT - FORCE HIGH SCORING
            prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] for i in range(len(self.data.players_data))]) >= 330
            
            # DIVERSITY CONSTRAINT: Prevent identical lineups
            if previous_lineups_players:
                # Force at least 3 different players from previous lineups
                for prev_lineup_players in previous_lineups_players:
                    prob += pulp.lpSum([player_vars[i] for i in prev_lineup_players]) <= 5  # Max 5 same players
            
            # TEAM DIVERSITY: Don't use the same team stack repeatedly
            if lineup_num > 0:
                # Get teams from previous lineups and limit exposure
                previous_teams = set()
                for prev_lineup in lineups:
                    for player in prev_lineup['players']:
                        previous_teams.add(player['team'])
                
                # Limit players from overused teams
                for team in list(previous_teams)[:3]:  # Top 3 most used teams
                    team_players = [i for i, p in enumerate(self.data.players_data) if p['team'] == team]
                    if len(team_players) > 0:
                        prob += pulp.lpSum([player_vars[i] for i in team_players]) <= 2  # Max 2 from same team
            
            # PLAYER TYPE DIVERSITY: Mix of studs and values
            stud_players = [i for i, p in enumerate(self.data.players_data) if p['salary'] >= 8000]
            value_players = [i for i, p in enumerate(self.data.players_data) if p['salary'] <= 5000 and p['projection'] >= 25]
            
            if len(stud_players) >= 2:
                prob += pulp.lpSum([player_vars[i] for i in stud_players]) >= 1  # At least 1 stud
                prob += pulp.lpSum([player_vars[i] for i in stud_players]) <= 3  # But no more than 3
            
            if len(value_players) >= 3:
                prob += pulp.lpSum([player_vars[i] for i in value_players]) >= 2  # At least 2 values
                prob += pulp.lpSum([player_vars[i] for i in value_players]) <= 4  # But no more than 4
            
            # Solve
            try:
                prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=20))
            except:
                prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status == pulp.LpStatusOptimal:
                lineup = self.extract_lineup(player_vars)
                if lineup and lineup['total_projection'] >= 330:
                    lineup_indices = [i for i, p in enumerate(self.data.players_data) if player_vars[i].value() == 1]
                    
                    # Check if this lineup is significantly different from previous ones
                    is_diverse = True
                    if previous_lineups_players:
                        for prev_lineup in previous_lineups_players:
                            common_players = set(lineup_indices) & set(prev_lineup)
                            if len(common_players) >= 6:  # Too similar (6+ same players)
                                is_diverse = False
                                break
                            
                    if is_diverse:
                        previous_lineups_players.append(lineup_indices)
                        lineups.append(lineup)
                        print(f"   ✅ High-Scoring Lineup {len(lineups)}: {lineup['total_projection']:.1f} pts")
                    else:
                        print(f"   🔄 Skipping similar lineup ({len(common_players)} same players)")
        
        if lineups:
            print(f"   🎯 Successfully built {len(lineups)} diverse lineups targeting 330+ points")
        else:
            print(f"   ⚠️ Could not build diverse lineups targeting 330+ points")
            # Fallback without diversity constraints
            lineups = self.build_cash_lineups_fallback(num_lineups, min_projection=300)
        
        return lineups
    
    def build_realistic_cash_lineups(self, num_lineups=3):
        """Build lineups with realistic projections and diversity"""
        print(f"   🎯 Building {num_lineups} REALISTIC cash lineups...")

        lineups = []
        previous_lineups = []  # Track previous lineups for diversity

        for i in range(num_lineups):
            prob = pulp.LpProblem(f"NBA_Cash_Realistic_{i}", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)

            self.add_injury_constraints(prob, player_vars)
            prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] for i in range(len(self.data.players_data))])

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

            guard_players = pg_players + sg_players
            forward_players = sf_players + pf_players
            prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
            prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3

            # DIVERSITY CONSTRAINT: Prevent identical lineups
            if previous_lineups:
                for prev_lineup in previous_lineups:
                    # Force at least 3 different players from each previous lineup
                    prob += pulp.lpSum([player_vars[i] for i in prev_lineup]) <= 5

            # PLAYER TYPE DIVERSITY: Different roster constructions
            if i == 0:
                # Lineup 1: Balanced (2 studs, 3 values)
                stud_players = [i for i, p in enumerate(self.data.players_data) if p['salary'] >= 8000]
                value_players = [i for i, p in enumerate(self.data.players_data) if p['salary'] <= 5000 and p['projection'] >= 20]
                if len(stud_players) >= 2 and len(value_players) >= 3:
                    prob += pulp.lpSum([player_vars[i] for i in stud_players]) >= 2
                    prob += pulp.lpSum([player_vars[i] for i in value_players]) >= 3
            elif i == 1:
                # Lineup 2: Stars and Scrubs (3 studs, 2 values)
                stud_players = [i for i, p in enumerate(self.data.players_data) if p['salary'] >= 8000]
                value_players = [i for i, p in enumerate(self.data.players_data) if p['salary'] <= 5000 and p['projection'] >= 20]
                if len(stud_players) >= 3 and len(value_players) >= 2:
                    prob += pulp.lpSum([player_vars[i] for i in stud_players]) >= 3
                    prob += pulp.lpSum([player_vars[i] for i in value_players]) >= 2
            else:
                # Lineup 3: Value Focus (1 stud, 4 values)
                stud_players = [i for i, p in enumerate(self.data.players_data) if p['salary'] >= 8000]
                value_players = [i for i, p in enumerate(self.data.players_data) if p['salary'] <= 5000 and p['projection'] >= 20]
                if len(stud_players) >= 1 and len(value_players) >= 4:
                    prob += pulp.lpSum([player_vars[i] for i in stud_players]) >= 1
                    prob += pulp.lpSum([player_vars[i] for i in value_players]) >= 4

            prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] for i in range(len(self.data.players_data))]) >= 250

            prob.solve(pulp.PULP_CBC_CMD(msg=0))

            if prob.status == pulp.LpStatusOptimal:
                lineup = self.extract_lineup(player_vars)
                if lineup:
                    # Get the player indices for this lineup
                    lineup_indices = [i for i, p in enumerate(self.data.players_data) if player_vars[i].value() == 1]
                    previous_lineups.append(lineup_indices)
                    lineups.append(lineup)
                    print(f"   ✅ Realistic Lineup {i+1}: {lineup['total_projection']:.1f} pts")

        return lineups
        
    # def build_high_scoring_cash_lineups(self, num_lineups=3):
    #     """Build lineups that actually target 330+ points"""
    #     print(f"   🎯 Building {num_lineups} HIGH-SCORING cash lineups (330+ target)...")

    #     lineups = []
    #     max_attempts = num_lineups * 5

    #     for attempt in range(max_attempts):
    #         if len(lineups) >= num_lineups:
    #             break

    #         prob = pulp.LpProblem(f"NBA_Cash_High_Score_{attempt}", pulp.LpMaximize)
    #         player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)

    #         # ADD INJURY CONSTRAINTS
    #         self.add_injury_constraints(prob, player_vars)

    #         # OBJECTIVE: Pure points maximization
    #         prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] for i in range(len(self.data.players_data))])

    #         # Standard constraints
    #         prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['salary'] for i in range(len(self.data.players_data))]) <= 50000
    #         prob += pulp.lpSum([player_vars[i] for i in range(len(self.data.players_data))]) == 8

    #         # Position constraints
    #         pg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PG']
    #         sg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SG']
    #         sf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SF']
    #         pf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PF']
    #         c_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'C']

    #         prob += pulp.lpSum([player_vars[i] for i in pg_players]) >= 1
    #         prob += pulp.lpSum([player_vars[i] for i in sg_players]) >= 1
    #         prob += pulp.lpSum([player_vars[i] for i in sf_players]) >= 1
    #         prob += pulp.lpSum([player_vars[i] for i in pf_players]) >= 1
    #         prob += pulp.lpSum([player_vars[i] for i in c_players]) == 1

    #         guard_players = pg_players + sg_players
    #         forward_players = sf_players + pf_players
    #         prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
    #         prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3

    #         # CRITICAL: MINIMUM PROJECTION CONSTRAINT - FORCE HIGH SCORING
    #         prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] for i in range(len(self.data.players_data))]) >= 330

    #         # Require at least 2 players with 50+ projections (studs)
    #         high_projections = [i for i, p in enumerate(self.data.players_data) if p['projection'] >= 50]
    #         if len(high_projections) >= 2:
    #             prob += pulp.lpSum([player_vars[i] for i in high_projections]) >= 2

    #         # Require at least 4 players with 30+ projections (solid producers)
    #         solid_projections = [i for i, p in enumerate(self.data.players_data) if p['projection'] >= 30]
    #         if len(solid_projections) >= 4:
    #             prob += pulp.lpSum([player_vars[i] for i in solid_projections]) >= 4

    #         # Solve
    #         try:
    #             prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
    #         except:
    #             prob.solve(pulp.PULP_CBC_CMD(msg=0))

    #         if prob.status == pulp.LpStatusOptimal:
    #             lineup = self.extract_lineup(player_vars)
    #             if lineup and lineup['total_projection'] >= 330:
    #                 lineups.append(lineup)
    #                 print(f"   ✅ High-Scoring Lineup {len(lineups)}: {lineup['total_projection']:.1f} pts")

    #     if lineups:
    #         print(f"   🎯 Successfully built {len(lineups)} lineups targeting 330+ points")
    #     else:
    #         print(f"   ⚠️ Could not build lineups targeting 330+ points, relaxing to 300+...")
    #         # Fallback with lower target
    #         lineups = self.build_cash_lineups_fallback(num_lineups, min_projection=300)

    #     return lineups

    # def build_cash_lineups_fallback(self, num_lineups=3, min_projection=300):
    #     """Fallback method if high-scoring fails"""
    #     lineups = []

    #     for i in range(num_lineups):
    #         prob = pulp.LpProblem(f"NBA_Cash_Fallback_{i}", pulp.LpMaximize)
    #         player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)

    #         self.add_injury_constraints(prob, player_vars)
    #         prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] for i in range(len(self.data.players_data))])

    #         # Standard constraints
    #         prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['salary'] for i in range(len(self.data.players_data))]) <= 50000
    #         prob += pulp.lpSum([player_vars[i] for i in range(len(self.data.players_data))]) == 8

    #         # Position constraints (same as above)
    #         pg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PG']
    #         sg_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SG']
    #         sf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'SF']
    #         pf_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'PF']
    #         c_players = [i for i, p in enumerate(self.data.players_data) if p['position'] == 'C']

    #         prob += pulp.lpSum([player_vars[i] for i in pg_players]) >= 1
    #         prob += pulp.lpSum([player_vars[i] for i in sg_players]) >= 1
    #         prob += pulp.lpSum([player_vars[i] for i in sf_players]) >= 1
    #         prob += pulp.lpSum([player_vars[i] for i in pf_players]) >= 1
    #         prob += pulp.lpSum([player_vars[i] for i in c_players]) == 1

    #         guard_players = pg_players + sg_players
    #         forward_players = sf_players + pf_players
    #         prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
    #         prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3

    #         # Lower minimum projection
    #         prob += pulp.lpSum([player_vars[i] * self.data.players_data[i]['projection'] for i in range(len(self.data.players_data))]) >= min_projection

    #         prob.solve(pulp.PULP_CBC_CMD(msg=0))

    #         if prob.status == pulp.LpStatusOptimal:
    #             lineup = self.extract_lineup(player_vars)
    #             if lineup:
    #                 lineups.append(lineup)
    #                 print(f"   ✅ Fallback Lineup {i+1}: {lineup['total_projection']:.1f} pts")

    #     return lineups
    
    def build_elite_anchor_cash_lineups(self, num_lineups=2):
        """Build cash lineups around 1-2 elite anchors"""
        lineups = []
        previous_lineups_players = []
        
        for lineup_num in range(num_lineups):
            prob = pulp.LpProblem(f"NBA_Cash_Elite_Anchor_{lineup_num}", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)
            
            # Objective focused on reliable production
            prob += pulp.lpSum([
                player_vars[i] * (
                    self.data.players_data[i]['projection'] * 0.7 +
                    self.data.players_data[i].get('floor_projection', 0) * 0.3
                ) for i in range(len(self.data.players_data))
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
            
            guard_players = pg_players + sg_players
            forward_players = sf_players + pf_players
            prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
            prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3
            
            # ELITE ANCHOR CONSTRAINTS
            
            # 1. Require exactly 1-2 elite players ($9000+)
            elite_players = [i for i, p in enumerate(self.data.players_data) 
                           if p['salary'] >= 9000 and p['projection'] >= 45]
            if len(elite_players) >= 1:
                prob += pulp.lpSum([player_vars[i] for i in elite_players]) >= 1
                prob += pulp.lpSum([player_vars[i] for i in elite_players]) <= 2
            
            # 2. Focus on high-usage reliable players for mid-range
            reliable_midrange = [i for i, p in enumerate(self.data.players_data) 
                               if 6000 <= p['salary'] <= 8000 and 
                               p.get('usage_rate', 0) >= 0.20 and
                               p.get('consistency_rating', 0) >= 65]
            if len(reliable_midrange) >= 2:
                prob += pulp.lpSum([player_vars[i] for i in reliable_midrange]) >= 2
            
            # 3. Value plays with secure minutes
            secure_value = [i for i, p in enumerate(self.data.players_data) 
                          if p['salary'] <= 5000 and 
                          p.get('projected_minutes', 0) >= 25 and
                          p.get('role_stability', 0) >= 60]
            if len(secure_value) >= 3:
                prob += pulp.lpSum([player_vars[i] for i in secure_value]) >= 3
            
            # Player diversity
            if lineup_num > 0 and previous_lineups_players:
                all_prev_players = []
                for prev_lineup in previous_lineups_players:
                    all_prev_players.extend(prev_lineup)
                all_prev_players = list(set(all_prev_players))
                
                if len(all_prev_players) > 0:
                    prob += pulp.lpSum([player_vars[i] for i in all_prev_players]) <= 4
            
            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status == pulp.LpStatusOptimal:
                lineup = self.extract_lineup(player_vars)
                if lineup:
                    lineup_indices = [i for i, p in enumerate(self.data.players_data) if player_vars[i].value() == 1]
                    previous_lineups_players.append(lineup_indices)
                    lineups.append(lineup)
        
        return lineups
    
    def build_value_focus_cash_lineups(self, num_lineups=2):
        """Build cash lineups focused on maximum value plays"""
        lineups = []
        previous_lineups_players = []
        
        for lineup_num in range(num_lineups):
            prob = pulp.LpProblem(f"NBA_Cash_Value_Focus_{lineup_num}", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts("Player", range(len(self.data.players_data)), 0, 1, pulp.LpBinary)
            
            # Objective: Maximize value (points per dollar)
            prob += pulp.lpSum([
                player_vars[i] * (
                    (self.data.players_data[i]['projection'] / self.data.players_data[i]['salary']) * 10000 +
                    self.data.players_data[i].get('bargain_rating', 0) * 0.1
                ) for i in range(len(self.data.players_data))
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
            
            guard_players = pg_players + sg_players
            forward_players = sf_players + pf_players
            prob += pulp.lpSum([player_vars[i] for i in guard_players]) >= 3
            prob += pulp.lpSum([player_vars[i] for i in forward_players]) >= 3
            
            # VALUE FOCUS CONSTRAINTS
            
            # 1. Heavy focus on value plays (<$6000 with good projection)
            strong_value = [i for i, p in enumerate(self.data.players_data) 
                          if p['salary'] <= 6000 and 
                          (p['projection'] / p['salary']) * 1000 >= 5.0]
            if len(strong_value) >= 4:
                prob += pulp.lpSum([player_vars[i] for i in strong_value]) >= 4
            
            # 2. Still require at least 1 reliable anchor
            reliable_anchor = [i for i, p in enumerate(self.data.players_data) 
                             if p['salary'] >= 7000 and 
                             p.get('consistency_rating', 0) >= 70]
            if len(reliable_anchor) >= 1:
                prob += pulp.lpSum([player_vars[i] for i in reliable_anchor]) >= 1
            
            # 3. Avoid extreme punts (players < $4000 with projection < 15)
            extreme_punts = [i for i, p in enumerate(self.data.players_data) 
                           if p['salary'] < 4000 and p['projection'] < 15]
            if len(extreme_punts) > 0:
                prob += pulp.lpSum([player_vars[i] for i in extreme_punts]) <= 1
            
            # 4. Focus on players with secure roles
            secure_roles = [i for i, p in enumerate(self.data.players_data) 
                          if p.get('role_stability', 0) >= 65]
            if len(secure_roles) >= 6:
                prob += pulp.lpSum([player_vars[i] for i in secure_roles]) >= 6
            
            # Player diversity
            if lineup_num > 0 and previous_lineups_players:
                all_prev_players = []
                for prev_lineup in previous_lineups_players:
                    all_prev_players.extend(prev_lineup)
                all_prev_players = list(set(all_prev_players))
                
                if len(all_prev_players) > 0:
                    prob += pulp.lpSum([player_vars[i] for i in all_prev_players]) <= 4
            
            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status == pulp.LpStatusOptimal:
                lineup = self.extract_lineup(player_vars)
                if lineup:
                    lineup_indices = [i for i, p in enumerate(self.data.players_data) if player_vars[i].value() == 1]
                    previous_lineups_players.append(lineup_indices)
                    lineups.append(lineup)
        
        return lineups

    def display_cash_lineup_analysis(self, lineup):
        """Display cash lineup with specific cash game analysis"""
        print("\n" + "=" * 120)
        print("💰 CASH GAME LINEUP ANALYSIS")
        print("=" * 120)
        
        # Calculate cash-specific metrics
        total_floor = sum(p.get('floor_projection', p['projection'] * 0.7) for p in lineup['players'])
        avg_consistency = sum(p.get('consistency_rating', 50) for p in lineup['players']) / len(lineup['players'])
        avg_stability = sum(p.get('role_stability', 50) for p in lineup['players']) / len(lineup['players'])
        high_risk_count = sum(1 for p in lineup['players'] if p.get('volatility_score', 1) > 1.2)
        secure_minutes_count = sum(1 for p in lineup['players'] if p.get('projected_minutes', 0) >= 28)
        value_plays_count = sum(1 for p in lineup['players'] if (p['projection'] / p['salary']) * 1000 >= 5.0)
        # NEW: Ownership metrics
        avg_ownership = sum(p.get('projected_ownership', 0) for p in lineup['players']) / len(lineup['players'])
        chalk_plays_count = sum(1 for p in lineup['players'] if p.get('projected_ownership', 0) >= 40)
        
        print(f"📊 CASH GAME METRICS:")
        print(f"  Total Floor: {total_floor:.1f} pts")
        print(f"  Avg Consistency: {avg_consistency:.1f}/100")
        print(f"  Avg Role Stability: {avg_stability:.1f}/100")
        print(f"  High-Risk Players: {high_risk_count}/8")
        print(f"  Secure Minutes (>28 min): {secure_minutes_count}/8")
        print(f"  Strong Value Plays: {value_plays_count}/8")
        print(f"  Avg Projected Ownership: {avg_ownership:.1f}%")
        print(f"  Chalk Plays (>40% owned): {chalk_plays_count}/8")
        
        # Display players
        for i, player in enumerate(lineup['players'], 1):
            value_score = (player['projection'] / player['salary']) * 1000
            projected_own = player.get('projected_ownership', 0)
            floor_ratio = player.get('floor_projection', player['projection'] * 0.7) / player['projection']
            
            # Cash game indicators
            consistency_indicator = "🛡️" if player.get('consistency_rating', 0) >= 70 else "📊" if player.get('consistency_rating', 0) >= 60 else "⚠️"
            stability_indicator = "🔒" if player.get('role_stability', 0) >= 70 else "📊" if player.get('role_stability', 0) >= 60 else "🔓"
            risk_indicator = "🚨" if player.get('volatility_score', 1) > 1.2 else "🟢"
            value_indicator = "💎" if value_score >= 6.0 else "💰" if value_score >= 5.0 else "💵"
            ownership_indicator = "🔥" if projected_own >= 50 else "📈" if projected_own >= 40 else "📊"
            
            print(f"{i:2d}. {player['position']:2} | {player['name']:20} | "
                  f"${player['salary']:5,} | {player['projection']:5.1f} pts | "
                  f"Floor: {player.get('floor_projection', player['projection'] * 0.7):4.1f} | "
                  f"Value: {value_score:4.2f} {value_indicator} | "
                  f"Own: {projected_own:2.0f}% {ownership_indicator} | "
                  f"Cons: {player.get('consistency_rating', 0):2.0f} {consistency_indicator} | "
                  f"Stab: {player.get('role_stability', 0):2.0f} {stability_indicator} | "
                  f"Risk: {risk_indicator}")
        
        print("=" * 120)

def parse_date_input(date_str):
    """Parse various date input formats"""
    try:
        # Try different date formats
        formats = ['%Y-%m-%d', '%m/%d/%Y', '%m/%d/%y', '%Y%m%d']
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # If no format works, try relative dates
        if date_str.lower() == 'yesterday':
            return datetime.now() - timedelta(days=1)
        elif date_str.lower() == 'today':
            return datetime.now()
        elif date_str.lower().startswith('-'):
            days_ago = int(date_str[1:])
            return datetime.now() - timedelta(days=days_ago)
        
        print(f"❌ Could not parse date: {date_str}")
        return None
    except Exception as e:
        print(f"❌ Error parsing date: {e}")
        return None
    
if __name__ == "__main__":
    print("🎯 ENHANCED PROJECTIONS NBA DFS LINEUP BUILDER v2")
    print("With date flexibility and enhanced projections")
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
    
    # Ask for target date
    print("\n📅 DATE SELECTION:")
    print("   Enter specific date or:")
    print("   - 'today' for today's slate")
    print("   - 'yesterday' for yesterday's slate") 
    print("   - '-1' for 1 day ago, '-2' for 2 days ago, etc.")
    print("   Examples: 2024-03-15, 03/15/2024, yesterday, -1")
    
    date_input = input("\nEnter target date (default: today): ").strip()
    
    if date_input == "":
        target_date = datetime.now()
        print(f"   Using today: {target_date.strftime('%Y-%m-%d')}")
    else:
        target_date = parse_date_input(date_input)
        if not target_date:
            print("   ❌ Invalid date, using today")
            target_date = datetime.now()
        else:
            print(f"   Using date: {target_date.strftime('%Y-%m-%d')}")
    
    # Ask user for strategy preference
    print("\n🎯 Available Lineup Strategies:")
    print("1. Balanced (default)")
    print("2. High Upside (tournament focus)")
    print("3. Stars & Scrubs (premium players + value picks)")
    print("4. High Floor (cash game focus)")
    print("5. Maximum Points (pure projection optimization)")
    print("6. Tournament Focus (20 lineups)")
    print("7. CASH GAME Strategies")
    print("8. Build ALL strategies")
    
    choice = input("\nSelect strategy (1-8, default 1): ").strip()
    
    if choice == "7":
        # CASH GAME Strategies
        print("\n💰 CASH GAME STRATEGIES:")
        print("1. Balanced Cash (recommended)")
        print("2. Elite Anchor Focus") 
        print("3. Value Focus")
        print("4. HIGH SCORING (330+ target)")  # NEW
        print("5. Build ALL Cash Strategies")
        
        cash_choice = input("\nSelect cash strategy (1-5, default 1): ").strip()
        
        cash_optimizer = CashGameNBAOptimizer(dk_file, target_date)
        
        if cash_choice == "2":
            lineups = cash_optimizer.build_cash_lineups(strategy='cash_elite_anchor', num_lineups=3)
        elif cash_choice == "3":
            lineups = cash_optimizer.build_cash_lineups(strategy='cash_value_focus', num_lineups=3)
        elif cash_choice == "4":  # NEW
            #lineups = cash_optimizer.build_realistic_cash_lineups(num_lineups=3) 
            lineups = cash_optimizer.build_cash_lineups(strategy='cash_high_scoring', num_lineups=3)
        elif cash_choice == "5":
            # Build all cash strategies
            print("\n🏗️  Building ALL cash game strategies...")
            lineups1 = cash_optimizer.build_cash_lineups(strategy='cash_balanced', num_lineups=2)
            lineups2 = cash_optimizer.build_cash_lineups(strategy='cash_elite_anchor', num_lineups=2)
            lineups3 = cash_optimizer.build_cash_lineups(strategy='cash_value_focus', num_lineups=2)
            lineups4 = cash_optimizer.build_cash_lineups(strategy='cash_high_scoring', num_lineups=1)  # NEW
            lineups = lineups1 + lineups2 + lineups3
        else:
            lineups = cash_optimizer.build_cash_lineups(strategy='cash_balanced', num_lineups=3)
        
        if lineups:
            for i, lineup in enumerate(lineups):
                if lineup:
                    print(f"\n💰 CASH LINEUP {i+1} for {target_date.strftime('%Y-%m-%d')}")
                    cash_optimizer.display_cash_lineup_analysis(lineup)
            success = True
        else:
            success = False
            
    elif choice == "2":
        # High Upside strategy
        num_lineups = input("How many high-upside lineups? (default 3): ").strip()
        num_lineups = int(num_lineups) if num_lineups.isdigit() else 3
        optimizer = EnhancedProjectionsNBAOptimizer(dk_file, target_date)
        success = optimizer.build_lineups(strategy='high_upside', num_lineups=num_lineups)
        
    elif choice == "3":
        # Stars & Scrubs strategy
        num_lineups = input("How many stars & scrubs lineups? (default 2): ").strip()
        num_lineups = int(num_lineups) if num_lineups.isdigit() else 2
        optimizer = EnhancedProjectionsNBAOptimizer(dk_file, target_date)
        success = optimizer.build_lineups(strategy='stars_and_scrubs', num_lineups=num_lineups)
        
    elif choice == "4":
        # High Floor strategy
        num_lineups = input("How many high-floor lineups? (default 2): ").strip()
        num_lineups = int(num_lineups) if num_lineups.isdigit() else 2
        optimizer = EnhancedProjectionsNBAOptimizer(dk_file, target_date)
        success = optimizer.build_lineups(strategy='high_floor', num_lineups=num_lineups)
        
    elif choice == "5":
        # Maximum Points strategy
        optimizer = EnhancedProjectionsNBAOptimizer(dk_file, target_date)
        max_points_lineup = optimizer.optimize_max_points_lineup()
        if max_points_lineup:
            print(f"\n💥 MAXIMUM POINTS LINEUP for {target_date.strftime('%Y-%m-%d')}")
            optimizer.display_lineup_enhanced(max_points_lineup)
            success = True
        else:
            success = False

    elif choice == "6":
        # Tournament Focus strategy
        num_lineups = input("How many tournament lineups? (default 20): ").strip()
        num_lineups = int(num_lineups) if num_lineups.isdigit() else 20
        optimizer = EnhancedProjectionsNBAOptimizer(dk_file, target_date)
        success = optimizer.build_lineups(strategy='tournament', num_lineups=num_lineups)
            
    elif choice == "8":
        # Build ALL strategies
        print(f"\n🏗️  Building ALL lineup strategies for {target_date.strftime('%Y-%m-%d')}...")
        
        optimizer = EnhancedProjectionsNBAOptimizer(dk_file, target_date)
        
        # Balanced
        print("\n" + "="*60)
        print("⚖️  BALANCED STRATEGY")
        print("="*60)
        balanced_success = optimizer.build_lineups(strategy='balanced', num_lineups=1)
        
        # High Upside
        print("\n" + "="*60)
        print("🚀 HIGH UPSIDE STRATEGY")
        print("="*60)
        upside_success = optimizer.build_lineups(strategy='high_upside', num_lineups=2)
        
        # Stars & Scrubs
        print("\n" + "="*60)
        print("⭐ STARS & SCRUBS STRATEGY")
        print("="*60)
        stars_success = optimizer.build_lineups(strategy='stars_and_scrubs', num_lineups=2)
        
        # High Floor
        print("\n" + "="*60)
        print("📊 HIGH FLOOR STRATEGY")
        print("="*60)
        floor_success = optimizer.build_lineups(strategy='high_floor', num_lineups=2)
        
        # Maximum Points
        print("\n" + "="*60)
        print("💥 MAXIMUM POINTS STRATEGY")
        print("="*60)
        max_points_lineup = optimizer.optimize_max_points_lineup()
        if max_points_lineup:
            optimizer.display_lineup_enhanced(max_points_lineup)
            max_success = True
        else:
            max_success = False
            
        success = balanced_success or upside_success or stars_success or floor_success or max_success
        
    else:
        # Default: Balanced strategy
        optimizer = EnhancedProjectionsNBAOptimizer(dk_file, target_date)
        success = optimizer.build_lineups(strategy='balanced', num_lineups=1)
    
    if not success:
        print(f"\n💥 Failed to build lineups for {target_date.strftime('%Y-%m-%d')}")
        sys.exit(1)
    else:
        print(f"\n✅ Successfully built lineups for {target_date.strftime('%Y-%m-%d')}")

