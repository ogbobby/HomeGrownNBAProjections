import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

class NBADataIntegrator:
    def __init__(self):
        self.base_url = "https://popcornmachine.net/api"
        self.session = requests.Session()
        
    def get_injury_data(self):
        """Get injury data from multiple sources"""
        print("🩹 Fetching injury data...")
        
        injuries = []
        
        # Method 1: Popcorn Machine injuries
        try:
            url = f"{self.base_url}/injuries"
            response = self.session.get(url)
            if response.status_code == 200:
                popcorn_injuries = response.json()
                for injury in popcorn_injuries:
                    injuries.append({
                        'player': injury.get('player_name'),
                        'team': injury.get('team'),
                        'status': injury.get('status'),
                        'injury': injury.get('injury'),
                        'source': 'popcorn_machine'
                    })
        except Exception as e:
            print(f"❌ Popcorn Machine injury error: {e}")
        
        # Method 2: Basketball Reference (scraping fallback)
        try:
            br_injuries = self._get_basketball_reference_injuries()
            injuries.extend(br_injuries)
        except Exception as e:
            print(f"❌ Basketball Reference error: {e}")
            
        return injuries
    
    def _get_basketball_reference_injuries(self):
        """Fallback injury data from Basketball Reference"""
        # This would require beautifulsoup4 installation
        try:
            import requests
            from bs4 import BeautifulSoup
            
            url = "https://www.basketball-reference.com/friv/injuries.fcgi"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            injuries = []
            table = soup.find('table', {'id': 'injuries'})
            if table:
                for row in table.find('tbody').find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        injuries.append({
                            'player': cells[0].text.strip(),
                            'team': cells[1].text.strip(),
                            'injury': cells[2].text.strip(),
                            'status': 'OUT' if 'out' in cells[2].text.lower() else 'QUESTIONABLE',
                            'source': 'basketball_reference'
                        })
            return injuries
        except ImportError:
            print("⚠️ Install beautifulsoup4 for Basketball Reference scraping")
            return []
    
    def get_team_data(self):
        """Get team stats and pace data from Popcorn Machine"""
        print("🏀 Fetching team data...")
        
        try:
            url = f"{self.base_url}/teams"
            response = self.session.get(url)
            if response.status_code == 200:
                teams = response.json()
                
                team_stats = {}
                for team in teams:
                    team_stats[team['abbreviation']] = {
                        'name': team['name'],
                        'pace': team.get('pace', 0),
                        'off_rating': team.get('offensive_rating', 0),
                        'def_rating': team.get('defensive_rating', 0),
                        'win_pct': team.get('win_pct', 0)
                    }
                return team_stats
        except Exception as e:
            print(f"❌ Team data error: {e}")
            return {}
    
    def get_player_stats(self, season='2024', last_n_games=10):
        """Get comprehensive player stats from Popcorn Machine"""
        print("📊 Fetching player statistics...")
        
        try:
            # Get all players
            players_url = f"{self.base_url}/players"
            players_response = self.session.get(players_url)
            
            if players_response.status_code != 200:
                return pd.DataFrame()
            
            players_data = players_response.json()
            player_stats = []
            
            for player in players_data[:50]:  # Limit for demo
                player_id = player['id']
                
                # Get detailed stats for each player
                stats_url = f"{self.base_url}/players/{player_id}/stats"
                stats_response = self.session.get(stats_url)
                
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                    if stats and 'per_game' in stats:
                        per_game = stats['per_game']
                        advanced = stats.get('advanced', {})
                        
                        player_stats.append({
                            'player_id': player_id,
                            'name': player['name'],
                            'team': player['team'],
                            'position': player['position'],
                            'games_played': per_game.get('games_played', 0),
                            'minutes': per_game.get('minutes', 0),
                            'points': per_game.get('points', 0),
                            'rebounds': per_game.get('rebounds_total', 0),
                            'assists': per_game.get('assists', 0),
                            'steals': per_game.get('steals', 0),
                            'blocks': per_game.get('blocks', 0),
                            'turnovers': per_game.get('turnovers', 0),
                            'usage_rate': advanced.get('usage_rate', 0),
                            'ts_percent': advanced.get('true_shooting_percent', 0)
                        })
                
                time.sleep(0.1)  # Be respectful to the API
            
            return pd.DataFrame(player_stats)
            
        except Exception as e:
            print(f"❌ Player stats error: {e}")
            return pd.DataFrame()
    
    def get_vegas_lines(self):
        """Get Vegas odds from free API"""
        print("🎰 Fetching Vegas lines...")
        
        try:
            # Using The Odds API (requires free API key)
            # Sign up at https://the-odds-api.com/
            api_key = "YOUR_API_KEY"  # You need to get this
            url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
            
            params = {
                'apiKey': api_key,
                'regions': 'us',
                'markets': 'spreads,totals',
                'oddsFormat': 'decimal'
            }
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                odds_data = response.json()
                games = []
                
                for game in odds_data:
                    home_team = game['home_team']
                    away_team = game['away_team']
                    
                    # Extract totals and spreads
                    total = self._extract_total(game)
                    spread = self._extract_spread(game)
                    
                    games.append({
                        'home_team': home_team,
                        'away_team': away_team,
                        'total': total,
                        'spread': spread,
                        'start_time': game['commence_time']
                    })
                
                return games
            else:
                print("⚠️ Using mock Vegas data - get API key for real data")
                return self._get_mock_vegas_data()
                
        except Exception as e:
            print(f"❌ Vegas lines error: {e}")
            return self._get_mock_vegas_data()
    
    def _extract_total(self, game_data):
        """Extract over/under total from odds data"""
        try:
            for bookmaker in game_data['bookmakers']:
                for market in bookmaker['markets']:
                    if market['key'] == 'totals':
                        return float(market['outcomes'][0]['point'])
        except:
            pass
        return 220.0  # Default fallback
    
    def _extract_spread(self, game_data):
        """Extract point spread from odds data"""
        try:
            for bookmaker in game_data['bookmakers']:
                for market in bookmaker['markets']:
                    if market['key'] == 'spreads':
                        return float(market['outcomes'][0]['point'])
        except:
            pass
        return -5.0  # Default fallback
    
    def _get_mock_vegas_data(self):
        """Mock Vegas data for demonstration"""
        return [
            {'home_team': 'LAC', 'away_team': 'GSW', 'total': 235.5, 'spread': -2.5},
            {'home_team': 'DEN', 'away_team': 'SAC', 'total': 228.0, 'spread': -5.5},
            {'home_team': 'BOS', 'away_team': 'PHO', 'total': 232.0, 'spread': -4.0}
        ]
    
    def get_daily_schedule(self, date=None):
        """Get today's NBA schedule"""
        print("📅 Fetching daily schedule...")
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            url = f"https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
            response = self.session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                games = data.get('scoreboard', {}).get('games', [])
                
                schedule = []
                for game in games:
                    schedule.append({
                        'game_id': game['gameId'],
                        'home_team': game['homeTeam']['teamTricode'],
                        'away_team': game['awayTeam']['teamTricode'],
                        'start_time': game['gameTimeUTC'],
                        'status': game['gameStatusText']
                    })
                
                return schedule
            else:
                return self._get_mock_schedule()
                
        except Exception as e:
            print(f"❌ Schedule error: {e}")
            return self._get_mock_schedule()
    
    def _get_mock_schedule(self):
        """Mock schedule data"""
        return [
            {'home_team': 'LAC', 'away_team': 'GSW', 'status': '7:30 PM ET'},
            {'home_team': 'DEN', 'away_team': 'SAC', 'status': '9:00 PM ET'}
        ]

class EnhancedDFSOptimizer:
    def __init__(self):
        self.data_integrator = NBADataIntegrator()
        self.players_df = None
        self.injuries = None
        self.schedule = None
        self.vegas_lines = None
        self.team_data = None
    
    def load_all_data(self):
        """Load all required data for optimization"""
        print("🔄 Loading all data sources...")
        
        # Load data in parallel (in production, use threading)
        self.injuries = self.data_integrator.get_injury_data()
        self.schedule = self.data_integrator.get_daily_schedule()
        self.vegas_lines = self.data_integrator.get_vegas_lines()
        self.team_data = self.data_integrator.get_team_data()
        self.players_df = self.data_integrator.get_player_stats()
        
        print("✅ All data loaded successfully!")
        
    def analyze_matchups(self):
        """Analyze matchups using integrated data"""
        print("\n🔍 Analyzing matchups...")
        
        target_games = []
        player_valuations = []
        
        # Analyze each game on schedule
        for game in self.schedule:
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Find Vegas line for this game
            vegas_info = next((g for g in self.vegas_lines 
                             if g['home_team'] == home_team or g['away_team'] == home_team), None)
            
            if vegas_info:
                total = vegas_info['total']
                spread = vegas_info['spread']
                
                # Game quality assessment
                if total > 230 and abs(spread) < 6:
                    target_games.append({
                        'game': f"{away_team} @ {home_team}",
                        'total': total,
                        'spread': spread,
                        'quality': 'ELITE'
                    })
                    
                    print(f"🎯 Elite Target: {away_team} @ {home_team} | Total: {total} | Spread: {spread}")
        
        return target_games
    
    def identify_value_plays(self):
        """Identify value plays based on injuries and matchups"""
        print("\n💰 Identifying value plays...")
        
        value_plays = []
        
        # Find players benefiting from injuries
        for injury in self.injuries:
            if injury['status'] in ['OUT', 'DOUBTFUL']:
                # Logic to find replacement players
                # This would need team depth chart data
                replacement = self._find_replacement_player(injury)
                if replacement:
                    value_plays.append({
                        'player': replacement,
                        'reason': f"{injury['player']} injury",
                        'projected_minutes': 30,  # Would be dynamic
                        'value_tier': 'HIGH'
                    })
        
        return value_plays
    
    def _find_replacement_player(self, injury):
        """Find likely replacement for injured player"""
        # Simplified logic - in practice, use depth charts
        team = injury['team']
        position = self._infer_position(injury['player'])
        
        # Look for bench players on same team
        if self.players_df is not None:
            team_players = self.players_df[self.players_df['team'] == team]
            bench_players = team_players[team_players['minutes'] < 25]
            
            if not bench_players.empty:
                return bench_players.iloc[0]['name']
        
        return f"Unknown {team} Player"
    
    def _infer_position(self, player_name):
        """Infer position from player data"""
        if self.players_df is not None:
            player = self.players_df[self.players_df['name'] == player_name]
            if not player.empty:
                return player.iloc[0]['position']
        return 'UNKNOWN'
    
    def generate_projections(self):
        """Generate fantasy projections using integrated data"""
        print("\n📈 Generating projections...")
        
        if self.players_df is None:
            print("❌ No player data available")
            return pd.DataFrame()
        
        # Enhanced projection formula using all data sources
        projections = []
        
        for _, player in self.players_df.iterrows():
            # Base stats
            base_projection = self._calculate_base_projection(player)
            
            # Adjust for matchup
            matchup_boost = self._calculate_matchup_adjustment(player)
            
            # Adjust for pace
            pace_boost = self._calculate_pace_adjustment(player)
            
            # Final projection
            final_projection = base_projection * (1 + matchup_boost + pace_boost)
            
            projections.append({
                'name': player['name'],
                'team': player['team'],
                'position': player['position'],
                'base_projection': base_projection,
                'matchup_adjustment': matchup_boost,
                'pace_adjustment': pace_boost,
                'final_projection': final_projection,
                'value_score': final_projection / 8000 * 1000  # Assuming $8k avg salary
            })
        
        return pd.DataFrame(projections)
    
    def _calculate_base_projection(self, player):
        """Calculate base fantasy projection"""
        return (
            player['points'] * 1.0 +
            player['rebounds'] * 1.2 +
            player['assists'] * 1.5 +
            player['steals'] * 3.0 +
            player['blocks'] * 3.0 -
            player['turnovers'] * 1.0
        )
    
    def _calculate_matchup_adjustment(self, player):
        """Calculate matchup-based adjustment"""
        if self.team_data and player['team'] in self.team_data:
            team_def = self.team_data[player['team']]['def_rating']
            # Simplified: worse defense = better matchup
            return (110 - team_def) * 0.01  # Adjust based on defense
        return 0
    
    def _calculate_pace_adjustment(self, player):
        """Calculate pace-based adjustment"""
        if self.team_data and player['team'] in self.team_data:
            team_pace = self.team_data[player['team']]['pace']
            # Higher pace = more opportunities
            return (team_pace - 100) * 0.005
        return 0

# Usage Example
if __name__ == "__main__":
    optimizer = EnhancedDFSOptimizer()
    
    # Load all data
    optimizer.load_all_data()
    
    # Analyze matchups
    target_games = optimizer.analyze_matchups()
    
    # Identify value plays
    value_plays = optimizer.identify_value_plays()
    
    # Generate projections
    projections = optimizer.generate_projections()
    
    print(f"\n🎯 Found {len(target_games)} target games")
    print(f"💰 Found {len(value_plays)} value plays")
    print(f"📊 Generated {len(projections)} player projections")
    
    # Display top value plays
    if not projections.empty:
        top_values = projections.nlargest(5, 'value_score')
        print("\n🏆 Top Value Plays:")
        for _, player in top_values.iterrows():
            print(f"  {player['name']} - Projection: {player['final_projection']:.1f}, Value: {player['value_score']:.2f}")