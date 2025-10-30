import pandas as pd
import requests
import numpy as np
from datetime import datetime
import pulp
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

class DraftKingsIntegration:
    def __init__(self):
        self.salary_data = None
        
    def load_dk_salaries(self, csv_path_or_url=None):
        """Load DraftKings salaries from CSV file or URL"""
        print("💰 Loading DraftKings salaries...")
        
        try:
            if csv_path_or_url and csv_path_or_url.startswith('http'):
                # Load from URL
                response = requests.get(csv_path_or_url)
                response.raise_for_status()
                self.salary_data = pd.read_csv(StringIO(response.text))
            elif csv_path_or_url:
                # Load from local file
                self.salary_data = pd.read_csv(csv_path_or_url)
            else:
                # Use mock data for demonstration
                self.salary_data = self._create_mock_salaries()
            
            # Standardize column names
            self.salary_data = self._standardize_columns(self.salary_data)
            print(f"✅ Loaded {len(self.salary_data)} players from DraftKings")
            return self.salary_data
            
        except Exception as e:
            print(f"❌ Error loading DraftKings salaries: {e}")
            print("🔄 Using mock salary data...")
            self.salary_data = self._create_mock_salaries()
            return self.salary_data
    
    def _standardize_columns(self, df):
        """Standardize DraftKings column names"""
        column_mapping = {
            'Name': 'name',
            'Position': 'position',
            'Salary': 'salary',
            'Game Info': 'game_info',
            'TeamAbbrev': 'team',
            'AvgPointsPerGame': 'avg_points',
            'Roster Position': 'roster_position'
        }
        
        # Rename columns
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Clean position data
        if 'position' in df.columns:
            df['position'] = df['position'].astype(str)
            # Handle multi-position players (e.g., "PG/SG")
            df['positions'] = df['position'].str.split('/')
        
        return df
    
    def _create_mock_salaries(self):
        """Create mock DraftKings salary data for demonstration"""
        mock_data = [
            {'name': 'Luka Doncic', 'position': 'PG', 'salary': 11600, 'team': 'DAL', 'avg_points': 58.5},
            {'name': 'Nikola Jokic', 'position': 'C', 'salary': 11200, 'team': 'DEN', 'avg_points': 56.2},
            {'name': 'Shai Gilgeous-Alexander', 'position': 'PG/SG', 'salary': 10500, 'team': 'OKC', 'avg_points': 52.1},
            {'name': 'Jayson Tatum', 'position': 'SF/PF', 'salary': 9800, 'team': 'BOS', 'avg_points': 48.3},
            {'name': 'Stephen Curry', 'position': 'PG', 'salary': 9500, 'team': 'GSW', 'avg_points': 46.8},
            {'name': 'Anthony Davis', 'position': 'PF/C', 'salary': 9400, 'team': 'LAL', 'avg_points': 47.2},
            {'name': 'Tyrese Haliburton', 'position': 'PG/SG', 'salary': 9200, 'team': 'IND', 'avg_points': 45.9},
            {'name': 'Kevin Durant', 'position': 'SF/PF', 'salary': 8900, 'team': 'PHX', 'avg_points': 44.1},
            {'name': 'Devin Booker', 'position': 'SG', 'salary': 8700, 'team': 'PHX', 'avg_points': 43.5},
            {'name': 'LeBron James', 'position': 'SF/PF', 'salary': 8600, 'team': 'LAL', 'avg_points': 42.8},
            {'name': 'Domantas Sabonis', 'position': 'PF/C', 'salary': 8500, 'team': 'SAC', 'avg_points': 42.3},
            {'name': 'Trae Young', 'position': 'PG', 'salary': 8300, 'team': 'ATL', 'avg_points': 41.7},
            {'name': 'Paul George', 'position': 'SG/SF', 'salary': 7800, 'team': 'LAC', 'avg_points': 38.9},
            {'name': 'James Harden', 'position': 'PG/SG', 'salary': 7700, 'team': 'LAC', 'avg_points': 38.2},
            {'name': 'Karl-Anthony Towns', 'position': 'PF/C', 'salary': 7600, 'team': 'MIN', 'avg_points': 37.8},
            {'name': 'Zion Williamson', 'position': 'PF', 'salary': 7500, 'team': 'NOP', 'avg_points': 37.1},
            {'name': 'Jaylen Brown', 'position': 'SG/SF', 'salary': 7400, 'team': 'BOS', 'avg_points': 36.8},
            {'name': 'Jalen Brunson', 'position': 'PG', 'salary': 7300, 'team': 'NYK', 'avg_points': 36.2},
            {'name': 'Donovan Mitchell', 'position': 'SG', 'salary': 7200, 'team': 'CLE', 'avg_points': 35.9},
            {'name': 'Bam Adebayo', 'position': 'C', 'salary': 7100, 'team': 'MIA', 'avg_points': 35.4},
            {'name': 'Value Player 1', 'position': 'SG', 'salary': 4500, 'team': 'LAC', 'avg_points': 22.5},
            {'name': 'Value Player 2', 'position': 'SF', 'salary': 4200, 'team': 'GSW', 'avg_points': 21.0},
            {'name': 'Value Player 3', 'position': 'PF', 'salary': 3800, 'team': 'DEN', 'avg_points': 19.5},
            {'name': 'Punt Play 1', 'position': 'PG', 'salary': 3500, 'team': 'SAC', 'avg_points': 17.8},
            {'name': 'Punt Play 2', 'position': 'C', 'salary': 3400, 'team': 'OKC', 'avg_points': 17.2}
        ]
        return pd.DataFrame(mock_data)

class ConstraintOptimizer:
    def __init__(self, salary_cap=50000, max_players_per_team=4):
        self.salary_cap = salary_cap
        self.max_players_per_team = max_players_per_team
        self.solver = pulp.PULP_CBC_CMD(msg=0)  # Silent solver
        
    def optimize_lineup(self, players_df, num_lineups=1, exposure_limits=None):
        """Optimize lineups using constraint programming"""
        print(f"🧠 Optimizing {num_lineups} lineup(s) using constraint programming...")
        
        if players_df.empty:
            print("❌ No players data available")
            return []
        
        # Create player dictionary for faster access
        players = players_df.to_dict('records')
        
        lineups = []
        for lineup_num in range(num_lineups):
            print(f"  Building lineup {lineup_num + 1}/{num_lineups}...")
            
            # Create optimization problem
            prob = pulp.LpProblem(f"DFS_Lineup_{lineup_num}", pulp.LpMaximize)
            
            # Decision variables
            player_vars = pulp.LpVariable.dicts("Player", range(len(players)), 0, 1, pulp.LpBinary)
            
            # Objective function: maximize projected points
            prob += pulp.lpSum([player_vars[i] * players[i]['projection'] for i in range(len(players))])
            
            # Add constraints
            self._add_base_constraints(prob, player_vars, players)
            
            # Add exposure limits for multiple lineups
            if exposure_limits and lineup_num > 0:
                self._add_exposure_constraints(prob, player_vars, players, lineups, exposure_limits)
            
            # Solve the problem
            prob.solve(self.solver)
            
            if prob.status == pulp.LpStatusOptimal:
                lineup = self._extract_lineup(player_vars, players)
                lineups.append(lineup)
            else:
                print(f"❌ No solution found for lineup {lineup_num + 1}")
        
        return lineups
    
    def _add_base_constraints(self, prob, player_vars, players):
        """Add base DFS constraints"""
        
        # Total salary constraint
        prob += pulp.lpSum([player_vars[i] * players[i]['salary'] for i in range(len(players))]) <= self.salary_cap
        
        # Total players constraint (8 for DraftKings)
        prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == 8
        
        # Position constraints
        pg_players = [i for i, p in enumerate(players) if 'PG' in p['position']]
        sg_players = [i for i, p in enumerate(players) if 'SG' in p['position']]
        sf_players = [i for i, p in enumerate(players) if 'SF' in p['position']]
        pf_players = [i for i, p in enumerate(players) if 'PF' in p['position']]
        c_players = [i for i, p in enumerate(players) if 'C' in p['position']]
        
        # PG: at least 1
        prob += pulp.lpSum([player_vars[i] for i in pg_players]) >= 1
        # SG: at least 1  
        prob += pulp.lpSum([player_vars[i] for i in sg_players]) >= 1
        # SF: at least 1
        prob += pulp.lpSum([player_vars[i] for i in sf_players]) >= 1
        # PF: at least 1
        prob += pulp.lpSum([player_vars[i] for i in pf_players]) >= 1
        # C: exactly 1
        prob += pulp.lpSum([player_vars[i] for i in c_players]) == 1
        
        # Guard position (PG or SG): at least 3 total
        g_players = list(set(pg_players + sg_players))
        prob += pulp.lpSum([player_vars[i] for i in g_players]) >= 3
        
        # Forward position (SF or PF): at least 3 total  
        f_players = list(set(sf_players + pf_players))
        prob += pulp.lpSum([player_vars[i] for i in f_players]) >= 3
        
        # Utility position (any): exactly 1
        # This is handled by the total players constraint
        
        # Team constraints
        teams = list(set(p['team'] for p in players))
        for team in teams:
            team_players = [i for i, p in enumerate(players) if p['team'] == team]
            prob += pulp.lpSum([player_vars[i] for i in team_players]) <= self.max_players_per_team
    
    def _add_exposure_constraints(self, prob, player_vars, players, previous_lineups, exposure_limits):
        """Add exposure constraints for multiple lineup generation"""
        max_exposure = exposure_limits.get('max_player_exposure', 0.5)
        
        for i, player in enumerate(players):
            # Count how many times this player has been used
            usage_count = sum(1 for lineup in previous_lineups if player['name'] in [p['name'] for p in lineup])
            max_usage = max_exposure * (len(previous_lineups) + 1)
            
            if usage_count >= max_usage:
                # Limit this player's exposure
                prob += player_vars[i] == 0
    
    def _extract_lineup(self, player_vars, players):
        """Extract the optimal lineup from solution"""
        lineup_players = []
        total_salary = 0
        total_projection = 0
        
        for i in range(len(players)):
            if player_vars[i].value() == 1:
                player = players[i].copy()
                lineup_players.append(player)
                total_salary += player['salary']
                total_projection += player['projection']
        
        # Sort by position for display
        position_order = {'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5}
        lineup_players.sort(key=lambda x: min(position_order.get(pos, 6) for pos in x['position'].split('/')))
        
        return {
            'players': lineup_players,
            'total_salary': total_salary,
            'total_projection': total_projection,
            'efficiency': total_projection / total_salary * 1000
        }

class AdvancedDFSOptimizer:
    def __init__(self):
        self.dk_integration = DraftKingsIntegration()
        self.optimizer = ConstraintOptimizer()
        self.players_df = None
        
    def load_and_prepare_data(self, dk_salaries_path=None):
        """Load all data and prepare for optimization"""
        print("🔄 Loading and preparing data...")
        
        # Load DraftKings salaries
        dk_data = self.dk_integration.load_dk_salaries(dk_salaries_path)
        
        # Generate projections (in practice, use your projection model)
        self.players_df = self._generate_projections(dk_data)
        
        print(f"✅ Prepared {len(self.players_df)} players for optimization")
        return self.players_df
    
    def _generate_projections(self, dk_data):
        """Generate fantasy projections for players"""
        projections = []
        
        for _, player in dk_data.iterrows():
            # Enhanced projection logic using multiple factors
            base_projection = self._calculate_player_projection(player)
            
            # Add randomness for demo (replace with real projections)
            projection_variance = base_projection * 0.15  # ±15% variance
            final_projection = base_projection + np.random.uniform(-projection_variance, projection_variance)
            
            projections.append({
                'name': player['name'],
                'position': player['position'],
                'salary': player['salary'],
                'team': player['team'],
                'projection': max(final_projection, 1),  # Ensure positive
                'value': final_projection / player['salary'] * 1000
            })
        
        return pd.DataFrame(projections)
    
    def _calculate_player_projection(self, player):
        """Calculate base projection from salary and historical data"""
        # Base projection based on salary (simplified - use real model)
        base_from_salary = player['salary'] / 200  # $1000 ≈ 5 points
        
        # Add position bonuses
        position_bonus = {
            'C': 1.1,  # Centers get rebound bonus
            'PG': 1.05, # PGs get assist bonus
            'SG': 1.0,
            'SF': 1.0,
            'PF': 1.03
        }
        
        pos = player['position'].split('/')[0]  # Use primary position
        bonus = position_bonus.get(pos, 1.0)
        
        return base_from_salary * bonus
    
    def generate_optimal_lineups(self, num_lineups=3, strategy='balanced', exposure_limits=None):
        """Generate optimal lineups based on strategy"""
        print(f"\n🎯 Generating {num_lineups} {strategy} lineups...")
        
        if self.players_df is None:
            print("❌ No player data available. Load data first.")
            return []
        
        # Filter players based on strategy
        filtered_players = self._apply_strategy_filter(strategy)
        
        # Generate lineups
        lineups = self.optimizer.optimize_lineup(
            filtered_players, 
            num_lineups=num_lineups,
            exposure_limits=exposure_limits
        )
        
        return lineups
    
    def _apply_strategy_filter(self, strategy):
        """Filter players based on lineup strategy"""
        players = self.players_df.copy()
        
        if strategy == 'cash':
            # High floor, popular plays
            players = players[players['value'] > 4.0]  # Minimum 4x value
            players = players[players['projection'] > 20]  # Minimum projection
            
        elif strategy == 'tournament':
            # High ceiling, lower ownership
            players = players[players['projection'] > 25]  # Higher ceiling threshold
            # Add some contrarian plays
            contrarian_players = self.players_df[self.players_df['value'] > 5.0]
            players = pd.concat([players, contrarian_players]).drop_duplicates()
            
        elif strategy == 'balanced':
            # Mix of value and safety
            players = players[players['value'] > 3.5]
            
        return players
    
    def analyze_lineup(self, lineup):
        """Analyze and display a lineup"""
        print(f"\n📊 Lineup Analysis:")
        print(f"Total Salary: ${lineup['total_salary']:,.0f}")
        print(f"Total Projection: {lineup['total_projection']:.1f} points")
        print(f"Efficiency: {lineup['efficiency']:.2f} points per $1k")
        
        print(f"\n🏀 Lineup:")
        for i, player in enumerate(lineup['players'], 1):
            print(f"  {i:2d}. {player['position']:6} {player['name']:25} "
                  f"${player['salary']:5,} | {player['projection']:5.1f} pts | "
                  f"Value: {player['projection']/player['salary']*1000:4.2f}")
    
    def run_complete_optimization(self, dk_salaries_path=None, num_lineups=3):
        """Run complete optimization pipeline"""
        print("🚀 Starting Complete DFS Optimization Pipeline")
        print("=" * 60)
        
        # Load data
        self.load_and_prepare_data(dk_salaries_path)
        
        # Generate different lineup strategies
        strategies = ['cash', 'tournament', 'balanced']
        
        all_lineups = {}
        for strategy in strategies:
            print(f"\n{'='*50}")
            print(f"Generating {strategy.upper()} lineups...")
            print(f"{'='*50}")
            
            lineups = self.generate_optimal_lineups(
                num_lineups=num_lineups,
                strategy=strategy,
                exposure_limits={'max_player_exposure': 0.4}
            )
            
            all_lineups[strategy] = lineups
            
            for i, lineup in enumerate(lineups):
                print(f"\n🏆 {strategy.upper()} Lineup #{i+1}:")
                self.analyze_lineup(lineup)
        
        return all_lineups

# Usage Example
if __name__ == "__main__":
    # Initialize optimizer
    dfs_optimizer = AdvancedDFSOptimizer()
    
    # Run complete optimization
    # For real usage, provide path to DraftKings CSV:
    # results = dfs_optimizer.run_complete_optimization("path/to/dk_salaries.csv")
    
    results = dfs_optimizer.run_complete_optimization()
    
    # Display summary
    print(f"\n{'='*60}")
    print("📈 OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    
    for strategy, lineups in results.items():
        if lineups:
            best_lineup = max(lineups, key=lambda x: x['total_projection'])
            print(f"{strategy.upper():12} | Best: {best_lineup['total_projection']:6.1f} pts | "
                  f"Salary: ${best_lineup['total_salary']:5,} | "
                  f"Eff: {best_lineup['efficiency']:5.2f}")