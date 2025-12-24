# nba_dfs_integration.py

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Import our existing components
from nba_data_integrator import NBADataIntegrator, EnhancedDFSOptimizer
from dk_optimizer import DraftKingsIntegration, ConstraintOptimizer

class NBA_DFS_Tools_Integration:
    def __init__(self, nba_dfs_tools_path):
        """
        Integrate with NBA-DFS-Tools repository
        
        Args:
            nba_dfs_tools_path (str): Path to the NBA-DFS-Tools repository
        """
        self.nba_dfs_tools_path = nba_dfs_tools_path
        self.setup_imports()
        
        # Initialize our components
        self.data_integrator = NBADataIntegrator()
        self.dk_integration = DraftKingsIntegration()
        self.optimizer = EnhancedDFSOptimizer()
        
    def setup_imports(self):
        """Add NBA-DFS-Tools to Python path and import key modules"""
        try:
            sys.path.insert(0, self.nba_dfs_tools_path)
            
            # Import key modules from NBA-DFS-Tools
            global Projections, Optimizer
            from projections import Projections
            from optimizer import Optimizer
            
            print("✅ Successfully imported NBA-DFS-Tools modules")
            
        except ImportError as e:
            print(f"❌ Error importing NBA-DFS-Tools: {e}")
            print("Please ensure the repository is cloned and path is correct")
            
    def generate_projections_for_nba_dfs_tools(self, dk_salaries_path=None):
        """
        Generate projections in the format expected by NBA-DFS-Tools
        
        Returns:
            DataFrame: Projections formatted for NBA-DFS-Tools
        """
        print("🔄 Generating projections for NBA-DFS-Tools...")
        
        # Load our data and generate projections
        players_df = self.optimizer.load_and_prepare_data(dk_salaries_path)
        
        # Convert to NBA-DFS-Tools format
        projections_df = self._convert_to_nba_dfs_tools_format(players_df)
        
        print(f"✅ Generated {len(projections_df)} projections for NBA-DFS-Tools")
        return projections_df
    
    def _convert_to_nba_dfs_tools_format(self, players_df):
        """
        Convert our projection format to NBA-DFS-Tools expected format
        """
        formatted_data = []
        
        for _, player in players_df.iterrows():
            # Map our data to their expected column names
            formatted_player = {
                'Player': player['name'],
                'Position': player['position'],
                'Team': player['team'],
                'Salary': player['salary'],
                'Projection': player['projection'],
                'Value': player['value'],
                # Additional fields that NBA-DFS-Tools might expect
                'Minutes': self._estimate_minutes(player),
                'Ceiling': player['projection'] * 1.3,  # 30% ceiling boost
                'Floor': player['projection'] * 0.7,    # 30% floor reduction
                'Ownership': self._estimate_ownership(player),
                'Uncertainty': np.random.uniform(0.1, 0.4)  # Projection uncertainty
            }
            formatted_data.append(formatted_player)
        
        return pd.DataFrame(formatted_data)
    
    def _estimate_minutes(self, player):
        """Estimate minutes based on salary and position"""
        base_minutes = player['salary'] / 300  # Rough estimate
        position_boost = {
            'PG': 2, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 2
        }
        
        primary_pos = player['position'].split('/')[0]
        boost = position_boost.get(primary_pos, 0)
        
        return min(38, max(20, base_minutes + boost))
    
    def _estimate_ownership(self, player):
        """Estimate ownership percentage based on value and salary"""
        value_score = player['value']
        salary = player['salary']
        
        # High value + low salary = high ownership
        if value_score > 6.0 and salary < 6000:
            return np.random.uniform(0.2, 0.4)
        elif value_score > 5.0:
            return np.random.uniform(0.1, 0.25)
        else:
            return np.random.uniform(0.05, 0.15)
    
    def create_nba_dfs_tools_projections_file(self, output_path=None):
        """
        Create a projections file that NBA-DFS-Tools can consume
        """
        if output_path is None:
            output_path = Path(self.nba_dfs_tools_path) / "data" / "custom_projections.csv"
        
        projections_df = self.generate_projections_for_nba_dfs_tools()
        
        # Save in format expected by NBA-DFS-Tools
        projections_df.to_csv(output_path, index=False)
        print(f"💾 Saved projections to: {output_path}")
        
        return output_path
    
    def run_nba_dfs_tools_optimizer(self, projections_file=None, num_lineups=20, overlap=4):
        """
        Run the NBA-DFS-Tools optimizer with our projections
        """
        try:
            if projections_file is None:
                projections_file = self.create_nba_dfs_tools_projections_file()
            
            # Initialize NBA-DFS-Tools optimizer
            dfs_optimizer = Optimizer(
                site='draftkings',
                sport='nba',
                projections_path=projections_file
            )
            
            # Generate lineups using their optimizer
            print(f"🎯 Running NBA-DFS-Tools optimizer for {num_lineups} lineups...")
            lineups = dfs_optimizer.optimize(
                num_lineups=num_lineups,
                overlap=overlap
            )
            
            return lineups
            
        except Exception as e:
            print(f"❌ Error running NBA-DFS-Tools optimizer: {e}")
            return None
    
    def compare_optimizers(self, dk_salaries_path=None, num_lineups=5):
        """
        Compare our optimizer vs NBA-DFS-Tools optimizer
        """
        print("🔍 Comparing Optimizers...")
        print("=" * 60)
        
        # Generate projections once
        projections_df = self.generate_projections_for_nba_dfs_tools(dk_salaries_path)
        
        # 1. Run our optimizer
        print("\n🧠 Our Optimizer Results:")
        print("-" * 30)
        our_lineups = self.optimizer.generate_optimal_lineups(
            num_lineups=num_lineups,
            strategy='balanced'
        )
        
        for i, lineup in enumerate(our_lineups):
            print(f"Lineup {i+1}: {lineup['total_projection']:.1f} pts | "
                  f"${lineup['total_salary']:,} | Eff: {lineup['efficiency']:.2f}")
        
        # 2. Run NBA-DFS-Tools optimizer
        print("\n🛠️ NBA-DFS-Tools Optimizer Results:")
        print("-" * 40)
        their_lineups = self.run_nba_dfs_tools_optimizer(num_lineups=num_lineups)
        
        if their_lineups:
            for i, lineup in enumerate(their_lineups):
                total_proj = sum(p['Projection'] for p in lineup)
                total_salary = sum(p['Salary'] for p in lineup)
                efficiency = total_proj / total_salary * 1000
                print(f"Lineup {i+1}: {total_proj:.1f} pts | "
                      f"${total_salary:,} | Eff: {efficiency:.2f}")
        
        return {
            'our_lineups': our_lineups,
            'their_lineups': their_lineups,
            'projections': projections_df
        }

# Enhanced version that integrates directly with NBA-DFS-Tools projection system
class HybridProjectionSystem:
    def __init__(self, nba_dfs_tools_path):
        self.nba_dfs_tools_path = nba_dfs_tools_path
        self.setup_imports()
        
    def setup_imports(self):
        """Import NBA-DFS-Tools projection system"""
        try:
            sys.path.insert(0, self.nba_dfs_tools_path)
            
            # Import their projection system
            global Projections
            from projections import Projections
            
            self.nba_dfs_projections = Projections()
            print("✅ Loaded NBA-DFS-Tools projection system")
            
        except ImportError as e:
            print(f"❌ Error importing NBA-DFS-Tools projections: {e}")
            self.nba_dfs_projections = None
    
    def generate_hybrid_projections(self, dk_salaries_path=None):
        """
        Combine our projections with NBA-DFS-Tools projections
        for improved accuracy
        """
        print("🔄 Generating hybrid projections...")
        
        # 1. Get our projections
        our_optimizer = EnhancedDFSOptimizer()
        our_players_df = our_optimizer.load_and_prepare_data(dk_salaries_path)
        
        # 2. Get NBA-DFS-Tools projections if available
        their_projections = None
        if self.nba_dfs_projections:
            try:
                their_projections = self.nba_dfs_projections.get_projections()
                print("✅ Loaded NBA-DFS-Tools projections")
            except:
                print("⚠️ Could not load NBA-DFS-Tools projections")
        
        # 3. Combine projections
        hybrid_projections = self._combine_projections(our_players_df, their_projections)
        
        return hybrid_projections
    
    def _combine_projections(self, our_projections, their_projections):
        """
        Combine our projections with NBA-DFS-Tools projections
        using weighted average
        """
        hybrid_data = []
        
        for _, our_player in our_projections.iterrows():
            player_name = our_player['name']
            
            # Try to find matching player in their projections
            their_player_proj = None
            if their_projections is not None:
                matching_players = their_projections[
                    their_projections['Player'].str.contains(player_name, case=False, na=False)
                ]
                if not matching_players.empty:
                    their_player_proj = matching_players.iloc[0]['Projection']
            
            # Calculate hybrid projection
            if their_player_proj is not None:
                # Use weighted average (60% our projection, 40% theirs)
                hybrid_projection = (our_player['projection'] * 0.6) + (their_player_proj * 0.4)
                source = "hybrid"
            else:
                hybrid_projection = our_player['projection']
                source = "our_only"
            
            hybrid_player = {
                'Player': player_name,
                'Position': our_player['position'],
                'Team': our_player['team'],
                'Salary': our_player['salary'],
                'Projection': hybrid_projection,
                'Our_Projection': our_player['projection'],
                'Their_Projection': their_player_proj,
                'Source': source,
                'Value': hybrid_projection / our_player['salary'] * 1000
            }
            hybrid_data.append(hybrid_player)
        
        return pd.DataFrame(hybrid_data)

# Usage example and main function
def main():
    # Path to your NBA-DFS-Tools repository
    NBA_DFS_TOOLS_PATH = "/path/to/NBA-DFS-Tools"  # Update this path
    
    if not os.path.exists(NBA_DFS_TOOLS_PATH):
        print("❌ NBA-DFS-Tools path not found. Please update the path.")
        return
    
    # Initialize integration
    integration = NBA_DFS_Tools_Integration(NBA_DFS_TOOLS_PATH)
    
    # Option 1: Generate projections for NBA-DFS-Tools
    print("1. Generating projections for NBA-DFS-Tools...")
    projections_file = integration.create_nba_dfs_tools_projections_file()
    
    # Option 2: Run NBA-DFS-Tools optimizer with our projections
    print("\n2. Running NBA-DFS-Tools optimizer...")
    lineups = integration.run_nba_dfs_tools_optimizer(
        projections_file=projections_file,
        num_lineups=10,
        overlap=3
    )
    
    # Option 3: Compare both optimizers
    print("\n3. Comparing optimizers...")
    results = integration.compare_optimizers(num_lineups=5)
    
    # Option 4: Use hybrid projection system
    print("\n4. Using hybrid projection system...")
    hybrid_system = HybridProjectionSystem(NBA_DFS_TOOLS_PATH)
    hybrid_projections = hybrid_system.generate_hybrid_projections()
    
    print(f"\n🎉 Integration complete! Generated {len(hybrid_projections)} hybrid projections")
    
    # Save hybrid projections
    hybrid_projections.to_csv("hybrid_projections.csv", index=False)
    print("💾 Saved hybrid projections to: hybrid_projections.csv")

if __name__ == "__main__":
    main()