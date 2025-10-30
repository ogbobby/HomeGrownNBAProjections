# config.py - Configuration for the integration

class IntegrationConfig:
    """Configuration for NBA-DFS-Tools integration"""
    
    # Paths
    NBA_DFS_TOOLS_PATH = "/path/to/NBA-DFS-Tools"
    OUTPUT_DIR = "./output"
    
    # Projection settings
    HYBRID_WEIGHTS = {
        'our_weight': 0.6,
        'their_weight': 0.4
    }
    
    # Optimizer settings
    OPTIMIZER_CONFIG = {
        'site': 'draftkings',
        'sport': 'nba',
        'num_lineups': 20,
        'overlap': 4,
        'salary_cap': 50000
    }
    
    # Data sources
    DATA_SOURCES = {
        'popcorn_machine': True,
        'basketball_reference': True,
        'vegas_odds': True,
        'dk_salaries': True
    }