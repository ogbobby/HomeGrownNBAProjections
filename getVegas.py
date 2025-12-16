import requests
from bs4 import BeautifulSoup
import re

def get_nba_odds():
    """
    Scrapes NBA odds from VegasInsider with proper identification of spreads vs totals.
    """
    url = 'https://www.vegasinsider.com/nba/odds/las-vegas/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the main table
        tables = soup.find_all('table')
        if not tables:
            return []
        
        main_table = tables[0]
        rows = main_table.find_all('tr')
        
        ## Debug: Show raw data structure
        #print("DEBUG: Raw table data structure:")
        #for i, row in enumerate(rows):
        #    cells = row.find_all('td')
        #    if cells:
        #        row_data = [cell.get_text(strip=True) for cell in cells]
        #        print(f"Row {i}: {row_data}")
        #
        #print("\n" + "="*60 + "\n")
        
        # The table has this structure:
        # Row 0: Headers
        # Row 1: Team 1 spread data
        # Row 2: Team 2 spread data  
        # Row 3: "MatchupView Picks3" separator
        # Row 4: Team 1 total data
        # Row 5: Team 2 total data
        # Row 6: "MatchupView Picks3" separator
        # Row 7: Team 1 moneyline data
        # Row 8: Team 2 moneyline data
        
        games_data = {}
        current_teams = []
        
        for i, row in enumerate(rows):
            cells = row.find_all('td')
            if len(cells) < 2:
                continue
            
            # Get the team identifier from first cell
            first_cell = cells[0].get_text(strip=True)
            
            # Skip separator rows
            if 'Matchup' in first_cell or 'View' in first_cell:
                continue
            
            # Extract team number and name
            match = re.match(r'(\d+)(\D+)', first_cell)
            if not match:
                continue
                
            team_num = match.group(1)
            team_name = match.group(2).strip()
            
            # Based on the row index, determine what type of data this is
            # Spread rows come first, then totals, then moneylines
            row_type = None
            
            # Check row content to determine type
            row_text = ' '.join([cell.get_text(strip=True) for cell in cells])
            
            if any(cell.get_text(strip=True).lower().startswith(('o', 'u')) for cell in cells):
                row_type = 'total'
            elif any('+' in cell.get_text(strip=True) or '-' in cell.get_text(strip=True) for cell in cells[1:]):
                # Check if it's spread or moneyline
                spread_found = False
                for cell in cells[1:]:
                    text = cell.get_text(strip=True)
                    if text and '.' in text and ('+' in text or '-' in text):
                        spread_found = True
                        break
                
                if spread_found:
                    row_type = 'spread'
                else:
                    # Look for pure moneyline values without decimals
                    moneyline_found = False
                    for cell in cells[1:]:
                        text = cell.get_text(strip=True)
                        if text and (text.startswith('+') or text.startswith('-')) and '.' not in text:
                            moneyline_found = True
                            break
                    
                    if moneyline_found:
                        row_type = 'moneyline'
            
            # Initialize team data if not exists
            if team_name not in games_data:
                games_data[team_name] = {
                    'team_name': team_name,
                    'spread': None,
                    'total': None,
                    'moneyline': None,
                    'team_number': team_num
                }
            
            # Extract the appropriate value based on row type
            # The main value is usually in the second column (index 1)
            if len(cells) > 1:
                main_value = cells[1].get_text(strip=True)
                
                if row_type == 'spread' and main_value:
                    games_data[team_name]['spread'] = main_value
                elif row_type == 'total' and main_value:
                    games_data[team_name]['total'] = main_value
                elif row_type == 'moneyline' and main_value:
                    games_data[team_name]['moneyline'] = main_value
                
                # Debug output
                #print(f"DEBUG: Row {i} - Team: {team_name} - Type: {row_type} - Value: {main_value}")
        
        # Pair teams based on team numbers (501 & 502, 503 & 504, etc.)
        paired_games = []
        
        # Group by game (teams with consecutive numbers)
        teams_by_num = {}
        for team_name, data in games_data.items():
            team_num = data['team_number']
            teams_by_num[team_num] = team_name
        
        # Sort team numbers and pair them
        sorted_nums = sorted(teams_by_num.keys(), key=int)
        
        for i in range(0, len(sorted_nums), 2):
            if i + 1 < len(sorted_nums):
                team1_num = sorted_nums[i]
                team2_num = sorted_nums[i + 1]
                
                team1_name = teams_by_num[team1_num]
                team2_name = teams_by_num[team2_num]
                
                game = {
                    'game_number': len(paired_games) + 1,
                    'team1': {
                        'team_name': team1_name,
                        'spread': games_data[team1_name]['spread'],
                        'total': games_data[team1_name]['total'],
                        'moneyline': games_data[team1_name]['moneyline']
                    },
                    'team2': {
                        'team_name': team2_name,
                        'spread': games_data[team2_name]['spread'],
                        'total': games_data[team2_name]['total'],
                        'moneyline': games_data[team2_name]['moneyline']
                    },
                    'matchup': f"{team1_name} vs {team2_name}"
                }
                paired_games.append(game)
        
        return paired_games
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return []