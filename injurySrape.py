#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

def get_injuries():
    """Scrape ESPN NBA injury data and return a cleaned DataFrame."""
    
    def fetch_html(url):
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text    
    
    def parse_injuries(html):
        soup = BeautifulSoup(html, "html.parser")
        data = []   
        sections = soup.select("div.ResponsiveTable")   
        for section in sections:
            team_header = section.find_previous("h2")
            team_name = team_header.get_text(strip=True) if team_header else "Unknown Team"
            
            table = section.find("table")
            if not table:
                continue
            
            rows = table.find_all("tr")
            for row in rows[1:]:  # skip header row
                cols = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cols) < 4:
                    continue
                #player_name = cols[0].strip().strip("'\"")
                player_name = re.sub(r"^[‘'\"`]+|[’'\"`]+$", "", cols[0].strip())
                position = cols[1].strip()
                status = cols[3].strip()
                data.append({
                    #"Team": team_name,
                    "Player": player_name,
                    "Position": position,
                    "Status": status
                })
        return pd.DataFrame(data)
    
    html = fetch_html("https://www.espn.com/nba/injuries/_/1000")
    df = parse_injuries(html)
    return dict(zip(df["Player"], df["Status"]))

