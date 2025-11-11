#!/usr/bin/env python3
"""
Scrape ESPN NBA Injuries page: https://www.espn.com/nba/injuries/_/1000
Outputs only Player, Position, and Status (plus Team for context).
"""

import requests
from bs4 import BeautifulSoup
import csv

URL = "https://www.espn.com/nba/injuries/_/1000"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
}
OUTPUT_FILE = "nba_injuries_clean.csv"


def fetch_html(url):
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def parse_injuries(html):
    soup = BeautifulSoup(html, "html.parser")
    data = []

    # Each team is wrapped in a div.ResponsiveTable
    sections = soup.select("div.ResponsiveTable")

    for section in sections:
        # Find the team name (appears in previous h2)
        team_header = section.find_previous("h2")
        team_name = team_header.get_text(strip=True) if team_header else "Unknown Team"

        table = section.find("table")
        if not table:
            continue

        rows = table.find_all("tr")
        for row in rows[1:]:  # skip header
            cols = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cols) < 4:
                continue

            player_name = cols[0]
            position = cols[1]
            status = cols[3]  # typically the 4th column: "Status"

            data.append({
                "Team": team_name,
                "Player": player_name,
                "Position": position,
                "Status": status
            })

    return data


def save_csv(data, filename):
    if not data:
        print("⚠️ No injury data found.")
        return

    keys = ["Team", "Player", "Position", "Status"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

    print(f"✅ Saved {len(data)} injury entries to {filename}")


def main():
    html = fetch_html(URL)
    data = parse_injuries(html)
    save_csv(data, OUTPUT_FILE)


if __name__ == "__main__":
    main()
