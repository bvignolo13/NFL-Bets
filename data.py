"""Download data: game results/schedule from nflverse, betting lines from The Odds API."""

import json
import statistics
from datetime import date

import pandas as pd
import requests

from teams import NFLVERSE_ALIASES, ODDS_API_TO_NFLVERSE

NFLVERSE_GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"


def load_games(source=NFLVERSE_GAMES_URL):
    """Load every NFL game (past scores + future schedule) from nflverse.

    `source` can be a URL or a local CSV path.
    """
    games = pd.read_csv(source, low_memory=False)
    games["gameday"] = pd.to_datetime(games["gameday"]).dt.date
    for col in ("home_team", "away_team"):
        games[col] = games[col].replace(NFLVERSE_ALIASES)
    return games


def this_weeks_games(games, today=None):
    """Return the unplayed games in the next upcoming week."""
    today = today or date.today()
    upcoming = games[games["result"].isna() & (games["gameday"] >= today)]
    if upcoming.empty:
        return upcoming
    first = upcoming.sort_values("gameday").iloc[0]
    return upcoming[(upcoming["season"] == first["season"]) & (upcoming["week"] == first["week"])]


def fetch_odds(api_key, regions="us"):
    """Download current NFL spreads and totals (raw JSON list of events)."""
    resp = requests.get(
        ODDS_API_URL,
        params={
            "apiKey": api_key,
            "regions": regions,
            "markets": "spreads,totals",
            "oddsFormat": "american",
        },
        timeout=30,
    )
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining")
    if remaining is not None:
        print(f"(The Odds API requests remaining this month: {remaining})")
    return resp.json()


def load_odds_file(path):
    with open(path) as f:
        return json.load(f)


def parse_odds(events, bookmaker=None):
    """Turn Odds API events into one row per game with the home spread and total.

    The home spread uses sportsbook convention: -3.5 means the home team is
    favored by 3.5. When several books are available, the median line is used
    unless a specific `bookmaker` key (e.g. "draftkings") is requested.
    """
    rows = []
    for ev in events:
        home = ODDS_API_TO_NFLVERSE.get(ev["home_team"])
        away = ODDS_API_TO_NFLVERSE.get(ev["away_team"])
        if home is None or away is None:
            continue
        spreads, totals = [], []
        for book in ev.get("bookmakers", []):
            if bookmaker and book["key"] != bookmaker:
                continue
            for market in book.get("markets", []):
                for outcome in market.get("outcomes", []):
                    if market["key"] == "spreads" and outcome["name"] == ev["home_team"]:
                        spreads.append(outcome["point"])
                    elif market["key"] == "totals" and outcome["name"] == "Over":
                        totals.append(outcome["point"])
        rows.append(
            {
                "home_team": home,
                "away_team": away,
                "book_spread": statistics.median(spreads) if spreads else None,
                "book_total": statistics.median(totals) if totals else None,
            }
        )
    return pd.DataFrame(rows, columns=["home_team", "away_team", "book_spread", "book_total"])
