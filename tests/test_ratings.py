import os
import sys
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import parse_odds  # noqa: E402
from ratings import fit_ratings, predict_game  # noqa: E402


def test_better_team_is_favored():
    rows = []
    for i in range(6):
        rows.append({"season": 2026, "gameday": date(2026, 9, 7) + timedelta(weeks=i), "location": "Home",
                     "home_team": "KC", "away_team": "LV", "home_score": 31, "away_score": 13, "result": 18})
        rows.append({"season": 2026, "gameday": date(2026, 9, 8) + timedelta(weeks=i), "location": "Home",
                     "home_team": "LV", "away_team": "KC", "home_score": 17, "away_score": 27, "result": -10})
    model = fit_ratings(pd.DataFrame(rows), today=date(2026, 11, 1))
    spread, total = predict_game(model, "KC", "LV")
    assert spread < -5          # KC clearly favored at home
    assert 35 < total < 55
    assert model["ratings"].at["KC", "power"] > model["ratings"].at["LV", "power"]


def test_parse_odds_median_home_spread():
    events = [{
        "home_team": "Kansas City Chiefs", "away_team": "New York Jets",
        "bookmakers": [
            {"key": "a", "markets": [
                {"key": "spreads", "outcomes": [{"name": "Kansas City Chiefs", "point": -9.5},
                                                {"name": "New York Jets", "point": 9.5}]},
                {"key": "totals", "outcomes": [{"name": "Over", "point": 47.5},
                                               {"name": "Under", "point": 47.5}]}]},
            {"key": "b", "markets": [
                {"key": "spreads", "outcomes": [{"name": "Kansas City Chiefs", "point": -8.5},
                                                {"name": "New York Jets", "point": 8.5}]}]},
        ],
    }]
    df = parse_odds(events)
    assert df.iloc[0]["home_team"] == "KC"
    assert df.iloc[0]["book_spread"] == -9.0
    assert df.iloc[0]["book_total"] == 47.5
