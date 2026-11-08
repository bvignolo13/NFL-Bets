"""Team power ratings from game scores.

Model: each team has an offense rating (points scored above average) and a
defense rating (points allowed above average; lower is better). For every game:

    home points = avg + home_field/2 + home_off + away_def
    away points = avg - home_field/2 + away_off + home_def

The ratings are fit with weighted least squares, where recent games count more
(the weight halves every `half_life_days`). A small ridge penalty pulls teams
toward average so ratings are stable early in the season.

Power rating = offense - defense = expected point margin vs. an average team
on a neutral field.
"""

from datetime import date

import numpy as np
import pandas as pd


def fit_ratings(games, today=None, half_life_days=60, ridge=2.0, seasons_back=1):
    today = today or date.today()
    played = games[games["result"].notna()].copy()
    played = played[played["season"] >= played["season"].max() - seasons_back]
    if played.empty:
        raise ValueError("No completed games found to build ratings from.")

    teams = sorted(set(played["home_team"]) | set(played["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    days_ago = np.array([(today - d).days for d in played["gameday"]], dtype=float)
    game_w = 0.5 ** (np.maximum(days_ago, 0) / half_life_days)
    is_home = (played["location"] != "Neutral").to_numpy(dtype=float)

    # Columns: [intercept, home field (+/- 0.5), off_0..off_n-1, def_0..def_n-1]
    m = len(played)
    X = np.zeros((2 * m, 2 + 2 * n))
    y = np.zeros(2 * m)
    w = np.repeat(game_w, 2)
    for k, (h, a, hs, as_) in enumerate(
        played[["home_team", "away_team", "home_score", "away_score"]].itertuples(index=False)
    ):
        # Row for the home team's points
        r = 2 * k
        X[r, 0], X[r, 1] = 1, 0.5 * is_home[k]
        X[r, 2 + idx[h]], X[r, 2 + n + idx[a]] = 1, 1
        y[r] = hs
        # Row for the away team's points
        r += 1
        X[r, 0], X[r, 1] = 1, -0.5 * is_home[k]
        X[r, 2 + idx[a]], X[r, 2 + n + idx[h]] = 1, 1
        y[r] = as_

    penalty = np.full(X.shape[1], ridge)
    penalty[:2] = 0  # don't shrink the league average or home field
    Xw = X * w[:, None]
    coef = np.linalg.solve(Xw.T @ X + np.diag(penalty), Xw.T @ y)

    off = coef[2 : 2 + n]
    deff = coef[2 + n :]
    # Center so 0 = league average; shift the intercept to compensate.
    avg = coef[0] + off.mean() + deff.mean()
    off, deff = off - off.mean(), deff - deff.mean()

    table = pd.DataFrame({"team": teams, "offense": off, "defense": deff})
    table["power"] = table["offense"] - table["defense"]
    table = table.sort_values("power", ascending=False).reset_index(drop=True)
    return {"ratings": table.set_index("team"), "avg_points": avg, "home_field": coef[1]}


def predict_game(model, home, away, neutral=False):
    """Return (home spread, total). Spread uses sportsbook sign: negative = home favored."""
    r = model["ratings"]
    hfa = 0.0 if neutral else model["home_field"]
    home_pts = model["avg_points"] + hfa / 2 + r.at[home, "offense"] + r.at[away, "defense"]
    away_pts = model["avg_points"] - hfa / 2 + r.at[away, "offense"] + r.at[home, "defense"]
    return -(home_pts - away_pts), home_pts + away_pts
