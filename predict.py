"""Predict this week's NFL spreads/totals and flag games that differ from the sportsbook line.

Usage:
    export ODDS_API_KEY=your_key_here
    python predict.py
"""

import argparse
import os
import sys

import pandas as pd

from data import (
    NFLVERSE_GAMES_URL,
    fetch_odds,
    load_games,
    load_odds_file,
    parse_odds,
    this_weeks_games,
)
from ratings import fit_ratings, predict_game


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", type=float, default=2.0, help="Flag games where model and book differ by more than this many points (default 2)")
    p.add_argument("--bookmaker", help="Use one sportsbook (e.g. draftkings, fanduel) instead of the median of all books")
    p.add_argument("--games-csv", default=NFLVERSE_GAMES_URL, help="nflverse games.csv URL or local path")
    p.add_argument("--odds-file", help="Read odds from a saved Odds API JSON file instead of calling the API")
    p.add_argument("--save-odds", help="Save the raw Odds API response to this file (reuse it with --odds-file)")
    p.add_argument("--out", help="Also write the full prediction table to this CSV file")
    args = p.parse_args()

    # 1. Team data from nflverse -> power ratings
    print("Downloading games from nflverse...")
    games = load_games(args.games_csv)
    model = fit_ratings(games)
    print(f"\nTEAM POWER RATINGS  (points vs. average team; home field = {model['home_field']:.1f})")
    print(model["ratings"].round(1).to_string())

    # 2. This week's schedule + model predictions
    week = this_weeks_games(games)
    if week.empty:
        sys.exit("No upcoming games found in the nflverse schedule.")
    season, wk = int(week["season"].iloc[0]), int(week["week"].iloc[0])

    preds = []
    for g in week.itertuples():
        spread, total = predict_game(model, g.home_team, g.away_team, neutral=(g.location == "Neutral"))
        preds.append({"gameday": g.gameday, "away_team": g.away_team, "home_team": g.home_team,
                      "model_spread": spread, "model_total": total})
    preds = pd.DataFrame(preds)

    # 3. Sportsbook lines from The Odds API
    if args.odds_file:
        events = load_odds_file(args.odds_file)
    else:
        api_key = os.environ.get("ODDS_API_KEY")
        if not api_key:
            sys.exit("Set the ODDS_API_KEY environment variable (free key at https://the-odds-api.com).")
        print("\nDownloading odds from The Odds API...")
        events = fetch_odds(api_key)
        if args.save_odds:
            import json
            with open(args.save_odds, "w") as f:
                json.dump(events, f)
    lines = parse_odds(events, bookmaker=args.bookmaker)

    df = preds.merge(lines, on=["home_team", "away_team"], how="left")
    df["spread_diff"] = df["model_spread"] - df["book_spread"]
    df["total_diff"] = df["model_total"] - df["book_total"]

    print(f"\nWEEK {wk} {season} PREDICTIONS  (spread is for the home team; negative = home favored)")
    show = df[["gameday", "away_team", "home_team", "model_spread", "book_spread", "spread_diff",
               "model_total", "book_total", "total_diff"]]
    print(show.round(1).to_string(index=False, na_rep="-"))

    # 4. Flag the differences
    print(f"\nGAMES WHERE MODEL DIFFERS FROM THE BOOK BY MORE THAN {args.threshold:g} POINTS")
    found = False
    for r in df.itertuples():
        matchup = f"{r.away_team} @ {r.home_team}"
        if pd.notna(r.spread_diff) and abs(r.spread_diff) > args.threshold:
            # Model spread lower than book spread => model likes the home team more.
            if r.spread_diff < 0:
                pick = f"{r.home_team} {r.book_spread:+g}"
            else:
                pick = f"{r.away_team} {-r.book_spread:+g}"
            print(f"  SPREAD  {matchup:<12} model {r.model_spread:+.1f} vs book {r.book_spread:+g}  "
                  f"(diff {abs(r.spread_diff):.1f}) -> lean {pick}")
            found = True
        if pd.notna(r.total_diff) and abs(r.total_diff) > args.threshold:
            side = "OVER" if r.total_diff > 0 else "UNDER"
            print(f"  TOTAL   {matchup:<12} model {r.model_total:.1f} vs book {r.book_total:g}  "
                  f"(diff {abs(r.total_diff):.1f}) -> lean {side} {r.book_total:g}")
            found = True
    if not found:
        print("  None this week.")
    missing = df["book_spread"].isna().sum()
    if missing:
        print(f"\n({missing} game(s) had no sportsbook line — game already started, or line not posted yet.)")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved full table to {args.out}")


if __name__ == "__main__":
    main()
