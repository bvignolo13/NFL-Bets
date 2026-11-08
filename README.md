# NFL Bets

A small Python project that:

1. Downloads NFL game results and the schedule from **nflverse**
2. Builds **team power ratings** (offense, defense and overall)
3. Predicts the **spread and total** for this week's games
4. Downloads current lines from **The Odds API**
5. Lists games where the prediction differs from the sportsbook line by **more than 2 points**

## Setup

You need Python 3.9 or newer.

```bash
git clone https://github.com/bvignolo13/NFL-Bets.git
cd NFL-Bets
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at <https://the-odds-api.com> (the free tier gives 500 requests a month; one run of this script uses 2).

## Run it

```bash
export ODDS_API_KEY=your_key_here    # Windows PowerShell: $env:ODDS_API_KEY="your_key_here"
python predict.py
```

The script prints three things:

- **Team power ratings**: points better or worse than an average team on a neutral field
- **This week's predictions**: the model's spread and total next to the sportsbook's
- **Flagged games**: where the model and the book differ by more than 2 points, with the side the model leans toward

Example of the flagged list:

```
GAMES WHERE MODEL DIFFERS FROM THE BOOK BY MORE THAN 2 POINTS
  SPREAD  NYJ @ KC     model -4.7 vs book -9.5  (diff 4.8) -> lean NYJ +9.5
  TOTAL   GB @ NE      model 39.6 vs book 44.5  (diff 4.9) -> lean UNDER 44.5
```

Spreads are always shown for the **home team**, using sportsbook signs: `-9.5` means the home team is favored by 9.5.

### Options

| Option | What it does |
|---|---|
| `--threshold 3` | Flag differences larger than 3 points instead of 2 |
| `--bookmaker draftkings` | Use one book's lines instead of the median across all US books |
| `--save-odds odds.json` | Save the odds you downloaded |
| `--odds-file odds.json` | Reuse saved odds instead of calling the API, which saves your API quota |
| `--out predictions.csv` | Also write the full table to a CSV file |

## How the model works (`ratings.py`)

Each team gets an **offense** rating (points scored above average) and a **defense** rating (points allowed above average, so lower is better). Every completed game from this season and last season is modeled as:

```
home points = league avg + home field/2 + home offense + away defense
away points = league avg - home field/2 + away offense + home defense
```

The ratings come from a weighted least-squares fit:

- **Recent games count more.** A game's weight halves every 60 days, so last season matters early in the year and fades as this season goes on.
- **A small ridge penalty** pulls every team toward average, which keeps early-season ratings from swinging too much.
- **Home field** is estimated from the data, and it is set to zero for neutral-site games.

**Power rating = offense − defense.** Predicted spread = −(home points − away points), and predicted total = home points + away points.

To change the model, adjust `half_life_days`, `ridge` and `seasons_back` in `fit_ratings()`.

## Files

| File | Purpose |
|---|---|
| `predict.py` | Main script: run this |
| `data.py` | Downloads the nflverse games and the Odds API lines |
| `ratings.py` | Power-rating model and game predictions |
| `teams.py` | Maps Odds API team names to nflverse abbreviations |
| `tests/` | Basic tests (`pip install pytest && pytest`) |

## Caveats

This is a simple, score-only model. It doesn't know about injuries, quarterback changes, weather or rest, and a sportsbook line already accounts for all of those. Treat flagged games as a starting point for your own research, not as automatic bets.
