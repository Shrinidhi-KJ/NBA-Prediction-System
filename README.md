# 🏀 NBA Champion Predictor

A machine-learning system that estimates every team's probability of winning
the NBA championship.

Predicting *the* champion directly is a dead end — there's only one winner per
year, so almost no training signal. Instead this system uses the standard,
robust approach: **rate team strength → model each playoff series → simulate
the whole bracket thousands of times.** The share of simulations a team wins
is its championship probability.

## How it works

| Stage | What it does |
|-------|--------------|
| **1. Data** | Pulls every regular-season and playoff game (2010-11 → present) from the [balldontlie](https://www.balldontlie.io) API. |
| **2. Team strength** | Computes FiveThirtyEight-style **Elo ratings** game-by-game: +100 home-court advantage, a margin-of-victory multiplier, and 75% carry-over between seasons. |
| **3. Series model** | A logistic regression that maps the **Elo gap entering a series** to *P(home-court team wins the best-of-7)*, validated leave-one-season-out. |
| **4. Bracket simulation** | Builds the 16-team bracket (seeded by regular-season record per conference) and runs a **Monte Carlo simulation** (50k runs) to produce title odds. |

## Does it actually work?

Back-tested across all 16 seasons (`--backtest`), simulating each bracket:

- The eventual champion is assigned **20.1% mean title odds** vs the **6.25%**
  naive baseline (1/16) — **3.2× better than chance**.
- The champion was the model's **#1 favorite in 6/16 seasons (38%)** — random ≈ 6%.
- The champion finished in the model's **top 4 in 11/16 seasons**.

Sanity checks land where they should: **OKC favored at 40% for 2024-25** (the
actual champion), and Golden State favored in 2014-15 and 2016-17 (both won).
It also *correctly fails* to predict genuine upsets (2015-16 Cleveland over the
73-9 Warriors), which is the honest signature of a model that isn't overfit.

**Series-model validation (leave-one-season-out, n = 240 series):**

| Metric | Home-court baseline | Model |
|--------|--------------------:|------:|
| Log-loss | 0.596 | **0.583** |
| Brier | 0.203 | **0.198** |
| AUC | 0.50 | **0.609** |

AUC of ~0.61 is modest by design — playoff series are inherently high-variance.
What matters for the simulation is *calibrated probabilities*, and the model
beats the baseline on both log-loss and Brier.

## Project structure

```
NBA-Playoff-Predictions/
├── src/
│   ├── data/
│   │   └── collect_games_balldontlie.py   # Phase 1 — pull games
│   ├── features/
│   │   ├── build_elo.py                    # Phase 2 — Elo + team-season strength
│   │   └── build_series.py                 # Phase 3a — reconstruct best-of-7 series
│   ├── models/
│   │   └── train_series_model.py           # Phase 3b — train + validate series model
│   └── simulation/
│       ├── bracket.py                      # Phase 4 — bracket + Monte Carlo core
│       ├── simulate_bracket.py             # Phase 4 — CLI + historical back-test
│       └── preseason_forecast.py           # Phase 5 — preseason forecast (upcoming season)
├── data/                # raw/ + processed/ (gitignored — regenerate with the scripts)
├── models/              # saved model artifacts (gitignored — regenerate)
└── requirements.txt
```

> Data and model artifacts are **not** committed (they're regenerable and the
> raw play-by-play archive alone is 2.2 GB). Run the pipeline below to build them.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

Get a free API key at <https://www.balldontlie.io>, then create a `.env` file
(see `.env.example`):

```
BALLDONTLIE_API_KEY=your_key_here
```

The `.env` file is gitignored and is never committed.

## Usage

Run the full pipeline in order:

```bash
# 1. Collect games (2010 -> current). ~40 min on the free tier (5 req/min).
python src/data/collect_games_balldontlie.py

# 2. Build Elo ratings + team-season strength
python src/features/build_elo.py

# 3. Reconstruct playoff series, then train + validate the series model
python src/features/build_series.py
python src/models/train_series_model.py

# 4. Simulate a season's bracket into championship odds
python src/simulation/simulate_bracket.py --season 2024     # 2024-25
python src/simulation/simulate_bracket.py --backtest        # validate all seasons

# 5. Preseason forecast for the upcoming (not-yet-played) season
python src/simulation/preseason_forecast.py
```

Championship odds are written to `data/processed/title_odds.csv`;
preseason odds to `data/processed/preseason_title_odds.csv`.

### Preseason forecast (upcoming season)

For a season that hasn't been played yet there is no bracket to simulate, so
`preseason_forecast.py` carries each team's Elo over from the end of the last
completed season (regressed toward the mean) and Monte-Carlos a full season —
a simulated regular season for seeding, then the playoff bracket.

**Caveat:** a preseason forecast cannot account for offseason roster changes
(trades, free agency, the draft, retirements, injuries). It is a strength-
carryover baseline — the reigning champion is naturally the favorite because
its deep playoff run inflates its carry-over Elo — not a roster-aware projection.

## Limitations & notes

- Team strength uses **Elo entering the playoffs**, held constant across rounds
  (a standard simplification for a series-level forward simulation).
- The model intentionally uses a single feature (Elo gap) to avoid overfitting
  ~240 historical series. Injuries, rest, and matchup effects are not modeled.
- `stats.nba.com` / `nba_api` are unreachable from many networks, which is why
  this project uses balldontlie as its data source.

## License

MIT
