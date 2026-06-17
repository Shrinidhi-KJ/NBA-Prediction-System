# NBA Champion Predictor

A machine-learning system that estimates each team's probability of winning the
NBA championship. Instead of predicting the champion directly (only one winner
per year = almost no training signal), it uses the standard, robust approach:

1. **Rate team strength** — an Elo rating updated game-by-game, plus
   season-level team features (net rating, SRS, offensive/defensive rating, pace).
2. **Series model** — given two teams' strengths and home-court advantage,
   estimate the probability that Team A wins a best-of-7 series.
3. **Bracket simulation** — run the current playoff bracket forward thousands of
   times (Monte Carlo). The share of simulations a team wins = its title odds.

## Project Structure
```
NBA-Playoff-Predictions/
├── src/
│   ├── data/          # Data collection from nba_api
│   ├── features/      # Elo ratings + team strength features
│   ├── models/        # Series-win model (train + predict)
│   └── simulation/    # Monte Carlo bracket simulation
├── data/
│   ├── raw/           # Raw pulls (incl. archived play_by_play.csv)
│   └── processed/     # Cleaned, model-ready datasets
├── models/            # Saved model artifacts (.joblib)
├── notebooks/         # Exploratory analysis
└── docs/              # Project notes
```

## Roadmap
- [ ] **Phase 1 — Data:** regular-season team game logs + playoff series results & brackets (2010→present) via `nba_api`.
- [ ] **Phase 2 — Team strength:** Elo ratings + season team features.
- [ ] **Phase 3 — Series model:** train & validate P(win 7-game series) on historical playoffs.
- [ ] **Phase 4 — Bracket sim:** Monte Carlo the current bracket into per-team title odds.
- [ ] **Phase 5 — Output:** ranked championship-probability report (re-runnable live during playoffs).

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage (planned)
```bash
python src/data/collect_team_games.py     # Phase 1
python src/features/build_elo.py          # Phase 2
python src/models/train_series_model.py   # Phase 3
python src/simulation/simulate_bracket.py # Phase 4
```
