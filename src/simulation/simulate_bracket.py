"""
Phase 4 — Bracket simulation CLI + back-test.

Default: simulate the most recent season's bracket and print championship odds.
--backtest: validate across all seasons (does the model give the eventual
            champion meaningfully more than the 1/16 = 6.25% naive baseline?).

Usage:
    python src/simulation/simulate_bracket.py                 # latest season
    python src/simulation/simulate_bracket.py --season 2015   # 2015-16
    python src/simulation/simulate_bracket.py --backtest
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bracket import build_bracket, simulate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC = PROJECT_ROOT / "data" / "processed"
N_SIMS = 50_000


def season_label(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def actual_champion(season: int) -> str | None:
    """Champion = the team that won 4 playoff series that season."""
    series = pd.read_csv(PROC / "playoff_series.csv")
    s = series[series["season"] == season]
    wins: dict[str, int] = {}
    for r in s.itertuples(index=False):
        winner = r.team_a if r.a_won else r.team_b
        wins[winner] = wins.get(winner, 0) + 1
    if not wins:
        return None
    champ = max(wins, key=wins.get)
    return champ if wins[champ] == 4 else None


def run_season(season: int, n_sims: int) -> pd.DataFrame:
    bracket = build_bracket(season)
    odds = simulate(bracket, n_sims=n_sims)
    odds.insert(0, "season_label", season_label(season))
    return odds


def backtest(n_sims: int) -> None:
    series = pd.read_csv(PROC / "playoff_series.csv")
    seasons = sorted(series["season"].unique())
    champ_probs, top_pick_hits, rows = [], 0, []
    for season in seasons:
        odds = run_season(season, n_sims)
        champ = actual_champion(season)
        if champ is None:
            continue
        p = float(odds.loc[odds["team"] == champ, "title_odds"].iloc[0])
        rank = int(odds.reset_index(drop=True).index[odds["team"] == champ][0]) + 1
        champ_probs.append(p)
        top_pick_hits += (rank == 1)
        favorite = odds.iloc[0]
        rows.append({"season": season_label(season), "champion": champ,
                     "champ_title_odds": round(p, 3), "champ_odds_rank": rank,
                     "model_favorite": favorite["team"],
                     "favorite_odds": round(favorite["title_odds"], 3)})

    bt = pd.DataFrame(rows)
    print(bt.to_string(index=False))
    n = len(champ_probs)
    print(f"\nSeasons evaluated: {n}")
    print(f"Mean title-odds assigned to the eventual champion: {np.mean(champ_probs):.3f}")
    print(f"  (naive baseline = 1/16 = 0.0625)")
    print(f"Champion was the model's #1 favorite: {top_pick_hits}/{n} "
          f"({top_pick_hits / n:.0%})   (random baseline ~6%)")
    print(f"Champion finished in model's top 4: "
          f"{sum(r['champ_odds_rank'] <= 4 for r in rows)}/{n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the NBA playoff bracket.")
    parser.add_argument("--season", type=int, default=None,
                        help="Season start year (default: most recent).")
    parser.add_argument("--n-sims", type=int, default=N_SIMS)
    parser.add_argument("--backtest", action="store_true")
    args = parser.parse_args()

    if args.backtest:
        backtest(args.n_sims)
        return

    series = pd.read_csv(PROC / "playoff_series.csv")
    season = args.season if args.season is not None else int(series["season"].max())
    odds = run_season(season, args.n_sims)
    out = PROC / "title_odds.csv"
    odds.to_csv(out, index=False)
    print(f"\nChampionship odds — {season_label(season)} ({args.n_sims:,} simulations)\n")
    print(odds.to_string(index=False))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
