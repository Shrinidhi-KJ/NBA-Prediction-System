"""
Phase 3a — Reconstruct best-of-7 playoff series from game data.

balldontlie gives individual playoff games but no series/round labels. We
rebuild series by grouping playoff games in a season by the (unordered) pair
of teams. A genuine best-of-7 series is identified by its winner reaching
exactly 4 wins (this cleanly excludes play-in games, where the winner has 1).

By NBA format the higher seed hosts Game 1, so the Game-1 home team is the
home-court team ("team_a"). Strength features are joined from the team-season
strength table (Elo entering playoffs, win %, point differential).

Inputs:  data/processed/elo_games.csv, data/processed/team_season_strength.csv
Output:  data/processed/playoff_series.csv   (one row per series)

Usage:
    python src/features/build_series.py
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC = PROJECT_ROOT / "data" / "processed"


def build() -> pd.DataFrame:
    games = pd.read_csv(PROC / "elo_games.csv", parse_dates=["date"])
    strength = pd.read_csv(PROC / "team_season_strength.csv")

    po = games[games["postseason"]].copy()
    po["pair"] = po.apply(
        lambda r: "__".join(sorted([r["home_team"], r["away_team"]])), axis=1)

    rows = []
    for (season, pair), g in po.groupby(["season", "pair"]):
        g = g.sort_values("date")
        t1, t2 = pair.split("__")
        wins = {t1: 0, t2: 0}
        for r in g.itertuples(index=False):
            winner = r.home_team if r.home_win else r.away_team
            wins[winner] += 1
        winner_wins = max(wins.values())
        if winner_wins != 4:
            continue  # not a completed best-of-7 series (e.g. play-in)

        # Higher seed hosts Game 1 -> Game-1 home team has home court.
        game1 = g.iloc[0]
        team_a = game1["home_team"]          # home-court team
        team_b = game1["away_team"]
        a_won = 1 if wins[team_a] > wins[team_b] else 0

        rows.append({
            "season": season,
            "season_label": game1["season_label"],
            "team_a": team_a, "team_b": team_b,
            "a_wins": wins[team_a], "b_wins": wins[team_b],
            "total_games": int(g.shape[0]),
            "a_won": a_won,
            # Live Elo entering THIS series (Game-1 pre-game), so later rounds
            # reflect how each team played in earlier rounds.
            "a_elo": float(game1["home_elo_pre"]),
            "b_elo": float(game1["away_elo_pre"]),
        })

    series = pd.DataFrame(rows)

    # Join regular-season context features for both teams (Elo already set
    # above from live Game-1 ratings).
    feat = strength[["season", "team", "win_pct", "avg_pt_diff"]]
    for side in ("a", "b"):
        series = series.merge(
            feat.rename(columns={
                "team": f"team_{side}",
                "win_pct": f"{side}_win_pct",
                "avg_pt_diff": f"{side}_pt_diff",
            }),
            on=["season", f"team_{side}"], how="left",
        )

    series["elo_diff"] = series["a_elo"] - series["b_elo"]          # home-court team minus opponent
    series["win_pct_diff"] = series["a_win_pct"] - series["b_win_pct"]
    series["pt_diff_diff"] = series["a_pt_diff"] - series["b_pt_diff"]
    return series.sort_values(["season"]).reset_index(drop=True)


def main() -> None:
    series = build()
    out = PROC / "playoff_series.csv"
    series.to_csv(out, index=False)
    logging.info("Wrote %d playoff series -> %s", len(series), out)
    logging.info("Series per season:\n%s", series.groupby("season_label").size().to_string())
    base = series["a_won"].mean()
    logging.info("Home-court team series win rate: %.3f (n=%d)", base, len(series))
    missing = series[["a_elo", "b_elo"]].isna().any(axis=1).sum()
    if missing:
        logging.warning("%d series missing Elo features", missing)


if __name__ == "__main__":
    main()
