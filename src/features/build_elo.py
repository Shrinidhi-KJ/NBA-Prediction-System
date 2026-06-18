"""
Phase 2 — Team strength.

Computes Elo ratings game-by-game (FiveThirtyEight-style: home-court
advantage + margin-of-victory multiplier + between-season regression to the
mean), plus per-team-season summary features.

Inputs:  data/raw/games.csv            (from collect_games_balldontlie.py)
Outputs: data/processed/elo_games.csv          one row per game w/ pre/post Elo
         data/processed/team_season_strength.csv   per team-season features

Usage:
    python src/features/build_elo.py
"""

import logging
import math
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw" / "games.csv"
PROC = PROJECT_ROOT / "data" / "processed"

# --- Elo hyperparameters (FiveThirtyEight NBA defaults) ---
START_ELO = 1500.0
MEAN_ELO = 1505.0
K = 20.0
HOME_ADV = 100.0          # Elo points added to the home side
CARRYOVER = 0.75          # fraction of rating retained across seasons


def expected_score(elo_a: float, elo_b: float) -> float:
    """Win probability for A vs B (elo already includes any home adjustment)."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def mov_multiplier(margin: int, elo_diff_winner: float) -> float:
    """
    Margin-of-victory multiplier (538). `elo_diff_winner` is winner_elo minus
    loser_elo *including* home advantage, pre-game. Dampens autocorrelation so
    blowouts by already-strong teams don't over-inflate ratings.
    """
    return math.log(abs(margin) + 1.0) * (2.2 / (elo_diff_winner * 0.001 + 2.2))


def regress(elo: float) -> float:
    return CARRYOVER * elo + (1 - CARRYOVER) * MEAN_ELO


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(RAW)
    # Completed games only, sorted chronologically (tie-break by game_id).
    df = df[df["status"] == "Final"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)

    elos: dict[str, float] = {}
    current_season: int | None = None
    # elo of each team after its most recent regular-season game, per season
    pre_playoff_elo: dict[tuple[int, str], float] = {}

    out_rows = []
    for r in df.itertuples(index=False):
        season = r.season
        # Between-season regression (applied once, when season advances).
        if current_season is not None and season > current_season:
            for t in elos:
                elos[t] = regress(elos[t])
        current_season = season

        home, away = r.home_team_abbr, r.visitor_team_abbr
        eh = elos.get(home, START_ELO)
        ea = elos.get(away, START_ELO)

        exp_home = expected_score(eh + HOME_ADV, ea)
        home_win = 1 if r.home_score > r.visitor_score else 0
        margin = r.home_score - r.visitor_score

        # elo diff from winner's perspective, incl. home adv, pre-game
        if home_win:
            elo_diff_winner = (eh + HOME_ADV) - ea
        else:
            elo_diff_winner = ea - (eh + HOME_ADV)
        mult = mov_multiplier(margin, elo_diff_winner)

        delta = K * mult * (home_win - exp_home)
        eh_post, ea_post = eh + delta, ea - delta
        elos[home], elos[away] = eh_post, ea_post

        out_rows.append({
            "game_id": r.game_id, "date": r.date, "season_label": r.season_label,
            "season": season, "postseason": r.postseason,
            "home_team": home, "away_team": away,
            "home_elo_pre": eh, "away_elo_pre": ea,
            "home_elo_post": eh_post, "away_elo_post": ea_post,
            "home_win_prob": exp_home, "home_win": home_win,
            "home_score": r.home_score, "visitor_score": r.visitor_score,
        })

        if not r.postseason:
            pre_playoff_elo[(season, home)] = eh_post
            pre_playoff_elo[(season, away)] = ea_post

    elo_games = pd.DataFrame(out_rows)

    season_strength = _season_strength(df, elo_games, pre_playoff_elo)
    return elo_games, season_strength


def _season_strength(df, elo_games, pre_playoff_elo) -> pd.DataFrame:
    """Per team-season regular-season record, point differential, and Elo."""
    reg = df[~df["postseason"]].copy()
    # Long format: one row per team per game.
    home = reg.rename(columns={"home_team_abbr": "team", "visitor_team_abbr": "opp",
                               "home_score": "pf", "visitor_score": "pa"})
    away = reg.rename(columns={"visitor_team_abbr": "team", "home_team_abbr": "opp",
                               "visitor_score": "pf", "home_score": "pa"})
    cols = ["season", "season_label", "team", "pf", "pa"]
    long = pd.concat([home[cols], away[cols]], ignore_index=True)
    long["win"] = (long["pf"] > long["pa"]).astype(int)
    long["pt_diff"] = long["pf"] - long["pa"]

    g = long.groupby(["season", "season_label", "team"]).agg(
        reg_games=("win", "size"),
        wins=("win", "sum"),
        avg_pt_diff=("pt_diff", "mean"),
        ppg=("pf", "mean"),
        opp_ppg=("pa", "mean"),
    ).reset_index()
    g["losses"] = g["reg_games"] - g["wins"]
    g["win_pct"] = g["wins"] / g["reg_games"]
    g["elo_pre_playoffs"] = g.apply(
        lambda x: pre_playoff_elo.get((x["season"], x["team"])), axis=1)

    made_playoffs = (elo_games[elo_games["postseason"]]
                     .melt(id_vars="season", value_vars=["home_team", "away_team"],
                           value_name="team")[["season", "team"]].drop_duplicates())
    made_playoffs["made_playoffs"] = 1
    g = g.merge(made_playoffs, on=["season", "team"], how="left")
    g["made_playoffs"] = g["made_playoffs"].fillna(0).astype(int)
    return g.sort_values(["season", "win_pct"], ascending=[True, False]).reset_index(drop=True)


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    elo_games, season_strength = build()
    elo_games.to_csv(PROC / "elo_games.csv", index=False)
    season_strength.to_csv(PROC / "team_season_strength.csv", index=False)
    logging.info("Wrote %d game rows -> elo_games.csv", len(elo_games))
    logging.info("Wrote %d team-season rows -> team_season_strength.csv", len(season_strength))
    # Sanity: top teams by final pre-playoff Elo in the most recent season.
    last = season_strength["season"].max()
    top = (season_strength[season_strength["season"] == last]
           .nlargest(5, "elo_pre_playoffs")[["team", "win_pct", "elo_pre_playoffs"]])
    logging.info("Top 5 by pre-playoff Elo in %s:\n%s", last, top.to_string(index=False))


if __name__ == "__main__":
    main()
