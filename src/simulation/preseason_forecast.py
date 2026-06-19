"""
Phase 5 — Preseason championship forecast for an upcoming season.

When a season hasn't been played yet there is no bracket to simulate, so we
forecast instead. Each team's strength is its Elo carried over from the end of
the last completed season (regressed toward the mean, matching the in-season
carry-over rule). We then Monte-Carlo a full season:

  1. Simulate a balanced regular season (double round-robin) via game-level Elo
     to produce win totals -> conference seeding (this captures seeding luck).
  2. Seed the top 8 per conference and simulate the playoff bracket with the
     trained series-win model.
  3. Repeat many times; the share of titles won is each team's championship odds.

IMPORTANT CAVEAT: a preseason forecast made before the offseason cannot know
roster changes — trades, free agency, the draft, retirements, or injuries. It
reflects "if rosters stayed roughly as they ended last season." Treat it as a
strength-carryover baseline, not a roster-aware projection.

Usage:
    python src/simulation/preseason_forecast.py                 # next season
    python src/simulation/preseason_forecast.py --n-sims 20000
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bracket import EAST, WEST, BRACKET_ORDER, conference, load_series_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC = PROJECT_ROOT / "data" / "processed"

# Must match build_elo.py.
MEAN_ELO = 1505.0
CARRYOVER = 0.75
HOME_ADV = 100.0
N_SIMS = 20_000


def season_label(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def carryover_elo() -> tuple[int, dict[str, float]]:
    """End-of-last-season Elo per team, regressed toward the mean."""
    elo = pd.read_csv(PROC / "elo_games.csv", parse_dates=["date"])
    last = int(elo["season"].max())
    s = elo[elo["season"] == last].sort_values("date")

    final: dict[str, float] = {}
    for r in s.itertuples(index=False):       # later games overwrite earlier ones
        final[r.home_team] = r.home_elo_post
        final[r.away_team] = r.away_elo_post

    pre = {t: CARRYOVER * e + (1 - CARRYOVER) * MEAN_ELO for t, e in final.items()}
    return last, pre


def build_schedule(teams: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Balanced double round-robin: every ordered (home, away) pair once."""
    idx = {t: i for i, t in enumerate(teams)}
    home, away = [], []
    for h in teams:
        for a in teams:
            if h != a:
                home.append(idx[h])
                away.append(idx[a])
    return np.array(home), np.array(away)


def p_home_game(elo_home: float, elo_away: float) -> float:
    return 1.0 / (1.0 + 10 ** (-((elo_home + HOME_ADV) - elo_away) / 400.0))


def simulate(pre_elo: dict[str, float], n_sims: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = sorted(pre_elo)
    elo = np.array([pre_elo[t] for t in teams])
    n_teams = len(teams)
    confs = np.array([conference(t) for t in teams])

    # --- Regular season (vectorised across sims) ---
    home, away = build_schedule(teams)
    p_home = 1.0 / (1.0 + 10 ** (-((elo[home] + HOME_ADV) - elo[away]) / 400.0))
    G = len(home)
    home_win = rng.random((n_sims, G)) < p_home          # (n_sims, G)
    home_oh = np.zeros((G, n_teams)); home_oh[np.arange(G), home] = 1
    away_oh = np.zeros((G, n_teams)); away_oh[np.arange(G), away] = 1
    wins = home_win @ home_oh + (~home_win) @ away_oh     # (n_sims, n_teams)
    n_games = G // n_teams * 2  # games per team in a double round-robin = 2*(n_teams-1)

    intercept, coef = load_series_model()
    east_i = np.where(confs == "East")[0]
    west_i = np.where(confs == "West")[0]

    titles = np.zeros(n_teams)
    finals = np.zeros(n_teams)
    playoffs = np.zeros(n_teams)

    for s in range(n_sims):
        w = wins[s]
        champ = _simulate_one_playoffs(w, elo, east_i, west_i, intercept, coef, rng,
                                       finals, playoffs)
        titles[champ] += 1

    win_pct_mean = wins.mean(axis=0) / n_games
    df = pd.DataFrame({
        "team": teams,
        "conference": confs,
        "preseason_elo": np.round(elo, 1),
        "proj_win_pct": np.round(win_pct_mean, 3),
        "playoff_odds": playoffs / n_sims,
        "finals_odds": finals / n_sims,
        "title_odds": titles / n_sims,
    })
    return df.sort_values("title_odds", ascending=False).reset_index(drop=True)


def _seed_conf(conf_idx, w):
    """Return the conference's top-8 team indices ordered by seed (1..8)."""
    order = conf_idx[np.argsort(-w[conf_idx], kind="stable")]
    return order[:8]


def _p_series(hc_elo, opp_elo, intercept, coef):
    return 1.0 / (1.0 + np.exp(-(intercept + coef * (hc_elo - opp_elo))))


def _simulate_one_playoffs(w, elo, east_i, west_i, intercept, coef, rng, finals, playoffs):
    champ_by_conf = {}
    for conf_idx, name in ((east_i, "E"), (west_i, "W")):
        seeds = _seed_conf(conf_idx, w)          # indices, seed order 1..8
        playoffs[seeds] += 1
        # arrange by BRACKET_ORDER for correct adjacency
        bracket = [seeds[s - 1] for s in BRACKET_ORDER]
        seed_of = {int(t): i + 1 for i, t in enumerate(seeds)}
        while len(bracket) > 1:
            nxt = []
            for i in range(0, len(bracket), 2):
                a, b = bracket[i], bracket[i + 1]
                hc, opp = (a, b) if seed_of[int(a)] < seed_of[int(b)] else (b, a)
                p = _p_series(elo[hc], elo[opp], intercept, coef)
                nxt.append(hc if rng.random() < p else opp)
            bracket = nxt
        champ_by_conf[name] = bracket[0]

    e, wch = champ_by_conf["E"], champ_by_conf["W"]
    finals[e] += 1
    finals[wch] += 1
    # Finals home court to the better regular-season record this sim.
    hc, opp = (e, wch) if w[e] >= w[wch] else (wch, e)
    p = _p_series(elo[hc], elo[opp], intercept, coef)
    return hc if rng.random() < p else opp


def main() -> None:
    parser = argparse.ArgumentParser(description="Preseason championship forecast.")
    parser.add_argument("--n-sims", type=int, default=N_SIMS)
    args = parser.parse_args()

    last_season, pre_elo = carryover_elo()
    upcoming = last_season + 1
    logging.info("Carry-over Elo from %s -> forecasting %s (%d sims)",
                 season_label(last_season), season_label(upcoming), args.n_sims)

    df = simulate(pre_elo, args.n_sims)
    df.insert(0, "season_label", season_label(upcoming))
    out = PROC / "preseason_title_odds.csv"
    df.to_csv(out, index=False)

    print(f"\nPreseason championship forecast — {season_label(upcoming)} "
          f"({args.n_sims:,} sims)")
    print("(carry-over strength only — does NOT account for offseason roster moves)\n")
    print(df.head(16).to_string(index=False))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
