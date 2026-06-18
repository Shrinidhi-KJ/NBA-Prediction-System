"""
Phase 4 — Bracket Monte Carlo (core library).

Builds a season's 16-team playoff bracket (seeded by regular-season win %
within each conference) and simulates it many times using the trained
series-win model to produce championship probabilities.

Modeling note: each team's strength is its Elo entering the playoffs, held
constant across rounds (a standard, defensible simplification for a
series-level forward simulation — at prediction time we don't yet know how
teams will perform in earlier rounds).
"""

import math
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"

# Current NBA conference alignment (balldontlie uses current abbreviations
# for all franchises across history).
EAST = {"ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DET", "IND",
        "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"}
WEST = {"DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN",
        "NOP", "OKC", "PHX", "POR", "SAC", "SAS", "UTA"}

# Seed order so that within an 8-team conference bracket, adjacent pairs each
# round are the correct matchups: R1 (1v8)(4v5)(2v7)(3v6); 1 and 2 can only
# meet in the conference final.
BRACKET_ORDER = [1, 8, 4, 5, 2, 7, 3, 6]


def conference(team: str) -> str:
    if team in EAST:
        return "East"
    if team in WEST:
        return "West"
    raise KeyError(f"Unknown team abbreviation: {team}")


class Team:
    __slots__ = ("abbr", "seed", "elo", "win_pct")

    def __init__(self, abbr, seed, elo, win_pct):
        self.abbr, self.seed, self.elo, self.win_pct = abbr, seed, elo, win_pct

    def __repr__(self):
        return f"{self.abbr}(#{self.seed})"


def load_series_model():
    bundle = joblib.load(MODELS / "series_model.joblib")
    model = bundle["model"]
    return float(model.intercept_[0]), float(model.coef_[0][0])


def build_bracket(season: int) -> dict[str, list[Team]]:
    """Return {'East':[8 Teams seeded 1..8], 'West':[...]} for a season."""
    series = pd.read_csv(PROC / "playoff_series.csv")
    strength = pd.read_csv(PROC / "team_season_strength.csv")

    s = series[series["season"] == season]
    teams = sorted(set(s["team_a"]) | set(s["team_b"]))
    if not teams:
        raise ValueError(f"No playoff series found for season {season}")

    st = strength[strength["season"] == season].set_index("team")
    bracket: dict[str, list[Team]] = {"East": [], "West": []}
    pool: dict[str, list[tuple]] = {"East": [], "West": []}
    for t in teams:
        row = st.loc[t]
        pool[conference(t)].append((t, row["elo_pre_playoffs"], row["win_pct"]))

    for conf, lst in pool.items():
        lst.sort(key=lambda x: x[2], reverse=True)  # seed by win %
        bracket[conf] = [Team(t, i + 1, elo, wp) for i, (t, elo, wp) in enumerate(lst)]
    return bracket


def _p_homecourt_win(intercept, coef, hc_elo, opp_elo) -> float:
    return 1.0 / (1.0 + math.exp(-(intercept + coef * (hc_elo - opp_elo))))


def _play_series(a: Team, b: Team, intercept, coef, rng, by_record=False) -> Team:
    """Simulate one series; home court to the better seed (or record in Finals)."""
    if by_record:
        hc, opp = (a, b) if a.win_pct >= b.win_pct else (b, a)
    else:
        hc, opp = (a, b) if a.seed < b.seed else (b, a)
    p = _p_homecourt_win(intercept, coef, hc.elo, opp.elo)
    return hc if rng.random() < p else opp


def _sim_conference(seeded: list[Team], intercept, coef, rng) -> Team:
    teams = [seeded[s - 1] for s in BRACKET_ORDER]  # order for adjacency
    while len(teams) > 1:
        teams = [_play_series(teams[i], teams[i + 1], intercept, coef, rng)
                 for i in range(0, len(teams), 2)]
    return teams[0]


def simulate(bracket: dict[str, list[Team]], n_sims: int, seed: int = 0) -> pd.DataFrame:
    import random
    rng = random.Random(seed)
    intercept, coef = load_series_model()

    titles: dict[str, int] = {}
    finals: dict[str, int] = {}
    for t in bracket["East"] + bracket["West"]:
        titles[t.abbr] = 0
        finals[t.abbr] = 0

    for _ in range(n_sims):
        e = _sim_conference(bracket["East"], intercept, coef, rng)
        w = _sim_conference(bracket["West"], intercept, coef, rng)
        finals[e.abbr] += 1
        finals[w.abbr] += 1
        champ = _play_series(e, w, intercept, coef, rng, by_record=True)
        titles[champ.abbr] += 1

    info = {t.abbr: (t.seed, conference(t.abbr), t.elo, t.win_pct)
            for t in bracket["East"] + bracket["West"]}
    rows = [{
        "team": ab, "conference": info[ab][1], "seed": info[ab][0],
        "elo_pre_playoffs": round(info[ab][2], 1), "win_pct": round(info[ab][3], 3),
        "finals_odds": finals[ab] / n_sims, "title_odds": titles[ab] / n_sims,
    } for ab in titles]
    return pd.DataFrame(rows).sort_values("title_odds", ascending=False).reset_index(drop=True)
