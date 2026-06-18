"""
Phase 1 — Data collection (balldontlie API).

Pulls game results (one row per game) for a range of seasons from the
balldontlie v1 `/games` endpoint. Each game includes home/visitor teams,
final scores, the season, and a `postseason` flag (regular season vs playoffs).

Requires a free API key from https://www.balldontlie.io, provided via the
BALLDONTLIE_API_KEY environment variable (e.g. in a .env file).

Output: data/raw/games.csv

Usage:
    python src/data/collect_games_balldontlie.py                 # 2010 -> current
    python src/data/collect_games_balldontlie.py --start 2015 --end 2023
"""

import argparse
import logging
import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

API_URL = "https://api.balldontlie.io/v1/games"
PER_PAGE = 100
# Free tier = 5 requests/minute. 13s between calls keeps us safely under that.
REQUEST_SLEEP = 13.0
MAX_RETRIES = 4
RETRY_BACKOFF = 15.0


def season_label(start_year: int) -> str:
    """2010 -> '2010-11'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def current_season_start() -> int:
    today = date.today()
    return today.year if today.month >= 10 else today.year - 1


def get_api_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("BALLDONTLIE_API_KEY")
    if not key:
        raise SystemExit(
            "BALLDONTLIE_API_KEY not set. Get a free key at https://www.balldontlie.io "
            "and put it in a .env file (see .env.example)."
        )
    return key


def fetch_page(session: requests.Session, season: int, cursor: int | None) -> dict:
    """Fetch one page of games for a season, with retries and rate-limit handling."""
    params = {"seasons[]": season, "per_page": PER_PAGE}
    if cursor is not None:
        params["cursor"] = cursor

    delay = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(API_URL, params=params, timeout=30)
            if resp.status_code == 429:  # rate limited
                logging.warning("Rate limited (429); sleeping %.0fs", delay)
                time.sleep(delay)
                delay *= 1.5
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logging.warning("Attempt %d/%d failed (season %s): %s",
                            attempt, MAX_RETRIES, season, exc)
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 1.5
    raise RuntimeError(f"Failed to fetch games for season {season} after {MAX_RETRIES} attempts")


def flatten(game: dict) -> dict:
    """Flatten a balldontlie game object into a flat row."""
    home, vis = game["home_team"], game["visitor_team"]
    return {
        "game_id": game["id"],
        "date": game["date"],
        "season": game["season"],
        "season_label": season_label(game["season"]),
        "postseason": game["postseason"],
        "status": game.get("status"),
        "home_team_id": home["id"],
        "home_team_abbr": home["abbreviation"],
        "home_team_name": home["full_name"],
        "home_score": game["home_team_score"],
        "visitor_team_id": vis["id"],
        "visitor_team_abbr": vis["abbreviation"],
        "visitor_team_name": vis["full_name"],
        "visitor_score": game["visitor_team_score"],
    }


def collect(start_year: int, end_year: int, api_key: str) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"Authorization": api_key})

    rows = []
    for year in range(start_year, end_year + 1):
        logging.info("Fetching season %s ...", season_label(year))
        cursor, pages, season_rows = None, 0, 0
        while True:
            payload = fetch_page(session, year, cursor)
            for g in payload.get("data", []):
                rows.append(flatten(g))
                season_rows += 1
            pages += 1
            cursor = payload.get("meta", {}).get("next_cursor")
            time.sleep(REQUEST_SLEEP)
            if not cursor:
                break
        logging.info("  -> %d games (%d pages)", season_rows, pages)

    if not rows:
        raise RuntimeError("No games collected.")
    df = pd.DataFrame(rows)
    # Only completed games are useful for modeling.
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect NBA games from balldontlie.")
    parser.add_argument("--start", type=int, default=2010)
    parser.add_argument("--end", type=int, default=current_season_start())
    parser.add_argument("--out", type=str, default=str(RAW_DIR / "games.csv"))
    args = parser.parse_args()

    api_key = get_api_key()
    logging.info("Collecting seasons %s -> %s", season_label(args.start), season_label(args.end))
    df = collect(args.start, args.end, api_key)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logging.info("Saved %d games to %s", len(df), out_path)
    logging.info("Seasons: %s", sorted(df["season_label"].unique()))


if __name__ == "__main__":
    main()
