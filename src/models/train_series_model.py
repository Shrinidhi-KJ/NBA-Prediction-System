"""
Phase 3b — Train & validate the series-win model.

Predicts P(home-court team wins a best-of-7 series) from the Elo difference
between the two teams. A logistic regression is used deliberately: with only
~240 historical series, a single principled feature (elo_diff) avoids
overfitting, and the intercept naturally captures home-court advantage
(P > 0.5 when elo_diff = 0).

Validation is leave-one-season-out (LOSO) cross-validation, compared against:
  - "home court always wins"      (majority-class baseline)
  - constant base-rate probability (for log-loss / Brier reference)

Inputs:  data/processed/playoff_series.csv
Outputs: models/series_model.joblib
         models/series_model_metrics.json

Usage:
    python src/models/train_series_model.py
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"

FEATURES = ["elo_diff"]
TARGET = "a_won"


def load() -> pd.DataFrame:
    df = pd.read_csv(PROC / "playoff_series.csv")
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return df


def loso_cv(df: pd.DataFrame) -> dict:
    """Leave-one-season-out cross-validated out-of-fold predictions."""
    oof = np.zeros(len(df))
    for season in df["season"].unique():
        tr = df[df["season"] != season]
        te = df[df["season"] == season]
        model = LogisticRegression()
        model.fit(tr[FEATURES], tr[TARGET])
        oof[te.index] = model.predict_proba(te[FEATURES])[:, 1]

    y = df[TARGET].values
    base_rate = y.mean()
    return {
        "n_series": int(len(df)),
        "model": {
            "accuracy": float(accuracy_score(y, oof >= 0.5)),
            "log_loss": float(log_loss(y, oof)),
            "brier": float(brier_score_loss(y, oof)),
            "auc": float(roc_auc_score(y, oof)),
        },
        "baseline_home_court_always": {
            "accuracy": float(accuracy_score(y, np.ones_like(y))),
            "log_loss": float(log_loss(y, np.full_like(oof, base_rate))),
            "brier": float(brier_score_loss(y, np.full_like(oof, base_rate))),
        },
        "home_court_base_rate": float(base_rate),
    }


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    df = load()

    metrics = loso_cv(df)
    logging.info("LOSO-CV results (n=%d series):", metrics["n_series"])
    m, b = metrics["model"], metrics["baseline_home_court_always"]
    logging.info("  model    : acc=%.3f  log_loss=%.3f  brier=%.3f  auc=%.3f",
                 m["accuracy"], m["log_loss"], m["brier"], m["auc"])
    logging.info("  baseline : acc=%.3f  log_loss=%.3f  brier=%.3f  (home court always)",
                 b["accuracy"], b["log_loss"], b["brier"])

    # Final model on all data.
    model = LogisticRegression()
    model.fit(df[FEATURES], df[TARGET])
    joblib.dump({"model": model, "features": FEATURES}, MODELS / "series_model.joblib")

    coef = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    metrics["final_model"] = {"features": FEATURES, "coef_elo_diff": coef, "intercept": intercept}
    with open(MODELS / "series_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Illustrative probabilities for the home-court team at various Elo gaps.
    logging.info("P(home-court team wins series) by Elo gap:")
    for d in [-200, -100, 0, 100, 200, 300]:
        p = model.predict_proba(pd.DataFrame({"elo_diff": [d]}))[0][1]
        logging.info("  elo_diff=%+4d -> %.3f", d, p)
    logging.info("Saved model -> models/series_model.joblib")


if __name__ == "__main__":
    main()
