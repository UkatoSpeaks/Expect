"""Step 6: one-shot evaluation of the frozen model (src/config.py) on held-out seasons.

    .venv/bin/python -m src.evaluate_test --split test      # 2023/24 + 2024/25
    .venv/bin/python -m src.evaluate_test --split forward   # 2025/26 + 2026/27 played so far

Protocol (same as the CV in src/cv.py / src/cv_dc.py):
- For each evaluated season N, the logistic regression is fit on every season from
  FIRST_TRAIN_SEASON to N-1 and predicts all of N without refitting.
- Dixon-Coles is refit before every match date on matches strictly before that date.
- Final model = BLEND_WEIGHT_LR * LR + (1 - BLEND_WEIGHT_LR) * DC.
- Home-rate baseline = class frequencies (H/D/A) of the same training seasons.
- Market = margin-free implied probabilities (benchmark only).

Nothing here is tuned. Each run writes to a new directory and refuses to overwrite one
that exists: reports/test/ for the test split, reports/forward_<last played date>/ for the
forward split (so later runs with more 2026/27 matches add a new snapshot).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.config import (BLEND_WEIGHT_LR, DC_XI, FIRST_TRAIN_SEASON, FORWARD_SEASONS,
                        LR_C, TEST_SEASONS)
from src.cv import CLASSES, MARKET_COLS, REPORTS, bootstrap_ci, calibration_plot, lr_model, row_scores
from src.dixon_coles import dc_predictions
from src.features import MATCHES_PATH, build_features, feature_columns

KEYS = ["date", "home_team", "away_team"]
DC_COLS = ["dc_p_home", "dc_p_draw", "dc_p_away"]
DC_CACHE = MATCHES_PATH.parent / "dc_probs.csv"
METRICS = ["log_loss", "brier", "accuracy"]

BLEND = "Blend (LR + DC)"
LR = f"LR (C={LR_C})"
DC = "Dixon-Coles"
HOME = "Home-rate baseline"
MARKET = "Market"


def _frame(p, index, season) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(p, float), columns=CLASSES, index=index).assign(season=season)


def predict_seasons(feats: pd.DataFrame, cols: list[str], seasons: list[str]) -> dict[str, pd.DataFrame]:
    """Walk-forward predictions for every played match of `seasons`, one season at a time.
    `feats` must already carry the pre-match DC probabilities (DC_COLS); `cols` are the LR
    features (DC probabilities are not among them: the final model blends, it does not stack)."""
    order = list(dict.fromkeys(feats["season"]))
    assert not set(DC_COLS) & set(cols)
    fit_lr = lr_model(cols, LR_C)
    out: dict[str, list[pd.DataFrame]] = {m: [] for m in (BLEND, LR, DC, HOME, MARKET)}
    for s in seasons:
        train_seasons = order[order.index(FIRST_TRAIN_SEASON):order.index(s)]
        tr = feats[feats["season"].isin(train_seasons) & feats["result"].notna()]
        te = feats[(feats["season"] == s) & feats["result"].notna()]
        print(f"{s}: train {train_seasons[0]}-{train_seasons[-1]} ({len(tr)} matches), "
              f"evaluate {len(te)} matches")

        p_lr = fit_lr(tr, te)
        p_dc = te[DC_COLS].to_numpy()
        assert not np.isnan(p_dc).any(), f"missing DC predictions in {s}"
        freq = tr["result"].value_counts(normalize=True).reindex(CLASSES).to_numpy()
        p_mkt = te[MARKET_COLS].to_numpy()
        assert not np.isnan(p_mkt).any(), f"missing market odds in {s}"

        out[LR].append(_frame(p_lr, te.index, s))
        out[DC].append(_frame(p_dc, te.index, s))
        out[BLEND].append(_frame(BLEND_WEIGHT_LR * p_lr + (1 - BLEND_WEIGHT_LR) * p_dc, te.index, s))
        out[HOME].append(_frame(np.tile(freq, (len(te), 1)), te.index, s))
        out[MARKET].append(_frame(p_mkt, te.index, s))
    return {m: pd.concat(v) for m, v in out.items()}


def _diff_cells(s: pd.DataFrame, ref: pd.DataFrame, tag: str) -> dict[str, str]:
    cells = {}
    for metric in METRICS:
        d = (s[metric] - ref[metric]).to_numpy()
        lo, hi = bootstrap_ci(d)
        cells[f"{metric} vs {tag}"] = f"{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"
    return cells


def score_table(preds: dict[str, pd.DataFrame], y: pd.Series) -> pd.DataFrame:
    """Rows: (scope, model) for each season and 'pooled'. Metrics plus paired bootstrap
    95% CIs of the per-match difference vs the market and vs LR (negative = better for
    log loss / Brier, positive = better for accuracy)."""
    scores = {m: row_scores(p[CLASSES].to_numpy(), y.loc[p.index]).assign(season=p["season"])
              for m, p in preds.items()}
    scopes = list(dict.fromkeys(preds[MARKET]["season"])) + ["pooled"]
    rows = []
    for scope in scopes:
        sub = {m: (s if scope == "pooled" else s[s["season"] == scope]) for m, s in scores.items()}
        for m, s in sub.items():
            row = {"scope": scope, "model": m, "n": len(s), **s[METRICS].mean().round(4).to_dict()}
            if m != MARKET:
                row.update(_diff_cells(s, sub[MARKET], "market"))
            if m != LR:
                row.update(_diff_cells(s, sub[LR], "LR"))
            rows.append(row)
    return pd.DataFrame(rows).set_index(["scope", "model"])


def dc_with_cache_check(matches: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """Rebuild DC predictions for `seasons`, then compare with data/dc_probs.csv."""
    dc = dc_predictions(matches, DC_XI, seasons=seasons)
    if DC_CACHE.exists():
        cache = pd.read_csv(DC_CACHE, parse_dates=["date"])
        j = dc.merge(cache[KEYS + DC_COLS], on=KEYS, suffixes=("", "_cache"))
        diff = max((j[c] - j[f"{c}_cache"]).abs().max() for c in DC_COLS)
        print(f"DC rebuilt for {len(dc)} matches; max |diff| vs {DC_CACHE.name} "
              f"on {len(j)} shared matches: {diff:.2e}")
    return dc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["test", "forward"], required=True)
    args = ap.parse_args()
    seasons = TEST_SEASONS if args.split == "test" else FORWARD_SEASONS

    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    played = matches[matches["season"].isin(seasons) & matches["result"].notna()]
    last = played["date"].max().date()
    out_dir = REPORTS / ("test" if args.split == "test" else f"forward_{last}")
    if out_dir.exists():
        sys.exit(f"{out_dir} already exists: this split was already evaluated on this data. "
                 "Results are not regenerated (delete the directory by hand only if you mean to).")

    feats = build_features(matches)  # Elo from src/config.py; features use earlier matches only
    cols = feature_columns(feats)
    dc = dc_with_cache_check(matches, seasons)
    feats = feats.merge(dc[KEYS + DC_COLS], on=KEYS, how="left", validate="1:1")
    preds = predict_seasons(feats, cols, seasons)
    y = feats["result"]

    table = score_table(preds, y)
    out_dir.mkdir(parents=True)
    table.to_csv(out_dir / "scores.csv")
    long = pd.concat(preds, names=["model"]).reset_index(level=0)
    long.join(feats[KEYS + ["result"]]).to_csv(out_dir / "predictions.csv")

    n = len(preds[MARKET])
    span = f"{seasons[0]}–{played['season'].max()}"
    label = "Test" if args.split == "test" else f"Forward (played to {last})"
    calibration_plot({BLEND: preds[BLEND], LR: preds[LR], MARKET: preds[MARKET]}, y,
                     out_dir / "calibration.png",
                     title=f"Calibration, {label.lower()} {span} ({n:,} matches, 10 equal-count bins)")

    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(f"\n{label} results, {span}, {n} matches. "
          "Differences: model minus reference, 95% paired bootstrap CI.")
    print(table.to_string())
    print(f"\nwrote {out_dir}/scores.csv, predictions.csv, calibration.png")


if __name__ == "__main__":
    main()
