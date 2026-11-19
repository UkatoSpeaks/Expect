"""Step 6: live predictions for an upcoming gameweek with the frozen model (src/config.py).

    .venv/bin/python -m src.predict                 # next gameweek with no finished matches
    .venv/bin/python -m src.predict --gameweek 11

- Fits the LR on every played match from FIRST_TRAIN_SEASON to today (data/matches.csv),
  and Dixon-Coles before each fixture date on played matches inside its window.
- Fixtures come from FPL-Core-Insights (src/fplcore.py), team names mapped to
  football-data.co.uk names.
- Output: predictions/<season>/gw<N>.csv with P(H)/P(D)/P(A) (the blend), the LR and DC
  components, and the Elo and DC inputs used, stamped with generated_at (UTC).

Prediction files are append-only records: the script refuses to write a gameweek that
already has a file, or fixtures that have already kicked off. It also refuses when
data/matches.csv is behind FPL-Core (a finished match is missing), so stale ratings are
never used silently. Refresh data/raw/E0_<season>.csv and rerun src.load_data first.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BLEND_WEIGHT_LR, DC_XI, FIRST_TRAIN_SEASON, LR_C
from src.cv import CLASSES, lr_model
from src.dixon_coles import PriorBook, _history, _prep, fit_dc, predict as dc_predict
from src.features import MATCHES_PATH, build_features, feature_columns
from src.fplcore import load_pl_matches

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "predictions"
KEYS = ["date", "home_team", "away_team"]


def season_dir(season: str) -> Path:
    a, b = season.split("/")
    return PRED_DIR / f"{a}-{b}"


def gw_path(season: str, gameweek: int) -> Path:
    return season_dir(season) / f"gw{gameweek}.csv"


def load_fixtures(fpl: pd.DataFrame, season: str, gameweek: int | None) -> tuple[int, pd.DataFrame]:
    s = fpl[fpl["season"] == season]
    if gameweek is None:
        open_gws = s.groupby("gameweek")["finished"].sum()
        gameweek = int(open_gws[open_gws == 0].index.min())
    fx = s[(s["gameweek"] == gameweek) & ~s["finished"]]
    if fx.empty:
        sys.exit(f"{season} GW{gameweek}: no unfinished fixtures in FPL-Core")
    dup = pd.concat([fx["home_team"], fx["away_team"]]).value_counts()
    if (dup > 1).any():
        sys.exit(f"GW{gameweek}: teams with two fixtures {dup[dup > 1].index.tolist()}; "
                 "form features for the second one would depend on the first. Not supported.")
    return gameweek, fx.sort_values(["kickoff_utc", "home_team"]).reset_index(drop=True)


def check_data_current(matches: pd.DataFrame, fpl: pd.DataFrame, season: str) -> None:
    """Every finished FPL-Core match of the season must be in matches.csv with a result."""
    fin = fpl[(fpl["season"] == season) & fpl["finished"]]
    played = matches[matches["result"].notna()]
    j = fin[["home_team", "away_team"]].merge(
        played.loc[played["season"] == season, ["home_team", "away_team"]], how="left", indicator=True)
    missing = j[j["_merge"] == "left_only"]
    if len(missing):
        sys.exit(f"data/matches.csv is missing {len(missing)} finished {season} matches, e.g.\n"
                 f"{missing.head().to_string()}\nUpdate data/raw and rerun `python -m src.load_data`.")


def build_predictions(matches: pd.DataFrame, fixtures: pd.DataFrame, season: str) -> pd.DataFrame:
    played = matches[matches["result"].notna()].copy()
    future = pd.DataFrame({"season": season, "date": fixtures["date"],
                           "home_team": fixtures["home_team"], "away_team": fixtures["away_team"],
                           "dq_ok": True})
    if (future["date"] <= played["date"].max()).any():
        raise ValueError("fixture dated on or before the last played match")
    known = set(played.loc[played["season"] == season, "home_team"])
    unknown = (set(future["home_team"]) | set(future["away_team"])) - known
    if unknown:
        raise ValueError(f"teams not seen in {season} matches.csv: {unknown}")
    allm = pd.concat([played, future], ignore_index=True)

    # LR: features are built from earlier matches only; fixtures have no outcome columns
    feats = build_features(allm)
    cols = feature_columns(feats)
    is_fx = feats["result"].isna()
    order = list(dict.fromkeys(feats["season"]))
    train = feats[~is_fx & feats["season"].isin(order[order.index(FIRST_TRAIN_SEASON):])]
    fx_feats = feats[is_fx]
    p_lr = lr_model(cols, LR_C)(train, fx_feats)
    out = fx_feats[["season"] + KEYS + ["elo_home", "elo_away", "elo_diff"]].copy()
    out[["lr_p_home", "lr_p_draw", "lr_p_away"]] = p_lr

    # Dixon-Coles: fit on played matches before each fixture date (same as dc_predictions)
    df = _prep(allm)
    prior = PriorBook(df, DC_XI).get(season)
    dc_rows = {}
    for date in map(pd.Timestamp, sorted(out["date"].unique())):
        hist = _history(df, date)
        fit = fit_dc(hist, date, DC_XI, prior)
        for i, r in out[out["date"] == date].iterrows():
            dc_rows[i] = (*dc_predict(fit, r.home_team, r.away_team, prior), len(hist))
    dc = pd.DataFrame.from_dict(dc_rows, orient="index",
                                columns=["dc_p_home", "dc_p_draw", "dc_p_away",
                                         "dc_exp_home", "dc_exp_away", "dc_n_matches"])
    out = out.join(dc)

    w = BLEND_WEIGHT_LR
    for side in ["home", "draw", "away"]:
        out[f"p_{side}"] = w * out[f"lr_p_{side}"] + (1 - w) * out[f"dc_p_{side}"]
    assert np.allclose(out[["p_home", "p_draw", "p_away"]].sum(1), 1)
    out["lr_n_train"] = len(train)
    out["data_through"] = played["date"].max().date().isoformat()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gameweek", type=int, help="default: next gameweek with no finished matches")
    ap.add_argument("--season", help="e.g. 2026/27 (default: latest season in FPL-Core)")
    args = ap.parse_args()

    now = pd.Timestamp(datetime.now(timezone.utc))
    fpl = load_pl_matches()
    season = args.season or fpl["season"].max()
    gameweek, fixtures = load_fixtures(fpl, season, args.gameweek)
    path = gw_path(season, gameweek)
    if path.exists():
        sys.exit(f"{path} already exists. Saved predictions are never overwritten.")
    started = fixtures[fixtures["kickoff_utc"] <= now]
    if len(started):
        sys.exit(f"{len(started)} GW{gameweek} fixtures have already kicked off; "
                 "predictions must be made before kick-off.")

    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    check_data_current(matches, fpl, season)

    pred = build_predictions(matches, fixtures, season)
    meta = fixtures[["home_team", "away_team", "match_id", "kickoff_utc"]]
    pred = pred.merge(meta, on=["home_team", "away_team"], validate="1:1")
    pred.insert(1, "gameweek", gameweek)
    pred["generated_at"] = now.isoformat(timespec="seconds")
    cols = (["season", "gameweek", "match_id", "kickoff_utc", "date", "home_team", "away_team",
             "p_home", "p_draw", "p_away", "lr_p_home", "lr_p_draw", "lr_p_away",
             "dc_p_home", "dc_p_draw", "dc_p_away", "dc_exp_home", "dc_exp_away",
             "elo_home", "elo_away", "elo_diff", "dc_n_matches", "lr_n_train",
             "data_through", "generated_at"])
    pred = pred[cols]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", newline="") as f:  # "x": fail if the file appeared meanwhile
        pred.to_csv(f, index=False, float_format="%.6f")

    show = pred[["date", "home_team", "away_team", "p_home", "p_draw", "p_away",
                 "elo_diff", "dc_exp_home", "dc_exp_away"]].copy()
    show["date"] = show["date"].dt.date
    pd.set_option("display.width", 200)
    print(f"{season} GW{gameweek} (data through {pred['data_through'].iloc[0]}, "
          f"LR trained on {pred['lr_n_train'].iloc[0]} matches)")
    print(show.round(3).to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
