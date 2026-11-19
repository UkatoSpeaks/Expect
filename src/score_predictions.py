"""Step 6: score saved live predictions against results and the market.

    .venv/bin/python -m src.score_predictions

For every predictions/<season>/gw*.csv, matches that now have a result in data/matches.csv
are scored (log loss, Brier, correct pick) for the model and for the market's margin-free
closing probabilities, and appended to predictions/track_record.csv.

The track record is append-only: a match already in it is never rescored or rewritten.
Matches are joined on season + home team + away team (each pairing happens once a season),
so a rescheduled fixture is still matched; its played date is recorded. A postponed match
predicted again in predictions/<season>/gwN_rescheduled_<date>.csv is scored on that latest
prediction (the original gameweek file is left as it was).
Refresh data/raw/E0_<season>.csv and rerun src.load_data first, so results and closing
odds are current.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.cv import CLASSES, bootstrap_ci, row_scores
from src.features import MATCHES_PATH
from src.predict import PRED_DIR

TRACK = PRED_DIR / "track_record.csv"
JOIN = ["season", "home_team", "away_team"]
MODEL_P = ["p_home", "p_draw", "p_away"]
MKT_P = ["mkt_p_home", "mkt_p_draw", "mkt_p_away"]
COLUMNS = (["season", "gameweek", "match_id", "kickoff_utc", "date_played", "home_team",
            "away_team", "home_goals", "away_goals", "result"] + MODEL_P + MKT_P
           + ["model_log_loss", "mkt_log_loss", "model_brier", "mkt_brier",
              "model_correct", "mkt_correct", "generated_at", "made_before_kickoff",
              "prediction_file", "scored_at"])


def load_saved_predictions(pred_dir: Path = PRED_DIR) -> pd.DataFrame:
    files = sorted(pred_dir.glob("*/gw*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f).assign(prediction_file=str(f.relative_to(pred_dir)))
                      for f in files], ignore_index=True)


def load_track(path: Path = TRACK) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    t = pd.read_csv(path)
    if list(t.columns) != COLUMNS:
        raise ValueError(f"{path} has unexpected columns; refusing to append")
    return t


def is_rescheduled(prediction_file: pd.Series) -> pd.Series:
    return prediction_file.str.contains("_rescheduled_", regex=False)


def latest_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per match. A postponed match can be predicted again on its own
    (gwN_rescheduled_<date>.csv); the latest rescheduled prediction supersedes the gameweek
    file's row. The same match in two gameweek files is an error."""
    if preds.empty:
        return preds
    resched = is_rescheduled(preds["prediction_file"])
    dup = preds[~resched].duplicated(JOIN, keep=False)
    if dup.any():
        raise ValueError(f"match predicted in more than one gameweek file:\n"
                         f"{preds[~resched].loc[dup, JOIN + ['prediction_file']]}")
    order = preds.assign(_r=resched, _t=pd.to_datetime(preds["generated_at"], utc=True))
    order = order.sort_values(["_r", "_t", "prediction_file"], kind="stable")
    return order.drop_duplicates(JOIN, keep="last").drop(columns=["_r", "_t"]).sort_index()


def score_new(preds: pd.DataFrame, matches: pd.DataFrame, track: pd.DataFrame) -> pd.DataFrame:
    preds = latest_predictions(preds)
    played = matches[matches["result"].notna()].rename(columns={"date": "date_played"})
    new = preds.merge(played[JOIN + ["date_played", "home_goals", "away_goals", "result",
                                     "p_home_implied", "p_draw_implied", "p_away_implied"]],
                      on=JOIN, how="inner", validate="1:1")
    done = set(map(tuple, track[JOIN].to_numpy()))
    new = new[[tuple(r) not in done for r in new[JOIN].to_numpy()]].copy()
    if new.empty:
        return new

    new[MKT_P] = new[["p_home_implied", "p_draw_implied", "p_away_implied"]].to_numpy()
    y = new["result"]
    s = row_scores(new[MODEL_P].to_numpy(), y)
    new["model_log_loss"], new["model_brier"], new["model_correct"] = s["log_loss"], s["brier"], s["accuracy"]
    has_mkt = new[MKT_P].notna().all(1)
    m = row_scores(new[MKT_P].fillna(1 / 3).to_numpy(), y).where(has_mkt)
    new["mkt_log_loss"], new["mkt_brier"], new["mkt_correct"] = m["log_loss"], m["brier"], m["accuracy"]
    new["made_before_kickoff"] = (pd.to_datetime(new["generated_at"], utc=True)
                                  < pd.to_datetime(new["kickoff_utc"], utc=True))
    new["date_played"] = pd.to_datetime(new["date_played"]).dt.date.astype(str)
    new["scored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return new[COLUMNS].sort_values(["season", "gameweek", "kickoff_utc", "home_team"])


def summary(track: pd.DataFrame) -> None:
    t = track[track["made_before_kickoff"].astype(bool)]
    if t.empty:
        print("track record is empty")
        return
    both = t.dropna(subset=["mkt_log_loss"])
    print(f"\nTrack record: {len(t)} matches scored ({len(both)} with market odds). "
          "Differences are model minus market, 95% paired bootstrap CI.")
    by_gw = t.groupby(["season", "gameweek"]).agg(
        n=("result", "size"), model_ll=("model_log_loss", "mean"), mkt_ll=("mkt_log_loss", "mean"),
        model_brier=("model_brier", "mean"), mkt_brier=("mkt_brier", "mean"),
        model_acc=("model_correct", "mean"), mkt_acc=("mkt_correct", "mean"))
    print(by_gw.round(4).to_string())
    for metric in ["log_loss", "brier", "correct"]:
        d = (both[f"model_{metric}"] - both[f"mkt_{metric}"]).to_numpy(float)
        lo, hi = bootstrap_ci(d) if len(d) > 1 else (np.nan, np.nan)
        print(f"  {metric:9s} model {both[f'model_{metric}'].mean():.4f}  "
              f"market {both[f'mkt_{metric}'].mean():.4f}  diff {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    late = len(track) - len(t)
    if late:
        print(f"  ({late} predictions made after kick-off are recorded but excluded here)")


def main() -> None:
    preds = load_saved_predictions()
    if preds.empty:
        print(f"no saved predictions under {PRED_DIR}")
        return
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    track = load_track()
    new = score_new(preds, matches, track)
    pending = len(latest_predictions(preds)) - len(track) - len(new)
    if len(new):
        new.to_csv(TRACK, mode="a", header=not TRACK.exists(), index=False, float_format="%.6f")
        print(f"appended {len(new)} matches to {TRACK}")
    else:
        print("no newly finished predicted matches")
    print(f"{pending} predicted matches still awaiting a result")
    full = new if track.empty else track if new.empty else pd.concat([track, new], ignore_index=True)
    if len(full):
        summary(full)


if __name__ == "__main__":
    main()
