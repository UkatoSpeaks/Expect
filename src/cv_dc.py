"""Step 5: Dixon-Coles candidates on the same walk-forward folds as src/cv.py.

1. Tune DC time decay xi on pooled walk-forward log loss of DC alone.
2. Build pre-match DC probabilities for every match with the best xi -> data/dc_probs.csv.
3. Compare on 2017/18-2022/23 (pooled, 2,280 matches), with paired bootstrap CIs
   against the full LR (current pick) and against the market:
     a) DC alone
     b) Full LR + DC probabilities as extra features
     c) Simple average of full LR and DC probabilities
Test (2023/24-2024/25) and forward (2025/26+) seasons are not evaluated.
"""

from __future__ import annotations

import pandas as pd

from src.cv import (CLASSES, FOLD_SEASONS, REPORTS, bootstrap_ci, calibration_plot,
                    cv_only, lr_model, market_model, row_scores, walk_forward)
from src.dixon_coles import dc_predictions
from src.features import FEATURES_PATH, MATCHES_PATH, feature_columns

ROOT = MATCHES_PATH.parents[1]
DC_PATH = ROOT / "data" / "dc_probs.csv"
XI_GRID = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.006]  # per day
LR_C = 0.003
LR_DC_C_GRID = [0.001, 0.003, 0.01]
DC_COLS = ["dc_p_home", "dc_p_draw", "dc_p_away"]
KEYS = ["date", "home_team", "away_team"]


def dc_frame(p: pd.DataFrame) -> pd.DataFrame:
    return p[DC_COLS].set_axis(CLASSES, axis=1).assign(season=p["season"])


def tune_xi(matches: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    m = cv_only(matches)
    rows = []
    for xi in XI_GRID:
        p = dc_predictions(m, xi, seasons=FOLD_SEASONS)
        s = row_scores(p[DC_COLS].to_numpy(), _results(m, p)).mean()
        rows.append({"xi_per_day": xi, "half_life_days": round(0.693 / xi), **s})
        print(f"  xi={xi}: log loss {s.log_loss:.4f}")
    grid = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    return float(grid["xi_per_day"].iloc[0]), grid


def _results(matches: pd.DataFrame, p: pd.DataFrame) -> pd.Series:
    r = p[KEYS].merge(matches[KEYS + ["result"]].assign(date=lambda d: pd.to_datetime(d["date"])),
                      on=KEYS, how="left")
    return pd.Series(r["result"].to_numpy(), index=p.index)


def main() -> None:
    pd.set_option("display.width", 200)
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])

    print("Tuning DC xi (pooled walk-forward log loss of DC alone):")
    xi, grid = tune_xi(matches)
    grid.to_csv(REPORTS / "dc_xi_tuning.csv", index=False)
    print(grid.round(4).to_string())
    print("best xi:", xi)

    dc = dc_predictions(matches, xi)
    dc.assign(xi=xi).to_csv(DC_PATH, index=False)

    feats = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    df = cv_only(feats.merge(dc[KEYS + DC_COLS], on=KEYS, how="left", validate="1:1"))
    assert len(df) == len(cv_only(feats))
    y = df["result"]
    base_cols = feature_columns(feats)

    preds = {"Market": walk_forward(df, market_model),
             f"Full LR (C={LR_C})": walk_forward(df, lr_model(base_cols, LR_C))}

    dc_only = df[DC_COLS].set_axis(CLASSES, axis=1).assign(season=df["season"])
    preds["a) Dixon-Coles"] = dc_only[dc_only["season"].isin(FOLD_SEASONS)]

    best = None
    for C in LR_DC_C_GRID:
        p = walk_forward(df, lr_model(base_cols + DC_COLS, C))
        ll = row_scores(p[CLASSES].to_numpy(), y.loc[p.index])["log_loss"].mean()
        print(f"  LR + DC, C={C}: log loss {ll:.4f}")
        if best is None or ll < best[0]:
            best = (ll, C, p)
    preds[f"b) LR + DC features (C={best[1]})"] = best[2]

    lr = preds[f"Full LR (C={LR_C})"]
    avg = (lr[CLASSES] + preds["a) Dixon-Coles"].loc[lr.index, CLASSES]) / 2
    preds["c) Average(LR, DC)"] = avg.assign(season=lr["season"])

    scores = {n: row_scores(p[CLASSES].to_numpy(), y.loc[p.index]).assign(season=p["season"])
              for n, p in preds.items()}
    ref_lr = scores[f"Full LR (C={LR_C})"]
    rows = []
    for n, s in scores.items():
        row = {"model": n, **s[["log_loss", "brier", "accuracy"]].mean().round(4).to_dict()}
        for ref_name, ref in [("LR", ref_lr), ("mkt", scores["Market"])]:
            if s is ref:
                continue
            for metric in ["log_loss", "brier"]:
                d = (s[metric] - ref[metric]).to_numpy()
                lo, hi = bootstrap_ci(d)
                row[f"{metric} vs {ref_name}"] = f"{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"
        rows.append(row)
    pooled = pd.DataFrame(rows).set_index("model")
    pooled.to_csv(REPORTS / "cv_dc_pooled.csv")

    per_fold = pd.concat({n: s.groupby("season")["log_loss"].mean() for n, s in scores.items()},
                         axis=1).round(4)
    per_fold.to_csv(REPORTS / "cv_dc_folds.csv")
    print("\nPer-fold log loss:")
    print(per_fold.to_string())
    print("\nPooled walk-forward (2,280 matches); differences with 95% paired bootstrap CIs:")
    print(pooled.to_string())

    out = pd.concat(preds, names=["model"]).reset_index(level=0)
    out.join(df[KEYS + ["result"]]).to_csv(REPORTS / "cv_dc_predictions.csv")

    calibration_plot({"Full LR": lr, "Average(LR, DC)": preds["c) Average(LR, DC)"],
                      "Market": preds["Market"]}, y, REPORTS / "figures" / "calibration_dc.png")


if __name__ == "__main__":
    main()
