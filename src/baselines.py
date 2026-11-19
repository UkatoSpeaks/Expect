"""Step 3 baselines on 2022/23 only. SUPERSEDED by src/cv.py (walk-forward, Elo tuned there).

- Always home: accuracy of predicting H; probabilities = training-set class frequencies.
- Elo-only multinomial logistic regression (elo_diff).
- Full-feature multinomial logistic regression (impute -> scale -> LR in one Pipeline).
- Market benchmark: margin-free implied probabilities (benchmark only, never a feature).

Elo K / home advantage / goal-difference multiplier are tuned on 2022/23 log loss of the
Elo-only model. Test (2023/24-2024/25) and forward (2025/26+) seasons are not touched.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import (MATCHES_PATH, EloParams, build_features, compute_elo,
                          feature_columns, prepare_matches)

CLASSES = ["H", "D", "A"]
BURN_IN = ["2000/01"]
TRAIN = [f"{y}/{str(y + 1)[2:]}" for y in range(2001, 2022)]
VAL = ["2022/23"]

ELO_GRID = {"k": [10, 15, 20, 25, 30, 40, 50, 60], "home_adv": [0, 40, 80, 120], "gd_multiplier": [False, True]}


def onehot(y: pd.Series) -> np.ndarray:
    return pd.get_dummies(y).reindex(columns=CLASSES, fill_value=0).to_numpy(float)


def metrics(p: np.ndarray, y: pd.Series) -> dict[str, float]:
    """p columns ordered H, D, A. Brier = sum over the 3 classes, averaged over matches."""
    Y = onehot(y)
    p = np.clip(p, 1e-15, 1)
    return {
        "accuracy": float((p.argmax(1) == Y.argmax(1)).mean()),
        "log_loss": float(-np.log((p * Y).sum(1)).mean()),
        "brier": float(((p - Y) ** 2).sum(1).mean()),
    }


def lr_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=2000)),
    ])


def fit_predict(train: pd.DataFrame, val: pd.DataFrame, cols: list[str], C: float = 1.0) -> np.ndarray:
    model = lr_pipeline(C).fit(train[cols], train["result"])
    proba = model.predict_proba(val[cols])
    order = [list(model.classes_).index(c) for c in CLASSES]
    return proba[:, order]


def tune_elo(matches: pd.DataFrame) -> tuple[EloParams, pd.DataFrame]:
    df = prepare_matches(matches)
    rows = []
    for k, ha, gd in itertools.product(*ELO_GRID.values()):
        params = EloParams(k=k, home_adv=ha, gd_multiplier=gd)
        d = df[["season", "result"]].join(compute_elo(df, params))
        tr, va = d[d.season.isin(TRAIN)], d[d.season.isin(VAL)]
        m = metrics(fit_predict(tr, va, ["elo_diff"]), va["result"])
        rows.append({"k": k, "home_adv": ha, "gd_multiplier": gd, **m})
    grid = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    best = grid.iloc[0]
    return EloParams(k=float(best.k), home_adv=float(best.home_adv),
                     gd_multiplier=bool(best.gd_multiplier)), grid


def evaluate(feats: pd.DataFrame) -> pd.DataFrame:
    tr, va = feats[feats.season.isin(TRAIN)], feats[feats.season.isin(VAL)]
    y = va["result"]
    res = {}

    freq = tr["result"].value_counts(normalize=True).reindex(CLASSES).to_numpy()
    res["Always home (train base rates)"] = metrics(np.tile(freq, (len(va), 1)), y)
    res["Elo-only LR"] = metrics(fit_predict(tr, va, ["elo_diff"]), y)
    res["Full-feature LR"] = metrics(fit_predict(tr, va, feature_columns(feats)), y)
    mkt = va[["p_home_implied", "p_draw_implied", "p_away_implied"]].to_numpy()
    res["Market (closing, benchmark)"] = metrics(mkt, y)
    return pd.DataFrame(res).T


def main() -> None:
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    best, grid = tune_elo(matches)
    pd.set_option("display.width", 140)
    print("Elo tuning on 2022/23 (Elo-only LR log loss), top 10 of", len(grid))
    print(grid.head(10).round(4).to_string())
    print("default EloParams:", EloParams(), "\nbest:", best)

    feats = build_features(matches, best)
    print(f"\nTrain {TRAIN[0]}-{TRAIN[-1]} ({feats.season.isin(TRAIN).sum()} matches), "
          f"validation {VAL[0]} ({feats.season.isin(VAL).sum()} matches), "
          f"{len(feature_columns(feats))} features")
    print(evaluate(feats).round(4).to_string())


if __name__ == "__main__":
    main()
