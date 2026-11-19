"""Step 4: walk-forward validation, Elo re-tuning, logistic regression and XGBoost.

Folds: for each season N in 2017/18..2022/23, train on 2001/02..N-1, evaluate on N.
Pooled metrics are computed over all 6 x 380 = 2,280 out-of-fold predictions.
The market (margin-free implied probabilities) is scored on the same matches as a benchmark.

Test (2023/24-2024/25) and forward (2025/26+) seasons are never read here.

Outputs: reports/cv_folds.csv, reports/cv_pooled.csv, reports/cv_predictions.csv,
reports/elo_tuning.csv, reports/xgb_grid.csv, reports/figures/calibration.png.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import (FEATURES_PATH, MATCHES_PATH, EloParams, build_features,
                          compute_elo, feature_columns, prepare_matches)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

CLASSES = ["H", "D", "A"]
FIRST_TRAIN = "2001/02"  # 2000/01 is Elo burn-in
FOLD_SEASONS = [f"{y}/{str(y + 1)[2:]}" for y in range(2017, 2023)]
HELD_OUT = {"2023/24", "2024/25", "2025/26", "2026/27"}  # test + forward: never used here
MARKET_COLS = ["p_home_implied", "p_draw_implied", "p_away_implied"]

ELO_GRID = {
    "k": [15, 20, 25, 30, 40, 50],
    "home_adv": [0, 50, 100],
    "regress": [0.0, 0.25, 1 / 3, 0.5],
    "gd_multiplier": [False, True],
}
LR_C_GRID = [0.001, 0.003, 0.01, 0.1, 1.0]
XGB_FIXED = dict(objective="multi:softprob", num_class=3, learning_rate=0.03,
                 subsample=0.8, colsample_bytree=0.8, n_estimators=3000,
                 early_stopping_rounds=100, eval_metric="mlogloss", tree_method="hist",
                 n_jobs=4, random_state=0)
XGB_GRID = {"max_depth": [2, 3, 4], "min_child_weight": [20, 50], "reg_lambda": [1.0, 10.0]}

N_BOOT = 5000


# --------------------------------------------------------------------------- scoring

def onehot(y: pd.Series) -> np.ndarray:
    return pd.get_dummies(y).reindex(columns=CLASSES, fill_value=0).to_numpy(float)


def row_scores(p: np.ndarray, y: pd.Series) -> pd.DataFrame:
    """Per-match log loss, Brier (summed over 3 classes) and correctness."""
    Y = onehot(y)
    return pd.DataFrame({
        "log_loss": -np.log(np.clip((p * Y).sum(1), 1e-15, 1)),
        "brier": ((p - Y) ** 2).sum(1),
        "accuracy": (p.argmax(1) == Y.argmax(1)).astype(float),
    }, index=y.index)


def bootstrap_ci(diff: np.ndarray, n: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of the mean of paired per-match differences."""
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------------- folds

def folds(df: pd.DataFrame):
    seasons = list(dict.fromkeys(df["season"]))
    assert not HELD_OUT & set(df["season"]), "held-out seasons must be removed before CV"
    for s in FOLD_SEASONS:
        train_seasons = seasons[seasons.index(FIRST_TRAIN):seasons.index(s)]
        yield s, df[df["season"].isin(train_seasons)], df[df["season"] == s]


def walk_forward(df: pd.DataFrame, fit_predict) -> pd.DataFrame:
    """Out-of-fold probabilities (columns H, D, A) for every fold-season match."""
    out = []
    for season, tr, va in folds(df):
        p = fit_predict(tr, va)
        out.append(pd.DataFrame(p, columns=CLASSES, index=va.index).assign(season=season))
    return pd.concat(out)


def cv_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["season"].isin(HELD_OUT)]


# --------------------------------------------------------------------------- models

def _lr(C: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=3000)),
    ])


def _ordered(model, proba: np.ndarray) -> np.ndarray:
    return proba[:, [list(model.classes_).index(c) for c in CLASSES]]


def lr_model(cols: list[str], C: float = 1.0):
    def fit_predict(tr, va):
        m = _lr(C).fit(tr[cols], tr["result"])
        return _ordered(m, m.predict_proba(va[cols]))
    return fit_predict


def xgb_model(cols: list[str], params: dict, log: list | None = None):
    """Early stopping on the last training season, then refit on all training seasons
    with the chosen number of trees."""
    from xgboost import XGBClassifier

    label = {c: i for i, c in enumerate(CLASSES)}

    def fit_predict(tr, va):
        last = tr["season"].iloc[-1]
        inner_tr, inner_va = tr[tr["season"] != last], tr[tr["season"] == last]
        m = XGBClassifier(**XGB_FIXED, **params)
        m.fit(inner_tr[cols], inner_tr["result"].map(label),
              eval_set=[(inner_va[cols], inner_va["result"].map(label))], verbose=False)
        n_trees = m.best_iteration + 1
        final = XGBClassifier(**{**XGB_FIXED, "n_estimators": n_trees,
                                 "early_stopping_rounds": None}, **params)
        final.fit(tr[cols], tr["result"].map(label), verbose=False)
        if log is not None:
            log.append({"fold": va["season"].iloc[0], "key": _key(params), "n_trees": n_trees})
        return final.predict_proba(va[cols])  # classes 0,1,2 = H,D,A
    return fit_predict


def _key(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def market_model(tr, va):
    return va[MARKET_COLS].to_numpy()


# --------------------------------------------------------------------------- Elo tuning

def tune_elo(matches: pd.DataFrame) -> tuple[EloParams, pd.DataFrame]:
    df = prepare_matches(cv_only(matches))
    rows = []
    for k, ha, reg, gd in itertools.product(*ELO_GRID.values()):
        params = EloParams(k=k, home_adv=ha, regress=reg, gd_multiplier=gd)
        d = df[["season", "result"]].join(compute_elo(df, params))
        p = walk_forward(d, lr_model(["elo_diff"]))
        s = row_scores(p[CLASSES].to_numpy(), d.loc[p.index, "result"]).mean()
        rows.append({"k": k, "home_adv": ha, "regress": round(reg, 4), "gd_multiplier": gd, **s})
    grid = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    b = grid.iloc[0]
    best = EloParams(k=float(b.k), home_adv=float(b.home_adv), regress=float(b.regress),
                     gd_multiplier=bool(b.gd_multiplier))
    return best, grid


# --------------------------------------------------------------------------- reporting

def summarize(preds: dict[str, pd.DataFrame], y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = {name: row_scores(p[CLASSES].to_numpy(), y.loc[p.index]).assign(season=p["season"])
              for name, p in preds.items()}
    per_fold = pd.concat({n: s.groupby("season").mean() for n, s in scores.items()},
                         names=["model"]).round(4)

    mkt = scores["Market"]
    rows = []
    for name, s in scores.items():
        row = {"model": name, **s[["log_loss", "brier", "accuracy"]].mean().to_dict()}
        if name != "Market":
            for metric in ["log_loss", "brier", "accuracy"]:
                d = (s[metric] - mkt[metric]).to_numpy()
                lo, hi = bootstrap_ci(d)
                row[f"d_{metric}_vs_mkt"] = d.mean()
                row[f"d_{metric}_ci"] = f"[{lo:+.4f}, {hi:+.4f}]"
        rows.append(row)
    return per_fold, pd.DataFrame(rows).set_index("model")


CV_CAL_TITLE = "Calibration, pooled walk-forward 2017/18–2022/23 (2,280 matches, 10 equal-count bins)"


def calibration_plot(preds: dict[str, pd.DataFrame], y: pd.Series, path: Path,
                     title: str = CV_CAL_TITLE) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # validated categorical slots 1-3 and recessive chrome (same as notebooks/01_eda.ipynb)
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 11, "font.size": 10,
    })
    Y = onehot(y)
    names = {"H": "Home win", "D": "Draw", "A": "Away win"}
    n = len(preds)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.9), sharey=True, layout="constrained")
    for j, (cls, ax) in enumerate(zip(CLASSES, axes)):
        ax.plot([0, 0.8], [0, 0.8], color=INK2, lw=1, ls="--", label="Perfect calibration")
        for (name, p), color in zip(preds.items(), colors):
            prob = p[cls].to_numpy()
            obs = Y[y.index.get_indexer(p.index), j]
            # equal-count bins (10% of matches each)
            bins = np.quantile(prob, np.linspace(0, 1, 11))
            idx = np.clip(np.digitize(prob, bins[1:-1]), 0, 9)
            g = pd.DataFrame({"b": idx, "p": prob, "o": obs}).groupby("b").mean()
            ax.plot(g["p"], g["o"], color=color, lw=2, marker="o", ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8); ax.set_aspect("equal")
        ax.set_title(names[cls]); ax.set_xlabel("Predicted probability")
    axes[0].set_ylabel("Observed frequency")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=n + 1, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main

def main() -> None:
    pd.set_option("display.width", 200)
    REPORTS.mkdir(exist_ok=True)
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])

    # 1. Elo re-tuning on pooled walk-forward log loss
    best_elo, grid = tune_elo(matches)
    grid.to_csv(REPORTS / "elo_tuning.csv", index=False)
    print(f"Elo grid ({len(grid)} combos), top 8 by pooled walk-forward log loss:")
    print(grid.head(8).round(4).to_string())
    print("current EloParams default:", EloParams())
    print("best:", best_elo)

    # 2. rebuild features with the tuned Elo (full history, so features.csv stays complete)
    feats = build_features(matches, best_elo)
    feats.to_csv(FEATURES_PATH, index=False)
    df = cv_only(feats)
    y = df["result"]
    cols = feature_columns(df)

    # 3. models
    preds = {"Market": walk_forward(df, market_model),
             "Elo-only LR": walk_forward(df, lr_model(["elo_diff"]))}
    lr_scores = {}
    for C in LR_C_GRID:
        p = walk_forward(df, lr_model(cols, C))
        lr_scores[C] = row_scores(p[CLASSES].to_numpy(), y.loc[p.index])["log_loss"].mean()
        preds[f"Full LR (C={C})"] = p
    print("\nFull LR pooled log loss by C:", {c: round(v, 4) for c, v in lr_scores.items()})

    xgb_rows, xgb_preds, tree_log = [], {}, []
    for combo in itertools.product(*XGB_GRID.values()):
        params = dict(zip(XGB_GRID, combo))
        p = walk_forward(df, xgb_model(cols, params, tree_log))
        s = row_scores(p[CLASSES].to_numpy(), y.loc[p.index]).mean()
        key = _key(params)
        xgb_rows.append({"key": key, **params, **s})
        xgb_preds[key] = p
        print(f"  XGB {key}: log loss {s.log_loss:.4f}")
    xgb_grid = pd.DataFrame(xgb_rows).sort_values("log_loss").reset_index(drop=True)
    xgb_grid.to_csv(REPORTS / "xgb_grid.csv", index=False)
    trees = pd.DataFrame(tree_log)
    trees.to_csv(REPORTS / "xgb_trees.csv", index=False)
    print("\nXGB grid by pooled log loss:")
    print(xgb_grid.round(4).to_string())

    best_C = min(lr_scores, key=lr_scores.get)
    best_x = xgb_grid["key"].iloc[0]
    preds = {"Market": preds["Market"], "Elo-only LR": preds["Elo-only LR"],
             f"Full LR (C={best_C})": preds[f"Full LR (C={best_C})"],
             f"XGB ({best_x})": xgb_preds[best_x]}
    best_trees = trees[trees["key"] == best_x]
    print("trees chosen per fold for best XGB:", best_trees.set_index("fold")["n_trees"].to_dict())

    # 4. tables
    per_fold, pooled = summarize(preds, y)
    per_fold.to_csv(REPORTS / "cv_folds.csv")
    pooled.to_csv(REPORTS / "cv_pooled.csv")
    print("\nPer-fold:")
    print(per_fold.unstack("model").swaplevel(axis=1).sort_index(axis=1).to_string())
    print("\nPooled walk-forward (2,280 matches), differences vs market with 95% bootstrap CIs:")
    print(pooled.round(4).to_string())

    # pairwise CI between the two best models
    names = list(preds)
    a, b = names[2], names[3]
    sa = row_scores(preds[a][CLASSES].to_numpy(), y.loc[preds[a].index])
    sb = row_scores(preds[b][CLASSES].to_numpy(), y.loc[preds[b].index])
    for m in ["log_loss", "brier"]:
        d = (sb[m] - sa[m]).to_numpy()
        print(f"{b} minus {a}, {m}: {d.mean():+.4f} {bootstrap_ci(d)}")

    out = pd.concat({n: p for n, p in preds.items()}, names=["model"]).reset_index(level=0)
    out = out.join(df[["date", "home_team", "away_team", "result"]])
    out.to_csv(REPORTS / "cv_predictions.csv")

    # 5. calibration
    calibration_plot({"Full LR": preds[a], "XGBoost": preds[b], "Market": preds["Market"]}, y,
                     REPORTS / "figures" / "calibration.png")
    print("wrote", REPORTS / "figures" / "calibration.png")


if __name__ == "__main__":
    main()
