"""Pre-match features: one row per match, every feature built only from earlier matches.

Input: data/matches.csv (from src/load_data.py). Output: data/features.csv.

Features
--------
- Elo (own implementation): elo_home, elo_away, elo_diff (pre-match ratings).
- Rolling form over the team's last 5 / 10 league matches (shift(1) before rolling,
  carried across seasons): points, goals, shots, shots on target, corners, for and against.
- Venue form: home team's last 5 home games, away team's last 5 away games.
- Head-to-head: home team's mean points over the last 5 meetings (either venue), NaN if none.
- Rest days since each team's previous league match (capped at REST_CAP).
- Flags: promoted_home / promoted_away, no_crowd.
- Differences (home minus away) for the main features.

Leakage rules
-------------
- Rows with dq_ok == False have their shot / shots-on-target stats treated as NaN.
- Each team's rows are shifted by one match before any rolling window, so the
  current match never contributes to its own features.
- "Relegated" teams (used to seed promoted teams' Elo) are the bottom 3 of the previous
  season's final table, which is known before the new season starts.
- Odds are not used as features (benchmark only).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ELO as _ELO

ROOT = Path(__file__).resolve().parents[1]
MATCHES_PATH = ROOT / "data" / "matches.csv"
FEATURES_PATH = ROOT / "data" / "features.csv"

WINDOWS = (5, 10)
VENUE_WINDOW = 5
H2H_WINDOW = 5
REST_CAP = 21  # longer gaps are summer / international breaks; the exact length carries no signal

# Matches played behind closed doors: all of 2020/21 and the 2019/20 restart (from 17 June 2020).
NO_CROWD_START = pd.Timestamp("2020-06-17")
NO_CROWD_END = pd.Timestamp("2021-05-23")

# per-team stats used for rolling form: name -> (home-side column, away-side column)
TEAM_STATS = {
    "goals": ("home_goals", "away_goals"),
    "shots": ("home_shots", "away_shots"),
    "sot": ("home_shots_on_target", "away_shots_on_target"),
    "corners": ("home_corners", "away_corners"),
}
SHOT_COLS = ["home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target"]
# columns describing the match outcome; no feature may depend on these for the same match
OUTCOME_COLS = ["home_goals", "away_goals", "result", "ht_home_goals", "ht_away_goals",
                "ht_result", "home_shots", "away_shots", "home_shots_on_target",
                "away_shots_on_target", "home_corners", "away_corners"]


# --------------------------------------------------------------------------- Elo

@dataclass(frozen=True)
class EloParams:
    # Defaults are the frozen values in src/config.py (tuned in src/cv.py).
    k: float = _ELO["k"]
    home_adv: float = _ELO["home_adv"]
    regress: float = _ELO["regress"]  # fraction pulled back toward `base` between seasons
    gd_multiplier: bool = _ELO["gd_multiplier"]  # World Football Elo goal-difference multiplier
    base: float = _ELO["base"]


def _gd_mult(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


def season_tables(df: pd.DataFrame) -> dict[str, list[str]]:
    """Final table per season: teams sorted best -> worst (points, goal difference, goals)."""
    rows = []
    for side, opp in [("home", "away"), ("away", "home")]:
        gf, ga = df[f"{side}_goals"], df[f"{opp}_goals"]
        rows.append(pd.DataFrame({
            "season": df["season"], "team": df[f"{side}_team"],
            "pts": np.select([gf > ga, gf == ga], [3, 1], 0), "gd": gf - ga, "gf": gf,
        }))
    long = pd.concat(rows).groupby(["season", "team"])[["pts", "gd", "gf"]].sum()
    return {s: g.sort_values(["pts", "gd", "gf"], ascending=False).index.get_level_values("team").tolist()
            for s, g in long.groupby(level="season")}


def compute_elo(df: pd.DataFrame, params: EloParams = EloParams()) -> pd.DataFrame:
    """Pre-match Elo for each row of `df` (must be sorted by date). Returns elo_home/away/diff.

    - First season: everyone starts at `base`.
    - Season boundary: all ratings regress toward `base` by `params.regress`.
    - Promoted team (not in the previous season): starts at the mean (regressed) rating
      of the previous season's bottom three.
    """
    seasons = df["season"].to_numpy()
    home, away = df["home_team"].to_numpy(), df["away_team"].to_numpy()
    hg, ag = df["home_goals"].to_numpy(dtype=float), df["away_goals"].to_numpy(dtype=float)
    season_order = list(dict.fromkeys(seasons))
    tables = season_tables(df)

    ratings: dict[str, float] = {}
    prev_teams: set[str] = set()
    relegated_seed = params.base
    cur_season = None
    seeded: set[str] = set()  # promoted teams already seeded this season
    elo_h = np.empty(len(df))
    elo_a = np.empty(len(df))

    for i in range(len(df)):
        s = seasons[i]
        if s != cur_season:
            if cur_season is not None:
                prev_teams = set(tables[cur_season])
                for t in ratings:
                    ratings[t] += params.regress * (params.base - ratings[t])
                bottom3 = tables[cur_season][-3:]
                relegated_seed = float(np.mean([ratings[t] for t in bottom3]))
            cur_season = s
            seeded = set()
            first_season = s == season_order[0]
        for t in (home[i], away[i]):
            if not first_season and t not in prev_teams and t not in seeded:
                ratings[t] = relegated_seed
                seeded.add(t)
            ratings.setdefault(t, params.base)
        rh, ra = ratings[home[i]], ratings[away[i]]
        elo_h[i], elo_a[i] = rh, ra
        if np.isnan(hg[i]) or np.isnan(ag[i]):
            continue  # unplayed fixture: no update
        exp_h = 1.0 / (1.0 + 10 ** ((ra - rh - params.home_adv) / 400.0))
        score_h = 1.0 if hg[i] > ag[i] else 0.5 if hg[i] == ag[i] else 0.0
        k = params.k * (_gd_mult(int(hg[i] - ag[i])) if params.gd_multiplier else 1.0)
        delta = k * (score_h - exp_h)
        ratings[home[i]] = rh + delta
        ratings[away[i]] = ra - delta

    return pd.DataFrame({"elo_home": elo_h, "elo_away": elo_a, "elo_diff": elo_h - elo_a},
                        index=df.index)


# --------------------------------------------------------------------------- team form

def _team_long(df: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match (one per team) with for/against stats from that team's view."""
    parts = []
    for side, opp, is_home in [("home", "away", 1), ("away", "home", 0)]:
        p = pd.DataFrame({
            "match_idx": df.index, "date": df["date"], "season": df["season"],
            "team": df[f"{side}_team"], "opp": df[f"{opp}_team"], "is_home": is_home,
        })
        for name, (hcol, acol) in TEAM_STATS.items():
            mine, theirs = (hcol, acol) if is_home else (acol, hcol)
            p[f"{name}_for"] = df[mine].astype(float)
            p[f"{name}_against"] = df[theirs].astype(float)
        gf, ga = p["goals_for"], p["goals_against"]
        p["points"] = np.where(gf.isna() | ga.isna(), np.nan,
                               np.select([gf > ga, gf == ga], [3.0, 1.0], 0.0))
        parts.append(p)
    long = pd.concat(parts, ignore_index=True)
    return long.sort_values(["date", "match_idx", "is_home"], kind="stable").reset_index(drop=True)


def _prior_rolling(long: pd.DataFrame, keys: list[str], cols: list[str], window: int) -> pd.DataFrame:
    """Mean of each col over the previous `window` rows of the same group (current row excluded)."""
    shifted = long.groupby(keys, sort=False)[cols].shift(1)
    shifted[keys] = long[keys]
    out = (shifted.groupby(keys, sort=False)[cols]
           .rolling(window, min_periods=1).mean()
           .reset_index(level=list(range(len(keys))), drop=True))
    return out.reindex(long.index)


FORM_COLS = ["points"] + [f"{n}_{d}" for n in TEAM_STATS for d in ("for", "against")]
VENUE_COLS = ["points", "goals_for", "goals_against"]


def team_features(df: pd.DataFrame) -> pd.DataFrame:
    long = _team_long(df)
    feats = pd.DataFrame(index=long.index)

    for w in WINDOWS:
        r = _prior_rolling(long, ["team"], FORM_COLS, w)
        feats[[f"{c}_l{w}" for c in FORM_COLS]] = r.to_numpy()

    r = _prior_rolling(long, ["team", "is_home"], VENUE_COLS, VENUE_WINDOW)
    feats[[f"venue_{c}_l{VENUE_WINDOW}" for c in VENUE_COLS]] = r.to_numpy()

    r = _prior_rolling(long, ["team", "opp"], ["points"], H2H_WINDOW)
    feats[f"h2h_points_l{H2H_WINDOW}"] = r["points"].to_numpy()
    feats["h2h_n"] = long.groupby(["team", "opp"], sort=False).cumcount().clip(upper=H2H_WINDOW)

    prev_date = long.groupby("team", sort=False)["date"].shift(1)
    feats["rest_days"] = (long["date"] - prev_date).dt.days.clip(upper=REST_CAP)

    feats["match_idx"] = long["match_idx"]
    feats["is_home"] = long["is_home"]
    return feats


def promoted_flags(df: pd.DataFrame) -> pd.DataFrame:
    """1 if the team was not in the previous season's league, NaN in the first season."""
    teams_by_season = {s: set(g["home_team"]) | set(g["away_team"]) for s, g in df.groupby("season")}
    order = list(dict.fromkeys(df["season"]))
    prev = {s: (teams_by_season[order[i - 1]] if i > 0 else None) for i, s in enumerate(order)}
    out = {}
    for side in ("home", "away"):
        out[f"promoted_{side}"] = [
            np.nan if prev[s] is None else float(t not in prev[s])
            for s, t in zip(df["season"], df[f"{side}_team"])
        ]
    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------- assembly

DIFF_FEATURES = (
    ["elo"]
    + [f"{c}_l{w}" for w in WINDOWS for c in FORM_COLS]
    + [f"venue_{c}_l{VENUE_WINDOW}" for c in VENUE_COLS]
    + ["rest_days"]
)


def prepare_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)
    bad = ~df["dq_ok"].astype(bool)
    df[SHOT_COLS] = df[SHOT_COLS].astype(float)
    df.loc[bad, SHOT_COLS] = np.nan
    return df


def build_features(matches: pd.DataFrame, elo_params: EloParams = EloParams()) -> pd.DataFrame:
    """One row per match: identifiers, target, market probs (benchmark only), features."""
    df = prepare_matches(matches)
    id_cols = ["season", "date", "home_team", "away_team", "result",
               "p_home_implied", "p_draw_implied", "p_away_implied"]
    out = df[[c for c in id_cols if c in df.columns]].copy()

    out = out.join(compute_elo(df, elo_params))

    tf = team_features(df)
    feat_cols = [c for c in tf.columns if c not in ("match_idx", "is_home")]
    for side, flag in (("home", 1), ("away", 0)):
        part = tf[tf["is_home"] == flag].set_index("match_idx")[feat_cols]
        out = out.join(part.add_suffix(f"_{side}"))

    # h2h is symmetric in information; keep the home team's view only
    out = out.drop(columns=[f"h2h_points_l{H2H_WINDOW}_away", "h2h_n_away"])
    out = out.rename(columns={f"h2h_points_l{H2H_WINDOW}_home": f"h2h_points_l{H2H_WINDOW}",
                              "h2h_n_home": "h2h_n"})

    out = out.join(promoted_flags(df))
    out["no_crowd"] = df["date"].between(NO_CROWD_START, NO_CROWD_END).astype(float)

    for f in DIFF_FEATURES:
        if f == "elo":
            continue  # elo_diff already present
        out[f"{f}_diff"] = out[f"{f}_home"] - out[f"{f}_away"]
    return out


ID_COLS = ["season", "date", "home_team", "away_team", "result",
           "p_home_implied", "p_draw_implied", "p_away_implied"]


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c not in ID_COLS]


def main() -> None:
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    feats = build_features(matches)
    feats.to_csv(FEATURES_PATH, index=False)
    print(f"Wrote {len(feats)} rows x {len(feature_columns(feats))} features -> {FEATURES_PATH}")


if __name__ == "__main__":
    main()
