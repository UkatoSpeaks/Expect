import numpy as np
import pandas as pd
import pytest

from src.features import (MATCHES_PATH, OUTCOME_COLS, EloParams, build_features,
                          compute_elo, feature_columns, prepare_matches)

KEYS = ["date", "home_team", "away_team"]


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return pd.read_csv(MATCHES_PATH, parse_dates=["date"])


@pytest.fixture(scope="module")
def full(matches) -> pd.DataFrame:
    return build_features(matches)


def _scramble_outcome(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Overwrite a match's outcome with extreme values, so any leak would change features."""
    df = df.copy()
    df.loc[mask, ["home_goals", "ht_home_goals"]] = 9
    df.loc[mask, ["away_goals", "ht_away_goals"]] = 0
    df.loc[mask, ["result", "ht_result"]] = "H"
    df.loc[mask, ["home_shots", "home_shots_on_target", "home_corners"]] = 40
    df.loc[mask, ["away_shots", "away_shots_on_target", "away_corners"]] = 0
    return df


def _row(feats: pd.DataFrame, key: pd.Series) -> pd.Series:
    m = (feats[KEYS] == key[KEYS].values).all(axis=1)
    assert m.sum() == 1
    return feats.loc[m, feature_columns(feats)].iloc[0]


def _sample(matches: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    # random matches plus edge cases: first match of a season and promoted teams' debuts
    rng = np.random.default_rng(seed)
    picks = matches.iloc[rng.choice(len(matches), n, replace=False)]
    firsts = matches.sort_values("date").groupby("season").head(1).sample(3, random_state=seed)
    return pd.concat([picks, firsts]).drop_duplicates(KEYS)


def test_features_use_only_prior_matches(matches, full):
    """Recompute each sampled match's features from rows dated strictly before it
    (plus the match itself with a scrambled outcome) and compare to the full build."""
    for _, target in _sample(matches, 25, seed=0).iterrows():
        is_target = (matches[KEYS] == target[KEYS].values).all(axis=1)
        prior = matches[(matches["date"] < target["date"]) | is_target]
        prior = _scramble_outcome(prior, is_target.loc[prior.index])
        got = _row(build_features(prior), target)
        want = _row(full, target)
        pd.testing.assert_series_equal(got, want, check_names=False, rtol=1e-9,
                                       obj=f"{target['date'].date()} {target['home_team']} v {target['away_team']}")


def test_current_match_outcome_not_used(matches, full):
    """Changing a match's goals/shots/result must not change its own features,
    but must change a later match's features (proving the check can fail)."""
    target = matches[(matches["season"] == "2015/16")].iloc[100]
    is_target = (matches[KEYS] == target[KEYS].values).all(axis=1)
    altered = build_features(_scramble_outcome(matches, is_target))

    pd.testing.assert_series_equal(_row(altered, target), _row(full, target), check_names=False)

    later = matches[(matches["date"] > target["date"])
                    & (matches["home_team"] == target["home_team"])].iloc[0]
    assert not _row(altered, later).equals(_row(full, later))


def test_feature_columns_exclude_outcomes(full):
    assert not set(feature_columns(full)) & set(OUTCOME_COLS)


def test_dq_rows_have_nan_shot_stats(matches):
    df = prepare_matches(matches)
    bad = ~df["dq_ok"].astype(bool)
    assert bad.any()
    assert df.loc[bad, ["home_shots", "home_shots_on_target"]].isna().all().all()


def _toy_season(season: str, teams: list[str], start: str) -> pd.DataFrame:
    rows, d = [], pd.Timestamp(start)
    for h in teams:
        for a in teams:
            if h != a:
                # alphabetically earlier team always wins 1-0
                rows.append({"season": season, "date": d, "home_team": h, "away_team": a,
                             "home_goals": int(h < a), "away_goals": int(h > a)})
                d += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def test_elo_promoted_team_seeded_from_relegated():
    s1 = _toy_season("2000/01", ["A", "B", "C", "D", "E"], "2000-08-01")
    s2 = _toy_season("2001/02", ["A", "B", "F", "G", "H"], "2001-08-01")  # C, D, E relegated
    df = pd.concat([s1, s2], ignore_index=True)
    p = EloParams(k=20, home_adv=0, regress=1 / 3, gd_multiplier=False)
    elo = compute_elo(df, p)

    def final(team):
        last = df[(df.season == "2000/01") & ((df.home_team == team) | (df.away_team == team))].index[-1]
        r = elo.loc[last, "elo_home" if df.loc[last, "home_team"] == team else "elo_away"]
        hg, ag = df.loc[last, ["home_goals", "away_goals"]]
        exp = 1 / (1 + 10 ** (-(r - elo.loc[last, "elo_away" if df.loc[last, "home_team"] == team else "elo_home"]) / 400))
        score = (1.0 if hg > ag else 0.0) if df.loc[last, "home_team"] == team else (1.0 if ag > hg else 0.0)
        return r + p.k * (score - exp)

    seed = np.mean([final(t) + p.regress * (1500 - final(t)) for t in ["C", "D", "E"]])
    first_s2 = df.index[df.season == "2001/02"][0]
    row = df.loc[first_s2]
    assert row.home_team == "A" and row.away_team == "B"
    first_f = df.index[(df.season == "2001/02") & ((df.home_team == "F") | (df.away_team == "F"))][0]
    col = "elo_home" if df.loc[first_f, "home_team"] == "F" else "elo_away"
    assert elo.loc[first_f, col] == pytest.approx(seed)
    assert elo.loc[df.season == "2000/01"].iloc[0]["elo_home"] == 1500


# --------------------------------------------------------------------------- Dixon-Coles

DC_PATH = MATCHES_PATH.parent / "dc_probs.csv"
DC_COLS = ["dc_p_home", "dc_p_draw", "dc_p_away"]


@pytest.fixture(scope="module")
def dc_probs() -> pd.DataFrame:
    if not DC_PATH.exists():
        pytest.skip("data/dc_probs.csv missing: run `python -m src.cv_dc`")
    return pd.read_csv(DC_PATH, parse_dates=["date"])


def test_dc_cache_uses_frozen_xi(dc_probs):
    from src.config import DC_XI
    assert (dc_probs["xi"] == DC_XI).all()


def test_dc_probs_use_only_prior_matches(matches, dc_probs):
    """Scramble every outcome on or after the target date, refit DC cold (no warm start),
    and compare with the stored warm-started pre-match probabilities.
    Future fixture lists stay (the season's teams are known before it starts); only
    future results are scrambled."""
    from src.config import DC_XI
    from src.dixon_coles import dc_predict_one

    cv = matches[matches["season"].between("2001/02", "2022/23")]
    targets = _sample(cv, 5, seed=1)
    # also a promoted team's first match of a season (uses the relegated-teams prior)
    promoted = build_features(matches).query("promoted_home == 1 and season > '2001/02'")
    targets = pd.concat([targets, cv.merge(promoted[KEYS].head(1), on=KEYS)])

    for _, t in targets.iterrows():
        scrambled = matches.copy()
        future = scrambled["date"] >= t["date"]
        scrambled.loc[future, ["home_goals", "away_goals"]] = [9, 0]
        scrambled.loc[future, "result"] = "H"
        got = dc_predict_one(scrambled, t["date"], t["home_team"], t["away_team"], DC_XI)[:3]
        want = dc_probs.merge(t[KEYS].to_frame().T.astype({"date": "datetime64[ns]"}),
                              on=KEYS)[DC_COLS].iloc[0].to_numpy()
        np.testing.assert_allclose(got, want, atol=1e-5,
                                   err_msg=f"{t['date'].date()} {t['home_team']} v {t['away_team']}")
