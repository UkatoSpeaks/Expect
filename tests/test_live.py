"""Live prediction and scoring: replay a played gameweek as if it were upcoming."""

import numpy as np
import pandas as pd
import pytest

from src.features import MATCHES_PATH, build_features
from src.predict import build_predictions
from src.score_predictions import COLUMNS, JOIN, score_new

KEYS = ["date", "home_team", "away_team"]
REPLAY_START, REPLAY_END = pd.Timestamp("2026-11-06"), pd.Timestamp("2026-11-08")  # 2026/27 GW10


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return pd.read_csv(MATCHES_PATH, parse_dates=["date"])


@pytest.fixture(scope="module")
def replay(matches):
    gw = matches[matches["date"].between(REPLAY_START, REPLAY_END)]
    assert len(gw) == 10
    before = matches[matches["date"] < REPLAY_START]
    fixtures = gw[KEYS].reset_index(drop=True)
    return gw, build_predictions(before, fixtures, "2026/27")


def test_replay_matches_historical_pipeline(matches, replay):
    """Same Elo as the full-history features and same DC as data/dc_probs.csv, so the live
    path uses exactly the pre-match information the evaluated pipeline used."""
    gw, pred = replay
    full = build_features(matches).merge(gw[KEYS], on=KEYS)
    j = pred.merge(full[KEYS + ["elo_home", "elo_away"]], on=KEYS, suffixes=("", "_hist"))
    assert len(j) == 10
    np.testing.assert_allclose(j["elo_home"], j["elo_home_hist"])
    np.testing.assert_allclose(j["elo_away"], j["elo_away_hist"])

    # DC: only the gameweek's first day is comparable. The historical run refits before every
    # date, so later days there already include this gameweek's earlier results.
    dc = pd.read_csv(MATCHES_PATH.parent / "dc_probs.csv", parse_dates=["date"])
    j = pred[pred["date"] == REPLAY_START].merge(dc, on=KEYS, suffixes=("", "_hist"))
    assert len(j) >= 1
    for c in ["dc_p_home", "dc_p_draw", "dc_p_away"]:
        np.testing.assert_allclose(j[c], j[f"{c}_hist"], atol=1e-4)
    np.testing.assert_allclose(pred[["p_home", "p_draw", "p_away"]].sum(1), 1)


def test_rejects_fixtures_already_played(matches, replay):
    """If the fixtures' own results are in the data, refuse rather than predict with them."""
    gw, _ = replay
    before = matches[matches["date"] <= REPLAY_END]
    with pytest.raises(ValueError, match="on or before the last played"):
        build_predictions(before, gw[KEYS].reset_index(drop=True), "2026/27")


def test_score_new_is_append_only(matches, replay):
    gw, pred = replay
    preds = pred.assign(gameweek=10, match_id="m", kickoff_utc="2026-11-06T20:00:00+00:00",
                        generated_at="2026-11-01T00:00:00+00:00", prediction_file="x")
    empty = pd.DataFrame(columns=COLUMNS)
    new = score_new(preds, matches, empty)
    assert len(new) == 10 and list(new.columns) == COLUMNS
    assert new["made_before_kickoff"].all()
    # log loss recomputed by hand
    p = np.select([new["result"] == "H", new["result"] == "D"],
                  [new["p_home"], new["p_draw"]], new["p_away"])
    np.testing.assert_allclose(new["model_log_loss"], -np.log(p))
    # already-recorded matches are never rescored
    assert score_new(preds, matches, new).empty
    assert len(score_new(preds, matches, new.iloc[:4])) == 6
    assert set(map(tuple, new[JOIN].to_numpy())) == set(map(tuple, gw[["season", "home_team", "away_team"]].to_numpy()))
