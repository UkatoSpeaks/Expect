"""src.pipeline: idempotent, never overwrites a gameweek file, never predicts after kick-off,
writes nothing when a source fails or the sources disagree.

Runs offline in a temp dir holding a snapshot of the repository data as of the end of
2026/27 GW10 (CUTOFF), so the tests keep passing after the daily job commits new results.
The downloads are faked: football-data.co.uk serves data/raw/E0_2627.csv cut at CUTOFF, and
FPL-Core files are synthesised from data/matches.csv (GW1-10, finished),
predictions/2026-27/gw11.csv (GW11 fixtures with real kick-off times) and an invented GW12,
so the tests do not need the FPL-Core clone in data/external/ (which is not committed).
"""

import functools

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import pipeline as pl
from src.features import build_features
from src.load_data import build_matches
from src.predict import PRED_COLS
from src.team_name_map import FPL_TO_FD

REPO = Path(__file__).resolve().parents[1]
SEASON = pl.Season(2026)
GW11 = REPO / "predictions" / "2026-27" / "gw11.csv"
BEFORE_GW11 = pd.Timestamp("2026-11-19 12:00", tz="UTC")
FIRST_KICKOFF_GW11 = pd.Timestamp("2026-11-21 12:30", tz="UTC")
CUTOFF = pd.Timestamp("2026-11-08")  # last day of GW10


@functools.cache
def fd_snapshot() -> bytes:
    """The committed football-data file without any match after CUTOFF."""
    lines = (REPO / "data" / "raw" / SEASON.raw_name).read_bytes().split(b"\n")
    col = lines[0].decode("utf-8-sig").split(",").index("Date")
    keep = [ln for ln in lines[1:] if ln.strip(b",\r ")
            and pd.to_datetime(ln.decode("latin-1").split(",")[col], format="%d/%m/%Y") <= CUTOFF]
    return b"\n".join([lines[0], *keep]) + b"\n"


@functools.cache
def derived_snapshot() -> tuple[bytes, bytes]:
    """data/matches.csv and data/features.csv as src.pipeline builds them from the snapshot."""
    raw = REPO / "data" / "raw"
    files = sorted(p for p in raw.glob("E0_*.csv") if p.name != SEASON.raw_name)
    with tempfile.TemporaryDirectory() as tmp:
        cut = Path(tmp) / SEASON.raw_name
        cut.write_bytes(fd_snapshot())
        text = pl.matches_csv_text(build_matches(files=sorted(files + [cut], key=lambda p: p.name)))
    feats = build_features(pl.read_csv_text(text, parse_dates=["date"]))
    return text.encode(), feats.to_csv(index=False).encode()


# --------------------------------------------------------------------------- fake sources

def _fpl_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pl.read_csv_text(derived_snapshot()[0].decode(), parse_dates=["date"])
    played = m[(m["season"] == SEASON.label) & m["result"].notna()].sort_values(["date", "home_team"])
    assert len(played) == 100 and played["date"].max() == CUTOFF
    fd_to_fpl = {fd: fpl for fpl, fd in reversed(FPL_TO_FD.items())}
    teams = sorted(set(played["home_team"]))
    teams_df = pd.DataFrame({"code": range(1, 21), "name": [fd_to_fpl[t] for t in teams]})
    code = dict(zip(teams, teams_df["code"]))

    rows = []
    for i, r in enumerate(played.itertuples()):
        rows.append((i // 10 + 1, f"{r.date:%Y-%m-%d}T14:00:00", r.home_team, r.away_team,
                     r.home_goals, r.away_goals, True))
    gw11 = pd.read_csv(GW11)
    for r in gw11.itertuples():
        ko = pd.Timestamp(r.kickoff_utc).tz_convert("UTC")
        rows.append((11, f"{ko:%Y-%m-%dT%H:%M:%S}", r.home_team, r.away_team, np.nan, np.nan, False))
    used = set(zip(played["home_team"], played["away_team"])) | set(zip(gw11["home_team"], gw11["away_team"]))
    rng = np.random.default_rng(0)
    while True:  # GW12: 10 unused pairings, each team once
        order = list(rng.permutation(teams))
        pairs = list(zip(order[::2], order[1::2]))
        if not used & set(pairs):
            break
    for h, a in pairs:
        rows.append((12, "2026-11-28T15:00:00", h, a, np.nan, np.nan, False))

    fx = pd.DataFrame(rows, columns=["gameweek", "kickoff_time", "home", "away", "home_score",
                                     "away_score", "finished"])
    fx["home_team"] = fx["home"].map(code).astype(float)
    fx["away_team"] = fx["away"].map(code).astype(float)
    fx["match_id"] = "26-27-prem-" + fx["home"].str.lower() + "-vs-" + fx["away"].str.lower()
    return teams_df, fx.drop(columns=["home", "away"])


class FakeSources:
    """URL -> bytes, built from repo data; tests mutate `fd`, `fixtures` or `fail`."""

    def __init__(self):
        self.fd = fd_snapshot()
        self.teams, self.fixtures = _fpl_frames()
        self.fail: set[str] = set()  # substrings of URLs that are unreachable

    def __call__(self, url: str) -> bytes:
        if any(f in url for f in self.fail):
            raise pl.SourceError(f"unreachable: {url}")
        if url == pl.FD_URL.format(code=SEASON.fd_code):
            return self.fd
        for gw in range(1, pl.MAX_GAMEWEEKS + 1):
            if url.endswith(f"/GW{gw}/matches.csv"):
                d = self.fixtures[self.fixtures["gameweek"] == gw]
                if d.empty:
                    raise pl.NotFound(url)
                return d.to_csv(index=False).encode()
        if url.endswith(f"/{SEASON.fpl_dir}/teams.csv"):
            return self.teams.to_csv(index=False).encode()
        raise pl.NotFound(url)


# --------------------------------------------------------------------------- repo copy

@pytest.fixture
def root(tmp_path) -> Path:
    shutil.copytree(REPO / "data" / "raw", tmp_path / "data" / "raw")
    (tmp_path / "data" / "raw" / SEASON.raw_name).write_bytes(fd_snapshot())
    matches, features = derived_snapshot()
    (tmp_path / "data" / "matches.csv").write_bytes(matches)
    (tmp_path / "data" / "features.csv").write_bytes(features)
    for rel in ["predictions/2026-27/gw11.csv", "reports/cv_dc_pooled.csv",
                "reports/test/scores.csv", "reports/forward_2026-11-08/scores.csv"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, tmp_path / rel)
    return tmp_path


@pytest.fixture
def src() -> FakeSources:
    return FakeSources()


def snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def run(root, src, now, **kw) -> pl.Result:
    return pl.run(pl.Paths(root), now=now, get=src, season=SEASON, **kw)


def gw11_path(root: Path) -> Path:
    return root / "predictions" / "2026-27" / "gw11.csv"


# --------------------------------------------------------------------------- tests

def test_idempotent_when_nothing_is_due(root, src):
    first = run(root, src, BEFORE_GW11)
    assert {p.relative_to(root).as_posix() for p in first.changed} == {
        "site_data/current.json", "site_data/track_record.json", "site_data/meta.json"}
    assert any("GW11 already predicted" in n for n in first.notes)
    before = snapshot(root)
    second = run(root, src, BEFORE_GW11 + pd.Timedelta(hours=6))
    assert second.changed == [] and second.summary == "nothing to do"
    assert snapshot(root) == before


def test_predicts_due_gameweek_once(root, src):
    gw11_path(root).unlink()
    first = run(root, src, BEFORE_GW11)
    assert "predicted 2026/27 GW11" in first.summary
    pred = pd.read_csv(gw11_path(root))
    assert list(pred.columns) == PRED_COLS and len(pred) == 10
    assert (pred["generated_at"] == BEFORE_GW11.isoformat(timespec="seconds")).all()
    np.testing.assert_allclose(pred[["p_home", "p_draw", "p_away"]].sum(axis=1), 1, atol=1e-5)
    # same model, same data -> same probabilities as the committed GW11 file
    ref = pd.read_csv(GW11)
    np.testing.assert_allclose(pred.sort_values("home_team")[["p_home", "p_draw", "p_away"]],
                               ref.sort_values("home_team")[["p_home", "p_draw", "p_away"]], atol=1e-6)

    before = snapshot(root)
    second = run(root, src, BEFORE_GW11 + pd.Timedelta(hours=1))
    assert second.changed == [] and snapshot(root) == before


def test_never_overwrites_existing_gameweek_file(root, src):
    # an existing GW11 file with different content, while GW11 is otherwise due
    p = gw11_path(root)
    p.write_text(p.read_text().replace("2026-11-19T14:49:58+00:00", "2026-11-18T09:00:00+00:00"))
    original = p.read_bytes()
    res = run(root, src, BEFORE_GW11)
    assert p.read_bytes() == original
    assert p not in res.changed and not res.new_files
    assert any("GW11 already predicted" in n for n in res.notes)


@pytest.mark.parametrize("now", [FIRST_KICKOFF_GW11, FIRST_KICKOFF_GW11 + pd.Timedelta(minutes=1),
                                 pd.Timestamp("2026-11-22 18:00", tz="UTC")])
def test_refuses_to_predict_after_kickoff(root, src, now):
    gw11_path(root).unlink()
    res = run(root, src, now)
    assert not gw11_path(root).exists() and not res.new_files
    assert "predicted" not in res.summary
    assert any("NOT predicted" in n and "kick-off" in n for n in res.notes)


def _drop_last_gw10_result(root, src) -> tuple[pd.Series, bytes]:
    """Remove GW10's last match from football-data (and from data/matches.csv, as if it had
    never been published). Returns that match and its football-data line."""
    m = pd.read_csv(root / "data" / "matches.csv")
    last = m[(m["season"] == SEASON.label) & m["result"].notna()].iloc[-1]
    m.drop(index=last.name).to_csv(root / "data" / "matches.csv", index=False)
    lines = src.fd.split(b"\n")
    hit = [i for i, line in enumerate(lines)
           if f",{last.home_team},{last.away_team},".encode() in line]
    assert len(hit) == 1
    src.fd = b"\n".join(lines[:hit[0]] + lines[hit[0] + 1:])
    return last, lines[hit[0]]


def _fixture_index(src, home: str, away: str):
    code = src.teams.set_index("name")["code"]
    fd_to_fpl = {fd: fpl for fpl, fd in reversed(FPL_TO_FD.items())}
    f = src.fixtures
    i = f.index[(f["home_team"] == code[fd_to_fpl[home]]) & (f["away_team"] == code[fd_to_fpl[away]])]
    assert len(i) == 1
    return i[0]


def _postpone_last_gw10(root, src, new_kickoff: str | None) -> tuple[pd.Series, bytes]:
    """GW10's last match (8 Nov) is postponed: no result in either source. FPL-Core either
    keeps the original kick-off (new_kickoff=None) or moves it to `new_kickoff` (UTC)."""
    last, line = _drop_last_gw10_result(root, src)
    i = _fixture_index(src, last.home_team, last.away_team)
    src.fixtures.loc[i, ["home_score", "away_score"]] = np.nan
    src.fixtures.loc[i, "finished"] = False
    if new_kickoff:
        src.fixtures.loc[i, "kickoff_time"] = new_kickoff
    return last, line


def test_waits_until_previous_gameweek_is_in_both_sources(root, src):
    """GW10's last match finished in FPL-Core but not yet in football-data: no GW11 prediction."""
    gw11_path(root).unlink()
    _drop_last_gw10_result(root, src)

    res = run(root, src, BEFORE_GW11)
    assert not gw11_path(root).exists()
    assert any("waiting for football-data" in n for n in res.notes)
    assert any("1 matches finished in FPL-Core but not yet in football-data" in n for n in res.notes)
    assert not res.alarms  # first kick-off is 48.5h away


@pytest.mark.parametrize("new_kickoff", [None, "2026-12-16T19:30:00"])
def test_postponed_match_does_not_block_next_gameweek(root, src, new_kickoff):
    """GW10's last match is postponed (kick-off >48h ago with no result anywhere, or moved by
    FPL-Core to after GW11 starts): GW11 is still predicted, without waiting for it."""
    gw11_path(root).unlink()
    last, _ = _postpone_last_gw10(root, src, new_kickoff)

    res = run(root, src, BEFORE_GW11)
    assert "predicted 2026/27 GW11" in res.summary and not res.alarms
    pred = pd.read_csv(gw11_path(root))
    assert len(pred) == 10 and (pred["gameweek"] == 11).all()
    assert any(n.startswith(f"postponed: {last.home_team} v {last.away_team} (GW10)") for n in res.notes)
    # a far-off new date is not predicted yet: GW11 and GW12 end before it and are not played
    assert not list((root / "predictions" / "2026-27").glob("*_rescheduled_*"))
    if new_kickoff:
        assert any("not predicted yet" in n and "GW[11, 12]" in n for n in res.notes)


def test_rescheduled_match_predicted_once_before_new_kickoff(root, src):
    """GW10's last match was predicted in gw10.csv, then postponed and moved by FPL-Core into
    GW11's weekend. It is predicted on its own, once, before its new kick-off; the result is
    scored against that prediction; gw10.csv is never touched."""
    new_ko = pd.Timestamp("2026-11-22 19:00", tz="UTC")
    last, fd_line = _postpone_last_gw10(root, src, f"{new_ko:%Y-%m-%dT%H:%M:%S}")
    gw10 = pd.DataFrame([{c: 0.0 for c in PRED_COLS}])
    gw10[["season", "gameweek", "match_id", "date", "home_team", "away_team"]] = [
        SEASON.label, 10, "orig", "2026-11-08", last.home_team, last.away_team]
    gw10[["p_home", "p_draw", "p_away"]] = [0.2, 0.3, 0.5]
    gw10[["kickoff_utc", "data_through", "generated_at"]] = [
        "2026-11-08 14:00:00+00:00", "2026-11-01", "2026-11-06T00:00:00+00:00"]
    gw10_file = root / "predictions" / "2026-27" / "gw10.csv"
    gw10.to_csv(gw10_file, index=False)
    gw10_bytes = gw10_file.read_bytes()
    resched = root / "predictions" / "2026-27" / "gw10_rescheduled_2026-11-22.csv"

    res = run(root, src, BEFORE_GW11)
    assert f"predicted rescheduled {last.home_team} v {last.away_team} (GW10)" in res.summary
    p = pd.read_csv(resched)
    assert list(p.columns) == PRED_COLS and len(p) == 1
    r = p.iloc[0]
    assert (r.home_team, r.away_team, r.gameweek, r.date) == (last.home_team, last.away_team, 10, "2026-11-22")
    assert pd.Timestamp(r.kickoff_utc) == new_ko
    assert pd.Timestamp(r.generated_at) == BEFORE_GW11 < new_ko
    assert gw10_file.read_bytes() == gw10_bytes
    cur = pl.json.loads((root / "site_data" / "current.json").read_text())
    assert cur["gameweek"] == 11 and len(cur["fixtures"]) == 10
    assert [(f["home_team"], f["status"]) for f in cur["rescheduled"]] == [(last.home_team, "upcoming")]

    # predicted once: later runs, before and after the new kick-off, add nothing
    before = snapshot(root)
    for now in [BEFORE_GW11 + pd.Timedelta(hours=12), new_ko - pd.Timedelta(minutes=5),
                new_ko + pd.Timedelta(hours=1)]:
        again = run(root, src, now)
        assert not again.new_files and not again.alarms
        assert snapshot(root) == before

    # the match is played: scored once, against the rescheduled prediction
    src.fd = src.fd.rstrip(b"\n") + b"\n" + fd_line.replace(b"08/11/2026", b"22/11/2026") + b"\n"
    i = _fixture_index(src, last.home_team, last.away_team)
    src.fixtures.loc[i, ["home_score", "away_score", "finished"]] = [last.home_goals, last.away_goals, True]
    res = run(root, src, pd.Timestamp("2026-11-23 06:00", tz="UTC"))
    assert "scored 1 matches (GW10)" in res.summary and not res.new_files
    track = pd.read_csv(root / "predictions" / "track_record.csv")
    assert len(track) == 1
    assert track.loc[0, "prediction_file"] == "2026-27/gw10_rescheduled_2026-11-22.csv"
    assert bool(track.loc[0, "made_before_kickoff"])
    assert track.loc[0, "p_home"] == pytest.approx(r.p_home, abs=1e-6)
    tj = pl.json.loads((root / "site_data" / "track_record.json").read_text())
    assert tj["matches"][0]["rescheduled"] and tj["totals"]["n"] == 1
    cur = pl.json.loads((root / "site_data" / "current.json").read_text())
    assert [(f["home_team"], f["status"]) for f in cur["rescheduled"]] == [(last.home_team, "finished")]
    assert gw10_file.read_bytes() == gw10_bytes


def test_fails_loudly_when_prediction_is_about_to_be_missed(root, src, monkeypatch, capsys, tmp_path):
    """football-data has not published GW10's last result 12h before GW11 kicks off: the run
    still writes everything else, but reports an alarm and main() exits 1."""
    gw11_path(root).unlink()
    _drop_last_gw10_result(root, src)
    now = FIRST_KICKOFF_GW11 - pd.Timedelta(hours=12)

    res = run(root, src, now, dry_run=True)
    assert len(res.alarms) == 1
    assert "GW11 not predicted: waiting for football-data" in res.alarms[0]
    assert "is in 12.0h" in res.alarms[0]

    real_run = pl.run
    monkeypatch.setattr(pl, "run", lambda season=None, dry_run=False: real_run(
        pl.Paths(root), now=now, get=src, season=SEASON, dry_run=dry_run))
    summary = tmp_path / "summary.txt"
    assert pl.main(["--summary-file", str(summary)]) == 1
    err = capsys.readouterr().err
    assert "PREDICTION ABOUT TO BE MISSED" in err and "GW11" in err
    assert summary.exists()
    assert (root / "site_data" / "current.json").exists() and not gw11_path(root).exists()

    # once football-data catches up, the same run predicts GW11 and exits 0
    src.fd = fd_snapshot()
    assert pl.main([]) == 0 and gw11_path(root).exists()


def test_scores_finished_predictions_once(root, src):
    """A GW10 prediction file made before kick-off gets scored; a second run adds nothing."""
    m = pd.read_csv(root / "data" / "matches.csv")
    gw10 = m[(m["season"] == SEASON.label) & m["result"].notna()].tail(10)
    fake = pd.DataFrame({c: 0.0 for c in PRED_COLS}, index=gw10.index)
    fake[["season", "date", "home_team", "away_team"]] = gw10[["season", "date", "home_team", "away_team"]]
    fake[["p_home", "p_draw", "p_away"]] = [0.45, 0.27, 0.28]
    fake["gameweek"], fake["match_id"] = 10, "m" + gw10.index.astype(str)
    fake["kickoff_utc"] = gw10["date"] + " 15:00:00+00:00"
    fake["data_through"], fake["generated_at"] = "2026-11-01", "2026-11-02T00:00:00+00:00"
    fake.to_csv(root / "predictions" / "2026-27" / "gw10.csv", index=False)

    res = run(root, src, BEFORE_GW11)
    assert "scored 10 matches (GW10)" in res.summary
    track = pd.read_csv(root / "predictions" / "track_record.csv")
    assert len(track) == 10 and track["made_before_kickoff"].all()
    tj = pl.export_track(track)
    assert tj["totals"]["n"] == 10
    assert tj["matches"][-1]["running"]["model"]["log_loss"] == pytest.approx(
        track["model_log_loss"].mean(), abs=1e-4)

    before = snapshot(root)
    assert run(root, src, BEFORE_GW11 + pd.Timedelta(days=1)).changed == []
    assert snapshot(root) == before


@pytest.mark.parametrize("unreachable", ["football-data.co.uk", "teams.csv", "/GW7/"])
def test_unreachable_source_writes_nothing(root, src, unreachable):
    gw11_path(root).unlink()
    before = snapshot(root)
    src.fail.add(unreachable)
    with pytest.raises(pl.SourceError):
        run(root, src, BEFORE_GW11)
    assert snapshot(root) == before


def _corrupt_fpl_score(src):
    i = src.fixtures.index[src.fixtures["finished"]][5]
    src.fixtures.loc[i, "home_score"] += 1


def _corrupt_fpl_date(src):
    i = src.fixtures.index[src.fixtures["finished"]][5]
    src.fixtures.loc[i, "kickoff_time"] = "2026-08-01T14:00:00"


def _drop_fpl_fixture(src):
    src.fixtures = src.fixtures.drop(index=src.fixtures.index[src.fixtures["finished"]][5])


def _change_recorded_fd_result(src):
    # rewrite one recorded result consistently in both sources: still refused, because
    # data/matches.csv already holds the old score
    lines = src.fd.decode("utf-8-sig").split("\n")
    header = lines[0].rstrip("\r").split(",")
    cells = lines[3].rstrip("\r").split(",")
    h, a = cells[header.index("HomeTeam")], cells[header.index("AwayTeam")]
    cells[header.index("FTHG")] = str(int(cells[header.index("FTHG")]) + 5)
    cells[header.index("FTR")] = "H"
    lines[3] = ",".join(cells) + ("\r" if lines[3].endswith("\r") else "")
    src.fd = "\n".join(lines).encode()
    code = src.teams.set_index("name")["code"]
    fd_to_fpl = {fd: fpl for fpl, fd in reversed(FPL_TO_FD.items())}
    f = src.fixtures
    i = f.index[(f["home_team"] == code[fd_to_fpl[h]]) & (f["away_team"] == code[fd_to_fpl[a]])]
    assert len(i) == 1
    f.loc[i, "home_score"] += 5


@pytest.mark.parametrize("corrupt", [_corrupt_fpl_score, _corrupt_fpl_date, _drop_fpl_fixture,
                                     _change_recorded_fd_result])
def test_inconsistent_sources_write_nothing(root, src, corrupt):
    gw11_path(root).unlink()
    before = snapshot(root)
    corrupt(src)
    with pytest.raises(pl.InconsistentData):
        run(root, src, BEFORE_GW11)
    assert snapshot(root) == before


def test_dry_run_writes_nothing(root, src):
    gw11_path(root).unlink()
    before = snapshot(root)
    res = run(root, src, BEFORE_GW11, dry_run=True)
    assert "predicted 2026/27 GW11" in res.summary
    assert snapshot(root) == before


def test_season_labels():
    s = pl.Season.for_date(pd.Timestamp("2027-03-01", tz="UTC"))
    assert (s.label, s.fd_code, s.fpl_dir, s.raw_name) == ("2026/27", "2627", "2026-2027", "E0_2627.csv")
    assert pl.Season.for_date(pd.Timestamp("2027-07-01", tz="UTC")).label == "2027/28"
    assert pl.Season.parse("2099/00").fd_code == "9900"
    with pytest.raises(ValueError):
        pl.Season.parse("2026/28")
