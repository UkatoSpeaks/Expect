"""Step 7: the daily pipeline. One idempotent command, safe to run any number of times.

    .venv/bin/python -m src.pipeline              # refresh, score, predict if due, export
    .venv/bin/python -m src.pipeline --dry-run    # same checks, print what would change, write nothing

Steps
-----
a) Refresh: download the current season's football-data.co.uk file and the season's
   FPL-Core-Insights fixtures/results (teams.csv + every published GW matches.csv) into a
   temporary directory, rebuild data/matches.csv and data/features.csv from them.
b) Score: newly finished predicted matches -> predictions/track_record.csv (append-only,
   same code as src.score_predictions).
c) Predict: the next gameweek (lowest gameweek with no finished match) is predicted only if
   it has no predictions file, every earlier gameweek is finished in FPL-Core *and* its
   results are in football-data, and its first kick-off is in the future. Postponed
   fixtures (see below) are left out of both checks and out of the gameweek file.
c') Rescheduled: a postponed fixture that has a new, future kick-off K is predicted on its
   own into predictions/<season>/gw<N>_rescheduled_<date>.csv once every gameweek that ends
   before K is finished in both sources (the same rule as for a gameweek). It is predicted
   again only if FPL-Core moves it to a later date than the one it was last predicted for.
d) Export site_data/current.json, track_record.json and meta.json.
e) Print a summary. Exit 0, whether or not anything changed, unless a prediction is about
   to be missed: the next gameweek (or a rescheduled fixture) kicks off within 24h and
   cannot be predicted yet. Then everything else is still written, and the exit code is 1
   so the scheduled job fails and GitHub sends an email.

Postponed fixtures
------------------
A fixture with no result in either source counts as postponed if FPL-Core gives it no
kick-off time, if its kick-off was more than 48h ago, or if FPL-Core has moved it to a later
date: after the next gameweek's first kick-off, or to a later date than the one in its
latest prediction file. The original gameweek file is never changed; the latest rescheduled
prediction is the one scored.

Safety
------
Everything is computed in memory first; files are written only at the very end, and only
those whose content changed. The run stops with exit code 1 and writes nothing if a source
is unreachable or unparseable, or if the sources disagree: a football-data result with no
FPL-Core fixture, different scores or dates for a finished match, or a result already in
data/matches.csv that changed or disappeared. One source being a few hours behind the other
is not an error: those matches are reported, and prediction waits until both have them.

Existing prediction files are never touched (new ones are opened with mode "x"), and the
track record is only appended to. Running twice in a row changes nothing the second time:
no output contains the wall-clock time of the run except new rows (generated_at of a new
prediction file, scored_at of newly scored matches).

FPL-Core downloads stay in a temporary directory: the clone in data/external/ used by
src.predict / src.fplcore is not modified.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src import config
from src.features import build_features
from src.fplcore import load_pl_matches
from src.load_data import build_matches
from src.predict import gw_path, prediction_file, rescheduled_path
from src.score_predictions import (is_rescheduled, latest_predictions, load_saved_predictions,
                                   load_track, score_new)

ROOT = Path(__file__).resolve().parents[1]
FD_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
FPLCORE_URL = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data/{season}/{path}"
FPLCORE_GW_FILE = "By Tournament/Premier League/GW{gw}/matches.csv"
MAX_GAMEWEEKS = 38
PAIR = ["home_team", "away_team"]
POSTPONED_AFTER = pd.Timedelta(hours=48)  # no result this long after kick-off -> postponed
ALARM_WITHIN = pd.Timedelta(hours=24)     # blocked prediction this close to kick-off -> exit 1


# --------------------------------------------------------------------------- errors

class Abort(Exception):
    """Stop the run; nothing has been written."""


class SourceError(Abort):
    """A source is unreachable or returned something unusable."""


class NotFound(SourceError):
    """HTTP 404."""


class InconsistentData(Abort):
    """football-data.co.uk and FPL-Core disagree, or a recorded result changed."""


# --------------------------------------------------------------------------- paths / season

@dataclass(frozen=True)
class Paths:
    root: Path = ROOT

    @property
    def raw_dir(self) -> Path: return self.root / "data" / "raw"
    @property
    def matches(self) -> Path: return self.root / "data" / "matches.csv"
    @property
    def features(self) -> Path: return self.root / "data" / "features.csv"
    @property
    def pred_dir(self) -> Path: return self.root / "predictions"
    @property
    def track(self) -> Path: return self.pred_dir / "track_record.csv"
    @property
    def site_dir(self) -> Path: return self.root / "site_data"
    @property
    def reports_dir(self) -> Path: return self.root / "reports"


@dataclass(frozen=True)
class Season:
    start: int  # 2026 for 2026/27

    @classmethod
    def for_date(cls, now: pd.Timestamp) -> "Season":
        """Seasons switch over on 1 July."""
        return cls(now.year if now.month >= 7 else now.year - 1)

    @classmethod
    def parse(cls, label: str) -> "Season":
        m = re.fullmatch(r"(\d{4})/(\d{2})", label)
        if not m or (int(m.group(1)) + 1) % 100 != int(m.group(2)):
            raise ValueError(f"season must look like 2026/27, got {label!r}")
        return cls(int(m.group(1)))

    @property
    def label(self) -> str: return f"{self.start}/{(self.start + 1) % 100:02d}"
    @property
    def fd_code(self) -> str: return f"{self.start % 100:02d}{(self.start + 1) % 100:02d}"
    @property
    def fpl_dir(self) -> str: return f"{self.start}-{self.start + 1}"
    @property
    def raw_name(self) -> str: return f"E0_{self.fd_code}.csv"


# --------------------------------------------------------------------------- fetching

Getter = Callable[[str], bytes]


def http_get(url: str, timeout: float = 30, retries: int = 3) -> bytes:
    err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "xpect-pipeline"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if not data:
                raise SourceError(f"empty response from {url}")
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise NotFound(f"404 Not Found: {url}") from e
            err = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            err = e
        if attempt < retries - 1:
            time.sleep(5 * 2 ** attempt)
    raise SourceError(f"unreachable after {retries} attempts: {url} ({err})")


def fetch_football_data(season: Season, get: Getter) -> bytes:
    data = get(FD_URL.format(code=season.fd_code))
    first = data[:400].decode("utf-8-sig", errors="replace").splitlines()[0] if data else ""
    if not {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"} <= set(first.split(",")):
        raise SourceError(f"football-data.co.uk {season.label}: not a results CSV (header {first[:80]!r})")
    return data


def fetch_fplcore(season: Season, get: Getter) -> dict[str, bytes]:
    """teams.csv plus GW1..GWk matches.csv, stopping at the first gameweek not published."""
    url = lambda path: FPLCORE_URL.format(season=season.fpl_dir, path=urllib.parse.quote(path))
    files = {"teams.csv": get(url("teams.csv"))}
    for gw in range(1, MAX_GAMEWEEKS + 1):
        rel = FPLCORE_GW_FILE.format(gw=gw)
        try:
            files[rel] = get(url(rel))
        except NotFound:
            break
    if len(files) == 1:
        raise SourceError(f"FPL-Core {season.fpl_dir}: no gameweek files published")
    return files


# --------------------------------------------------------------------------- refresh + checks

def matches_csv_text(df: pd.DataFrame) -> str:
    """Exactly what src.load_data writes."""
    df = df.copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df.to_csv(index=False)


def read_csv_text(text: str, **kw) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text), **kw)


def check_consistency(matches: pd.DataFrame, old: pd.DataFrame | None, fpl: pd.DataFrame,
                      season: Season) -> dict[str, pd.DataFrame]:
    """Raise InconsistentData on any disagreement. Returns the matches where one source is
    simply behind the other ('fd_behind': finished in FPL-Core, no football-data result;
    'fpl_behind': football-data result, not finished in FPL-Core)."""
    lab = season.label
    fd = matches[(matches["season"] == lab) & matches["result"].notna()]
    fc = fpl[fpl["season"] == lab]
    for name, d in [("football-data", fd), ("FPL-Core", fc)]:
        dup = d[d.duplicated(PAIR, keep=False)]
        if len(dup):
            raise InconsistentData(f"{name} {lab}: pairing listed twice:\n{dup[PAIR].to_string()}")
    bad_fin = fc[fc["finished"] & (fc["home_score"].isna() | fc["away_score"].isna())]
    if len(bad_fin):
        raise InconsistentData(f"FPL-Core: finished matches without a score:\n{bad_fin[PAIR].to_string()}")

    j = fd[PAIR + ["date", "home_goals", "away_goals"]].merge(
        fc[PAIR + ["date", "home_score", "away_score", "finished"]], on=PAIR, how="left",
        suffixes=("_fd", "_fpl"), indicator=True)
    orphan = j[j["_merge"] == "left_only"]
    if len(orphan):
        raise InconsistentData(f"football-data {lab} results with no FPL-Core fixture:\n"
                               f"{orphan[PAIR + ['date_fd']].to_string()}")
    both = j[j["finished"].astype(bool)]
    diff = both[(both["home_goals"] != both["home_score"]) | (both["away_goals"] != both["away_score"])
                | (both["date_fd"] != both["date_fpl"])]
    if len(diff):
        raise InconsistentData(f"football-data and FPL-Core disagree on {len(diff)} finished {lab} matches:\n"
                               f"{diff[PAIR + ['date_fd', 'date_fpl', 'home_goals', 'away_goals', 'home_score', 'away_score']].to_string()}")

    if old is not None:  # results we already recorded must survive unchanged
        prev = old[(old["season"] == lab) & old["result"].notna()]
        k = prev[PAIR + ["date", "home_goals", "away_goals"]].merge(
            fd[PAIR + ["date", "home_goals", "away_goals"]], on=PAIR, how="left",
            suffixes=("", "_new"), indicator=True)
        gone = k[k["_merge"] == "left_only"]
        changed = k[(k["_merge"] == "both") & ((k["date"] != k["date_new"])
                    | (k["home_goals"] != k["home_goals_new"]) | (k["away_goals"] != k["away_goals_new"]))]
        if len(gone) or len(changed):
            raise InconsistentData(f"new football-data file drops {len(gone)} and changes {len(changed)} "
                                   f"results already in data/matches.csv:\n"
                                   f"{pd.concat([gone, changed])[PAIR + ['date', 'home_goals', 'away_goals']].to_string()}")

    fin = fc[fc["finished"]].merge(fd[PAIR], on=PAIR, how="left", indicator=True)
    return {"fd_behind": fin[fin["_merge"] == "left_only"][PAIR + ["gameweek", "date"]],
            "fpl_behind": j[~j["finished"].astype(bool)][PAIR + ["date_fd"]]}


# --------------------------------------------------------------------------- prediction plan

def season_fixtures(fpl: pd.DataFrame, season: Season) -> pd.DataFrame:
    fc = fpl[(fpl["season"] == season.label) & fpl["gameweek"].notna()].copy()
    fc["gameweek"] = fc["gameweek"].astype(int)
    return fc


def pairs(df: pd.DataFrame) -> pd.Series:
    return pd.Series(list(zip(df["home_team"], df["away_team"])), index=df.index, dtype=object)


def played_pairs(matches: pd.DataFrame, season: Season) -> set[tuple[str, str]]:
    """Matches with a football-data result this season."""
    return set(pairs(matches[(matches["season"] == season.label) & matches["result"].notna()]))


def latest_predicted_dates(preds: pd.DataFrame, season: Season) -> dict[tuple[str, str], pd.Timestamp]:
    """(home, away) -> match date in the latest prediction for that match."""
    if preds.empty:
        return {}
    p = latest_predictions(preds[preds["season"] == season.label])
    return dict(zip(pairs(p), pd.to_datetime(p["date"])))


def gameweek_starts(fc: pd.DataFrame) -> dict[int, pd.Timestamp]:
    """First kick-off of each gameweek, ignoring fixtures brought forward to before the
    previous gameweek's median kick-off."""
    ko = fc.dropna(subset=["kickoff_utc"]).groupby("gameweek")["kickoff_utc"]
    median = ko.median()
    starts = {}
    for gw, k in ko:
        prev = median.get(gw - 1)
        k = k[k > prev] if prev is not None else k
        if len(k):
            starts[gw] = k.min()
    return starts


def postponed_fixtures(fc: pd.DataFrame, matches: pd.DataFrame, preds: pd.DataFrame,
                       season: Season, now: pd.Timestamp) -> pd.Series:
    """Why each postponed fixture is postponed, indexed like `fc`. A fixture with no result in
    either source is postponed if FPL-Core gives it no kick-off, if its kick-off was more than
    48h ago, or if FPL-Core has moved it to a later date: after the next gameweek's first
    kick-off, or to a later date than the one it was last predicted for."""
    played = played_pairs(matches, season)
    starts = gameweek_starts(fc)
    predicted = latest_predicted_dates(preds, season)
    why = {}
    for i, r in fc[~fc["finished"]].iterrows():
        pair = (r["home_team"], r["away_team"])
        if pair in played:
            continue
        ko, nxt, was = r["kickoff_utc"], starts.get(r["gameweek"] + 1), predicted.get(pair)
        if pd.isna(ko):
            why[i] = "no kick-off date in FPL-Core"
        elif ko <= now - POSTPONED_AFTER:
            why[i] = f"kick-off {ko:%Y-%m-%d %H:%M} UTC passed over 48h ago, no result in either source"
        elif nxt is not None and ko > nxt:
            why[i] = f"moved to {ko:%Y-%m-%d %H:%M} UTC, after GW{r['gameweek'] + 1} starts"
        elif was is not None and r["date"] > was:
            why[i] = f"moved from {was:%Y-%m-%d} to {r['date']:%Y-%m-%d}"
    return pd.Series(why, dtype=object)


def _fixture_name(r) -> str:
    return f"{r['home_team']} v {r['away_team']} (GW{r['gameweek']})"


def _until(t: pd.Timestamp, now: pd.Timestamp) -> str:
    return f"{(t - now) / pd.Timedelta(hours=1):.1f}h"


def completeness(fc: pd.DataFrame, matches: pd.DataFrame, season: Season) -> str | None:
    """None if every fixture in `fc` is finished in FPL-Core and has a football-data result,
    else what is missing."""
    unfinished = fc[~fc["finished"]]
    if len(unfinished):
        gws = sorted(unfinished["gameweek"].unique().tolist())
        return f"{len(unfinished)} fixtures of GW{gws} not finished in FPL-Core"
    missing = fc[~pairs(fc).isin(played_pairs(matches, season))]
    if len(missing):
        return f"waiting for football-data.co.uk to publish {len(missing)} finished results"
    return None


@dataclass
class Plan:
    gameweek: int | None
    fixtures: pd.DataFrame | None
    reason: str
    first_kickoff: pd.Timestamp | None = None  # of the gameweek considered
    blocked: bool = False  # due, but cannot be predicted yet


def plan_prediction(fc: pd.DataFrame, matches: pd.DataFrame, season: Season, pred_dir: Path,
                    now: pd.Timestamp, postponed: pd.Index) -> Plan:
    lab = season.label
    n_finished = fc.groupby("gameweek")["finished"].sum()
    open_gws = n_finished[n_finished == 0].index
    if len(open_gws) == 0:
        return Plan(None, None, f"no {lab} gameweek without a finished match (season over?)")
    gw = int(open_gws.min())
    live = fc[~fc.index.isin(postponed)]
    fixtures = live[live["gameweek"] == gw].sort_values(["kickoff_utc", "home_team"]).reset_index(drop=True)
    first = fixtures["kickoff_utc"].min()
    path = gw_path(lab, gw, pred_dir)
    if path.exists():
        return Plan(None, None, f"GW{gw} already predicted ({path.relative_to(pred_dir.parent)})", first)
    if fixtures.empty:
        return Plan(None, None, f"GW{gw}: every fixture is postponed")

    missing = completeness(live[live["gameweek"] < gw], matches, season)
    if missing:
        return Plan(None, None, f"GW{gw} not predicted: {missing}", first, blocked=True)
    if pd.isna(first) or first <= now:
        return Plan(None, None, f"GW{gw} NOT predicted: first kick-off {first} is not in the future "
                                "(predictions must be made before kick-off)", first)
    counts = pd.concat([fixtures["home_team"], fixtures["away_team"]]).value_counts()
    if (counts > 1).any():
        return Plan(None, None, f"GW{gw} not predicted: teams with two fixtures "
                                f"{counts[counts > 1].index.tolist()} (double gameweeks not supported)",
                    first, blocked=True)
    return Plan(gw, fixtures, f"predicting GW{gw} ({len(fixtures)} fixtures, first kick-off {first})", first)


@dataclass
class ReschedulePlan:
    due: list[pd.DataFrame] = field(default_factory=list)  # one frame per (gameweek, date)
    notes: list[str] = field(default_factory=list)
    alarms: list[str] = field(default_factory=list)


def plan_rescheduled(fc: pd.DataFrame, matches: pd.DataFrame, preds: pd.DataFrame, season: Season,
                     pred_dir: Path, now: pd.Timestamp, postponed: pd.Index) -> ReschedulePlan:
    """Fixtures to predict on their own: no result, a future kick-off K, and either postponed
    or left out of their gameweek's file; not yet predicted for their current date. Due once
    every gameweek that ends before K is finished in both sources."""
    out = ReschedulePlan()
    played = played_pairs(matches, season)
    predicted = latest_predicted_dates(preds, season)
    gw_done = {gw for gw in fc["gameweek"].unique() if gw_path(season.label, gw, pred_dir).exists()}
    cand = []
    for i, r in fc[~fc["finished"] & (fc["kickoff_utc"] > now)].iterrows():
        pair = (r["home_team"], r["away_team"])
        was = predicted.get(pair)
        if (pair not in played and (i in postponed or r["gameweek"] in gw_done)
                and (was is None or r["date"] > was)):
            cand.append(i)
    if not cand:
        return out
    settled = fc[~fc.index.isin(postponed) & ~fc.index.isin(cand)]
    ends = settled.groupby("gameweek")["kickoff_utc"].max()
    due = []
    for i in cand:
        r = fc.loc[i]
        before = ends[ends < r["kickoff_utc"]].index
        missing = completeness(settled[settled["gameweek"].isin(before)], matches, season)
        if missing is None:
            due.append(i)
            continue
        msg = (f"rescheduled {_fixture_name(r)}, kick-off {r['kickoff_utc']:%Y-%m-%d %H:%M} UTC, "
               f"not predicted yet: {missing}")
        out.notes.append(msg)
        if r["kickoff_utc"] - now <= ALARM_WITHIN:
            out.alarms.append(f"{msg}; kick-off in {_until(r['kickoff_utc'], now)}")
    for _, g in fc.loc[due].groupby(["gameweek", "date"]):
        out.due.append(g.sort_values(["kickoff_utc", "home_team"]).reset_index(drop=True))
    return out


# --------------------------------------------------------------------------- site export

def _f(x, nd: int = 4):
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        return None
    return round(float(x), nd)


def _iso_utc(x) -> str | None:
    if pd.isna(x):
        return None
    return pd.Timestamp(x).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _metrics(d: pd.DataFrame, who: str) -> dict:
    return {"log_loss": _f(d[f"{who}_log_loss"].mean()), "brier": _f(d[f"{who}_brier"].mean()),
            "accuracy": _f(d[f"{who}_correct"].mean())}


def _fixture_json(r, m, status: str) -> dict:
    probs = {"H": r.p_home, "D": r.p_draw, "A": r.p_away}
    show = status == "finished"
    return {
        "match_id": r.match_id, "kickoff_utc": _iso_utc(r.ko), "home_team": r.home_team,
        "away_team": r.away_team, "p_home": _f(r.p_home), "p_draw": _f(r.p_draw),
        "p_away": _f(r.p_away), "pick": max(probs, key=probs.get),
        "elo_home": _f(r.elo_home, 1), "elo_away": _f(r.elo_away, 1),
        "elo_gap": _f(r.elo_diff, 1), "dc_exp_goals_home": _f(r.dc_exp_home, 2),
        "dc_exp_goals_away": _f(r.dc_exp_away, 2),
        "status": status,
        "home_goals": int(m.home_goals) if show else None,
        "away_goals": int(m.away_goals) if show else None,
        "result": m.result if show else None,
    }


def export_current(preds: pd.DataFrame, matches: pd.DataFrame, season: Season,
                   postponed: frozenset[tuple[str, str]] = frozenset()) -> dict:
    """The latest gameweek file's fixtures, plus `rescheduled`: postponed fixtures predicted on
    their own that are still to be played or kick off on/after that gameweek's first kick-off.

    status: finished | upcoming | postponed (no result, no new prediction yet) |
    rescheduled (a gameweek-file row superseded by a rescheduled prediction)."""
    s = preds[preds["season"] == season.label] if len(preds) else preds
    empty = {"season": season.label, "gameweek": None, "generated_at": None,
             "data_through": None, "fixtures": [], "rescheduled": []}
    if s.empty:
        return empty
    s = s.assign(ko=pd.to_datetime(s["kickoff_utc"], utc=True),
                 gen=pd.to_datetime(s["generated_at"], utc=True)).sort_values(["ko", "home_team"])
    resched = is_rescheduled(s["prediction_file"])
    reg, rs = s[~resched], latest_predictions(s[resched]).sort_values(["ko", "home_team"])
    played = matches[(matches["season"] == season.label) & matches["result"].notna()]
    res = {(r.home_team, r.away_team): r for r in played.itertuples()}
    moved = set(pairs(rs))

    def status(r) -> str:
        pair = (r.home_team, r.away_team)
        return ("finished" if pair in res else "postponed" if pair in postponed else "upcoming")

    out = dict(empty)
    since = None
    if len(reg):
        gw = int(reg["gameweek"].max())
        g = reg[reg["gameweek"] == gw]
        since = g["ko"].min()
        out.update(gameweek=gw, generated_at=_iso_utc(g["gen"].min()), data_through=g["data_through"].iloc[0])
        out["fixtures"] = [
            _fixture_json(r, None, "rescheduled") if (r.home_team, r.away_team) in moved
            else _fixture_json(r, res.get((r.home_team, r.away_team)), status(r))
            for r in g.itertuples()]
    for r in rs.itertuples():
        st = status(r)
        if st != "finished" or since is None or r.ko >= since:
            out["rescheduled"].append({"gameweek": int(r.gameweek),
                                       **_fixture_json(r, res.get((r.home_team, r.away_team)), st)})
    return out


def export_track(track: pd.DataFrame) -> dict:
    """Per-match record plus running averages over the matches that count: predicted before
    kick-off and with market odds (so model and market are compared on the same matches)."""
    if track.empty:
        return {"counted_rule": COUNTED_RULE, "totals": None, "by_gameweek": [], "matches": []}
    t = track.copy()
    t["ko"] = pd.to_datetime(t["kickoff_utc"], utc=True)
    t = t.sort_values(["ko", "home_team"]).reset_index(drop=True)
    t["counted"] = t["made_before_kickoff"].astype(bool) & t["mkt_log_loss"].notna()
    c = t[t["counted"]]
    cum = {f"{who}_{m}": c[f"{who}_{m}"].expanding().mean()
           for who in ["model", "mkt"] for m in ["log_loss", "brier", "correct"]}
    rows = []
    for i, r in t.iterrows():
        running = None
        if r["counted"]:
            running = {"n": int(c.index.get_loc(i)) + 1}
            for who, name in [("model", "model"), ("mkt", "market")]:
                running[name] = {"log_loss": _f(cum[f"{who}_log_loss"][i]),
                                 "brier": _f(cum[f"{who}_brier"][i]),
                                 "accuracy": _f(cum[f"{who}_correct"][i])}
        rows.append({
            "season": r["season"], "gameweek": int(r["gameweek"]), "match_id": r["match_id"],
            "kickoff_utc": _iso_utc(r["ko"]), "date_played": r["date_played"],
            "home_team": r["home_team"], "away_team": r["away_team"],
            "home_goals": int(r["home_goals"]), "away_goals": int(r["away_goals"]), "result": r["result"],
            "model": {"p_home": _f(r["p_home"]), "p_draw": _f(r["p_draw"]), "p_away": _f(r["p_away"]),
                      "log_loss": _f(r["model_log_loss"]), "brier": _f(r["model_brier"]),
                      "correct": bool(r["model_correct"])},
            "market": None if pd.isna(r["mkt_log_loss"]) else {
                "p_home": _f(r["mkt_p_home"]), "p_draw": _f(r["mkt_p_draw"]), "p_away": _f(r["mkt_p_away"]),
                "log_loss": _f(r["mkt_log_loss"]), "brier": _f(r["mkt_brier"]),
                "correct": bool(r["mkt_correct"])},
            "prediction_file": r["prediction_file"],
            "rescheduled": "_rescheduled_" in str(r["prediction_file"]),
            "made_before_kickoff": bool(r["made_before_kickoff"]), "counted": bool(r["counted"]),
            "running": running,
        })
    by_gw = [{"season": s, "gameweek": int(gw), "n": len(d), "model": _metrics(d, "model"),
              "market": _metrics(d, "mkt")} for (s, gw), d in c.groupby(["season", "gameweek"])]
    totals = {"n": len(c), "n_excluded": int((~t["counted"]).sum()),
              "model": _metrics(c, "model"), "market": _metrics(c, "mkt")} if len(c) else None
    return {"counted_rule": COUNTED_RULE, "totals": totals, "by_gameweek": by_gw, "matches": rows}


COUNTED_RULE = "predicted before kick-off and closing market odds available"
_CI = re.compile(r"^\s*([+-]?\d+\.\d+)\s*\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]\s*$")


def _report_table(path: Path) -> list[dict] | None:
    """A reports/*.csv score table as records; 'x [lo, hi]' cells -> {diff, ci_95}."""
    if not path.exists():
        return None
    out = []
    for rec in pd.read_csv(path).to_dict("records"):
        row = {}
        for k, v in rec.items():
            key = k.replace(" ", "_").replace("vs_mkt", "vs_market")
            if isinstance(v, str) and (m := _CI.match(v)):
                row[key] = {"diff": float(m.group(1)), "ci_95": [float(m.group(2)), float(m.group(3))]}
            elif isinstance(v, (float, np.floating)):
                row[key] = _f(v)
            elif isinstance(v, (int, np.integer)):
                row[key] = int(v)
            else:
                row[key] = v
        out.append(row)
    return out


def export_meta(paths: Paths, preds: pd.DataFrame, track: pd.DataFrame, matches: pd.DataFrame,
                track_json: dict) -> dict:
    forward = sorted(p for p in paths.reports_dir.glob("forward_*") if p.is_dir())
    fwd = forward[-1] if forward else None
    stamps = []
    if len(preds):
        stamps += pd.to_datetime(preds["generated_at"], utc=True).tolist()
    if len(track):
        stamps += pd.to_datetime(track["scored_at"], utc=True).tolist()
    played = matches[matches["result"].notna()]
    return {
        "name": "Xpect",
        "description": ("Pre-match home/draw/away probabilities for Premier League matches, "
                        "built only from information available before kick-off. Bookmaker odds "
                        "are a benchmark, never a model input."),
        "model": {
            "blend": f"{config.BLEND_WEIGHT_LR:g} x logistic regression + "
                     f"{1 - config.BLEND_WEIGHT_LR:g} x Dixon-Coles",
            "logistic_regression": {
                "features": "Elo ratings, rolling form (last 5/10 matches: points, goals, shots, "
                            "shots on target, corners), venue form, head-to-head, rest days, "
                            "promoted-team and no-crowd flags",
                "C": config.LR_C, "trained_from": config.FIRST_TRAIN_SEASON},
            "dixon_coles": {"time_decay_xi_per_day": config.DC_XI,
                            "window_days": config.DC_WINDOW_DAYS, "ridge": config.DC_RIDGE},
            "elo": config.ELO,
            "frozen": "src/config.py; unchanged after looking at test or forward results",
            "market_benchmark": "closing 1X2 odds from football-data.co.uk, margin removed proportionally",
        },
        "data_sources": [
            {"name": "football-data.co.uk", "url": "https://www.football-data.co.uk/englandm.php"},
            {"name": "FPL-Core-Insights", "url": "https://github.com/olbauday/FPL-Core-Insights"},
        ],
        "evaluation": {
            "protocol": ("walk-forward: each season predicted by a model trained on all earlier "
                         "seasons; differences are model minus market (or minus LR) with 95% "
                         "paired bootstrap CIs; negative = model better"),
            "cv": {"seasons": config.CV_FOLD_SEASONS, "source": "reports/cv_dc_pooled.csv",
                   "rows": _report_table(paths.reports_dir / "cv_dc_pooled.csv")},
            "test": {"seasons": config.TEST_SEASONS, "source": "reports/test/scores.csv",
                     "rows": _report_table(paths.reports_dir / "test" / "scores.csv")},
            "forward": None if fwd is None else {
                "seasons": config.FORWARD_SEASONS, "through": fwd.name.removeprefix("forward_"),
                "source": f"reports/{fwd.name}/scores.csv",
                "rows": _report_table(fwd / "scores.csv")},
            "live": track_json["totals"],
        },
        "data_through": played["date"].max().date().isoformat() if len(played) else None,
        "last_updated": _iso_utc(max(stamps)) if stamps else None,
    }


def to_json(obj) -> bytes:
    return (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


# --------------------------------------------------------------------------- run

@dataclass
class Result:
    actions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    changed: list[Path] = field(default_factory=list)
    outputs: dict[Path, bytes] = field(default_factory=dict)  # overwritten if content differs
    new_files: dict[Path, bytes] = field(default_factory=dict)  # created exclusively, never overwritten
    alarms: list[str] = field(default_factory=list)  # a prediction is about to be missed -> exit 1

    @property
    def summary(self) -> str:
        return "; ".join(self.actions) if self.actions else "nothing to do"


def run(paths: Paths = Paths(), now: pd.Timestamp | None = None, get: Getter = http_get,
        season: Season | None = None, dry_run: bool = False) -> Result:
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    season = season or Season.for_date(now)
    res = Result()

    # a) refresh ------------------------------------------------------------------------
    try:
        fpl_files = fetch_fplcore(season, get)
    except NotFound:
        if now.month in (7, 8):
            res.notes.append(f"FPL-Core has not published {season.fpl_dir} yet")
            return res
        raise
    try:
        fd_bytes: bytes | None = fetch_football_data(season, get)
    except NotFound:
        fd_bytes = None  # allowed only before the season's first result (checked below)

    old_matches = pd.read_csv(paths.matches, parse_dates=["date"]) if paths.matches.exists() else None
    raw_target = paths.raw_dir / season.raw_name
    with tempfile.TemporaryDirectory(prefix="xpect-") as tmp:
        tmp = Path(tmp)
        fpl_root = tmp / "fplcore"
        for rel, data in fpl_files.items():
            p = fpl_root / season.fpl_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        try:
            fpl = load_pl_matches(fpl_root)
        except Exception as e:
            raise SourceError(f"FPL-Core {season.fpl_dir}: could not parse ({type(e).__name__}: {e})") from e

        raw_files = [p for p in paths.raw_dir.glob("E0_*.csv") if p.name != season.raw_name]
        if fd_bytes is not None:
            (tmp / season.raw_name).write_bytes(fd_bytes)
            raw_files.append(tmp / season.raw_name)
        elif fpl["finished"].any() or raw_target.exists():
            raise SourceError(f"football-data.co.uk {season.label} file not found (404), "
                              "but the season has started")
        try:
            built = build_matches(files=sorted(raw_files, key=lambda p: p.name))
        except Exception as e:
            raise SourceError(f"football-data.co.uk: could not build matches ({type(e).__name__}: {e})") from e
    matches_text = matches_csv_text(built)
    matches = read_csv_text(matches_text, parse_dates=["date"])
    lag = check_consistency(matches, old_matches, fpl, season)
    for key, what in [("fd_behind", "finished in FPL-Core but not yet in football-data"),
                      ("fpl_behind", "in football-data but not yet finished in FPL-Core")]:
        if len(lag[key]):
            res.notes.append(f"{len(lag[key])} matches {what}")

    if fd_bytes is not None:
        res.outputs[raw_target] = fd_bytes
    res.outputs[paths.matches] = matches_text.encode()
    res.outputs[paths.features] = build_features(matches).to_csv(index=False).encode()
    n_old = 0 if old_matches is None else int(old_matches["result"].notna().sum())
    n_new = int(matches["result"].notna().sum()) - n_old
    if n_new:
        res.actions.append(f"{n_new} new results")

    # b) score --------------------------------------------------------------------------
    preds = load_saved_predictions(paths.pred_dir)
    track = load_track(paths.track)
    track_text = paths.track.read_text() if paths.track.exists() else ""
    if len(preds):
        new_rows = score_new(preds, matches, track)
        if len(new_rows):
            track_text += new_rows.to_csv(index=False, header=not track_text, float_format="%.6f")
            gws = ", ".join(f"GW{g}" for g in sorted(new_rows["gameweek"].unique()))
            res.actions.append(f"scored {len(new_rows)} matches ({gws})")
    if track_text:
        res.outputs[paths.track] = track_text.encode()
        track = read_csv_text(track_text)

    # c) predict ------------------------------------------------------------------------
    fc = season_fixtures(fpl, season)
    why = postponed_fixtures(fc, matches, preds, season, now)
    for i, w in why.items():
        res.notes.append(f"postponed: {_fixture_name(fc.loc[i])}: {w}")
    postponed = why.index

    def add_prediction(path: Path, fixtures: pd.DataFrame, gameweek: int) -> None:
        nonlocal preds
        pred = prediction_file(matches, fixtures, season.label, gameweek, now)
        text = pred.to_csv(index=False, float_format="%.6f")
        res.new_files[path] = text.encode()
        pred = read_csv_text(text).assign(prediction_file=str(path.relative_to(paths.pred_dir)))
        preds = pred if preds.empty else pd.concat([preds, pred], ignore_index=True)

    plan = plan_prediction(fc, matches, season, paths.pred_dir, now, postponed)
    res.notes.append(plan.reason)
    if plan.gameweek is not None:
        try:
            add_prediction(gw_path(season.label, plan.gameweek, paths.pred_dir), plan.fixtures, plan.gameweek)
        except ValueError as e:
            raise InconsistentData(f"GW{plan.gameweek} prediction failed: {e}") from e
        res.actions.append(f"predicted {season.label} GW{plan.gameweek}")
    elif plan.blocked and pd.notna(plan.first_kickoff) and now < plan.first_kickoff <= now + ALARM_WITHIN:
        res.alarms.append(f"{plan.reason}; first kick-off {plan.first_kickoff:%Y-%m-%d %H:%M} UTC "
                          f"is in {_until(plan.first_kickoff, now)}")

    rplan = plan_rescheduled(fc, matches, preds, season, paths.pred_dir, now, postponed)
    res.notes += rplan.notes
    res.alarms += rplan.alarms
    for fixtures in rplan.due:
        gw, date = int(fixtures["gameweek"].iloc[0]), fixtures["date"].iloc[0]
        n = 1
        while (path := rescheduled_path(season.label, gw, date, paths.pred_dir, n)).exists() \
                or path in res.new_files:
            n += 1
        names = ", ".join(_fixture_name(r) for _, r in fixtures.iterrows())
        try:
            add_prediction(path, fixtures, gw)
        except ValueError as e:  # never blocks the rest of the run
            msg = f"rescheduled {names} not predicted: {e}"
            res.notes.append(msg)
            if fixtures["kickoff_utc"].min() - now <= ALARM_WITHIN:
                res.alarms.append(msg)
            continue
        res.actions.append(f"predicted rescheduled {names}")

    # d) export -------------------------------------------------------------------------
    track_json = export_track(track)
    # shown as postponed unless predicted for its current date and not over 48h past kick-off
    dates = latest_predicted_dates(preds, season)
    pp = frozenset(pair for pair, (_, r) in zip(pairs(fc.loc[postponed]), fc.loc[postponed].iterrows())
                   if not (r["kickoff_utc"] > now - POSTPONED_AFTER
                           and pair in dates and dates[pair] >= r["date"]))
    site = {"current.json": export_current(preds, matches, season, pp), "track_record.json": track_json,
            "meta.json": export_meta(paths, preds, track, matches, track_json)}
    for name, obj in site.items():
        res.outputs[paths.site_dir / name] = to_json(obj)

    # write (only now; nothing above touched the repository) ------------------------------
    for path, data in res.new_files.items():
        if path.exists():
            raise Abort(f"{path} appeared during the run; not overwriting")
    unchanged = [p for p, d in res.outputs.items() if p.exists() and p.read_bytes() == d]
    for p in unchanged:
        del res.outputs[p]
    res.changed = list(res.new_files) + list(res.outputs)
    if not dry_run:
        for path, data in res.new_files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "xb") as f:
                f.write(data)
        for path, data in res.outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(path.name + ".tmp")
            tmp_path.write_bytes(data)
            os.replace(tmp_path, path)
    if res.changed and not res.actions:
        names = sorted({p.parent.name if p.parent.name == "site_data" else p.name for p in res.changed})
        res.actions.append("refreshed " + ", ".join(names))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--season", help="e.g. 2026/27 (default: from today's date, switching on 1 July)")
    ap.add_argument("--dry-run", action="store_true", help="do everything except write files")
    ap.add_argument("--summary-file", type=Path, help="also write the one-line summary here")
    args = ap.parse_args(argv)
    try:
        res = run(season=Season.parse(args.season) if args.season else None, dry_run=args.dry_run)
    except Abort as e:
        print(f"ABORTED ({type(e).__name__}): {e}\nNothing was written.", file=sys.stderr)
        return 1
    for n in res.notes:
        print(f"  - {n}")
    verb = "would change" if args.dry_run else "changed"
    if res.changed:
        print(f"{verb} {len(res.changed)} files:")
        for p in res.changed:
            print(f"    {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
    print(f"summary: {res.summary}")
    if args.summary_file:
        args.summary_file.write_text(res.summary + "\n")
    if res.alarms:
        print("\nPREDICTION ABOUT TO BE MISSED. Everything else was written; exiting with code 1 "
              "so the scheduled job fails.", file=sys.stderr)
        for a in res.alarms:
            print(f"  ! {a}", file=sys.stderr)
        print("Predict manually before kick-off (see README, 'Not handled automatically').",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
