"""Load Premier League match-level stats from the FPL-Core-Insights clone.

Repo: https://github.com/olbauday/FPL-Core-Insights, cloned to data/external/.
Coverage: 2024/25 onward only (xG, possession, big chances, ...).

Layout differs by season:
- 2024-2025: data/2024-2025/matches/GW*/matches.csv (PL only, no tournament column)
- 2025-2026+: data/<season>/By Tournament/Premier League/GW*/matches.csv
  (the "By Gameweek" files also contain cup/European games; we avoid them).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.team_name_map import to_fd_name

ROOT = Path(__file__).resolve().parents[1]
FPLCORE_DIR = ROOT / "data" / "external" / "FPL-Core-Insights" / "data"


def _season_files(season_dir: Path) -> tuple[list[Path], Path]:
    old = sorted(season_dir.glob("matches/GW*/matches.csv"))
    if old:
        return old, season_dir / "teams" / "teams.csv"
    new = sorted(season_dir.glob("By Tournament/Premier League/GW*/matches.csv"))
    return new, season_dir / "teams.csv"


def load_pl_matches(root: Path = FPLCORE_DIR) -> pd.DataFrame:
    """All PL matches (finished and upcoming), with football-data team names and a UK date."""
    frames = []
    for season_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files, teams_path = _season_files(season_dir)
        if not files:
            continue
        m = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).copy()
        if "tournament" in m.columns:
            m = m[m["tournament"] == "prem"].copy()
        teams = pd.read_csv(teams_path).set_index("code")["name"]
        for side in ["home", "away"]:
            codes = m[f"{side}_team"].astype(int)
            missing = set(codes) - set(teams.index)
            if missing:
                raise ValueError(f"{season_dir.name}: team codes not in teams.csv: {missing}")
            m[f"{side}_team_fpl"] = codes.map(teams)
            m[f"{side}_team"] = m[f"{side}_team_fpl"].map(to_fd_name)
        a, b = season_dir.name.split("-")
        m.insert(0, "season", f"{a}/{b[2:]}")
        frames.append(m)
    df = pd.concat(frames, ignore_index=True).copy()
    # kickoff_time is UTC (some seasons tz-aware, some naive). Convert to UK local date,
    # which is what football-data.co.uk uses.
    ko = pd.to_datetime(df["kickoff_time"], format="mixed", utc=True)
    df["kickoff_utc"] = ko
    df["date"] = ko.dt.tz_convert("Europe/London").dt.tz_localize(None).dt.normalize()
    df["finished"] = df["finished"].astype(bool)
    return df.drop_duplicates("match_id").reset_index(drop=True)


def join_report(matches_csv: Path = ROOT / "data" / "matches.csv") -> pd.DataFrame:
    fd = pd.read_csv(matches_csv, parse_dates=["date"])
    fc = load_pl_matches()
    keys = ["date", "home_team", "away_team"]
    fin = fc[fc["finished"]]
    merged = fin.merge(fd[keys + ["season", "home_goals", "away_goals"]], on=keys,
                       how="left", suffixes=("", "_fd"), indicator=True)
    print("FPL-Core finished PL matches joined to matches.csv on date+home+away:")
    rep = merged.groupby("season").agg(
        fplcore=("match_id", "size"), joined=("_merge", lambda s: (s == "both").sum()))
    fd_counts = fd[fd["season"].isin(rep.index)].groupby("season").size()
    rep["matches_csv"] = fd_counts
    rep["join_rate"] = rep["joined"] / rep["fplcore"]
    print(rep.to_string())
    j = merged[merged["_merge"] == "both"]
    bad = j[(j["home_score"] != j["home_goals"]) | (j["away_score"] != j["away_goals"])]
    print(f"score disagreements among joined: {len(bad)}")
    unj = merged[merged["_merge"] == "left_only"]
    if len(unj):
        print("unjoined:\n", unj[["season", "date", "home_team", "away_team"]].to_string())
    return merged


if __name__ == "__main__":
    join_report()
