"""Combine football-data.co.uk Premier League season files into data/matches.csv.

Raw files live in data/raw/E0_{YYZZ}.csv (e.g. E0_0001.csv = 2000/01).

Notes on the raw files:
- Column sets drift between seasons; we map everything to stable snake_case names.
- Some files (2003/04, 2004/05) have data rows with more fields than the header.
  The extra fields are empty trailing commas, so we trim them (and assert that
  they really are empty). pandas.read_csv would otherwise silently skip those rows.
- Encodings vary (UTF-8 with BOM in recent seasons, Latin-1 in older ones).
- Dates mix dd/mm/yy and dd/mm/yyyy.

Odds: we pick one 1X2 odds triple per match, preferring true closing market
averages, and record which source was used (`odds_source`, `odds_is_closing`).
Implied probabilities have the bookmaker margin (overround) removed by
proportional normalisation.

Bad rows are never dropped: they get flags in `dq_flags`.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "matches.csv"
CROSSCHECK_PATH = ROOT / "data" / "epl_final.csv"

# raw column -> normalised column
COLUMN_MAP = {
    "Date": "date",
    "Time": "kickoff_time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HTR": "ht_result",
    "Referee": "referee",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
}

STAT_COLS = [
    "home_goals", "away_goals", "ht_home_goals", "ht_away_goals",
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners", "home_fouls", "away_fouls",
    "home_yellows", "away_yellows", "home_reds", "away_reds",
]

# Odds sources in priority order: (name, raw prefix, is_closing).
# Closing odds only exist from 2012/13 (Pinnacle) and 2019/20 (market average, Bet365).
ODDS_PRIORITY = [
    ("avg_closing", "AvgC", True),       # market average closing, 2019/20+
    ("pinnacle_closing", "PSC", True),   # Pinnacle closing, 2012/13+
    ("b365_closing", "B365C", True),     # Bet365 closing, 2019/20+
    ("avg_preclose", "BbAv", False),     # BetBrain market average, 2005/06-2018/19
    ("avg_preclose", "Avg", False),      # market average (pre-closing), 2019/20+
    ("b365_preclose", "B365", False),    # Bet365, 2002/03+
]
# 2000/01 and 2001/02 have no Bet365 / averages: use the mean of available books.
FALLBACK_BOOKS = ["GB", "IW", "LB", "SB", "WH", "SY", "BW", "SO", "VC"]

OVERROUND_RANGE = (1.0, 1.30)


def season_label(code: str) -> str:
    """'0001' -> '2000/01'."""
    start = 2000 + int(code[:2]) if int(code[:2]) < 90 else 1900 + int(code[:2])
    return f"{start}/{code[2:]}"


def read_raw(path: Path) -> pd.DataFrame:
    """Read one raw season file robustly (encoding, BOM, ragged rows, blank rows)."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    rows = list(csv.reader(io.StringIO(text)))
    header = [h.strip() for h in rows[0]]
    n = len(header)
    fixed = []
    for i, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue  # blank line (e.g. trailing ",,,," rows)
        if len(row) > n:
            extra = row[n:]
            if any(cell.strip() for cell in extra):
                raise ValueError(f"{path.name} line {i}: non-empty fields beyond header: {extra}")
            row = row[:n]
        elif len(row) < n:
            row = row + [""] * (n - len(row))
        fixed.append(row)

    df = pd.DataFrame(fixed, columns=header)
    # drop unnamed header columns created by trailing commas in the header
    df = df.loc[:, [c for c in df.columns if c]]
    return df.replace({"": np.nan})


def parse_dates(s: pd.Series) -> pd.Series:
    s = s.str.strip()
    long_fmt = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short_fmt = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return long_fmt.fillna(short_fmt)


def _triple(df: pd.DataFrame, prefix: str) -> pd.DataFrame | None:
    cols = [f"{prefix}{x}" for x in "HDA"]
    if not all(c in df.columns for c in cols):
        return None
    return df[cols].apply(pd.to_numeric, errors="coerce").set_axis(["H", "D", "A"], axis=1)


def select_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the best available odds triple per row, in ODDS_PRIORITY order."""
    out = pd.DataFrame(np.nan, index=df.index, columns=["H", "D", "A"])
    source = pd.Series(pd.NA, index=df.index, dtype="object")
    closing = pd.Series(pd.NA, index=df.index, dtype="boolean")

    def fill(triple: pd.DataFrame, name: str, is_closing: bool) -> None:
        valid = triple.notna().all(axis=1) & (triple > 1.0).all(axis=1)
        take = valid & source.isna()
        out.loc[take] = triple.loc[take].values
        source.loc[take] = name
        closing.loc[take] = is_closing

    for name, prefix, is_closing in ODDS_PRIORITY:
        t = _triple(df, prefix)
        if t is not None:
            fill(t, name, is_closing)

    books = [t for b in FALLBACK_BOOKS if (t := _triple(df, b)) is not None]
    if books:
        stacked = pd.concat(books, keys=range(len(books)))
        stacked = stacked.where(stacked > 1.0)
        mean_odds = stacked.groupby(level=1).mean()
        fill(mean_odds, "books_mean_preclose", False)

    res = out.rename(columns={"H": "odds_home", "D": "odds_draw", "A": "odds_away"})
    res["odds_source"] = source
    res["odds_is_closing"] = closing
    return res


def add_implied_probs(df: pd.DataFrame) -> pd.DataFrame:
    inv = 1.0 / df[["odds_home", "odds_draw", "odds_away"]]
    df["overround"] = inv.sum(axis=1)
    for col, side in zip(inv.columns, ["home", "draw", "away"]):
        df[f"p_{side}_implied"] = inv[col] / df["overround"]
    return df


def load_season(path: Path) -> pd.DataFrame:
    code = re.search(r"E0_(\d{4})\.csv$", path.name).group(1)
    raw = read_raw(path)
    df = raw[[c for c in COLUMN_MAP if c in raw.columns]].rename(columns=COLUMN_MAP)
    for col in COLUMN_MAP.values():
        if col not in df.columns:
            df[col] = np.nan
    df.insert(0, "season", season_label(code))
    df["date"] = parse_dates(df["date"])
    for col in ["home_team", "away_team", "referee", "result", "ht_result"]:
        df[col] = df[col].str.strip()
    for col in STAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = pd.concat([df, select_odds(raw)], axis=1)
    return df


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean dq_* columns and a combined dq_flags string. Nothing is dropped."""
    hg, ag = df["home_goals"], df["away_goals"]
    expected = np.select([hg > ag, hg < ag, hg == ag], ["H", "A", "D"], default=None)
    df["dq_result_mismatch"] = (df["result"] != expected) & hg.notna() & ag.notna()
    df["dq_sot_gt_shots"] = (
        (df["home_shots_on_target"] > df["home_shots"])
        | (df["away_shots_on_target"] > df["away_shots"])
    ).fillna(False).astype(bool)
    df["dq_missing_stats"] = df[STAT_COLS].isna().any(axis=1)
    df["dq_missing_core"] = df[["date", "home_team", "away_team", "home_goals",
                                "away_goals", "result"]].isna().any(axis=1)
    df["dq_missing_odds"] = df["odds_home"].isna()
    lo, hi = OVERROUND_RANGE
    df["dq_odds_suspect"] = df["overround"].notna() & ~df["overround"].between(lo, hi)
    df["dq_duplicate"] = df.duplicated(["date", "home_team", "away_team"], keep=False)

    flag_cols = [c for c in df.columns if c.startswith("dq_")]
    df["dq_flags"] = df[flag_cols].apply(
        lambda r: ";".join(c[3:] for c in flag_cols if r[c]), axis=1
    )
    df["dq_ok"] = df["dq_flags"] == ""
    return df


def build_matches(raw_dir: Path = RAW_DIR, files: list[Path] | None = None) -> pd.DataFrame:
    """All season files in raw_dir, or exactly `files` if given (used by src.pipeline to
    build from a freshly downloaded file before it replaces the one in data/raw)."""
    files = sorted(raw_dir.glob("E0_*.csv")) if files is None else list(files)
    if not files:
        raise FileNotFoundError(f"No E0_*.csv files in {raw_dir}")
    df = pd.concat([load_season(p) for p in files], ignore_index=True)
    df = add_implied_probs(df)
    df = add_quality_flags(df)
    df = df.sort_values(["date", "kickoff_time", "home_team"], na_position="first")
    return df.reset_index(drop=True)


def cross_check(matches: pd.DataFrame, path: Path = CROSSCHECK_PATH) -> pd.DataFrame:
    """Compare results/goals with the Kaggle epl_final.csv on overlapping matches."""
    ref = pd.read_csv(path, parse_dates=["MatchDate"]).rename(columns={
        "MatchDate": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FullTimeHomeGoals": "ref_home_goals", "FullTimeAwayGoals": "ref_away_goals",
        "FullTimeResult": "ref_result",
    })
    keys = ["date", "home_team", "away_team"]
    merged = matches.merge(ref[keys + ["ref_home_goals", "ref_away_goals", "ref_result"]],
                           on=keys, how="outer", indicator=True)
    both = merged[merged["_merge"] == "both"]
    mismatch = both[
        (both["home_goals"] != both["ref_home_goals"])
        | (both["away_goals"] != both["ref_away_goals"])
        | (both["result"] != both["ref_result"])
    ]
    print(f"Cross-check vs {path.name}: {len(both)} overlapping matches, "
          f"{len(mismatch)} disagreements")
    print(f"  only in matches.csv:   {(merged['_merge'] == 'left_only').sum()}")
    print(f"  only in epl_final.csv: {(merged['_merge'] == 'right_only').sum()}")
    return merged


def main() -> None:
    df = build_matches()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Wrote {len(df)} matches, {df['season'].nunique()} seasons -> {OUT_PATH}")
    if CROSSCHECK_PATH.exists():
        cross_check(df)


if __name__ == "__main__":
    main()
