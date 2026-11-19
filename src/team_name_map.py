"""Map FPL-Core-Insights team names to football-data.co.uk team names.

FPL-Core match files store teams as FPL team `code` integers; each season's
teams.csv resolves code -> `name` (FPL short-ish name). This module maps those
names onto the names used in data/matches.csv (football-data.co.uk).

Names not listed here are identical in both sources (e.g. "Arsenal", "Leeds").
"""

from __future__ import annotations

FPL_TO_FD: dict[str, str] = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    "Ipswich Town": "Ipswich",   # 2026/27 teams.csv uses the long form
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    # identical names, listed so unknown names can be detected
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "Bournemouth": "Bournemouth",
    "Brentford": "Brentford", "Brighton": "Brighton", "Burnley": "Burnley",
    "Chelsea": "Chelsea", "Crystal Palace": "Crystal Palace", "Everton": "Everton",
    "Fulham": "Fulham", "Ipswich": "Ipswich", "Leeds": "Leeds",
    "Leicester": "Leicester", "Liverpool": "Liverpool", "Man City": "Man City",
    "Newcastle": "Newcastle", "Nott'm Forest": "Nott'm Forest",
    "Southampton": "Southampton", "Sunderland": "Sunderland",
    "West Ham": "West Ham", "Wolves": "Wolves",
}


def to_fd_name(fpl_name: str) -> str:
    """Return the football-data.co.uk name; raise on unknown names so new clubs are noticed."""
    try:
        return FPL_TO_FD[fpl_name]
    except KeyError:
        raise KeyError(f"Unmapped FPL-Core team name: {fpl_name!r} — add it to FPL_TO_FD") from None
