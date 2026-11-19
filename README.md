# Xpect

Pre-match outcome probabilities (home win / draw / away win) for English Premier League
matches, built only from information available before kick-off. Bookmaker odds are used
only as a benchmark to compare against, never as model input.

## Data sources

- **[football-data.co.uk](https://www.football-data.co.uk/englandm.php)**: Premier League
  results, match stats and bookmaker odds, 2000/01 onward (`data/raw/E0_*.csv`).
- **[FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights)** by olbauday: match-level
  stats (xG, possession, big chances) and fixtures for 2024/25 onward, cloned into `data/external/`.

Thanks to both projects for publishing their data.

## Splits (fixed before modelling)

| Role | Seasons |
|---|---|
| Train / walk-forward CV | 2001/02 – 2022/23 (2000/01 is Elo burn-in) |
| Test | 2023/24, 2024/25 |
| Forward test (never used for tuning) | 2025/26, 2026/27 |

## How to run

```bash
# 1. raw data: put football-data.co.uk season files in data/raw/E0_YYZZ.csv
.venv/bin/python -m src.load_data        # -> data/matches.csv

# 2. optional external data
git clone https://github.com/olbauday/FPL-Core-Insights data/external/FPL-Core-Insights
.venv/bin/python -m src.fplcore          # join report vs data/matches.csv

# 3. features and models
.venv/bin/python -m src.features         # -> data/features.csv
.venv/bin/python -m src.cv               # walk-forward validation, Elo tuning, LR + XGBoost

# tests
.venv/bin/python -m pytest tests
```

XGBoost on macOS needs the OpenMP runtime: `brew install libomp`.
