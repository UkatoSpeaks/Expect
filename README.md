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
.venv/bin/python -m src.cv_dc            # Dixon-Coles xi tuning + final candidates -> data/dc_probs.csv

# 4. held-out evaluation of the frozen model (each refuses to overwrite an existing run)
.venv/bin/python -m src.evaluate_test --split test     # -> reports/test/
.venv/bin/python -m src.evaluate_test --split forward  # -> reports/forward_<last played date>/

# tests
.venv/bin/python -m pytest tests
```

Final model (frozen in `src/config.py`): 50/50 average of the full-feature logistic
regression and time-weighted Dixon-Coles probabilities.

## Results

Same walk-forward protocol everywhere: each season is predicted by a model trained on all
earlier seasons (from 2001/02), with Dixon-Coles refit before every match date. Nothing
was changed after looking at test or forward results. Brackets are 95% paired bootstrap
CIs of the per-match difference (negative log loss / Brier difference = model better).

| Matches | Blend log loss | Market log loss | Blend − market log loss | Blend − market Brier | Blend − LR log loss |
|---|---|---|---|---|---|
| CV 2017/18–2022/23 (2,280) | 0.9637 | 0.9499 | | | |
| **Test** 2023/24 (380) | 0.9192 | 0.9007 | +0.0185 [+0.0046, +0.0318] | +0.0136 [+0.0043, +0.0226] | +0.0003 [−0.0089, +0.0089] |
| **Test** 2024/25 (380) | 0.9823 | 0.9667 | +0.0155 [+0.0011, +0.0300] | +0.0119 [+0.0021, +0.0218] | −0.0057 [−0.0149, +0.0031] |
| **Test pooled** (760) | **0.9508** | 0.9337 | +0.0170 [+0.0072, +0.0271] | +0.0128 [+0.0060, +0.0196] | −0.0027 [−0.0090, +0.0035] |
| **Forward** 2025/26 (380) | 1.0292 | 1.0118 | +0.0174 [+0.0035, +0.0310] | +0.0114 [+0.0017, +0.0208] | −0.0092 [−0.0199, +0.0008] |
| **Forward** 2026/27 to 8 Nov (100) | 1.0177 | 1.0108 | +0.0069 [−0.0289, +0.0415] | +0.0034 [−0.0214, +0.0271] | −0.0043 [−0.0240, +0.0157] |
| **Forward pooled** (480) | **1.0268** | 1.0116 | +0.0152 [+0.0021, +0.0278] | +0.0097 [+0.0007, +0.0187] | −0.0082 [−0.0175, +0.0009] |

All models on the pooled sets (log loss / Brier / accuracy):

| Model | Test (760) | Forward (480) |
|---|---|---|
| Blend (LR + DC) | 0.9508 / 0.5637 / 55.5% | 1.0268 / 0.6168 / 47.5% |
| Logistic regression | 0.9535 / 0.5659 / 55.8% | 1.0349 / 0.6228 / 46.7% |
| Dixon-Coles | 0.9554 / 0.5667 / 55.9% | 1.0280 / 0.6164 / 48.3% |
| Home-rate baseline | 1.0709 / 0.6485 / 43.4% | 1.0822 / 0.6554 / 42.5% |
| Market (closing, margin removed) | 0.9337 / 0.5510 / 57.8% | 1.0116 / 0.6071 / 49.2% |

What this says:
- The model is far better than the home-rate baseline, and the CV gap to the market (about
  0.014 log loss) held up out of sample: about 0.017 on test and 0.015 on forward. The
  market is significantly better on both, except on the first 100 matches of 2026/27,
  which are too few to tell apart.
- Blending helped slightly over LR alone on every held-out season except 2023/24 (−0.003
  test, −0.008 forward), but none of these CIs exclude zero.
- 2025/26 was a harder season for everyone (market log loss 1.012, vs 0.90–0.97 in test).

Full tables (per season, every model, CIs vs market and vs LR, accuracy) and calibration
plots: `reports/test/` and `reports/forward_2026-11-08/`. `reports/invalid/` holds a
first run with a bug (DC probabilities leaked into the LR features); see its README.

## Live predictions

```bash
# after each gameweek: refresh data/raw/E0_2627.csv and the FPL-Core-Insights clone, then
.venv/bin/python -m src.load_data
.venv/bin/python -m src.score_predictions   # score finished predicted matches -> predictions/track_record.csv
.venv/bin/python -m src.predict             # next gameweek -> predictions/2026-27/gw<N>.csv
```

- `src.predict` trains on every played match, takes the gameweek's fixtures from FPL-Core,
  and writes the blend P(H)/P(D)/P(A), the LR and DC components, and the Elo and DC inputs
  used, with a `generated_at` timestamp. It refuses if the file already exists, if any
  fixture has kicked off, or if `data/matches.csv` is missing a finished match.
- A whole gameweek is predicted at once, so Sunday's predictions do not use Saturday's
  results (the historical evaluation refit DC before each date).
- `src.score_predictions` appends each newly finished match once (model and market log
  loss, Brier, correct pick) and never rewrites earlier rows.

XGBoost on macOS needs the OpenMP runtime: `brew install libomp`.
