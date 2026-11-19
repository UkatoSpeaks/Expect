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
# 0. environment (Python 3.11)
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt

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

## Automated pipeline

`src/pipeline.py` runs the whole weekly cycle in one idempotent command. It is safe to run
any number of times: when there is nothing new it changes no file.

```bash
.venv/bin/python -m src.pipeline             # refresh, score, predict if due, export
.venv/bin/python -m src.pipeline --dry-run   # same checks, list what would change, write nothing
```

Each run:

1. **Refresh**: downloads the current season's football-data.co.uk file and the season's
   FPL-Core-Insights fixtures/results (`teams.csv` and each published gameweek's
   `matches.csv`), then rebuilds `data/raw/E0_<season>.csv`, `data/matches.csv` and
   `data/features.csv`. The season is chosen from today's date, switching on 1 July
   (`--season 2026/27` overrides it).
2. **Score**: scores newly finished predicted matches into `predictions/track_record.csv`,
   using the same append-only code as `src.score_predictions`.
3. **Predict**: takes the next gameweek (the lowest one with no finished match) and predicts
   it only if all three hold: (a) it has no predictions file yet, (b) every earlier gameweek
   is finished in FPL-Core *and* its results are in football-data, and (c) its first
   kick-off is in the future. Otherwise it does nothing and says why. Postponed fixtures
   (below) are ignored by (b) and left out of the gameweek file.
4. **Rescheduled fixtures**: a postponed fixture with a new, future kick-off is predicted on
   its own into `predictions/<season>/gw<N>_rescheduled_<date>.csv` (N = its FPL-Core
   gameweek, date = new match date). It is due once every gameweek that ends before the new
   kick-off is finished in both sources, the same rule as for a gameweek. Files are created
   with mode `"x"` like gameweek files. The fixture is predicted again only if FPL-Core moves
   it to a later date than its latest prediction.
5. **Export**: writes the site data in `site_data/`:
   - `current.json`: fixtures of the latest predicted gameweek, with P(H/D/A), the pick,
     kick-off, Elo ratings and gap, Dixon-Coles expected goals for each side, the result
     once played, and `generated_at`. `status` is `finished`, `upcoming`, `postponed` (no
     new prediction yet) or `rescheduled` (superseded by a rescheduled prediction).
     `rescheduled` lists rescheduled predictions that are still to be played or kick off
     on or after the gameweek's first kick-off, each with its `gameweek`.
   - `track_record.json`: every scored match (model vs market vs result), with running
     log loss, Brier and accuracy for model and market, per-gameweek and overall totals.
     Only matches predicted before kick-off and with closing odds count towards the totals,
     so model and market are compared on the same matches. Each match carries its
     `prediction_file` and a `rescheduled` flag.
   - `meta.json`: model description and frozen settings, the CV/test/forward results from
     `reports/`, the live totals, `data_through` and `last_updated`.
6. Prints a summary and exits 0, including when there was nothing to do, unless a
   prediction is about to be missed (below).

**Postponed fixtures.** A fixture with no result in either source counts as postponed if
FPL-Core gives it no kick-off time, if its scheduled kick-off was more than 48h ago, or if
FPL-Core has moved it to a later date: after the next gameweek's first kick-off, or to a
later date than the one in its latest prediction file. (FPL-Core keeps a postponed match in
its original gameweek and changes its `kickoff_time`, as with Everton v Liverpool in
2024/25.) Postponed fixtures never block the next gameweek. The original gameweek file is
never changed. When a match has a rescheduled prediction, the track record scores that one,
the latest made before the match was played.

**Missed-prediction alarm.** If the next gameweek's first kick-off (or a rescheduled
fixture's kick-off) is less than 24h away and it still cannot be predicted, the run writes
everything else as usual, prints `PREDICTION ABOUT TO BE MISSED` with the reason, and exits
1. Typical reasons are football-data.co.uk not yet publishing the last results, an earlier
fixture not finished in FPL-Core within 48h, or a double gameweek. The workflow runs every
12h, so every kick-off gets at least one run inside that 24h window.

Safety:
- All outputs are computed in memory first. Files are written only at the end, and only if
  their content changed.
- The run stops with exit code 1 and writes nothing if a source is unreachable or returns
  something unparseable, or if the sources disagree. That covers a football-data result with
  no FPL-Core fixture, a finished match whose score or date differs between the sources, and
  a result already in `data/matches.csv` that changed or disappeared.
- One source lagging the other by a few hours is not treated as an error. The run reports
  the lagging matches, and prediction waits until both sources have them.
- Existing gameweek and rescheduled files are never overwritten (new ones are created with
  mode `"x"`), and the track record is only ever appended to.
- FPL-Core files are downloaded to a temporary directory. The clone in `data/external/`,
  used by the manual commands above, is not modified.

Not handled automatically (each one triggers the missed-prediction alarm before kick-off,
so the job fails and you get an email):

- **Double gameweeks.** If a team has two fixtures in the next gameweek, that gameweek is
  not predicted. FPL sometimes moves a rearranged fixture into another gameweek. If the
  fixture had already been predicted for an earlier date, it is handled as rescheduled.
  Otherwise it looks like an ordinary double gameweek and is not predicted.
- **football-data.co.uk running late.** Prediction waits for every earlier result. If the
  last results are still missing 24h before kick-off, nothing is predicted automatically.
- **A match neither finished nor postponed in FPL-Core yet.** A result can be overdue by
  less than 48h, with FPL-Core not yet showing a new date. If the next gameweek kicks off
  inside that window, the match still blocks it.
- **A rescheduled fixture whose date clashes with played results.** The model refuses a
  fixture dated on or before the last played match. It is reported, not predicted.

The alarm only tells you a prediction is about to be missed; there is no one-command fix.
`python -m src.predict` applies the same checks, so it also refuses double gameweeks and
missing results. Any manual workaround means predicting without the missing data. Decide
case by case, and commit a hand-made file before kick-off. The pipeline never overwrites a
predictions file and scores it like any other. If nothing is done, the gameweek is simply
not predicted and does not appear in the track record.

### GitHub Actions

`.github/workflows/pipeline.yml` runs twice a day, at 06:00 and 18:00 UTC. It installs
`requirements.txt` (with pip caching), runs `pytest`, runs the pipeline, and, if anything in
`predictions/`, `site_data/` or `data/` changed, commits it as `pipeline: <summary>` (for
example `pipeline: 10 new results; scored 10 matches (GW11); predicted 2026/27 GW12`) and
pushes it with the default `GITHUB_TOKEN`. If the tests fail or the pipeline aborts, the job
fails and nothing is committed. If the missed-prediction alarm fires, whatever the run wrote
is still committed, and then the job fails.

GitHub emails failed scheduled runs to the user who last edited the `cron` line, as long as
Actions notifications are enabled in their GitHub settings (Settings → Notifications →
Actions).

To trigger it manually:

- **GitHub UI:** open the **Actions** tab, select **pipeline**, then **Run workflow** (on `main`).
- **CLI:** `gh workflow run pipeline.yml`, then `gh run watch` to follow it.

If the push is rejected, check that the repository allows it under Settings → Actions →
General → Workflow permissions ("Read and write permissions"), and that branch protection
on `main` does not block `github-actions[bot]`.

`data/raw/` is committed so that CI can rebuild `data/matches.csv`. `data/external/` is not
committed.

XGBoost on macOS needs the OpenMP runtime: `brew install libomp`. It is only needed for the
research scripts (`src.cv`). The pipeline and tests need only `requirements.txt`.
