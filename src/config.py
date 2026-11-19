"""Frozen final-model configuration (Step 5, 2026-11-19).

Final model: equal-weight average of
  1. full-feature multinomial logistic regression (impute median + indicators ->
     StandardScaler -> LogisticRegression(C=LR_C)) on all features from src/features.py, and
  2. time-weighted Dixon-Coles probabilities (src/dixon_coles.py).

Selected on pooled walk-forward log loss, 2017/18-2022/23 (2,280 matches):
  Average(LR, DC) 0.9637 | DC alone 0.9659 | LR + DC features 0.9668 | LR 0.9704 | market 0.9499.
See reports/cv_dc_pooled.csv. Test and forward seasons were not used for any choice here.

Do not change these values after looking at test or forward results.
"""

# --- seasons
BURN_IN_SEASONS = ["2000/01"]
FIRST_TRAIN_SEASON = "2001/02"
CV_FOLD_SEASONS = ["2017/18", "2018/19", "2019/20", "2020/21", "2021/22", "2022/23"]
TEST_SEASONS = ["2023/24", "2024/25"]
FORWARD_SEASONS = ["2025/26", "2026/27"]

# --- Elo (tuned in src/cv.py on pooled walk-forward log loss of Elo-only LR)
ELO = {"k": 20.0, "home_adv": 0.0, "regress": 0.0, "gd_multiplier": True, "base": 1500.0}

# --- logistic regression (C chosen in src/cv.py from 0.001-1.0)
LR_C = 0.003
LR_MAX_ITER = 3000

# --- Dixon-Coles (xi tuned in src/cv_dc.py; window / ridge / grid fixed by design, not tuned)
DC_XI = 0.002          # per day; half-life ~ 346 days
DC_WINDOW_DAYS = 5 * 365
DC_RIDGE = 2.0
DC_MAX_GOALS = 10
DC_MIN_MATCHES = 150

# --- blend
BLEND_WEIGHT_LR = 0.5  # final P = w * P_LR + (1 - w) * P_DC
