"""Time-weighted Dixon-Coles model, refit before every match date.

Model (Dixon & Coles 1997):
    home goals ~ Poisson(lam), away goals ~ Poisson(mu), with
    log lam = c + home + att[h] - def[a]
    log mu  = c +        att[a] - def[h]
    and the low-score correction tau(x, y) on 0-0, 1-0, 0-1, 1-1 governed by rho.
Each past match is weighted by exp(-xi * days_ago). Matches older than WINDOW_DAYS are
dropped (their weight is small for the xi values we consider).

Shrinkage: attack/defence get a ridge penalty RIDGE * (param - prior)^2. The prior is 0
(an average PL team) except for teams that were not in the previous season, whose prior
is the mean fitted attack/defence of the previous season's bottom three. This mirrors
the Elo seeding of promoted teams and stops teams with 1-2 matches getting extreme values.

Leakage: predictions for matches on date d come from a fit on matches dated strictly
before d. Priors for season s come from a fit made before the last match date of
season s-1, using the final table of s-1 (known before s starts).

Output: data/dc_probs_xi{xi}.csv with dc_p_home, dc_p_draw, dc_p_away, dc_exp_home,
dc_exp_away per match (NaN where the fit window holds too few matches).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from src.config import DC_MAX_GOALS as MAX_GOALS
from src.config import DC_MIN_MATCHES as MIN_MATCHES  # fewer in window -> no prediction (early 2000/01)
from src.config import DC_RIDGE as RIDGE
from src.config import DC_WINDOW_DAYS as WINDOW_DAYS
from src.features import season_tables

ROOT = Path(__file__).resolve().parents[1]
MATCHES_PATH = ROOT / "data" / "matches.csv"

RHO_BOUNDS = (-0.25, 0.25)


@dataclass
class Fit:
    teams: list[str]
    c: float
    home: float
    rho: float
    att: np.ndarray
    dfn: np.ndarray

    def team_params(self, team: str) -> tuple[float, float] | None:
        try:
            i = self.teams.index(team)
        except ValueError:
            return None
        return self.att[i], self.dfn[i]


# --------------------------------------------------------------------------- likelihood

def _negll(theta, hi, ai, x, y, w, n, prior_att, prior_def, ridge):
    c, home, rho = theta[0], theta[1], theta[2]
    att, dfn = theta[3:3 + n], theta[3 + n:]
    log_lam = c + home + att[hi] - dfn[ai]
    log_mu = c + att[ai] - dfn[hi]
    lam, mu = np.exp(log_lam), np.exp(log_mu)

    ll = x * log_lam - lam + y * log_mu - mu
    g_loglam = x - lam
    g_logmu = y - mu
    g_rho = np.zeros_like(lam)

    tau = np.ones_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + mu[m10] * rho
    tau[m11] = 1 - rho
    tau = np.maximum(tau, 1e-10)
    ll = ll + np.log(tau)

    t = tau[m00]
    g_loglam[m00] += -lam[m00] * mu[m00] * rho / t
    g_logmu[m00] += -lam[m00] * mu[m00] * rho / t
    g_rho[m00] = -lam[m00] * mu[m00] / t
    t = tau[m01]
    g_loglam[m01] += lam[m01] * rho / t
    g_rho[m01] = lam[m01] / t
    t = tau[m10]
    g_logmu[m10] += mu[m10] * rho / t
    g_rho[m10] = mu[m10] / t
    g_rho[m11] = -1 / tau[m11]

    wl, wm = w * g_loglam, w * g_logmu
    grad = np.empty_like(theta)
    grad[0] = wl.sum() + wm.sum()
    grad[1] = wl.sum()
    grad[2] = (w * g_rho).sum()
    grad[3:3 + n] = np.bincount(hi, wl, n) + np.bincount(ai, wm, n)
    grad[3 + n:] = -np.bincount(ai, wl, n) - np.bincount(hi, wm, n)

    da, dd = att - prior_att, dfn - prior_def
    f = -(w * ll).sum() + ridge * (da @ da + dd @ dd)
    grad = -grad
    grad[3:3 + n] += 2 * ridge * da
    grad[3 + n:] += 2 * ridge * dd
    return f, grad


def fit_dc(hist: pd.DataFrame, as_of: pd.Timestamp, xi: float,
           prior: dict[str, tuple[float, float]], init: Fit | None = None) -> Fit | None:
    """Fit on `hist` (already restricted to dates < as_of and within the window)."""
    if len(hist) < MIN_MATCHES:
        return None
    teams = sorted(set(hist["home_team"]) | set(hist["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = hist["home_team"].map(idx).to_numpy()
    ai = hist["away_team"].map(idx).to_numpy()
    x = hist["home_goals"].to_numpy(float)
    y = hist["away_goals"].to_numpy(float)
    w = np.exp(-xi * (as_of - hist["date"]).dt.days.to_numpy(float))
    pa = np.array([prior.get(t, (0.0, 0.0))[0] for t in teams])
    pd_ = np.array([prior.get(t, (0.0, 0.0))[1] for t in teams])

    theta0 = np.concatenate([[np.log(1.3), 0.25, -0.05], pa, pd_])
    if init is not None:
        theta0[:3] = [init.c, init.home, init.rho]
        for t, i in idx.items():
            p = init.team_params(t)
            if p is not None:
                theta0[3 + i], theta0[3 + n + i] = p
    bounds = [(None, None), (None, None), RHO_BOUNDS] + [(None, None)] * (2 * n)
    res = minimize(_negll, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                   args=(hi, ai, x, y, w, n, pa, pd_, RIDGE),
                   options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7})
    th = res.x
    return Fit(teams, th[0], th[1], th[2], th[3:3 + n], th[3 + n:])


def predict(fit: Fit, home: str, away: str,
            prior: dict[str, tuple[float, float]]) -> tuple[float, float, float, float, float]:
    """P(H), P(D), P(A), expected home goals, expected away goals."""
    ah, dh = fit.team_params(home) or prior.get(home, (0.0, 0.0))
    aa, da = fit.team_params(away) or prior.get(away, (0.0, 0.0))
    lam = np.exp(fit.c + fit.home + ah - da)
    mu = np.exp(fit.c + aa - dh)
    g = np.arange(MAX_GOALS + 1)
    grid = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    grid[0, 0] *= 1 - lam * mu * fit.rho
    grid[0, 1] *= 1 + lam * fit.rho
    grid[1, 0] *= 1 + mu * fit.rho
    grid[1, 1] *= 1 - fit.rho
    grid /= grid.sum()
    return (float(np.tril(grid, -1).sum()), float(np.trace(grid)),
            float(np.triu(grid, 1).sum()), float(lam), float(mu))


# --------------------------------------------------------------------------- history / priors

def _prep(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches[["season", "date", "home_team", "away_team", "home_goals", "away_goals"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)


def _history(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    h = df[(df["date"] < as_of) & (df["date"] >= as_of - pd.Timedelta(days=WINDOW_DAYS))]
    return h.dropna(subset=["home_goals", "away_goals"])


class PriorBook:
    """Season priors: promoted teams -> mean params of the previous season's bottom three."""

    def __init__(self, df: pd.DataFrame, xi: float):
        self.df, self.xi = df, xi
        self.seasons = list(dict.fromkeys(df["season"]))
        self._cache: dict[str, dict] = {}

    def get(self, season: str) -> dict[str, tuple[float, float]]:
        if season in self._cache:
            return self._cache[season]
        i = self.seasons.index(season)
        prior: dict[str, tuple[float, float]] = {}
        if i > 0:
            prev = self.seasons[i - 1]
            prev_rows = self.df[self.df["season"] == prev]
            if prev_rows[["home_goals", "away_goals"]].notna().all().all():
                last_date = prev_rows["date"].max()
                fit = fit_dc(_history(self.df, last_date), last_date, self.xi, self.get(prev))
                if fit is not None:
                    table = season_tables(prev_rows)[prev]
                    params = [fit.team_params(t) for t in table[-3:]]
                    params = [p for p in params if p is not None]
                    if params:
                        seed = tuple(np.mean(params, axis=0))
                        prev_teams = set(table)
                        cur = self.df[self.df["season"] == season]
                        for t in set(cur["home_team"]) | set(cur["away_team"]):
                            if t not in prev_teams:
                                prior[t] = seed
        self._cache[season] = prior
        return prior


def dc_predictions(matches: pd.DataFrame, xi: float, seasons: list[str] | None = None) -> pd.DataFrame:
    """Pre-match DC predictions for every match (or only those in `seasons`)."""
    df = _prep(matches)
    priors = PriorBook(df, xi)
    target = df if seasons is None else df[df["season"].isin(seasons)]
    out = []
    prev_fit = None
    for date, day in target.groupby("date", sort=True):
        season = day["season"].iloc[0]
        prior = priors.get(season)
        fit = fit_dc(_history(df, date), date, xi, prior, init=prev_fit)
        prev_fit = fit or prev_fit
        for i, r in day.iterrows():
            vals = predict(fit, r.home_team, r.away_team, prior) if fit else (np.nan,) * 5
            out.append((i, *vals))
    res = pd.DataFrame(out, columns=["i", "dc_p_home", "dc_p_draw", "dc_p_away",
                                     "dc_exp_home", "dc_exp_away"]).set_index("i")
    return df[["season", "date", "home_team", "away_team"]].join(res, how="inner")


def dc_predict_one(matches: pd.DataFrame, date, home: str, away: str, xi: float) -> tuple:
    """Cold (no warm start) prediction for a single fixture; used by the leakage test."""
    df = _prep(matches)
    date = pd.Timestamp(date)
    season = df.loc[(df["date"] == date) & (df["home_team"] == home), "season"].iloc[0]
    prior = PriorBook(df, xi).get(season)
    fit = fit_dc(_history(df, date), date, xi, prior)
    return predict(fit, home, away, prior)


def cache_path(xi: float) -> Path:
    return ROOT / "data" / f"dc_probs_xi{xi:g}.csv"


def load_or_build(matches: pd.DataFrame, xi: float) -> pd.DataFrame:
    path = cache_path(xi)
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    res = dc_predictions(matches, xi)
    res.to_csv(path, index=False)
    return res


if __name__ == "__main__":
    import time
    m = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    t = time.time()
    r = dc_predictions(m, 0.002, seasons=["2022/23"])
    print(f"{len(r)} predictions in {time.time() - t:.1f}s")
    print(r.head())
