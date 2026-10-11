"""CalibrationEngine (spec §33, §53).

Parameters are estimated from observations - never hand-tuned and called calibrated.

Methods
  * nls / mle               - Bass (p, q, m) from an adoption series
  * bayesian                - random-walk Metropolis posterior for Bass (p, q, m)
  * abc                     - rejection ABC for simulator parameters using summary statistics
  * sbc                     - simulation-based calibration check of the Bayesian sampler
  * platform rates          - Beta-Binomial posteriors for share/engagement/adoption rates

Every result records the method, data version, diagnostics and whether the parameter is
CALIBRATED (fitted to observed data) or PRIOR (no data).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from dsdvc.core.literature import cites
from dsdvc.diffusion.bass import bass_cdf, fit_bass_mle, fit_bass_nls


@dataclass
class CalibrationResult:
    method: str
    parameters: dict
    intervals: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    data_version: str = "unversioned"
    status: str = "CALIBRATED"
    samples: np.ndarray | None = None

    def table(self) -> pd.DataFrame:
        rows = []
        for k, v in self.parameters.items():
            lo, hi = self.intervals.get(k, (np.nan, np.nan))
            rows.append({"parameter": k, "estimate": v, "ci_low": lo, "ci_high": hi, "method": self.method,
                         "status": self.status})
        return pd.DataFrame(rows)


def _bass_loglik(theta, y, t):
    p, q, m = theta
    if p <= 0 or q <= 0 or m < y.sum() or p > 1 or q > 3:
        return -np.inf
    mu = m * (bass_cdf(t, p, q) - bass_cdf(t - 1, p, q))
    return float(stats.poisson.logpmf(y, np.maximum(mu, 1e-9)).sum())


class CalibrationEngine:
    def __init__(self, data_version: str = "unversioned", seed: int = 0):
        self.data_version = data_version
        self.seed = seed

    # ---------------- Bass
    @cites("bass1969")
    def bass_nls(self, adoptions) -> CalibrationResult:
        f = fit_bass_nls(np.asarray(adoptions))
        iv = {k: (getattr(f, k) - 1.645 * s, getattr(f, k) + 1.645 * s) for k, s in f.se.items()}
        return CalibrationResult("nls", {"p": f.p, "q": f.q, "m": f.m}, iv,
                                 {"sse": f.sse, "aic": f.aic, "n": f.n}, self.data_version)

    @cites("bass1969")
    def bass_mle(self, adoptions) -> CalibrationResult:
        f = fit_bass_mle(np.asarray(adoptions))
        iv = {k: (getattr(f, k) - 1.645 * s, getattr(f, k) + 1.645 * s) for k, s in f.se.items()}
        return CalibrationResult("mle", {"p": f.p, "q": f.q, "m": f.m}, iv,
                                 {"loglik": f.loglik, "aic": f.aic, "n": f.n}, self.data_version)

    def bass_bayesian(self, adoptions, n_iter: int = 6000, burn: int = 2000, prior_m_scale: float = 10.0) -> CalibrationResult:
        """Random-walk Metropolis on log-parameters with weakly-informative priors:
        p ~ LogNormal(log 0.01, 1.5), q ~ LogNormal(log 0.3, 1.0), m ~ Uniform(sum y, scale * sum y)."""
        y = np.asarray(adoptions, dtype=float)
        t = np.arange(1, len(y) + 1, dtype=float)
        rng = np.random.default_rng(self.seed)
        start = fit_bass_nls(y)
        x = np.log([start.p, start.q, max(start.m - y.sum(), 1.0)])

        def logpost(z):
            p, q, extra = np.exp(z)
            m = y.sum() + extra
            if m > prior_m_scale * max(y.sum(), 1):
                return -np.inf
            lp = stats.norm.logpdf(z[0], np.log(0.01), 1.5) + stats.norm.logpdf(z[1], np.log(0.3), 1.0) + z[2]
            return _bass_loglik((p, q, m), y, t) + lp

        cur = logpost(x)
        step = np.array([0.05, 0.05, 0.1])
        chain, acc = [], 0
        for i in range(n_iter):
            prop = x + rng.normal(0, step)
            lp = logpost(prop)
            if np.log(rng.random()) < lp - cur:
                x, cur = prop, lp
                acc += 1
            if i < burn and i % 200 == 199:  # adapt during burn-in only
                rate = acc / (i + 1)
                step *= 1.3 if rate > 0.35 else (0.7 if rate < 0.15 else 1.0)
            chain.append(x.copy())
        ch = np.exp(np.array(chain[burn:]))
        ch[:, 2] += y.sum()
        names = ["p", "q", "m"]
        est = {k: float(np.median(ch[:, i])) for i, k in enumerate(names)}
        iv = {k: tuple(np.quantile(ch[:, i], [0.05, 0.95]).tolist()) for i, k in enumerate(names)}
        ess = {k: float(_ess(ch[:, i])) for i, k in enumerate(names)}
        warn = [f"low effective sample size for {k} ({v:.0f}); run longer chains" for k, v in ess.items() if v < 100]
        return CalibrationResult("bayesian_mcmc", est, iv, {"acceptance": acc / n_iter, "ess": ess, "warnings": warn},
                                 self.data_version, samples=ch)

    # ---------------- simulator ABC
    @cites("beaumont2002", "rand_rust2011")
    def abc(self, simulate: Callable[[dict, int], np.ndarray], observed_summary: np.ndarray,
            priors: dict[str, tuple[float, float]], n_sims: int = 200, accept_frac: float = 0.1,
            scale: np.ndarray | None = None) -> CalibrationResult:
        """Rejection ABC. ``simulate(params, seed) -> summary vector``; priors are uniform ranges."""
        rng = np.random.default_rng(self.seed)
        names = list(priors)
        thetas, summaries = [], []
        for k in range(n_sims):
            th = {n: float(rng.uniform(*priors[n])) for n in names}
            summaries.append(np.asarray(simulate(th, self.seed + k), dtype=float))
            thetas.append([th[n] for n in names])
        S = np.array(summaries)
        T = np.array(thetas)
        sc = scale if scale is not None else (S.std(axis=0) + 1e-9)
        d = np.sqrt((((S - observed_summary) / sc) ** 2).sum(axis=1))
        keep = np.argsort(d)[: max(5, int(accept_frac * n_sims))]
        post = T[keep]
        est = {n: float(np.median(post[:, i])) for i, n in enumerate(names)}
        iv = {n: tuple(np.quantile(post[:, i], [0.05, 0.95]).tolist()) for i, n in enumerate(names)}
        shrink = {n: float(1 - post[:, i].std() / T[:, i].std()) for i, n in enumerate(names)}
        weak = [n for n, v in shrink.items() if v < 0.2]
        warnings = [f"'{n}' is weakly identified by these summary statistics (posterior shrinkage "
                    f"{shrink[n]:.2f} < 0.2): treat its estimate as prior-dominated." for n in weak]
        return CalibrationResult("abc_rejection", est, iv, {"n_sims": n_sims, "accepted": len(keep),
                                                            "posterior_shrinkage": shrink,
                                                            "weakly_identified": weak, "warnings": warnings,
                                                            "tolerance": float(d[keep].max())},
                                 self.data_version, status="CALIBRATED" if not weak else "PARTIALLY_CALIBRATED",
                                 samples=post)

    @cites("talts2018")
    def sbc_bass(self, n_reps: int = 30, T: int = 30, n_iter: int = 2500, burn: int = 1000) -> dict:
        """Simulation-based calibration: draw (p,q) from the prior, simulate, fit, record rank of
        truth in the posterior. Uniform ranks (KS test) indicate a well-calibrated sampler."""
        rng = np.random.default_rng(self.seed)
        ranks = {"p": [], "q": []}
        for r in range(n_reps):
            p = float(np.exp(rng.normal(np.log(0.01), 0.5)))
            q = float(np.exp(rng.normal(np.log(0.3), 0.4)))
            m = 5000.0
            t = np.arange(1, T + 1)
            y = rng.poisson(m * (bass_cdf(t, p, q) - bass_cdf(t - 1, p, q)))
            eng = CalibrationEngine(seed=self.seed + r)
            res = eng.bass_bayesian(y, n_iter=n_iter, burn=burn)
            ranks["p"].append(float((res.samples[:, 0] < p).mean()))
            ranks["q"].append(float((res.samples[:, 1] < q).mean()))
        return {k: {"ks_p_value": float(stats.kstest(v, "uniform").pvalue), "ranks": v} for k, v in ranks.items()}

    # ---------------- platform rates
    def platform_rates(self, obs: pd.DataFrame, prior: tuple[float, float] = (1.0, 1.0)) -> pd.DataFrame:
        """Beta-Binomial posteriors for per-platform rates from aggregate counts
        (columns: platform, exposures, engagements, shares, adoptions, churns)."""
        rows = []
        for plat, g in obs.groupby("platform"):
            n = g.exposures.sum()
            for metric, num, den in (("engagement_rate", g.engagements.sum(), n), ("share_rate", g.shares.sum(), n),
                                     ("adoption_rate", g.adoptions.sum(), n),
                                     ("churn_rate", g.churns.sum(), g.adoptions.sum())):
                a, b = prior[0] + num, prior[1] + max(den - num, 0)
                rows.append({"platform": plat, "metric": metric, "estimate": a / (a + b),
                             "ci_low": stats.beta.ppf(0.05, a, b), "ci_high": stats.beta.ppf(0.95, a, b),
                             "n": int(den), "status": "CALIBRATED" if den > 0 else "PRIOR"})
        return pd.DataFrame(rows)


def _ess(x: np.ndarray) -> float:
    x = x - x.mean()
    n = len(x)
    if n < 10 or x.var() == 0:
        return float(n)
    ac = np.correlate(x, x, "full")[n - 1:] / (x.var() * np.arange(n, 0, -1))
    s = 0.0
    for k in range(1, min(n, 1000)):
        if ac[k] < 0.05:
            break
        s += ac[k]
    return float(n / (1 + 2 * s))


def abm_summary_stats(result) -> np.ndarray:
    """Summary statistics for ABC against the simulator: adoption level/timing, churn, share volume."""
    ts = result.timeseries
    n = result.config.n_agents
    cum = ts.cum_adopted.to_numpy() / n
    return np.array([cum[-1], cum[len(cum) // 3], ts.new_adoptions.idxmax() / len(ts),
                     ts.churned.iloc[-1] / n, (ts.shares_ig + ts.shares_fb + ts.shares_wa).sum() / n])


def make_abm_simulator(base_cfg, param_map: dict[str, Callable] | None = None):
    """Wrap the ABM for ABC: params dict -> config overrides -> summary statistics."""
    from dsdvc.diffusion.abm import run_simulation

    def sim(params: dict, seed: int) -> np.ndarray:
        cfg = copy.deepcopy(base_cfg)
        cfg.seed = seed
        for k, v in params.items():
            if param_map and k in param_map:
                param_map[k](cfg, v)
            else:
                setattr(cfg, k, v)
        return abm_summary_stats(run_simulation(cfg))

    return sim
