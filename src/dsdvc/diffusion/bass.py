"""Bass diffusion: the macro benchmark (spec §3).

Classical Bass (1969): hazard h(t) = p + q F(t), closed form

    F(t) = (1 - exp(-(p+q)t)) / (1 + (q/p) exp(-(p+q)t)).

The classical model assumes a homogeneous, fully-mixed population with a fixed market
potential m and no marketing-mix, network, churn or competition effects. DSDVC *extends*
it (generalised Bass, ABM); it does not claim the original assumptions hold for social
platforms. The Bass fit is retained as the baseline every richer model must beat (§60).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize
from scipy.special import gammaln

from dsdvc.core.literature import cites


def bass_cdf(t: np.ndarray, p: float, q: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    e = np.exp(-(p + q) * t)
    return (1 - e) / (1 + (q / max(p, 1e-12)) * e)


def bass_pdf(t: np.ndarray, p: float, q: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    e = np.exp(-(p + q) * t)
    return ((p + q) ** 2 / p) * e / (1 + (q / p) * e) ** 2


def bass_hazard(F: np.ndarray, p: float, q: float) -> np.ndarray:
    return p + q * np.asarray(F)


def bass_peak_time(p: float, q: float) -> float:
    if q <= p:
        return 0.0
    return float(np.log(q / p) / (p + q))


def generalized_bass_cdf(t_grid: np.ndarray, p: float, q: float, x: np.ndarray) -> np.ndarray:
    """Generalized Bass (Bass, Krishnan & Jain 1994): hazard (p + qF) x(t).

    ``x`` is the marketing-mix multiplier per period (1.0 = baseline). Solved by forward
    integration of dF/dt = (p + qF)(1 - F) x(t).
    """
    F = np.zeros(len(t_grid))
    for k in range(1, len(t_grid)):
        dt = t_grid[k] - t_grid[k - 1]
        Fk = F[k - 1]
        F[k] = min(1.0, Fk + dt * (p + q * Fk) * (1 - Fk) * x[k - 1])
    return F


@dataclass
class BassFit:
    p: float
    q: float
    m: float
    method: str
    se: dict = field(default_factory=dict)
    sse: float = float("nan")
    loglik: float = float("nan")
    aic: float = float("nan")
    n: int = 0

    def cumulative(self, t: np.ndarray) -> np.ndarray:
        return self.m * bass_cdf(t, self.p, self.q)

    def per_period(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return self.m * (bass_cdf(t, self.p, self.q) - bass_cdf(t - 1, self.p, self.q))

    @property
    def peak_time(self) -> float:
        return bass_peak_time(self.p, self.q)


@cites("bass1969", "mahajan_muller_bass1990")
def fit_bass_nls(adoptions: np.ndarray, m_upper: float | None = None) -> BassFit:
    """Nonlinear least squares on per-period adoptions (Srinivasan & Mason style)."""
    y = np.asarray(adoptions, dtype=float)
    t = np.arange(1, len(y) + 1, dtype=float)
    cum = y.sum()
    m_upper = m_upper or cum * 20

    def resid(theta):
        p, q, m = theta
        pred = m * (bass_cdf(t, p, q) - bass_cdf(t - 1, p, q))
        return pred - y

    best = None
    for p0 in (0.005, 0.02, 0.05):
        for q0 in (0.1, 0.4, 0.8):
            try:
                r = optimize.least_squares(resid, x0=[p0, q0, cum * 1.5],
                                           bounds=([1e-6, 1e-6, max(cum, 1.0)], [1.0, 3.0, m_upper]))
            except ValueError:
                continue
            if best is None or r.cost < best.cost:
                best = r
    p, q, m = best.x
    J = best.jac
    sse = float(2 * best.cost)
    dof = max(len(y) - 3, 1)
    try:
        cov = np.linalg.pinv(J.T @ J) * sse / dof
        se = dict(zip(["p", "q", "m"], np.sqrt(np.clip(np.diag(cov), 0, None)).tolist()))
    except np.linalg.LinAlgError:
        se = {}
    sigma2 = sse / len(y)
    ll = -0.5 * len(y) * (np.log(2 * np.pi * max(sigma2, 1e-12)) + 1)
    return BassFit(p=float(p), q=float(q), m=float(m), method="nls", se=se, sse=sse,
                   loglik=float(ll), aic=float(2 * 4 - 2 * ll), n=len(y))


@cites("bass1969")
def fit_bass_mle(adoptions: np.ndarray, m_upper: float | None = None) -> BassFit:
    """Maximum likelihood (Schmittlein & Mahajan 1982 style) with multinomial likelihood
    over periods plus a 'not yet adopted' cell, parameterised by (p, q, m)."""
    y = np.asarray(adoptions, dtype=float)
    T = len(y)
    t = np.arange(0, T + 1, dtype=float)
    cum = y.sum()
    m_upper = m_upper or cum * 20

    def nll(theta):
        p, q, m = np.exp(theta[0]), np.exp(theta[1]), cum + np.exp(theta[2])
        F = bass_cdf(t, p, q)
        probs = np.clip(np.diff(F), 1e-300, 1)
        rest = max(1 - F[-1], 1e-300)
        ll = (y * np.log(probs)).sum() + (m - cum) * np.log(rest)
        # multinomial normalising constant w.r.t. m
        ll += gammaln(m + 1) - gammaln(m - cum + 1) - gammaln(y + 1).sum()
        return -ll

    best = None
    for p0 in (0.005, 0.03):
        for q0 in (0.2, 0.6):
            for mm in (0.5, 2.0):
                r = optimize.minimize(nll, x0=[np.log(p0), np.log(q0), np.log(cum * mm + 1)], method="Nelder-Mead",
                                      options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
                if best is None or r.fun < best.fun:
                    best = r
    p, q, m = float(np.exp(best.x[0])), float(np.exp(best.x[1])), float(cum + np.exp(best.x[2]))
    m = min(m, m_upper)
    # numerical Hessian for SE on the log scale
    se = {}
    try:
        from scipy.optimize import approx_fprime
        h = np.zeros((3, 3))
        eps = 1e-4
        for i in range(3):
            def gi(x, i=i):
                return approx_fprime(x, nll, eps)[i]
            h[i] = approx_fprime(best.x, gi, eps)
        cov = np.linalg.pinv((h + h.T) / 2)
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        se = {"p": float(p * sd[0]), "q": float(q * sd[1]), "m": float((m - cum) * sd[2])}
    except Exception:  # noqa: BLE001 - SEs are best effort
        pass
    ll = -best.fun
    return BassFit(p=p, q=q, m=m, method="mle", se=se, loglik=float(ll), aic=float(2 * 3 - 2 * ll), n=T,
                   sse=float(((BassFit(p, q, m, "mle").per_period(np.arange(1, T + 1)) - y) ** 2).sum()))


def logistic_cumulative_fit(adoptions: np.ndarray) -> dict:
    """Simple logistic growth baseline (benchmark §60): N(t) = K / (1 + exp(-r (t - t0)))."""
    y = np.cumsum(np.asarray(adoptions, dtype=float))
    t = np.arange(1, len(y) + 1, dtype=float)

    def f(t, K, r, t0):
        return K / (1 + np.exp(-r * (t - t0)))

    try:
        (K, r, t0), _ = optimize.curve_fit(f, t, y, p0=[y[-1] * 1.5, 0.1, len(y) / 2],
                                           bounds=([y[-1], 1e-4, -len(y)], [y[-1] * 50 + 1, 5, 5 * len(y)]),
                                           maxfev=20000)
    except RuntimeError:
        K, r, t0 = y[-1], 0.1, len(y) / 2
    return {"K": float(K), "r": float(r), "t0": float(t0),
            "predict_cum": lambda tt: f(np.asarray(tt, dtype=float), K, r, t0)}


def simulate_bass_adoptions(p: float, q: float, m: float, T: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Synthetic adoption series for testing calibration (labelled SYNTHETIC by callers)."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, T + 1, dtype=float)
    mean = m * (bass_cdf(t, p, q) - bass_cdf(t - 1, p, q))
    if noise <= 0:
        return rng.poisson(mean).astype(float)
    return np.maximum(0, rng.normal(mean, noise * np.sqrt(mean + 1)))
