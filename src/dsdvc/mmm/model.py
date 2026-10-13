"""Marketing mix model (MMM): the measurement backbone for budget decisions.

    y_t = a + b·trend_t + seasonality_t + Σ_j γ_j·control_jt + Σ_c β_c · Hill(Adstock_c,t / scale_c) + ε_t

* **Adstock** — normalised geometric carryover: A_t = (1-θ)·x_t + θ·A_{t-1}. Because the weights
  sum to one, steady-state adstock equals weekly spend, so response curves read directly in
  "spend per week".
* **Saturation** — Hill curve h(a) = a^S / (a^S + K^S) on spend scaled by the channel's mean
  non-zero spend (diminishing returns; S > 1 gives an S-shape).
* **Estimation** — maximum a posteriori with weakly-informative priors (documented in ``PRIORS``),
  multiple restarts, then a **Laplace approximation** (Gaussian at the mode, from a numerical
  Hessian) for uncertainty. Draws propagate to contributions, ROI, response curves and plans.
* **Calibration with experiments** — lift tests (geo-lift / conversion-lift) enter as additional
  likelihood terms on the channel's modelled incremental contribution over the test window,
  following the practice of open-source MMMs (Google Meridian, Meta Robyn).

Status of outputs: contributions/ROI are ESTIMATED; plans are RECOMMENDED; everything carries
interval estimates and data/model versions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.signal import lfilter

from dsdvc import MODEL_VERSION
from dsdvc.core.reproducibility import stable_hash

PRIORS = {
    # unconstrained parameter -> (mean, sd) of a normal prior
    "theta_logit": (-0.4, 1.0),   # carryover θ = 0.95·sigmoid(z); prior median ≈ 0.38
    "log_K": (0.0, 0.6),          # half-saturation at ≈ the channel's average spend
    "log_S": (np.log(1.5), 0.35),  # shape
    "log_beta": (np.log(0.1), 1.5),  # channel effect at saturation, as a fraction of mean outcome
    "intercept": (0.5, 1.0),
    "trend": (0.0, 0.5),
    "fourier": (0.0, 0.3),
    "gamma": (0.0, 0.5),
    "log_sigma": (np.log(0.1), 1.0),
}
THETA_MAX = 0.95


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def geometric_adstock(x: np.ndarray, theta: float) -> np.ndarray:
    """A_t = (1-θ)·x_t + θ·A_{t-1}, A_{-1} = 0 (IIR filter; identical to the explicit recursion)."""
    return lfilter([1 - theta], [1, -theta], np.asarray(x, dtype=float))


def hill(a: np.ndarray, K: float, S: float) -> np.ndarray:
    a = np.maximum(a, 0.0)
    aS = np.power(a, S)
    return aS / (aS + K ** S + 1e-12)


@dataclass
class LiftTest:
    """An experiment result: incremental outcome caused by ``channel`` in [start, end] (inclusive dates)."""
    channel: str
    start: pd.Timestamp
    end: pd.Timestamp
    lift: float
    se: float


@dataclass
class MMMSpec:
    date_col: str = "date"
    target_col: str = "revenue"
    channels: list[str] = field(default_factory=list)       # spend columns
    controls: list[str] = field(default_factory=list)
    n_fourier: int | None = None                             # None -> 2 if >= 52 periods else 0
    period: float = 52.18                                     # weekly data, yearly seasonality


@dataclass
class MMMFit:
    spec: MMMSpec
    z_map: np.ndarray
    cov: np.ndarray
    draws: np.ndarray
    names: list[str]
    y_mean: float
    x_scale: dict[str, float]
    ctrl_mean: dict[str, float]
    ctrl_sd: dict[str, float]
    t_scale: float
    train_index: pd.DatetimeIndex
    data: pd.DataFrame
    diagnostics: dict
    lift_tests: list[LiftTest]
    data_version: str
    model_version: str = MODEL_VERSION


class MarketingMixModel:
    def __init__(self, spec: MMMSpec, n_restarts: int = 6, n_draws: int = 400, seed: int = 0):
        self.spec = spec
        self.n_restarts = n_restarts
        self.n_draws = n_draws
        self.seed = seed

    # ------------------------------------------------------------------ parameter layout
    def _layout(self, n_controls: int, nf: int) -> list[str]:
        names = []
        for c in self.spec.channels:
            names += [f"{c}:theta_logit", f"{c}:log_K", f"{c}:log_S", f"{c}:log_beta"]
        names += ["intercept", "trend"] + [f"fourier_{k}" for k in range(2 * nf)]
        names += [f"gamma:{c}" for c in self.spec.controls[:n_controls]] + ["log_sigma"]
        return names

    @staticmethod
    def _prior_key(name: str) -> str:
        if ":" in name and not name.startswith("gamma"):
            return name.split(":")[1]
        if name.startswith("gamma"):
            return "gamma"
        if name.startswith("fourier"):
            return "fourier"
        return name

    def _unpack(self, z: np.ndarray, names: list[str]) -> dict:
        p = dict(zip(names, z))
        ch = {}
        for c in self.spec.channels:
            ch[c] = {"theta": THETA_MAX * sigmoid(p[f"{c}:theta_logit"]), "K": np.exp(p[f"{c}:log_K"]),
                     "S": np.exp(p[f"{c}:log_S"]), "beta": np.exp(p[f"{c}:log_beta"])}
        return {"channels": ch, "intercept": p["intercept"], "trend": p["trend"],
                "fourier": np.array([p[n] for n in names if n.startswith("fourier")]),
                "gamma": np.array([p[n] for n in names if n.startswith("gamma")]),
                "sigma": np.exp(p["log_sigma"])}

    # ------------------------------------------------------------------ design matrices
    def _prepare(self, df: pd.DataFrame, fit_stats: dict | None = None):
        s = self.spec
        d = df.sort_values(s.date_col).reset_index(drop=True)
        T = len(d)
        if fit_stats is None:
            x_scale = {c: float(d[c][d[c] > 0].mean()) if (d[c] > 0).any() else 1.0 for c in s.channels}
            ctrl_mean = {c: float(d[c].mean()) for c in s.controls}
            ctrl_sd = {c: float(d[c].std()) or 1.0 for c in s.controls}
            t_scale = max(T - 1, 1)
            nf = s.n_fourier if s.n_fourier is not None else (2 if T >= 52 else 0)
        else:
            x_scale, ctrl_mean, ctrl_sd, t_scale, nf = (fit_stats[k] for k in ("x_scale", "ctrl_mean", "ctrl_sd", "t_scale", "nf"))
        X = {c: d[c].to_numpy(float) / x_scale[c] for c in s.channels}
        Z = np.column_stack([(d[c].to_numpy(float) - ctrl_mean[c]) / ctrl_sd[c] for c in s.controls]) if s.controls else np.zeros((T, 0))
        t = np.arange(T) / t_scale
        week = np.arange(T)
        F = np.column_stack([f(2 * np.pi * (k + 1) * week / s.period) for k in range(nf) for f in (np.sin, np.cos)]) if nf else np.zeros((T, 0))
        return d, X, Z, t, F, {"x_scale": x_scale, "ctrl_mean": ctrl_mean, "ctrl_sd": ctrl_sd, "t_scale": t_scale, "nf": nf}

    def _components(self, P: dict, X: dict, Z: np.ndarray, t: np.ndarray, F: np.ndarray, zero: dict | None = None) -> dict:
        """Per-component contributions in units of y/y_mean. ``zero`` = {channel: bool mask} zeroes spend."""
        out = {"base": np.full(len(t), P["intercept"]), "trend": P["trend"] * t,
               "seasonality": F @ P["fourier"] if F.shape[1] else np.zeros(len(t)),
               "controls": Z @ P["gamma"] if Z.shape[1] else np.zeros(len(t))}
        for c, q in P["channels"].items():
            x = X[c]
            if zero and c in zero:
                x = np.where(zero[c], 0.0, x)
            out[c] = q["beta"] * hill(geometric_adstock(x, q["theta"]), q["K"], q["S"])
        return out

    # ------------------------------------------------------------------ fitting
    def fit(self, df: pd.DataFrame, lift_tests: list[LiftTest] | None = None, data_version: str | None = None) -> MMMFit:
        s = self.spec
        lift_tests = lift_tests or []
        d, X, Z, t, F, stats = self._prepare(df)
        y = d[s.target_col].to_numpy(float)
        y_mean = float(y.mean()) or 1.0
        ys = y / y_mean
        names = self._layout(Z.shape[1], stats["nf"])
        prior = np.array([PRIORS[self._prior_key(n)] for n in names])
        dates = pd.to_datetime(d[s.date_col])
        lt_masks = [(lt, (dates >= lt.start).to_numpy() & (dates <= lt.end).to_numpy(), (dates >= lt.start).to_numpy())
                    for lt in lift_tests if lt.channel in s.channels]

        def neg_log_post(z):
            P = self._unpack(z, names)
            comp = self._components(P, X, Z, t, F)
            mu = sum(comp.values())
            r = (ys - mu) / P["sigma"]
            nll = 0.5 * np.sum(r * r) + len(ys) * np.log(P["sigma"])
            nlp = 0.5 * np.sum(((z - prior[:, 0]) / prior[:, 1]) ** 2)
            for lt, window, after in lt_masks:
                cf = self._components(P, X, Z, t, F, zero={lt.channel: window})[lt.channel]
                model_lift = float(np.sum((comp[lt.channel] - cf)[after])) * y_mean
                nll += 0.5 * ((model_lift - lt.lift) / max(lt.se, 1e-9)) ** 2
            return nll + nlp

        rng = np.random.default_rng(self.seed)
        best = None
        for k in range(self.n_restarts):
            z0 = prior[:, 0] + (0 if k == 0 else 1) * rng.normal(0, 1, len(names)) * prior[:, 1] * 0.7
            z0[names.index("intercept")] = 0.5 if k == 0 else rng.uniform(0.2, 0.9)
            r = optimize.minimize(neg_log_post, z0, method="L-BFGS-B", options={"maxiter": 3000, "maxfun": 15000})
            if best is None or r.fun < best.fun:
                best = r
        # polish the best restart with a larger evaluation budget so convergence is genuine
        best = optimize.minimize(neg_log_post, best.x, method="L-BFGS-B",
                                 options={"maxiter": 20000, "maxfun": 200000, "ftol": 1e-10, "gtol": 1e-6})
        z_map = best.x
        H = _hessian(neg_log_post, z_map)
        w, V = np.linalg.eigh((H + H.T) / 2)
        w = np.maximum(w, 1e-6)
        cov = (V / w) @ V.T
        draws = rng.multivariate_normal(z_map, cov, size=self.n_draws, check_valid="ignore")

        P = self._unpack(z_map, names)
        comp = self._components(P, X, Z, t, F)
        mu = sum(comp.values()) * y_mean
        resid = y - mu
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        diagnostics = {
            "converged": bool(best.success), "optimizer_message": str(best.message), "n_obs": int(len(y)), "n_params": len(names),
            "r2": 1 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan"),
            "mape": float(np.mean(np.abs(resid / np.where(y == 0, np.nan, y))[~np.isnan(y)])),
            "nrmse": float(np.sqrt(np.mean(resid ** 2)) / (y.max() - y.min())) if y.max() > y.min() else float("nan"),
            "durbin_watson": float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2)) if np.sum(resid ** 2) > 0 else float("nan"),
            "posterior": "Laplace approximation at the MAP",
            "warnings": _identifiability_warnings(d, s, names, prior, cov),
        }
        dv = data_version or stable_hash(d[[s.date_col, s.target_col] + s.channels + s.controls].astype(str).to_dict("list"))
        return MMMFit(spec=s, z_map=z_map, cov=cov, draws=draws, names=names, y_mean=y_mean, x_scale=stats["x_scale"],
                      ctrl_mean=stats["ctrl_mean"], ctrl_sd=stats["ctrl_sd"], t_scale=stats["t_scale"],
                      train_index=pd.DatetimeIndex(dates), data=d, diagnostics=diagnostics | {"nf": stats["nf"]},
                      lift_tests=lift_tests, data_version=dv)


def _hessian(f, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    n = len(x)
    H = np.zeros((n, n))
    fx = f(x)
    h = eps * np.maximum(1.0, np.abs(x))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        fpp, fmm = f(x + ei), f(x - ei)
        H[i, i] = (fpp - 2 * fx + fmm) / h[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def _identifiability_warnings(d: pd.DataFrame, s: MMMSpec, names, prior, cov) -> list[str]:
    w = []
    if len(d) < 52:
        w.append(f"Only {len(d)} periods of history: seasonality is not modelled and estimates are less stable (≥ 104 weeks recommended).")
    sp = d[s.channels]
    for c in s.channels:
        cv = sp[c].std() / sp[c].mean() if sp[c].mean() > 0 else 0
        if cv < 0.15:
            w.append(f"'{c}' spend barely varies (CV {cv:.2f}); its effect is mostly determined by the prior. Vary spend or run a lift test.")
    if len(s.channels) > 1:
        C = sp.corr().to_numpy()
        for i in range(len(s.channels)):
            for j in range(i + 1, len(s.channels)):
                if abs(C[i, j]) > 0.85:
                    w.append(f"'{s.channels[i]}' and '{s.channels[j]}' spend move together (r={C[i, j]:.2f}); the model cannot "
                             "separate their effects well. A lift test on one of them would resolve this.")
    for c in s.channels:
        i = names.index(f"{c}:log_beta")
        if np.sqrt(cov[i, i]) > 0.8 * prior[i, 1]:
            w.append(f"'{c}' effect is weakly identified by the data (posterior ≈ prior).")
    return w
