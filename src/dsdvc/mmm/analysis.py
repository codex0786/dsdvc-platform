"""MMM outputs: decomposition, ROI with intervals, response curves, budget plans, back-tests.

All posterior summaries use the Laplace draws stored in ``MMMFit.draws``; intervals are 90%.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from dsdvc.mmm.model import MarketingMixModel, MMMFit, hill


def _model(fit: MMMFit) -> MarketingMixModel:
    return MarketingMixModel(fit.spec)


def _stats(fit: MMMFit) -> dict:
    return {"x_scale": fit.x_scale, "ctrl_mean": fit.ctrl_mean, "ctrl_sd": fit.ctrl_sd, "t_scale": fit.t_scale,
            "nf": fit.diagnostics["nf"]}


def components(fit: MMMFit, df: pd.DataFrame | None = None, z: np.ndarray | None = None) -> pd.DataFrame:
    """Contribution of every component (in outcome units) for ``df`` (default: training data)."""
    m = _model(fit)
    d, X, Z, t, F, _ = m._prepare(fit.data if df is None else df, _stats(fit))
    P = m._unpack(fit.z_map if z is None else z, fit.names)
    comp = m._components(P, X, Z, t, F)
    out = pd.DataFrame({k: v * fit.y_mean for k, v in comp.items()})
    out.insert(0, "date", pd.to_datetime(d[fit.spec.date_col]).values)
    return out


def decomposition(fit: MMMFit) -> pd.DataFrame:
    c = components(fit)
    c["fitted"] = c.drop(columns="date").sum(axis=1)
    c["actual"] = fit.data[fit.spec.target_col].to_numpy(float)
    return c


def _draw_components(fit: MMMFit, n: int | None = None, df: pd.DataFrame | None = None) -> list[pd.DataFrame]:
    D = fit.draws if n is None else fit.draws[:n]
    return [components(fit, df, z) for z in D]


def channel_summary(fit: MMMFit, n_draws: int = 200) -> pd.DataFrame:
    s = fit.spec
    spend = fit.data[s.channels].sum()
    draws = _draw_components(fit, n_draws)
    rows = []
    m = _model(fit)
    d, X, Z, t, F, _ = m._prepare(fit.data, _stats(fit))
    for c in s.channels:
        contrib = np.array([dc[c].sum() for dc in draws])
        # marginal ROI: +1% spend on this channel in every period
        mroi = []
        for z in fit.draws[:n_draws]:
            P = m._unpack(z, fit.names)
            base = m._components(P, X, Z, t, F)[c].sum()
            X2 = dict(X)
            X2[c] = X[c] * 1.01
            bumped = m._components(P, X2, Z, t, F)[c].sum()
            mroi.append((bumped - base) * fit.y_mean / max(0.01 * spend[c], 1e-9))
        mroi = np.array(mroi)
        q = lambda a: np.quantile(a, [0.05, 0.5, 0.95])  # noqa: E731
        cq, rq, mq = q(contrib), q(contrib / max(spend[c], 1e-9)), q(mroi)
        rows.append({"channel": c, "spend": float(spend[c]), "contribution": cq[1], "contribution_low": cq[0],
                     "contribution_high": cq[2], "roi": rq[1], "roi_low": rq[0], "roi_high": rq[2],
                     "mroi": mq[1], "mroi_low": mq[0], "mroi_high": mq[2], "p_roi_above_1": float(np.mean(contrib / max(spend[c], 1e-9) > 1))})
    out = pd.DataFrame(rows)
    out["share_of_spend"] = out.spend / out.spend.sum()
    out["share_of_effect"] = out.contribution / out.contribution.clip(lower=0).sum()
    return out


def channel_params(fit: MMMFit) -> pd.DataFrame:
    m = _model(fit)
    rows = []
    P0 = m._unpack(fit.z_map, fit.names)
    Ps = [m._unpack(z, fit.names) for z in fit.draws]
    for c in fit.spec.channels:
        th = np.array([p["channels"][c]["theta"] for p in Ps])
        rows.append({"channel": c, "carryover_theta": P0["channels"][c]["theta"],
                     "theta_low": np.quantile(th, 0.05), "theta_high": np.quantile(th, 0.95),
                     "half_life_weeks": float(np.log(0.5) / np.log(max(P0["channels"][c]["theta"], 1e-6))),
                     "half_saturation_spend": P0["channels"][c]["K"] * fit.x_scale[c], "shape_S": P0["channels"][c]["S"]})
    return pd.DataFrame(rows)


def response_curves(fit: MMMFit, max_multiple: float = 3.0, points: int = 40) -> pd.DataFrame:
    """Steady-state weekly incremental outcome vs weekly spend, with 90% band."""
    m = _model(fit)
    Ps = [m._unpack(z, fit.names) for z in fit.draws[:200]]
    rows = []
    for c in fit.spec.channels:
        cur = float(fit.data[c].mean())
        grid = np.linspace(0, max(cur, fit.x_scale[c]) * max_multiple, points)
        vals = np.array([[p["channels"][c]["beta"] * hill(grid / fit.x_scale[c], p["channels"][c]["K"], p["channels"][c]["S"])
                          for p in Ps]]).squeeze(0) * fit.y_mean
        for i, g in enumerate(grid):
            rows.append({"channel": c, "weekly_spend": g, "weekly_effect": float(np.median(vals[:, i])),
                         "low": float(np.quantile(vals[:, i], 0.05)), "high": float(np.quantile(vals[:, i], 0.95)),
                         "current_weekly_spend": cur})
    return pd.DataFrame(rows)


@dataclass
class Plan:
    table: pd.DataFrame          # channel, current, recommended, change, expected effects
    weekly_budget: float
    objective: str
    uplift: float                # expected weekly incremental outcome vs current mix
    uplift_low: float
    uplift_high: float
    p_better: float
    notes: list[str]


def optimize_budget(fit: MMMFit, weekly_budget: float | None = None, bounds: dict[str, tuple[float, float]] | None = None,
                    max_change: float = 0.5, objective: str = "outcome", margin: float = 1.0, n_draws: int = 200) -> Plan:
    """Reallocate a weekly budget across channels using steady-state response curves.

    objective="outcome": maximise incremental outcome with the budget fully spent.
    objective="profit":  maximise margin·outcome − spend with spend ≤ budget (may under-spend).
    Default bounds keep each channel within ±``max_change`` of its current weekly spend - large
    moves extrapolate beyond the data the model has seen.
    """
    m = _model(fit)
    chans = fit.spec.channels
    cur = np.array([float(fit.data[c].mean()) for c in chans])
    B = float(weekly_budget if weekly_budget is not None else cur.sum())
    if bounds is None:
        scale = B / cur.sum() if cur.sum() > 0 else 1.0
        bounds = {c: (max(0.0, cur[i] * scale * (1 - max_change)), cur[i] * scale * (1 + max_change)) for i, c in enumerate(chans)}
    bnds = [bounds[c] for c in chans]
    P_map = m._unpack(fit.z_map, fit.names)

    def effect(x, P):
        return np.array([P["channels"][c]["beta"] * hill(x[i] / fit.x_scale[c], P["channels"][c]["K"], P["channels"][c]["S"])
                         for i, c in enumerate(chans)]) * fit.y_mean

    def neg(x):
        e = effect(x, P_map).sum()
        return -(margin * e - x.sum()) if objective == "profit" else -e

    cons = [{"type": "eq" if objective == "outcome" else "ineq", "fun": (lambda x: B - x.sum())}]
    x0 = np.clip(cur * (B / cur.sum() if cur.sum() > 0 else 1), [b[0] for b in bnds], [b[1] for b in bnds])
    r = optimize.minimize(neg, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 500})
    x = r.x
    notes = []
    if objective == "outcome" and abs(x.sum() - B) > 1e-3 * B:
        notes.append("Budget could not be fully allocated within the channel bounds; widen bounds or lower the budget.")
    cur_scaled = cur * (B / cur.sum()) if cur.sum() > 0 else cur
    Ps = [m._unpack(z, fit.names) for z in fit.draws[:n_draws]]
    gains = np.array([effect(x, P).sum() - effect(cur_scaled, P).sum() for P in Ps])
    e_cur = effect(cur, P_map)
    e_new = effect(x, P_map)
    tab = pd.DataFrame({"channel": chans, "current_weekly": cur, "recommended_weekly": x,
                        "change_pct": np.where(cur > 0, x / np.maximum(cur, 1e-9) - 1, np.nan),
                        "current_weekly_effect": e_cur, "recommended_weekly_effect": e_new,
                        "lower_bound": [b[0] for b in bnds], "upper_bound": [b[1] for b in bnds]})
    at_bound = [c for c, xi, b in zip(chans, x, bnds) if xi >= b[1] * 0.999 or xi <= b[0] * 1.001 + 1e-9]
    if at_bound:
        from dsdvc.exports.deck import display_name
        notes.append("At their limits: " + ", ".join(display_name(c) for c in at_bound) + ". The optimum may lie further out, but that is outside "
                     "the spend range the model has evidence for; test before moving further.")
    if abs(B - cur.sum()) > 1e-6 * max(B, 1):
        notes.append(f"Uplift is measured against the current mix scaled to the same budget ({B:,.0f}/week).")
    return Plan(table=tab, weekly_budget=B, objective=objective, uplift=float(np.median(gains)),
                uplift_low=float(np.quantile(gains, 0.05)), uplift_high=float(np.quantile(gains, 0.95)),
                p_better=float(np.mean(gains > 0)), notes=notes)


@dataclass
class Backtest:
    holdout_weeks: int
    metrics: dict
    series: pd.DataFrame
    verdict: str


def backtest(model: MarketingMixModel, df: pd.DataFrame, holdout_weeks: int = 12, lift_tests=None) -> Backtest:
    """Fit on all but the last ``holdout_weeks``; predict them from actual spend/controls only.
    The 90% predictive interval includes parameter uncertainty and observation noise."""
    s = model.spec
    d = df.sort_values(s.date_col).reset_index(drop=True)
    train = d.iloc[:-holdout_weeks]
    fit = model.fit(train, lift_tests)
    rng = np.random.default_rng(model.seed)
    preds = []
    for z in fit.draws[:300]:
        comp = components(fit, d, z)
        mu = comp.drop(columns="date").sum(axis=1).to_numpy()
        sigma = np.exp(dict(zip(fit.names, z))["log_sigma"]) * fit.y_mean
        preds.append(mu + rng.normal(0, sigma, len(mu)))
    Pm = np.array(preds)
    y = d[s.target_col].to_numpy(float)
    h = slice(len(d) - holdout_weeks, len(d))
    med, lo, hi = np.median(Pm, 0), np.quantile(Pm, 0.05, 0), np.quantile(Pm, 0.95, 0)
    yh, ph = y[h], med[h]
    mape = float(np.mean(np.abs((yh - ph) / np.where(yh == 0, np.nan, yh))))
    total_err = float((ph.sum() - yh.sum()) / yh.sum())
    cover = float(np.mean((yh >= lo[h]) & (yh <= hi[h])))
    ss = np.sum((yh - yh.mean()) ** 2)
    metrics = {"holdout_weeks": holdout_weeks, "mape": mape, "total_error_pct": total_err,
               "r2": float(1 - np.sum((yh - ph) ** 2) / ss) if ss > 0 else float("nan"),
               "interval_coverage_90": cover, "train_r2": fit.diagnostics["r2"]}
    if mape <= 0.10 and abs(total_err) <= 0.05 and cover >= 0.7:
        verdict = "PASS — accurate enough to support budget decisions"
    elif mape <= 0.20 and abs(total_err) <= 0.12:
        verdict = "CAUTION — directionally useful; treat reallocations as hypotheses to test"
    else:
        verdict = "FAIL — do not base budget decisions on this model yet; improve data or run lift tests"
    series = pd.DataFrame({"date": pd.to_datetime(d[s.date_col]), "actual": y, "predicted": med, "low": lo, "high": hi,
                           "holdout": np.arange(len(d)) >= len(d) - holdout_weeks})
    return Backtest(holdout_weeks, metrics, series, verdict)
