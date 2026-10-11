"""Causal / counterfactual engine (spec §31, §8).

Questions are translated into config interventions and evaluated as paired simulation
experiments with common random numbers. Output: Baseline / Treatment / Difference / CI.
These are *model-based* counterfactuals: their validity is conditional on the simulator's
calibration. Real incrementality still requires field experiments (attribution.incrementality_test).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from dsdvc.core.literature import cites
from dsdvc.diffusion.abm import SimConfig
from dsdvc.experiments.design import ab_simulation, get_param, run_batch, set_param


@dataclass
class Scenario:
    name: str
    question: str
    overrides: dict = field(default_factory=dict)
    transform: Callable[[SimConfig], None] | None = None

    def apply(self, cfg: SimConfig) -> dict:
        """Return concrete overrides (resolving relative changes against cfg)."""
        if self.transform is None:
            return dict(self.overrides)
        c = copy.deepcopy(cfg)
        self.transform(c)
        return _diff(cfg, c)


def _diff(a: SimConfig, b: SimConfig) -> dict:
    out = {}
    keys = ["platforms.instagram.social_reach", "platforms.instagram.algo_reach", "platforms.facebook.social_reach",
            "platforms.facebook.algo_reach", "platforms.whatsapp.social_reach", "platforms.whatsapp.trust",
            "platforms.instagram.trust", "platforms.facebook.trust", "homophily_scale", "seeding_strategy",
            "seeding_budget", "stepps.emotion", "stepps.social_currency", "stepps.practical_value", "stepps.public",
            "stepps.stories", "stepps.triggers", "retention_investment", "complex_share"]
    for k in keys:
        va, vb = get_param(a, k), get_param(b, k)
        if va != vb:
            out[k] = vb
    for i, (ba, bb) in enumerate(zip(a.brands, b.brands)):
        for attr in ("price", "quality", "brand_equity", "marketing"):
            if getattr(ba, attr) != getattr(bb, attr):
                out[f"brands.{i}.{attr}"] = getattr(bb, attr)
    return out


def _scale(path: str, factor: float) -> Callable[[SimConfig], None]:
    def f(c: SimConfig):
        set_param(c, path, float(np.clip(get_param(c, path) * factor, 0, 1e9)))
    return f


def _compose(*fs):
    def f(c):
        for g in fs:
            g(c)
    return f


def _competitor_price(factor: float):
    def f(c: SimConfig):
        for b in c.brands:
            if not b.is_focal:
                b.price *= factor
    return f


SCENARIO_LIBRARY: dict[str, Scenario] = {
    "instagram_reach_minus_30": Scenario("instagram_reach_minus_30", "What if Instagram reach falls 30%?",
                                         transform=_compose(_scale("platforms.instagram.social_reach", 0.7),
                                                            _scale("platforms.instagram.algo_reach", 0.7))),
    "celebrity_to_micro": Scenario("celebrity_to_micro", "What if we replace celebrity seeding with micro-influencers?",
                                   overrides={"seeding_strategy": "micro_network"}),
    "whatsapp_trust_plus_15": Scenario("whatsapp_trust_plus_15", "What if WhatsApp referral trust increases 15%?",
                                       transform=_scale("platforms.whatsapp.trust", 1.15)),
    "competitor_price_minus_10": Scenario("competitor_price_minus_10", "What if competitor price falls 10%?",
                                          transform=_competitor_price(0.9)),
    "stepps_emotion_up": Scenario("stepps_emotion_up", "What if STEPPS emotion increases (+0.2)?",
                                  transform=lambda c: c.stepps.__setitem__("emotion", min(1.0, c.stepps["emotion"] + 0.2))),
    "more_segregated": Scenario("more_segregated", "What if the network becomes more segregated?",
                                overrides={"homophily_scale": 1.25}),
}


@cites("watts_dodds2007", "rand_rust2011")
def run_counterfactual(cfg: SimConfig, scenario: Scenario | str, n_reps: int = 8, workers: int = 1,
                       metrics=("reach", "ever_adopted", "active_customers", "revenue", "cost", "profit", "roi")) -> dict:
    sc = SCENARIO_LIBRARY[scenario] if isinstance(scenario, str) else scenario
    if sc.name == "celebrity_to_micro" and cfg.seeding_strategy == "micro_network":
        base_cfg = copy.deepcopy(cfg)
        base_cfg.seeding_strategy = "celebrity_first"
    else:
        base_cfg = cfg
    ov = sc.apply(base_cfg)
    res, _ = ab_simulation(base_cfg, ov, metrics, n_reps, workers)
    table = pd.DataFrame([{"metric": r.metric, "baseline": r.baseline_mean, "treatment": r.treatment_mean,
                           "difference": r.difference, "ci_low": r.ci_low, "ci_high": r.ci_high} for r in res])
    return {"question": sc.question, "scenario": sc.name, "interventions": ov, "n_reps": n_reps,
            "table": table, "status": "SIMULATED",
            "caveat": "Model-based counterfactual (90% CI over stochastic replications, common random numbers). "
                      "Valid only to the extent the simulator is calibrated; confirm with a field experiment."}


@cites("watts_dodds2007", "deveirman2017")
def compare_seeding_strategies(cfg: SimConfig, strategies=("celebrity_first", "micro_network", "community_seeding", "hybrid"),
                               n_reps: int = 6, workers: int = 1, cascade_threshold: float = 0.05) -> pd.DataFrame:
    """Strategy A-D comparison on reach, cost, cascade probability, conversion, retention, ROI (spec §8)."""
    seeds = list(range(cfg.seed, cfg.seed + n_reps))
    df = run_batch(cfg, [{"seeding_strategy": s} for s in strategies], seeds, workers)
    df["large_cascade"] = df["adoption_rate"] > cascade_threshold
    df["retention"] = df["active_customers"] / df["ever_adopted"].clip(lower=1)
    g = df.groupby("seeding_strategy")
    out = g.agg(reach=("reach", "mean"), cost=("cost", "mean"), cascade_probability=("large_cascade", "mean"),
                adopters=("ever_adopted", "mean"), conversion_of_reached=("conversion_rate_of_reached", "mean"),
                retention=("retention", "mean"), profit=("profit", "mean"), roi=("roi", "mean"),
                roi_sd=("roi", "std"), seeds_used=("seeds", "mean"))
    from scipy import stats
    tq = stats.t.ppf(0.95, max(n_reps - 1, 1))  # 90% two-sided
    out["roi_ci_low"] = out["roi"] - tq * out["roi_sd"] / np.sqrt(n_reps)
    out["roi_ci_high"] = out["roi"] + tq * out["roi_sd"] / np.sqrt(n_reps)
    return out.sort_values("roi", ascending=False)
