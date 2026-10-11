"""MarketingResearchAgent (spec §37-§39, §43, §62, §63).

A deterministic, auditable orchestrator: it parses the brief, builds the 14-step research
plan, runs each step with the tool layer, records every result and assumption in
ResearchMemory, and renders a report whose every recommendation is traced to
source/data/model/experiment. An LLM is optional and only narrates computed evidence
(``research_agent.llm``); it never supplies numbers.

Data modes
  authorized (default): uses supplied/authorised data; missing data -> DATA_UNAVAILABLE.
  synthetic (explicit opt-in): fills gaps with SYNTHETIC data, labelled throughout.
The simulator's agent population is always synthetic (the digital twin's synthetic layer);
its parameters are PRIORS unless calibrated to observed data.
"""
from __future__ import annotations

import copy
import re
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dsdvc.core.literature import LITERATURE
from dsdvc.data.quality import DataQualityEngine
from dsdvc.diffusion.abm import BrandSpec, SimConfig, run_simulation
from dsdvc.research_agent import tools as T
from dsdvc.research_agent.memory import ResearchMemory

PLAN = ["Define market", "Identify competitors", "Identify audience", "Discover relevant public social ecosystems",
        "Discover creators", "Analyze communities", "Analyze content", "Extract STEPPS features", "Estimate diffusion",
        "Estimate conversion", "Estimate CAC/LTV", "Simulate strategies", "Identify risks", "Produce recommendations"]

CATEGORIES = ["skincare", "haircare", "makeup", "cosmetics", "fashion", "apparel", "footwear", "food", "beverage",
              "fitness", "wellness", "fintech", "edtech", "gaming", "electronics", "travel", "jewellery", "jewelry"]
COUNTRIES = {"india": "IN", "united states": "US", "usa": "US", "uk": "GB", "united kingdom": "GB", "indonesia": "ID",
             "brazil": "BR", "uae": "AE", "singapore": "SG", "germany": "DE", "nigeria": "NG"}


def parse_brief(text: str) -> dict:
    t = text.lower()
    cat = next((c for c in CATEGORIES if c in t), "general")
    country = next((v for k, v in COUNTRIES.items() if k in t), None)
    ages = re.search(r"(\d{2})\s*[-–to]+\s*(\d{2})", t)
    gender = "women" if re.search(r"\bwomen\b|\bfemale\b|\bgirls\b", t) else ("men" if re.search(r"\bmen\b|\bmale\b", t) else "all")
    tier = "premium" if re.search(r"premium|luxury|high-end", t) else ("value" if re.search(r"budget|affordable|mass", t) else "mid")
    return {"category": cat, "country": country, "age_min": int(ages.group(1)) if ages else None,
            "age_max": int(ages.group(2)) if ages else None, "gender": gender, "price_tier": tier, "raw": text}


@dataclass
class ResearchInputs:
    creators: pd.DataFrame | None = None          # candidate creators (authorised/public professional data)
    content: pd.DataFrame | None = None           # content with outcomes (authorised exports)
    daily: pd.DataFrame | None = None             # observed daily metrics: adoptions, engagement, shares, ...
    customer_panel: pd.DataFrame | None = None    # prospects/customers for conversion & churn models
    market_data_paths: list[str] = field(default_factory=list)
    competitors: list[dict] = field(default_factory=list)   # [{"name", "price", "quality", "brand_equity"}]
    usernames: list[str] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    price_per_month: float | None = None
    budget: float = 600_000.0
    population: float | None = None               # addressable consumers (if known; else assumption)


class MarketingResearchAgent:
    def __init__(self, allow_synthetic: bool = False, n_agents: int = 1000, steps: int = 90, reps: int = 3,
                 seed: int = 0, narrator=None):
        self.allow_synthetic = allow_synthetic
        self.n_agents, self.steps, self.reps, self.seed = n_agents, steps, reps, seed
        self.narrator = narrator

    def plan(self, brief: dict) -> list[dict]:
        return [{"step": i + 1, "name": s, "status": "pending"} for i, s in enumerate(PLAN)]

    def _mark(self, mem: ResearchMemory, step: int, status: str, note: str = ""):
        mem.plan[step - 1].update({"status": status, "note": note})

    def run(self, request: str, inputs: ResearchInputs | None = None) -> tuple[ResearchMemory, dict]:
        inp = inputs or ResearchInputs()
        brief = parse_brief(request)
        mem = ResearchMemory(project_id=f"proj_{uuid.uuid4().hex[:8]}", request=request)
        mem.plan = self.plan(brief)
        S: dict = {}  # report sections
        mode = "SYNTHETIC-ALLOWED" if self.allow_synthetic else "AUTHORIZED-ONLY"
        mem.parameters["data_mode"] = mode

        # 1 market
        mem.market_definition = {k: brief[k] for k in ("category", "country", "price_tier")}
        md = T.search_market_data(inp.market_data_paths)
        if md.ok:
            mem.sources.append({"type": "market_data", **md.provenance})
            S["market_size"] = md.data.to_dict(orient="records")
            self._mark(mem, 1, "done", "user-supplied market data")
        else:
            mem.unavailable.append({"item": "market size", "message": md.message})
            S["market_size"] = "DATA_UNAVAILABLE - " + md.message
            self._mark(mem, 1, "DATA_UNAVAILABLE", md.message)
        population = inp.population or 200_000
        if inp.population is None:
            mem.assume("addressable_population", population, "Placeholder for simulation scaling only; replace with sourced market size.")

        # 2 competitors
        if inp.competitors:
            mem.competitors = inp.competitors
            mem.sources.append({"type": "competitors", "source": "user_provided"})
            self._mark(mem, 2, "done")
        else:
            web = T.search_web(f"{brief['category']} brands {brief.get('country') or ''} market share")
            if web.ok:
                mem.sources.append({"type": "web_leads", **web.provenance})
                S["competitor_leads"] = web.data
                self._mark(mem, 2, "partial", "web leads only; competitor attributes need verification")
            else:
                mem.unavailable.append({"item": "competitors", "message": web.message})
                self._mark(mem, 2, "DATA_UNAVAILABLE", web.message)
        S["competitive_landscape"] = mem.competitors or "DATA_UNAVAILABLE - supply competitor names, prices and positioning."

        # 3 audience
        mem.target_audience = {k: brief[k] for k in ("gender", "age_min", "age_max")}
        mem.target_audience["note"] = "Audience definition from the brief; DSDVC does not infer sensitive attributes of individuals."
        self._mark(mem, 3, "done")

        # 4 ecosystems
        if inp.hashtags:
            sp = T.search_public_social_data(inp.hashtags)
            if sp.ok:
                S["public_social"] = {"items": len(sp.data)}
                mem.sources.append({"type": "public_social", **sp.provenance})
                self._mark(mem, 4, "done")
            else:
                mem.unavailable.append({"item": "public social data", "message": sp.message})
                self._mark(mem, 4, "DATA_UNAVAILABLE", sp.message)
        else:
            self._mark(mem, 4, "skipped", "no hashtags/keywords for public ecosystem search")

        # 5 creators
        inf = T.discover_influencers(brief["category"], inp.keywords or [brief["category"]], inp.creators, inp.usernames or None,
                                     inp.content, allow_synthetic=self.allow_synthetic, seed=self.seed)
        if inf.ok:
            top = inf.data.head(15)
            S["influencer_landscape"] = {"status": inf.status, "table": top, "column_status": inf.provenance.get("column_status")}
            mem.results["influencers_top"] = top[["creator_id", "platform", "followers", "roi_score", "roi_score_low",
                                                   "roi_score_high", "rank_by_followers"]].to_dict(orient="records")
            corr = float(pd.Series(inf.data["rank_by_roi"]).corr(pd.Series(inf.data["rank_by_followers"]), method="spearman"))
            mem.conclude(f"Ranking creators by expected incremental value per cost differs from ranking by followers "
                         f"(Spearman rank correlation {corr:.2f}).", "predicted", "medium",
                         inf.provenance.get("source", "creators"), "creator table", "score_creators")
            self._mark(mem, 5, "done" if inf.status == "OK" else "SYNTHETIC")
        else:
            S["influencer_landscape"] = "DATA_UNAVAILABLE - " + inf.message
            mem.unavailable.append({"item": "creators", "message": inf.message})
            self._mark(mem, 5, "DATA_UNAVAILABLE", inf.message)

        # 6 communities (synthetic twin unless an observed network is supplied)
        from dsdvc.networks.multiplex import synthetic_multiplex
        mx = synthetic_multiplex(self.n_agents, 8, seed=self.seed)
        net = T.analyze_network(mx)
        S["community_structure"] = {"status": "SIMULATED (synthetic twin; no observed graph supplied)", **{
            k: {kk: v["structure"][kk] for kk in ("modularity", "edge_homophily", "avg_clustering", "community_count")}
            | {"detected_communities": v["detected_communities"]} for k, v in net.data.items()}}
        self._mark(mem, 6, "SIMULATED", "community structure of the synthetic twin; supply authorised aggregate graph data to observe it")

        # 7-8 content + STEPPS
        content = inp.content
        if content is None and self.allow_synthetic:
            from dsdvc.data.synthetic import synthetic_content
            content = synthetic_content(2000, seed=self.seed + 3)
        dq = None
        if content is not None:
            dq = DataQualityEngine().evaluate(content)
            S["data_quality"] = {"score": dq.score, "checks": dq.table().to_dict(orient="records")}
            ac = T.analyze_content(content)
            cdf = ac.data["content"]
            from dsdvc.content.stepps import DIMENSIONS
            from dsdvc.content.stepps_models import compare_stepps_forms
            S["content_landscape"] = {"status": ac.status, "n": len(cdf), "topics": ac.data["topics"],
                                      "stepps_means": cdf[[f"stepps_{d}" for d in DIMENSIONS]].mean().round(3).to_dict()}
            if {"shares", "views"} <= set(cdf.columns):
                comp = compare_stepps_forms(cdf[[f"stepps_{d}" for d in DIMENSIONS]].values,
                                            (cdf.shares / cdf.views.clip(lower=1)).values, seed=self.seed)
                S["stepps_forms"] = comp["table"]
                mem.experiments.append({"id": "stepps_form_comparison", "design": "5-fold CV", "winner": comp["best_form"]})
                mem.conclude(f"Best-supported STEPPS functional form: {comp['best_form']} (out-of-sample RMSE).",
                             "estimated", "medium", ac.status.lower(), "content corpus", "compare_stepps_forms",
                             "stepps_form_comparison")
            self._mark(mem, 7, ac.status)
            self._mark(mem, 8, ac.status)
        else:
            S["content_landscape"] = "DATA_UNAVAILABLE - supply authorised content exports with outcomes."
            mem.unavailable.append({"item": "content", "message": "no content supplied"})
            self._mark(mem, 7, "DATA_UNAVAILABLE")
            self._mark(mem, 8, "DATA_UNAVAILABLE")

        # virality model (if content with outcomes)
        if content is not None and {"shares", "views", "timestamp", "followers"} <= set(content.columns) and len(content) >= 300:
            from dsdvc.content.virality import ViralityPredictor
            vp = ViralityPredictor(seed=self.seed)
            rep = vp.fit(content)
            S["virality_benchmarks"] = rep["benchmarks"]
            mem.experiments.append({"id": "virality_benchmark", "design": "temporal 60/20/20 split",
                                    "definition": rep["viral_definition"]})
            pred = vp.predict(content.sort_values("timestamp").tail(200))
            ranked = sorted(pred, key=lambda r: -r["viral"]["prediction"])[:10]
            S["viral_opportunities"] = [{"content_id": r["content_id"], "summary": r["summary"],
                                         "expected_conversion_rate": r.get("expected_conversion_rate", {}).get("prediction")}
                                        for r in ranked]
            conv_ranked = sorted(pred, key=lambda r: -(r.get("expected_conversion_rate", {}) or {}).get("prediction", 0))[:10]
            S["conversion_opportunities"] = [{"content_id": r["content_id"],
                                              "expected_conversion_rate": r["expected_conversion_rate"]["prediction"],
                                              "p_viral": r["viral"]["prediction"]} for r in conv_ranked]
            self.virality = vp
        else:
            S["viral_opportunities"] = "DATA_UNAVAILABLE - needs >= 300 content items with outcomes and timestamps."
            S["conversion_opportunities"] = S["viral_opportunities"]

        # 9 diffusion calibration
        cfg = SimConfig(n_agents=self.n_agents, steps=self.steps, seed=self.seed, population_scale=population / self.n_agents,
                        seeding_budget=inp.budget * 0.4, rewire_every=30)
        price = inp.price_per_month or (60.0 if brief["price_tier"] == "premium" else 30.0)
        brands = [BrandSpec("focal", 0.72 if brief["price_tier"] == "premium" else 0.6, price, 0.3, is_focal=True)]
        for c in (mem.competitors or []):
            brands.append(BrandSpec(c.get("name", "competitor"), c.get("quality", 0.6), c.get("price", price * 0.8),
                                    c.get("brand_equity", 0.5), c.get("marketing", 0.4)))
        if len(brands) == 1:
            brands += [BrandSpec("generic_competitor_1", 0.6, price * 0.7, 0.6, 0.4),
                       BrandSpec("generic_competitor_2", 0.5, price * 0.45, 0.4, 0.3)]
            mem.assume("competitors", "two generic competitors", "No competitor data supplied; simulation uses placeholder competitors.")
        cfg.brands = brands
        cfg.cost_model.population_scale = cfg.population_scale
        adopt = inp.daily["adoptions"].to_numpy(float) if inp.daily is not None and "adoptions" in inp.daily else None
        cal = T.run_calibration(adopt)
        if cal.ok:
            b = cal.data["nls"]
            S["diffusion"] = {"bass_nls": b.table().to_dict(orient="records"),
                              "bass_bayes": cal.data["bayes"].table().to_dict(orient="records")}
            mem.parameters["bass"] = b.parameters
            cfg.p_innovation = float(np.clip(b.parameters["p"], 1e-4, 0.05))
            mem.conclude(f"Observed adoption implies Bass p={b.parameters['p']:.4f}, q={b.parameters['q']:.3f}.",
                         "estimated", "medium", "observed daily adoptions", "daily series", "Bass NLS + MCMC")
            self._mark(mem, 9, "CALIBRATED")
        else:
            S["diffusion"] = "PRIORS ONLY - " + cal.message
            self._mark(mem, 9, "PRIOR", cal.message)

        # 10 conversion
        if inp.customer_panel is not None:
            from dsdvc.conversion.model import ConversionModel, benchmark_conversion_models
            cm = ConversionModel().fit(inp.customer_panel)
            S["conversion_model"] = {"by_platform": cm.coefficients_by_platform(),
                                     "benchmark": benchmark_conversion_models(inp.customer_panel)}
            self._mark(mem, 10, "ESTIMATED")
        else:
            self._mark(mem, 10, "SIMULATED", "conversion from simulator priors (no customer panel)")

        # 11-12 simulate strategies
        from dsdvc.experiments.counterfactual import compare_seeding_strategies, run_counterfactual
        strat = compare_seeding_strategies(cfg, n_reps=self.reps)
        mem.experiments.append({"id": "seeding_strategy_comparison", "design": f"Monte Carlo x{self.reps}, common random numbers",
                                "strategies": list(strat.index)})
        best = strat.index[0]
        S["strategy_comparison"] = strat
        best_cfg = copy.deepcopy(cfg)
        best_cfg.seeding_strategy = best
        res = run_simulation(best_cfg)
        summ = res.summary()
        mem.results["best_strategy_run"] = {k: v for k, v in summ.items()}
        from dsdvc.optimization.budget import curves_from_simulator, optimize_allocation
        chans = ["instagram_paid", "facebook_paid", "creators", "referral_whatsapp", "retention"]
        curves, curve_tab = curves_from_simulator(best_cfg, chans, [inp.budget * 0.1, inp.budget * 0.3, inp.budget * 0.6],
                                                  seeds=tuple(range(self.seed, self.seed + 2)))
        alloc = optimize_allocation(curves, inp.budget)
        S["budget_allocation"] = alloc["allocation"]
        mem.experiments.append({"id": "budget_response_curves", "design": "spend sweeps on top of baseline plan",
                                "channels": chans})
        S["expected_outcomes"] = summ
        self._mark(mem, 11, "SIMULATED")
        self._mark(mem, 12, "SIMULATED")

        # scenarios
        scen = {}
        for k in ("instagram_reach_minus_30", "competitor_price_minus_10", "whatsapp_trust_plus_15"):
            r = run_counterfactual(best_cfg, k, n_reps=self.reps)
            scen[r["question"]] = r["table"]
            mem.experiments.append({"id": f"cf_{k}", "design": "paired simulation", "question": r["question"]})
        S["scenarios"] = scen

        # 13 risks
        from dsdvc.forecasting.early_warning import platform_dependency
        dep = platform_dependency({k: v for k, v in summ["acquisition_share_by_platform"].items() if k != "external"})
        S["platform_risk"] = dep
        churn_share = summ["churned"] / max(summ["ever_adopted"], 1)
        S["churn_risk"] = {"simulated_churned_share_of_adopters": churn_share,
                           "note": "Simulated with prior churn coefficients unless a customer panel was supplied."}
        if inp.daily is not None and len(inp.daily) >= 28:
            S["forecast"] = T.run_forecast(inp.daily).data
        self._mark(mem, 13, "done")

        # 14 recommendations (each traced to an experiment)
        roi_row = strat.loc[best]
        mem.conclude(f"Lead with the '{best}' seeding strategy: highest simulated ROI {roi_row.roi:.2f} "
                     f"(90% CI {roi_row.roi_ci_low:.2f} to {roi_row.roi_ci_high:.2f}) among the strategies tested.",
                     "recommended", "medium" if self.reps >= 5 else "low", "simulation (synthetic twin, prior parameters)",
                     "SimConfig priors" + (" + calibrated Bass p" if cal.ok else ""), "MarketSimulator",
                     "seeding_strategy_comparison", recommendation=True)
        top_alloc = alloc["allocation"].sort_values("spend", ascending=False)
        funded = top_alloc[top_alloc.spend > 0.02 * inp.budget]
        mem.conclude("Allocate budget to " + ", ".join(f"{r.channel} ({r.spend:,.0f})" for r in funded.itertuples())
                     + f"; total {alloc['total_spend']:,.0f} of {inp.budget:,.0f}.", "recommended", "low",
                     "simulation response curves", "spend sweeps", "optimize_allocation (SLSQP)",
                     "budget_response_curves", recommendation=True)
        if dep["flag"]:
            others = [p for p in ("instagram", "facebook", "whatsapp") if p != dep["max_platform"]]
            mem.conclude(dep["message"] + f" - diversify acquisition into {others[0]} and {others[1]} before scaling spend.",
                         "recommended", "medium",
                         "simulation", "acquisition channels", "platform_dependency", "seeding_strategy_comparison",
                         recommendation=True)
        S["recommended_experiments"] = [
            {"experiment": f"Geo or audience holdout for the '{best}' creator programme", "measures": "incremental conversions vs control"},
            {"experiment": "STEPPS creative A/B (weakest dimension strengthened)", "measures": "share rate, conversion rate, 90-day retention"},
            {"experiment": "WhatsApp referral incentive test (randomised invitation)", "measures": "referral conversions, CAC, churn"},
            {"experiment": "Price sensitivity test vs main competitor", "measures": "conversion, churn, LTV"},
        ]
        self._mark(mem, 14, "done")
        S["literature"] = sorted({LITERATURE[k]["citation"] for k in ("bass1969", "schelling1971", "berger2013", "centola_macy2007",
                                                                       "watts_dodds2007", "deveirman2017", "mcfadden1974", "gupta2006",
                                                                       "lewis_rao2015", "rand_rust2011")})
        S["data_quality_score"] = dq.score if dq else None
        if self.narrator is not None:
            S["narrative"] = self.narrator.narrate(mem, S)
        return mem, S
