"""Report rendering for the research agent (spec §43, §63). Every section states its epistemic
status (OBSERVED / ESTIMATED / SIMULATED / PREDICTED / RECOMMENDED / DATA_UNAVAILABLE)."""
from __future__ import annotations

import pandas as pd

SECTION_ORDER = ["Executive Summary", "Market Size", "Competitive Landscape", "Consumer Segments", "Platform Landscape",
                 "Community Structure", "Influencer Landscape", "Content Landscape", "Viral Opportunities",
                 "Conversion Opportunities", "Recommended Campaign", "Budget Allocation", "Expected Reach",
                 "Expected Conversion", "Expected CAC", "Expected LTV", "Expected ROI", "Churn Risk", "Platform Risk",
                 "Scenario Analysis", "Confidence & Uncertainty", "Data Limitations", "Recommended Experiments", "Sources"]


def _fmt(x) -> str:
    if isinstance(x, pd.DataFrame):
        return x.round(4).to_markdown() if len(x) else "_empty_"
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], dict):
        return pd.DataFrame(x).round(4).to_markdown(index=False)
    if isinstance(x, dict):
        return "\n".join(f"- **{k}**: {_fmt(v) if isinstance(v, (pd.DataFrame, dict)) else v}" for k, v in x.items())
    return str(x)


def render_markdown(mem, S: dict) -> str:
    out = [f"# Market research report — {mem.market_definition.get('category', '').title()}",
           f"_Project {mem.project_id} · data mode: {mem.parameters.get('data_mode')} · request: “{mem.request}”_", ""]
    e = S.get("expected_outcomes", {})

    def sec(name, body, status=None):
        out.append(f"## {name}" + (f"  `[{status}]`" if status else ""))
        out.append(body if isinstance(body, str) else _fmt(body))
        out.append("")

    recs = "\n".join(f"- {c.statement} _(status: {c.status}; confidence: {c.confidence}; experiment: {c.trace.experiment})_"
                     for c in mem.recommendations)
    unav = len(mem.unavailable)
    sec("Executive Summary", (S.get("narrative") or "") + "\n\n" + recs +
        f"\n\n> {unav} input(s) were DATA_UNAVAILABLE; simulated figures use prior parameters unless marked CALIBRATED. "
        "Probabilities are conditional on the stated data and models.")
    sec("Market Size", S.get("market_size"), "OBSERVED" if isinstance(S.get("market_size"), list) else "DATA_UNAVAILABLE")
    sec("Competitive Landscape", S.get("competitive_landscape"))
    sec("Consumer Segments", {**mem.target_audience, "segmentation": "communities in the synthetic twin (see Community Structure)"})
    sec("Platform Landscape", "Instagram = discovery & creator reach (estimated algorithmic exposure); Facebook = groups & "
                              "community diffusion; WhatsApp = high-trust private forwarding (aggregate/authorised data only). "
                              "Platform parameters are priors unless calibrated.", "PRIOR")
    sec("Community Structure", S.get("community_structure"), "SIMULATED")
    inf = S.get("influencer_landscape")
    if isinstance(inf, dict):
        cols = [c for c in ["creator_id", "platform", "followers", "engagement_rate", "trust_proxy", "niche_relevance",
                            "estimated_cost", "expected_incremental_conversions", "estimated_cac", "roi_score",
                            "roi_score_low", "roi_score_high", "rank_by_followers"] if c in inf["table"]]
        sec("Influencer Landscape", inf["table"][cols], inf["status"])
        out.append("_Column status: followers/engagement = observed; trust/relevance/cost = estimated; conversions/ROI = predicted._\n")
    else:
        sec("Influencer Landscape", inf, "DATA_UNAVAILABLE")
    sec("Content Landscape", S.get("content_landscape"))
    if "stepps_forms" in S:
        sec("STEPPS functional-form test", S["stepps_forms"], "ESTIMATED")
    if "virality_benchmarks" in S:
        sec("Virality model benchmarks (test set)", S["virality_benchmarks"], "PREDICTED")
    sec("Viral Opportunities", S.get("viral_opportunities"), "PREDICTED")
    sec("Conversion Opportunities", S.get("conversion_opportunities"), "PREDICTED")
    sec("Recommended Campaign", S.get("strategy_comparison"), "SIMULATED")
    sec("Budget Allocation", S.get("budget_allocation"), "RECOMMENDED")
    sec("Expected Reach", f"{e.get('reach', 0):,.0f} people reached (simulated)", "SIMULATED")
    sec("Expected Conversion", f"{e.get('ever_adopted', 0):,.0f} adopters; {e.get('conversion_rate_of_reached', 0):.1%} of reached",
        "SIMULATED")
    sec("Expected CAC", f"{e.get('cac', float('nan')):,.2f} per adopter", "SIMULATED")
    sec("Expected LTV", f"{e.get('realised_ltv_per_customer', 0):,.2f} realised discounted margin per agent-customer over the horizon",
        "SIMULATED")
    sec("Expected ROI", f"{e.get('roi', float('nan')):.2f}", "SIMULATED")
    sec("Churn Risk", S.get("churn_risk"), "SIMULATED")
    sec("Platform Risk", S.get("platform_risk"), "SIMULATED")
    sc = S.get("scenarios", {})
    sec("Scenario Analysis", "\n\n".join(f"**{q}**\n\n{_fmt(t)}" for q, t in sc.items()), "SIMULATED")
    sec("Confidence & Uncertainty", "Simulation intervals reflect stochastic replication only, not parameter uncertainty, "
        "unless parameters were calibrated. Data quality score: "
        f"{S.get('data_quality_score') if S.get('data_quality_score') is not None else 'n/a'}.\n\n" +
        "\n".join(f"- {a['name']}: {a['value']} — {a['rationale']}" for a in mem.assumptions))
    sec("Data Limitations", "\n".join(f"- {u['item']}: {u['message']}" for u in mem.unavailable) or "None recorded.")
    sec("Recommended Experiments", S.get("recommended_experiments"))
    sec("Sources", "\n".join(f"- {s}" for s in mem.sources) + "\n\n**Literature**\n" + "\n".join(f"- {c}" for c in S.get("literature", [])))
    out.append("## Research plan trace")
    out.append(pd.DataFrame(mem.plan).to_markdown(index=False))
    return "\n".join(out)
