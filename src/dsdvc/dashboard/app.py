"""DSDVC Streamlit dashboard (spec §44-§48).

Every panel shows an epistemic badge (OBSERVED / ESTIMATED / SIMULATED / PREDICTED /
RECOMMENDED / SYNTHETIC / DATA_UNAVAILABLE). With no uploads, analyses that need observed data
show DATA_UNAVAILABLE unless the user explicitly switches on the labelled synthetic demo.

Chart conventions (dataviz reference palette): entity-bound categorical colours in fixed
order (Instagram, Facebook, WhatsApp = slots 1-3), single-hue sequential scales, 2px lines,
legend whenever >= 2 series, one y-axis per chart.
"""
from __future__ import annotations

import copy
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dsdvc import MODEL_VERSION, __version__

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PLATFORM_COLOR = {"instagram": SERIES[0], "facebook": SERIES[1], "whatsapp": SERIES[2], "external": "#8a8985"}
SEQ = "Blues"
BADGE = {"OBSERVED": "#1b7a3a", "ESTIMATED": "#2a78d6", "SIMULATED": "#4a3aa7", "PREDICTED": "#0f6e8c",
         "RECOMMENDED": "#6b4f00", "SYNTHETIC": "#a1460f", "DATA_UNAVAILABLE": "#8a1c1c", "PRIOR": "#52514e"}

TABS = ["Overview", "Market", "Consumers", "Networks", "Instagram", "Facebook", "WhatsApp", "Influencers", "Communities",
        "Content", "Virality", "Conversions", "Churn", "Competition", "Simulation", "Forecast", "Scenarios",
        "Attribution", "Calibration", "Validation", "Research Agent", "Reports"]


def badge(status: str, note: str = ""):
    c = BADGE.get(status, "#52514e")
    st.markdown(f"<span style='background:{c};color:white;padding:2px 8px;border-radius:4px;font-size:0.75rem;"
                f"font-weight:600'>{status}</span> <span style='color:#52514e;font-size:0.85rem'>{note}</span>",
                unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(template="simple_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.12, x=0), font=dict(size=12))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#ececea")
    fig.update_traces(selector=dict(type="scatter", mode="lines"), line=dict(width=2))
    return fig


def unavailable(what: str, remedy: str):
    badge("DATA_UNAVAILABLE")
    st.info(f"**{what}** is not available. {remedy}")


# ------------------------------------------------------------------ cached computations
@st.cache_data(show_spinner=False)
def sim_runs(cfg_items: tuple, n_seeds: int):
    from dsdvc.diffusion.abm import SimConfig, run_simulation
    cfg = SimConfig(**dict(cfg_items))
    cfg.cost_model.population_scale = cfg.population_scale
    out = []
    for s in range(n_seeds):
        c = copy.deepcopy(cfg)
        c.seed = cfg.seed + s
        r = run_simulation(c)
        out.append({"ts": r.timeseries, "summary": r.summary(), "agents": r.agents, "shares": r.shares,
                    "paths": r.paths, "events": r.event_counts, "run": r.run_record.model_dump(mode="json")})
    return out


@st.cache_data(show_spinner=False)
def demo_content(n: int = 2500):
    from dsdvc.data.synthetic import synthetic_content, synthetic_creators
    cr = synthetic_creators(250, 1)
    return synthetic_content(n, cr, 2), cr


@st.cache_data(show_spinner=False)
def demo_panel():
    from dsdvc.data.synthetic import synthetic_customer_panel
    return synthetic_customer_panel(8000)


@st.cache_data(show_spinner=False)
def demo_daily():
    from dsdvc.data.synthetic import synthetic_daily_metrics
    return synthetic_daily_metrics(180)


@st.cache_data(show_spinner=False)
def demo_journeys():
    from dsdvc.data.synthetic import synthetic_journeys
    return synthetic_journeys(8000)


@st.cache_resource(show_spinner=False)
def fit_virality(df: pd.DataFrame, top_pct: float):
    from dsdvc.content.virality import ViralityPredictor
    vp = ViralityPredictor(top_pct=top_pct)
    vp.fit(df, data_version="dashboard")
    return vp


@st.cache_data(show_spinner=False)
def network(n: int, homophily: float, seed: int):
    from dsdvc.networks.multiplex import synthetic_multiplex
    return synthetic_multiplex(n, 8, homophily, seed=seed)


def read_upload(label: str, key: str) -> pd.DataFrame | None:
    f = st.sidebar.file_uploader(label, type=["csv"], key=key)
    if f is None:
        return None
    df = pd.read_csv(f)
    for c in ("timestamp", "date"):
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df


# ------------------------------------------------------------------ app
def main():
    st.set_page_config(page_title="DSDVC Marketing Intelligence", layout="wide")
    st.title("DSDVC — Social Diffusion, Virality, Conversion & Marketing Intelligence")
    st.caption(f"v{__version__} · {MODEL_VERSION} · Decision support, not a vanity-metrics dashboard: "
               "Reach ≠ Engagement ≠ Sharing ≠ Adoption ≠ Conversion ≠ Retention.")

    sb = st.sidebar
    sb.header("Data")
    synthetic = sb.toggle("Use labelled SYNTHETIC demo data where observed data is missing", value=False,
                          help="Off = authorised-data-only mode. Synthetic data is always badged SYNTHETIC.")
    content_up = read_upload("Content export (CSV)", "content")
    creators_up = read_upload("Creator candidates (CSV)", "creators")
    daily_up = read_upload("Daily metrics (CSV)", "daily")
    panel_up = read_upload("Customer panel (CSV)", "panel")
    journeys_up = read_upload("Touchpoint paths (CSV: user_id, path ';'-separated, converted)", "paths")

    sb.header("Simulation")
    n_agents = sb.slider("Agents", 300, 4000, 1000, 100)
    steps = sb.slider("Days", 30, 180, 90, 10)
    n_seeds = sb.slider("Monte Carlo replications", 2, 12, 4)
    strategy = sb.selectbox("Seeding strategy", ["micro_network", "celebrity_first", "community_seeding", "hybrid", "random"])
    budget = sb.number_input("Creator/seeding budget", 0.0, 5e7, 200_000.0, 10_000.0)
    homophily = sb.slider("Homophily scale (Schelling-inspired)", 0.5, 1.4, 1.0, 0.05)
    stepps_form = sb.selectbox("STEPPS functional form", ["multiplicative", "additive", "interaction"])
    with sb.expander("STEPPS of focal content"):
        stepps = {k: st.slider(k, 0.0, 1.0, v, 0.05) for k, v in
                  {"social_currency": 0.6, "triggers": 0.5, "emotion": 0.7, "public": 0.6, "practical_value": 0.5,
                   "stories": 0.6}.items()}
    with sb.expander("Paid media (daily)"):
        ig_paid = st.number_input("Instagram paid / day", 0.0, 1e6, 0.0, 500.0)
        fb_paid = st.number_input("Facebook paid / day", 0.0, 1e6, 0.0, 500.0)
    complex_share = sb.slider("Share of complex-contagion adopters", 0.0, 1.0, 0.5, 0.05)
    retention = sb.slider("Retention investment", 0.0, 1.0, 0.2, 0.05)
    pop_scale = sb.number_input("Real consumers per agent", 1.0, 10000.0, 100.0, 10.0)

    cfg_items = tuple(sorted({"n_agents": n_agents, "steps": steps, "seeding_strategy": strategy, "seeding_budget": budget,
                              "homophily_scale": homophily, "stepps_form": stepps_form, "stepps": stepps,
                              "paid_daily_budget": {"instagram": ig_paid, "facebook": fb_paid},
                              "complex_share": complex_share, "retention_investment": retention,
                              "population_scale": pop_scale, "rewire_every": 15}.items(), key=lambda kv: kv[0]))
    cfg_items = tuple((k, v) for k, v in cfg_items)

    content = content_up if content_up is not None else (demo_content()[0] if synthetic else None)
    content_status = "OBSERVED" if content_up is not None else ("SYNTHETIC" if synthetic else "DATA_UNAVAILABLE")
    creators = creators_up if creators_up is not None else (demo_content()[1] if synthetic else None)
    daily = daily_up if daily_up is not None else (demo_daily() if synthetic else None)
    panel = panel_up if panel_up is not None else (demo_panel() if synthetic else None)
    journeys = None
    if journeys_up is not None:
        journeys = journeys_up.assign(path=journeys_up["path"].astype(str).str.split(";"),
                                      converted=journeys_up["converted"].astype(bool))
    elif synthetic:
        journeys = demo_journeys()

    with st.spinner("Running Monte Carlo simulation…"):
        runs = sim_runs(cfg_items, n_seeds)
    summaries = pd.DataFrame([{k: v for k, v in r["summary"].items() if not isinstance(v, dict)} for r in runs])

    tabs = st.tabs(TABS)
    T = dict(zip(TABS, tabs))

    # ---------------- Overview (executive KPIs with intervals)
    with T["Overview"]:
        badge("SIMULATED", "Executive KPIs: mean across replications with 90% interval. Parameters are PRIORS unless calibrated.")

        def kpi(col, label, series, fmt="{:,.0f}"):
            s = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                col.metric(label, "n/a")
                return
            lo, hi = s.quantile(0.05), s.quantile(0.95)
            col.metric(label, fmt.format(s.mean()), help=f"90% interval {fmt.format(lo)} – {fmt.format(hi)}")
            col.caption(f"{fmt.format(lo)} – {fmt.format(hi)}")

        r1 = st.columns(5)
        kpi(r1[0], "Market size (simulated population)", [n_agents * pop_scale] * len(summaries))
        kpi(r1[1], "Potential customers reached", summaries.reach)
        kpi(r1[2], "Predicted adopters", summaries.ever_adopted)
        kpi(r1[3], "Active customers (end)", summaries.active_customers)
        kpi(r1[4], "Conversion of reached", summaries.conversion_rate_of_reached, "{:.1%}")
        r2 = st.columns(5)
        kpi(r2[0], "CAC", summaries.cac, "{:,.2f}")
        kpi(r2[1], "Realised LTV / customer (agent)", summaries.realised_ltv_per_customer, "{:,.2f}")
        kpi(r2[2], "Churned share of adopters", summaries.churned / summaries.ever_adopted.clip(lower=1), "{:.1%}")
        kpi(r2[3], "ROI", summaries.roi, "{:.2f}")
        kpi(r2[4], "Platform dependency", summaries.platform_dependency, "{:.0%}")
        from dsdvc.networks.metrics import structure_report
        mx = network(min(n_agents, 1500), homophily, 0)
        rep = structure_report(mx.layers["facebook"], mx.community)
        r3 = st.columns(3)
        r3[0].metric("Network density (FB layer)", f"{rep['density']:.4f}")
        r3[1].metric("Community fragmentation (modularity)", f"{rep['modularity']:.2f}")
        cross = np.mean([r["shares"].groupby("agent").platform.nunique().gt(1).mean() if len(r["shares"]) else 0 for r in runs])
        r3[2].metric("Cross-platform sharers", f"{cross:.1%}")
        if content is not None:
            st.caption(f"Viral probability and content KPIs: see Virality tab ({content_status}).")
        ts = pd.concat([r["ts"].assign(rep=i) for i, r in enumerate(runs)])
        g = ts.groupby("t")
        band = pd.DataFrame({"t": g.active.mean().index, "mean": g.active.mean().values,
                             "lo": g.active.quantile(0.05).values, "hi": g.active.quantile(0.95).values})
        fig = go.Figure([go.Scatter(x=band.t, y=band.hi, line=dict(width=0), showlegend=False, hoverinfo="skip"),
                         go.Scatter(x=band.t, y=band.lo, fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
                                    line=dict(width=0), name="90% interval"),
                         go.Scatter(x=band.t, y=band["mean"], mode="lines", line=dict(color=SERIES[0], width=2),
                                    name="Active customers (agents)")])
        st.plotly_chart(style(fig.update_layout(title="Active customers over time")), width="stretch")

    # ---------------- Market
    with T["Market"]:
        st.subheader("Market definition & size")
        unavailable("Observed market size", "Upload sourced market-size tables to the research agent; the simulation "
                    "uses 'Agents × Real consumers per agent' as a scaling assumption only.")
        badge("SIMULATED")
        acq = pd.DataFrame([r["summary"]["acquisition_share_by_platform"] for r in runs]).mean().rename("share").reset_index()
        acq.columns = ["platform", "share"]
        st.plotly_chart(style(px.bar(acq, x="platform", y="share", color="platform", color_discrete_map=PLATFORM_COLOR,
                                     title="Acquisition share by platform (simulated)")), width="stretch")

    # ---------------- Consumers
    with T["Consumers"]:
        badge("SIMULATED", "Journey stages in the synthetic population")
        ag = runs[0]["agents"]
        order = ["UNAWARE", "AWARE", "INTERESTED", "CONSIDERING", "TRIAL", "ADOPTED", "ENGAGED", "ADVOCATE",
                 "DISSATISFIED", "CHURNED", "POSSIBLE_RECOVERY", "COMPETITOR"]
        sc = ag.stage.value_counts().reindex(order).fillna(0).reset_index()
        sc.columns = ["stage", "agents"]
        st.plotly_chart(style(px.bar(sc, x="stage", y="agents", title="Final journey stage distribution",
                                     color_discrete_sequence=[SERIES[0]])), width="stretch")
        from dsdvc.conversion.journey import funnel_table
        st.dataframe(funnel_table(ag.stage.value_counts().to_dict()), width="stretch")
        ts0 = runs[0]["ts"]
        cols = [c for c in ["n_aware", "n_interested", "n_considering", "n_trial", "active", "churned", "competitor"] if c in ts0]
        m = ts0.melt("t", cols, "stage", "agents")
        st.plotly_chart(style(px.line(m, x="t", y="agents", color="stage", color_discrete_sequence=SERIES,
                                      title="Stage counts over time (one replication)")), width="stretch")

    # ---------------- Networks
    with T["Networks"]:
        badge("SYNTHETIC", "Synthetic multiplex prior — replace with authorised aggregate graph data where available")
        from dsdvc.networks.metrics import structure_report
        rows = [{"layer": k, **structure_report(g, mx.community)} for k, g in mx.layers.items()]
        st.dataframe(pd.DataFrame(rows).set_index("layer").round(4), width="stretch")
        layer = st.selectbox("Layer to draw", list(mx.layers), key="netlayer")
        import networkx as nx
        g = mx.layers[layer].to_undirected() if mx.layers[layer].is_directed() else mx.layers[layer]
        sub = g.subgraph(list(range(min(400, g.number_of_nodes()))))
        pos = nx.spring_layout(sub, seed=1, k=0.15)
        ex, ey = [], []
        for a, b in sub.edges():
            ex += [pos[a][0], pos[b][0], None]
            ey += [pos[a][1], pos[b][1], None]
        nodes = list(sub.nodes)
        comm = [int(mx.community[i]) for i in nodes]
        fig = go.Figure([go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#d5d4cf", width=0.5), hoverinfo="skip", showlegend=False),
                         go.Scatter(x=[pos[i][0] for i in nodes], y=[pos[i][1] for i in nodes], mode="markers",
                                    marker=dict(size=7, color=[SERIES[c % 8] for c in comm], line=dict(width=1, color="white")),
                                    text=[f"agent {i} · community {c}" for i, c in zip(nodes, comm)], hoverinfo="text",
                                    showlegend=False)])
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        st.plotly_chart(style(fig, 520).update_layout(title=f"{layer}: first 400 agents, coloured by community "
                                                             "(identity also in hover text)"), width="stretch")
        st.subheader("Schelling-inspired rewiring")
        if st.button("Run 30 rewiring steps on the Facebook layer"):
            from dsdvc.networks.schelling import SchellingRewirer
            rw = SchellingRewirer(mx.layers["facebook"].copy(), mx.taste, seed=0)
            h = pd.DataFrame(rw.run(30, mx.community, 5))
            st.plotly_chart(style(px.line(h, x="step", y=["edge_homophily", "modularity", "mean_cross_community_exposure"],
                                          color_discrete_sequence=SERIES, title="Individual preference → aggregate structure")),
                            width="stretch")

    # ---------------- Platform tabs
    shares_all = pd.concat([r["shares"].assign(rep=i) for i, r in enumerate(runs)]) if any(len(r["shares"]) for r in runs) else pd.DataFrame()
    for plat, tab in (("instagram", "Instagram"), ("facebook", "Facebook"), ("whatsapp", "WhatsApp")):
        with T[tab]:
            badge("SIMULATED", {"instagram": "Discovery, creators, Reels; social + ESTIMATED algorithmic exposure "
                                             "(not Instagram's proprietary algorithm)",
                                "facebook": "Friends, Pages, Groups; community-level diffusion",
                                "whatsapp": "High-trust private layer; aggregate/authorised data only — no private chats"}[plat])
            pre = {"instagram": "shares_ig", "facebook": "shares_fb", "whatsapp": "shares_wa"}[plat]
            ts = pd.concat([r["ts"][["t", pre]].assign(rep=i) for i, r in enumerate(runs)])
            mean = ts.groupby("t")[pre].mean().reset_index()
            st.plotly_chart(style(px.line(mean, x="t", y=pre, title=f"New {tab} shares per day (mean of replications)",
                                          color_discrete_sequence=[PLATFORM_COLOR[plat]])), width="stretch")
            if len(shares_all):
                gen = shares_all[shares_all.platform == plat].groupby("generation").size().reset_index(name="shares")
                st.plotly_chart(style(px.bar(gen, x="generation", y="shares", color_discrete_sequence=[PLATFORM_COLOR[plat]],
                                             title="Cascade generations (0 = seed/algorithmic root)")), width="stretch")
            ch = runs[0]["agents"].acquisition_channel.value_counts()
            ch = ch[ch.index.str.startswith(plat[:2] if plat != "whatsapp" else "wa")]
            st.dataframe(ch.rename("acquired customers (agents)"), width="stretch")
            if content is not None and "platform" in content:
                c = content[content.platform == plat]
                if len(c):
                    badge(content_status, f"{len(c)} content items")
                    st.dataframe(c[["views", "likes", "comments", "shares", "saves"]].describe().round(1), width="stretch")
            from dsdvc.platforms.transmission import TransmissionMatrix
            if plat == "instagram":
                tm = TransmissionMatrix()
                st.caption("Cross-platform transmission matrix (prior; Bayesian-updated when transfer counts are supplied)")
                lo, hi = tm.interval()
                st.dataframe(tm.mean().round(3).astype(str) + " [" + lo.round(3).astype(str) + "–" + hi.round(3).astype(str) + "]",
                             width="stretch")

    # ---------------- Influencers
    with T["Influencers"]:
        if creators is None:
            unavailable("Creator candidates", "Upload a candidate table (public professional-account data or authorised "
                        "exports) or enable the labelled synthetic demo.")
        else:
            from dsdvc.networks.influencers import COLUMN_STATUS, score_creators
            badge("OBSERVED" if creators_up is not None else "SYNTHETIC", "observed vs modelled columns are listed below")
            kw = st.text_input("Category / keywords", "skincare")
            ranked = score_creators(creators, category=kw, content=content if content is not None and "creator_id" in content else None)
            rank_by = st.radio("Rank by", ["roi_score", "followers", "expected_incremental_conversions", "trust_proxy",
                                           "estimated_influence"], horizontal=True)
            cols = ["creator_id", "platform", "followers", "engagement_rate", "trust_proxy", "niche_relevance",
                    "community_centrality", "bridge_centrality", "expected_reach", "expected_incremental_conversions",
                    "expected_ltv", "estimated_cost", "estimated_cac", "roi_score", "roi_score_low", "roi_score_high"]
            st.dataframe(ranked.sort_values(rank_by, ascending=False)[[c for c in cols if c in ranked]].head(40).round(4),
                         width="stretch")
            st.dataframe(pd.Series(COLUMN_STATUS, name="status").to_frame().T, width="stretch")
            fig = px.scatter(ranked, x="followers", y="roi_score", color="platform", log_x=True,
                             color_discrete_map=PLATFORM_COLOR, hover_data=["creator_id"],
                             title="Followers vs expected incremental value per cost")
            st.plotly_chart(style(fig), width="stretch")
        st.subheader("Seeding strategy test (Watts & Dodds)")
        if st.button("Compare celebrity / micro / community / hybrid"):
            from dsdvc.diffusion.abm import SimConfig
            from dsdvc.experiments.counterfactual import compare_seeding_strategies
            cfg = SimConfig(**dict(cfg_items))
            cfg.cost_model.population_scale = pop_scale
            with st.spinner("Simulating strategies…"):
                st.session_state["strat"] = compare_seeding_strategies(cfg, n_reps=n_seeds)
        if "strat" in st.session_state:
            badge("SIMULATED")
            st.dataframe(st.session_state["strat"].round(3), width="stretch")

    # ---------------- Communities
    with T["Communities"]:
        badge("SYNTHETIC", "community discovery on the synthetic twin")
        from dsdvc.networks.communities import compare_methods, detect_communities
        layer = st.selectbox("Layer", list(mx.layers), key="commlayer")
        method = st.selectbox("Algorithm", ["leiden", "louvain", "label_propagation", "spectral", "hdbscan"])
        cr = detect_communities(mx.layers[layer], method)
        st.metric("Detected communities", cr.n_communities, help=cr.notes)
        st.metric("Modularity", f"{cr.modularity:.3f}")
        st.dataframe(pd.DataFrame(cr.per_community).round(4), width="stretch")
        st.write("Top bridge accounts (bridging centrality):", cr.bridges[:10])
        st.write("Isolated clusters:", cr.isolated_clusters, " · micro-communities:", cr.micro_communities)
        if st.checkbox("Compare all algorithms"):
            st.dataframe(pd.DataFrame(compare_methods(mx.layers[layer])), width="stretch")

    # ---------------- Content / Virality
    vp = None
    with T["Content"]:
        if content is None:
            unavailable("Content", "Upload an authorised content export or enable the synthetic demo.")
        else:
            badge(content_status)
            from dsdvc.content.stepps import DIMENSIONS, STEPPSAnalyzer
            an = STEPPSAnalyzer()
            sample = content.head(600).copy()
            S = np.array([an.analyze(t, m).vector() for t, m in zip(sample.text.fillna(""), sample.get("media_type", "image"))])
            for k, d in enumerate(DIMENSIONS):
                sample[d] = S[:, k]
            st.plotly_chart(style(px.bar(sample[list(DIMENSIONS)].mean().reset_index(name="mean score").rename(columns={"index": "dimension"}),
                                         x="dimension", y="mean score", color_discrete_sequence=[SERIES[0]],
                                         title="Mean STEPPS profile (lexicon coder; hypothesis, not guarantee)")),
                            width="stretch")
            if {"shares", "views"} <= set(sample.columns):
                from dsdvc.content.stepps_models import compare_stepps_forms
                cmp = compare_stepps_forms(sample[list(DIMENSIONS)].values, (sample.shares / sample.views.clip(lower=1)).values)
                badge("ESTIMATED", f"best-supported form: {cmp['best_form']}")
                st.dataframe(cmp["table"].round(5), width="stretch")
            st.subheader("Viral Content Copilot")
            txt = st.text_area("Concept / caption / script", "My journey with acne: 3 tips under ₹500 that finally worked. Tag a friend!")
            from dsdvc.content.copilot import ContentCopilot
            if {"shares", "views", "timestamp", "followers"} <= set(content.columns) and len(content) >= 300:
                vp = fit_virality(content, 0.05)
            rep = ContentCopilot(vp).analyze(caption=txt)
            if rep.viral:
                badge("PREDICTED", rep.language_note)
                st.write(f"Estimated probability of top-5% performance: **{rep.viral['prediction']:.0%}** "
                         f"(90% interval {rep.viral['lower']:.0%}–{rep.viral['upper']:.0%}; confidence {rep.viral['confidence']}), "
                         "conditional on the historical benchmark and current data.")
            st.json({"stepps": rep.stepps["scores"], "recommendations": rep.recommendations, "risks": rep.risks})
            if rep.variants:
                st.dataframe(pd.DataFrame(rep.variants)[["strengthen", "example_copy", "p_viral", "expected_conversion_rate"]],
                             width="stretch")
            if rep.ab_test:
                st.json(rep.ab_test)

    with T["Virality"]:
        if content is None or not {"shares", "views", "timestamp", "followers"} <= set(content.columns):
            unavailable("Virality model", "Needs ≥300 content items with views, shares, followers and timestamps.")
        else:
            top = st.select_slider("Viral = top x% of comparable content", [0.01, 0.05, 0.10], 0.05)
            vp = fit_virality(content, top)
            badge("PREDICTED", vp.report["viral_definition"])
            st.dataframe(vp.report["benchmarks"].round(4), width="stretch")
            st.write("Learned ensemble weights:", vp.report["weights"], "· tree backend:", vp.report["tree_backend"])
            cc = pd.DataFrame(vp.report["calibration_curve"])
            fig = go.Figure([go.Scatter(x=[0, cc.mean_pred.max()], y=[0, cc.mean_pred.max()], mode="lines",
                                        line=dict(dash="dot", color="#8a8985", width=1), name="perfect calibration"),
                             go.Scatter(x=cc.mean_pred, y=cc.observed, mode="lines+markers", name="ensemble (test set)",
                                        line=dict(color=SERIES[0], width=2), marker=dict(size=8))])
            st.plotly_chart(style(fig.update_layout(title="Calibration curve", xaxis_title="predicted", yaxis_title="observed")),
                            width="stretch")
            recent = content.sort_values("timestamp").tail(150)
            preds = vp.predict(recent)
            board = pd.DataFrame([{"content_id": p["content_id"], "p_viral": p["viral"]["prediction"],
                                   "p_viral_low": p["viral"]["lower"], "p_viral_high": p["viral"]["upper"],
                                   "confidence": p["viral"]["confidence"],
                                   "conversion_rate": (p.get("expected_conversion_rate") or {}).get("prediction"),
                                   "expected_conversions": (p.get("expected_conversions") or {}).get("prediction")} for p in preds])
            board = board.merge(recent[["content_id", "platform", "creator_id"]], on="content_id")
            margin = st.number_input("Margin per conversion", 0.0, 1e5, 500.0)
            board["predicted_profit"] = board.expected_conversions * margin
            st.subheader("Content leaderboard — three separate rankings")
            c1, c2, c3 = st.columns(3)
            c1.write("**By virality**")
            c1.dataframe(board.sort_values("p_viral", ascending=False).head(10)[["content_id", "platform", "p_viral", "confidence"]].round(3))
            c2.write("**By conversion**")
            c2.dataframe(board.sort_values("conversion_rate", ascending=False).head(10)[["content_id", "platform", "conversion_rate"]].round(4))
            c3.write("**By profit**")
            c3.dataframe(board.sort_values("predicted_profit", ascending=False).head(10)[["content_id", "platform", "predicted_profit"]].round(0))
            pick = st.selectbox("Why? — explain a prediction", board.content_id.head(30))
            p = next(x for x in preds if x["content_id"] == pick)
            st.code(p["summary"] + "\n\n" + p["drivers_text"])
            from dsdvc.content.velocity import classify_trends
            st.write("Early-velocity trend (12h):", classify_trends(recent.head(50)).value_counts().to_dict())

    # ---------------- Conversions / Churn
    with T["Conversions"]:
        if panel is None:
            unavailable("Customer panel", "Upload prospects/customers with awareness, social_proof, trust, price_fit, stepps, "
                        "community_fit, product_fit, platform, converted.")
        else:
            badge("ESTIMATED" if panel_up is not None else "SYNTHETIC")
            from dsdvc.conversion.model import ConversionModel, benchmark_conversion_models
            cm = ConversionModel().fit(panel)
            coef = cm.coefficients_by_platform()
            fig = px.bar(coef, x="feature", y="beta", color="platform", barmode="group", color_discrete_map=PLATFORM_COLOR,
                         error_y=coef.ci_high - coef.beta, title="Platform-specific conversion coefficients (90% CI)")
            st.plotly_chart(style(fig), width="stretch")
            if st.checkbox("Benchmark vs pooled logistic and gradient boosting (5-fold CV)"):
                st.dataframe(benchmark_conversion_models(panel).round(4), width="stretch")

    with T["Churn"]:
        if panel is None:
            unavailable("Customer panel", "Upload customer data with churn outcomes.")
        else:
            badge("ESTIMATED" if panel_up is not None else "SYNTHETIC")
            from dsdvc.churn.ltv import ltv_by
            from dsdvc.churn.model import ChurnModel, churn_breakdown
            cust = panel[panel.converted]
            st.dataframe(ChurnModel().fit(cust).table().round(3), width="stretch")
            by = st.selectbox("Churn / LTV by", ["platform", "acquisition_channel", "community", "influencer"])
            c1, c2 = st.columns(2)
            c1.dataframe(churn_breakdown(cust, by).round(3))
            c2.dataframe(ltv_by(cust, by).round(1))

    # ---------------- Competition
    with T["Competition"]:
        badge("SIMULATED", "multinomial logit with network effects inside the simulator")
        fs = summaries.focal_market_share
        st.metric("Focal brand share of category buyers", f"{fs.mean():.1%}", help=f"90% interval {fs.quantile(.05):.1%}–{fs.quantile(.95):.1%}")
        from dsdvc.competition.mnl import MNLParams, compare_network_specs, simulate_choices
        if st.checkbox("Test network-effect specifications on simulated choice data"):
            rng = np.random.default_rng(0)
            X = rng.normal(0, 1, (3000, 4, 6))
            N = np.array([50, 10, 30, 5.]) * rng.uniform(0.5, 1.5, (3000, 4))
            ch = simulate_choices(X, N, MNLParams(network_spec="log", alpha=0.5), 1)
            badge("SYNTHETIC", "truth = log specification")
            st.dataframe(compare_network_specs(X, N, ch).round(3), width="stretch")

    # ---------------- Simulation
    with T["Simulation"]:
        badge("SIMULATED")
        ts = pd.concat([r["ts"].assign(rep=i) for i, r in enumerate(runs)])
        metric = st.selectbox("Series", ["active", "cum_adopted", "new_adoptions", "reach", "churned", "competitor",
                                         "cum_revenue", "cum_cost"])
        fig = px.line(ts, x="t", y=metric, line_group="rep", title=f"{metric} — each line is one replication",
                      color_discrete_sequence=[SERIES[0]])
        st.plotly_chart(style(fig), width="stretch")
        st.dataframe(summaries.describe().T.round(3), width="stretch")
        st.write("Event counts (replication 1):")
        st.dataframe(runs[0]["events"].sum().rename("events"), width="stretch")
        st.write("Reproducibility record:")
        st.json(runs[0]["run"])
        st.download_button("Download time series CSV", ts.to_csv(index=False), "dsdvc_simulation.csv")

    # ---------------- Forecast
    with T["Forecast"]:
        if daily is None:
            unavailable("Daily metrics", "Upload ≥28 days of observed daily metrics (adoptions, engagement, shares, conversions, churn, revenue).")
        else:
            badge("PREDICTED", "observed history" if daily_up is not None else "SYNTHETIC history")
            from dsdvc.forecasting.early_warning import scan
            from dsdvc.forecasting.forecast import Forecaster
            from dsdvc.forecasting.regime import detect_regime
            F = Forecaster()
            st.dataframe(F.forecast_frame(daily).round(1), width="stretch")
            m = st.selectbox("Path", [c for c in ("adoptions", "engagement", "shares", "conversions", "churn", "revenue") if c in daily])
            fr = F.forecast(daily[m].to_numpy(float), m)
            hist = daily[[m]].reset_index(drop=True)
            x0 = len(hist)
            fig = go.Figure([go.Scatter(x=list(range(x0)), y=hist[m], mode="lines", name="history", line=dict(color="#52514e", width=2)),
                             go.Scatter(x=list(range(x0, x0 + len(fr.path))), y=fr.path.upper, line=dict(width=0), showlegend=False),
                             go.Scatter(x=list(range(x0, x0 + len(fr.path))), y=fr.path.lower, fill="tonexty",
                                        fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), name="90% interval"),
                             go.Scatter(x=list(range(x0, x0 + len(fr.path))), y=fr.path.forecast, mode="lines",
                                        name=f"forecast ({fr.model})", line=dict(color=SERIES[0], width=2))])
            st.plotly_chart(style(fig.update_layout(title=f"{m}: 90-day forecast")), width="stretch")
            st.dataframe(fr.model_comparison.round(3), width="stretch")
            reg = detect_regime(daily)
            st.write(f"**Regime:** {reg['regime']}")
            st.json(reg["evidence"])
            acq_by_platform = {k: v for k, v in runs[0]["summary"]["acquisition_share_by_platform"].items() if k != "external"}
            st.write("**Early warnings:**", scan(daily, conversions_by_platform=acq_by_platform) or "none")

    # ---------------- Scenarios
    with T["Scenarios"]:
        badge("SIMULATED", "paired simulation with common random numbers; Baseline / Treatment / Difference / CI")
        from dsdvc.experiments.counterfactual import SCENARIO_LIBRARY, run_counterfactual
        q = st.selectbox("Question", list(SCENARIO_LIBRARY), format_func=lambda k: SCENARIO_LIBRARY[k].question)
        if st.button("Run counterfactual"):
            from dsdvc.diffusion.abm import SimConfig
            cfg = SimConfig(**dict(cfg_items))
            cfg.cost_model.population_scale = pop_scale
            with st.spinner("Simulating…"):
                r = run_counterfactual(cfg, q, n_reps=max(n_seeds, 4))
            st.write(r["question"], r["interventions"])
            st.dataframe(r["table"].round(3), width="stretch")
            st.caption(r["caveat"])

    # ---------------- Attribution
    with T["Attribution"]:
        paths = journeys if journeys is not None else runs[0]["paths"]
        badge("OBSERVED" if journeys_up is not None else ("SYNTHETIC" if journeys is not None else "SIMULATED"),
              "Attributed conversions are NOT incremental conversions")
        from dsdvc.attribution.models import all_attributions, incrementality_test
        att = all_attributions(paths)
        m = att.reset_index(names="channel").melt("channel", var_name="model", value_name="attributed")
        st.plotly_chart(style(px.bar(m, x="channel", y="attributed", color="model", barmode="group",
                                     color_discrete_sequence=SERIES, title="Attributed conversions by model")),
                        width="stretch")
        st.subheader("Incrementality (experiment)")
        c = st.columns(4)
        tc, tn = c[0].number_input("Treatment conversions", 0, 10**9, 520), c[1].number_input("Treatment n", 1, 10**9, 20000)
        cc_, cn = c[2].number_input("Control conversions", 0, 10**9, 430), c[3].number_input("Control n", 1, 10**9, 20000)
        st.json(incrementality_test(tc, tn, cc_, cn))

    # ---------------- Calibration
    with T["Calibration"]:
        from dsdvc.calibration.engine import CalibrationEngine
        if daily is None or "adoptions" not in daily:
            unavailable("Observed adoption series", "Parameters remain PRIORS. Upload daily adoptions to calibrate Bass p, q, m.")
            badge("PRIOR")
        else:
            badge("ESTIMATED" if daily_up is not None else "SYNTHETIC")
            e = CalibrationEngine("dashboard")
            y = daily.adoptions.to_numpy(float)
            st.dataframe(pd.concat([e.bass_nls(y).table(), e.bass_mle(y).table(), e.bass_bayesian(y, 3000, 1000).table()]).round(5),
                         width="stretch")
        st.subheader("Platform rate posteriors (Beta-Binomial)")
        from pathlib import Path
        sample = Path(__file__).resolve().parents[3] / "legacy" / "dsdvc_mp_v3" / "sample_platform_observations.csv"
        if sample.exists():
            badge("SYNTHETIC", "legacy v3 sample observations (illustrative)")
            st.dataframe(CalibrationEngine().platform_rates(pd.read_csv(sample)).round(5), width="stretch")

    # ---------------- Validation
    with T["Validation"]:
        st.write("The complex model must earn its complexity out-of-sample.")
        if daily is not None and "adoptions" in daily and st.button("Run adoption benchmark (rolling origin)"):
            from dsdvc.diffusion.abm import SimConfig
            from dsdvc.validation.benchmarks import adoption_benchmark
            st.dataframe(adoption_benchmark(daily.adoptions.to_numpy(float),
                                            abm_cfg=SimConfig(n_agents=600, steps=len(daily), rewire_every=0)).round(3))
        if st.button("Run ablation report (Bass → +Networks → … → +Churn)"):
            from dsdvc.diffusion.abm import SimConfig, run_simulation
            from dsdvc.validation.benchmarks import ablation_report
            target = daily.adoptions.to_numpy(float)[:90] if daily is not None and "adoptions" in daily else \
                run_simulation(SimConfig(n_agents=600, steps=60, seed=123, rewire_every=0)).timeseries.new_adoptions.to_numpy()
            badge("OBSERVED" if daily_up is not None else "SYNTHETIC", "target series")
            with st.spinner("Calibrating each variant…"):
                st.dataframe(ablation_report(target, SimConfig(n_agents=600, rewire_every=0), grid=np.linspace(-5.5, -2.5, 5),
                                             seeds=(0,)).round(3))
        if vp is not None:
            st.write("Virality benchmarks:")
            st.dataframe(vp.report["benchmarks"].round(4))

    # ---------------- Research agent
    with T["Research Agent"]:
        req = st.text_area("Research request", "I want to launch a premium skincare brand in India targeting women 18-30.")
        b = st.number_input("Campaign budget", 10_000.0, 1e9, 600_000.0, 10_000.0)
        if st.button("Run research plan"):
            from dsdvc.research_agent.agent import MarketingResearchAgent, ResearchInputs
            from dsdvc.research_agent.report import render_markdown
            inputs = ResearchInputs(creators=creators_up, content=content_up, daily=daily_up, customer_panel=panel_up, budget=b)
            with st.spinner("Executing 14-step research plan…"):
                mem, S = MarketingResearchAgent(allow_synthetic=synthetic, n_agents=min(n_agents, 1200), reps=max(2, min(n_seeds, 4))).run(req, inputs)
            st.session_state["report"] = render_markdown(mem, S)
            st.session_state["plan"] = pd.DataFrame(mem.plan)
        if "plan" in st.session_state:
            st.dataframe(st.session_state["plan"], width="stretch")

    with T["Reports"]:
        if "report" in st.session_state:
            st.download_button("Download report (Markdown)", st.session_state["report"], "dsdvc_report.md")
            st.markdown(st.session_state["report"])
        else:
            st.info("Run the Research Agent to generate a report.")


if __name__ == "__main__":
    main()
