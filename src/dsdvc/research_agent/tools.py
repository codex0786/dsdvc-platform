"""Research-agent tool interfaces (spec §38).

Every tool returns a ``ToolResult`` with a status of OK / DATA_UNAVAILABLE / SYNTHETIC and
provenance. No tool fabricates platform data: synthetic data is produced only when the
caller explicitly sets ``allow_synthetic=True`` and is labelled as such.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dsdvc.core.provenance import DataUnavailable, unavailable


@dataclass
class ToolResult:
    tool: str
    status: str                      # OK | DATA_UNAVAILABLE | SYNTHETIC
    data: Any = None
    provenance: dict = field(default_factory=dict)
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("OK", "SYNTHETIC")


def _unavail(tool: str, u: DataUnavailable) -> ToolResult:
    return ToolResult(tool, "DATA_UNAVAILABLE", None, {}, f"{u.reason}. {u.remedy or ''}".strip())


def search_web(query: str, count: int = 10, session=None) -> ToolResult:
    """Web search via a configured provider (Brave Search API: BRAVE_SEARCH_API_KEY)."""
    key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if not key:
        return _unavail("search_web", unavailable("search_web", "no web-search provider configured",
                                                  "Set BRAVE_SEARCH_API_KEY, or attach market reports via search_market_data(paths=...)."))
    import requests
    s = session or requests.Session()
    try:
        r = s.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": count},
                  headers={"X-Subscription-Token": key, "Accept": "application/json"}, timeout=20)
        r.raise_for_status()
        hits = [{"title": w.get("title"), "url": w.get("url"), "snippet": w.get("description")}
                for w in r.json().get("web", {}).get("results", [])]
    except Exception as e:  # noqa: BLE001
        return _unavail("search_web", unavailable("search_web", f"search failed: {type(e).__name__}"))
    return ToolResult("search_web", "OK", hits, {"source": "brave_search_api", "query": query},
                      "Search results are leads to verify, not evidence by themselves.")


def search_market_data(paths: list[str] | None = None) -> ToolResult:
    """Load user-supplied market data (CSV with columns metric, value, unit, source, year)."""
    if not paths:
        return _unavail("search_market_data", unavailable("market_data", "no market data supplied",
                                                          "Provide market-size/penetration tables with cited sources."))
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return ToolResult("search_market_data", "OK", df, {"source": "user_provided", "files": paths})


def search_public_social_data(hashtags: list[str]) -> ToolResult:
    from dsdvc.platforms.meta import InstagramGraphConnector
    ig = InstagramGraphConnector()
    out = []
    for h in hashtags[:30]:  # Instagram limits unique hashtag queries
        r = ig.hashtag_top_media(h)
        if isinstance(r, DataUnavailable):
            return _unavail("search_public_social_data", r)
        out.append(r.assign(hashtag=h))
    return ToolResult("search_public_social_data", "OK", pd.concat(out, ignore_index=True),
                      {"source": "instagram_graph_api.hashtag", "scope": "public hashtag media"})


def fetch_authorized_platform_data(platform: str, user_export: str | None = None, attestation: str | None = None) -> ToolResult:
    from dsdvc.platforms import meta
    if user_export:
        if not attestation:
            return ToolResult("fetch_authorized_platform_data", "DATA_UNAVAILABLE",
                              message="An authorisation attestation is required for user-provided exports.")
        return ToolResult("fetch_authorized_platform_data", "OK", meta.load_user_export(user_export, platform, attestation),
                          {"source": "user_export", "authorization_scope": attestation})
    conn = {"instagram": lambda: meta.InstagramGraphConnector().own_media(),
            "facebook": lambda: meta.FacebookPageConnector().page_posts()}.get(platform)
    if platform == "whatsapp":
        from datetime import datetime, timedelta, timezone
        end = datetime.now(timezone.utc)
        r = meta.WhatsAppBusinessConnector().message_analytics(end - timedelta(days=90), end)
    elif conn:
        r = conn()
    else:
        return ToolResult("fetch_authorized_platform_data", "DATA_UNAVAILABLE", message=f"unknown platform {platform}")
    if isinstance(r, DataUnavailable):
        return _unavail("fetch_authorized_platform_data", r)
    return ToolResult("fetch_authorized_platform_data", "OK", r, {"source": platform, "method": "official API"})


def discover_influencers(category: str, keywords: list[str], creators: pd.DataFrame | None = None,
                         usernames: list[str] | None = None, content: pd.DataFrame | None = None,
                         allow_synthetic: bool = False, seed: int = 0) -> ToolResult:
    """Influencer Discovery Engine. Candidate sources, in order: user-supplied candidate table,
    official business_discovery lookups for named public professional accounts, synthetic (opt-in)."""
    from dsdvc.networks.influencers import COLUMN_STATUS, score_creators
    status = "OK"
    prov = {}
    if creators is None and usernames:
        from dsdvc.platforms.meta import InstagramGraphConnector
        ig = InstagramGraphConnector()
        rows = []
        for u in usernames:
            bd = ig.business_discovery(u)
            if isinstance(bd, DataUnavailable):
                return _unavail("discover_influencers", bd)
            media = (bd.get("media") or {}).get("data", [])
            eng = np.mean([(m.get("like_count", 0) + m.get("comments_count", 0)) for m in media]) if media else np.nan
            rows.append({"creator_id": bd["username"], "handle": bd["username"], "platform": "instagram",
                         "followers": bd.get("followers_count"), "following": bd.get("follows_count"),
                         "engagement_rate": eng / max(bd.get("followers_count") or 1, 1), "niche": category})
        creators = pd.DataFrame(rows)
        prov = {"source": "instagram_graph_api.business_discovery"}
    if creators is None:
        if not allow_synthetic:
            return ToolResult("discover_influencers", "DATA_UNAVAILABLE",
                              message="No candidate creators: supply a candidate table, named public professional accounts "
                                      "with Graph API credentials, or explicitly allow SYNTHETIC mode.")
        from dsdvc.data.synthetic import synthetic_content, synthetic_creators
        creators = synthetic_creators(200, seed)
        content = synthetic_content(1500, creators, seed + 1)
        status, prov = "SYNTHETIC", {"source": "synthetic_creators", "note": "SYNTHETIC creators; not real accounts"}
    ranked = score_creators(creators, category=category, keywords=keywords, content=content)
    return ToolResult("discover_influencers", status, ranked, {**prov, "column_status": COLUMN_STATUS})


def analyze_content(content: pd.DataFrame, predictor=None) -> ToolResult:
    from dsdvc.content.stepps import DIMENSIONS, STEPPSAnalyzer
    from dsdvc.content.features import topic_model
    an = STEPPSAnalyzer()
    S = np.array([an.analyze(t, m).vector() for t, m in zip(content["text"].fillna(""), content.get("media_type", "image"))])
    df = content.copy()
    for k, d in enumerate(DIMENSIONS):
        df[f"stepps_{d}"] = S[:, k]
    topics = topic_model(df["text"].fillna("").tolist()) if len(df) >= 20 else {"topics": {}, "doc_topic": np.zeros(len(df))}
    df["topic"] = topics["doc_topic"]
    status = "SYNTHETIC" if (content.get("data_status") == "synthetic").any() else "OK"
    return ToolResult("analyze_content", status, {"content": df, "topics": topics["topics"]},
                      {"model": "STEPPSAnalyzer(lexicon) + NMF topics"})


def analyze_network(mx) -> ToolResult:
    from dsdvc.networks.communities import detect_communities
    from dsdvc.networks.metrics import structure_report
    out = {}
    for layer, g in mx.layers.items():
        cr = detect_communities(g, "leiden")
        out[layer] = {"structure": structure_report(g, mx.community), "detected_communities": cr.n_communities,
                      "modularity_detected": cr.modularity, "bridges": cr.bridges[:10], "per_community": cr.per_community}
    status = "SYNTHETIC" if mx.provenance and mx.provenance.observed_or_estimated.value == "synthetic" else "OK"
    return ToolResult("analyze_network", status, out, {"model": "leiden + structure metrics"})


def run_simulation(cfg) -> ToolResult:
    from dsdvc.diffusion.abm import run_simulation as rs
    r = rs(cfg)
    return ToolResult("run_simulation", "OK", r, {"run_id": r.run_record.run_id, "status": "SIMULATED"})


def run_calibration(adoptions: np.ndarray | None) -> ToolResult:
    from dsdvc.calibration.engine import CalibrationEngine
    if adoptions is None or len(adoptions) < 10:
        return ToolResult("run_calibration", "DATA_UNAVAILABLE",
                          message="No observed adoption series: parameters remain PRIORS (not calibrated).")
    e = CalibrationEngine()
    return ToolResult("run_calibration", "OK", {"nls": e.bass_nls(adoptions), "bayes": e.bass_bayesian(adoptions)},
                      {"methods": ["nls", "bayesian_mcmc"]})


def run_counterfactual(cfg, scenario: str, n_reps: int = 6) -> ToolResult:
    from dsdvc.experiments.counterfactual import run_counterfactual as rc
    r = rc(cfg, scenario, n_reps=n_reps)
    return ToolResult("run_counterfactual", "OK", r, {"status": "SIMULATED"})


def run_forecast(daily: pd.DataFrame | None) -> ToolResult:
    from dsdvc.forecasting.forecast import Forecaster
    if daily is None or len(daily) < 28:
        return ToolResult("run_forecast", "DATA_UNAVAILABLE", message="Need >= 28 days of observed daily metrics to forecast.")
    return ToolResult("run_forecast", "OK", Forecaster().forecast_frame(daily), {"model": "rolling-origin selected"})


def generate_report(memory, sections: dict) -> ToolResult:
    from dsdvc.research_agent.report import render_markdown
    return ToolResult("generate_report", "OK", render_markdown(memory, sections), {})
