"""Influencer scoring and ranking (spec §7, §15, §48).

Influence is NOT proportional to followers (De Veirman et al. 2017). We keep separate:

  audience size        (OBSERVED: followers)
  audience engagement  (OBSERVED: engagement rate on recent posts)
  audience relevance   (ESTIMATED: niche/keyword/topic match)
  audience trust       (ESTIMATED proxy: comment/like ratio, saves, consistency, size penalty)
  network position     (ESTIMATED: community + bridge centrality when a graph is available)
  conversion ability   (ESTIMATED/PREDICTED: expected incremental conversions)

Ranking objective:
    score = E[IncrementalConversions] x E[LTV] / E[Cost]
with a Monte Carlo interval from the uncertain inputs. Every column is tagged in
``COLUMN_STATUS`` as observed / estimated / predicted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dsdvc.agents.seeding import CostModel
from dsdvc.core.literature import cites

COLUMN_STATUS = {
    "followers": "observed", "following": "observed", "follower_following_ratio": "observed",
    "engagement_rate": "observed", "posting_frequency": "observed",
    "audience_quality": "estimated", "content_consistency": "estimated", "niche_relevance": "estimated",
    "audience_overlap_with_brand": "estimated", "community_centrality": "estimated",
    "bridge_centrality": "estimated", "trust_proxy": "estimated", "brand_fit": "estimated",
    "estimated_influence": "estimated", "expected_reach": "predicted",
    "expected_incremental_conversions": "predicted", "expected_ltv": "predicted",
    "estimated_cost": "estimated", "estimated_cac": "predicted", "roi_score": "predicted",
    "roi_score_low": "predicted", "roi_score_high": "predicted",
}


@dataclass
class RankingAssumptions:
    reach_rate_by_platform: dict | None = None    # share of followers reached per post
    click_rate: float = 0.012
    base_conversion_rate: float = 0.03            # conversions per click for a relevant, trusted creator
    incrementality: float = 0.6                   # share of attributed conversions that are incremental
    incrementality_sd: float = 0.15
    ltv: float = 2500.0
    ltv_sd: float = 600.0
    posts: int = 3

    def reach_rate(self, platform: str) -> float:
        d = self.reach_rate_by_platform or {"instagram": 0.25, "facebook": 0.10, "whatsapp": 0.6}
        return d.get(platform, 0.15)


def _minmax(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 0 else pd.Series(0.5, index=x.index)


@cites("deveirman2017", "watts_dodds2007", "aral_walker2012")
def score_creators(creators: pd.DataFrame, category: str | None = None, keywords: list[str] | None = None,
                   content: pd.DataFrame | None = None, centrality: pd.DataFrame | None = None,
                   assumptions: RankingAssumptions | None = None, costs: CostModel | None = None,
                   draws: int = 2000, seed: int = 0) -> pd.DataFrame:
    """creators: creator_id, platform, followers, following, [engagement_rate | true_engagement_rate],
    [posts_last_30d], [niche], [audience_quality]. content (optional): creator_id, text, shares, views.
    centrality (optional): creator_id, community_centrality, bridge_centrality."""
    a = assumptions or RankingAssumptions()
    costs = costs or CostModel(population_scale=1.0)
    rng = np.random.default_rng(seed)
    df = creators.copy()
    if "engagement_rate" not in df:
        df["engagement_rate"] = df.get("true_engagement_rate", np.nan)
    df["follower_following_ratio"] = df["followers"] / df["following"].clip(lower=1)
    df["posting_frequency"] = df.get("posts_last_30d", pd.Series(np.nan, index=df.index)) / 30
    df["audience_quality"] = df.get("audience_quality", pd.Series(0.8, index=df.index)).fillna(0.8)
    kw = [k.lower() for k in (keywords or [])] + ([category.lower()] if category else [])
    if "niche" in df and kw:
        df["niche_relevance"] = df["niche"].fillna("").str.lower().apply(lambda s: 1.0 if any(k in s for k in kw) else 0.3)
    else:
        df["niche_relevance"] = 0.5
    df["content_consistency"] = 0.5
    stepps_cols = []
    if content is not None and len(content):
        from dsdvc.content.stepps import DIMENSIONS, STEPPSAnalyzer
        an = STEPPSAnalyzer()
        prof = content.copy()
        vec = np.array([an.analyze(t).vector() for t in prof["text"].fillna("")])
        for k, d in enumerate(DIMENSIONS):
            prof[f"stepps_{d}"] = vec[:, k]
        agg = prof.groupby("creator_id")[[f"stepps_{d}" for d in DIMENSIONS]].mean()
        df = df.merge(agg, left_on="creator_id", right_index=True, how="left")
        stepps_cols = [f"stepps_{d}" for d in DIMENSIONS]
        if kw:
            rel = prof.assign(hit=prof["text"].str.lower().apply(lambda s: float(any(k in s for k in kw))))
            df = df.merge(rel.groupby("creator_id")["hit"].mean().rename("kw_rate"), left_on="creator_id",
                          right_index=True, how="left")
            df["niche_relevance"] = 0.5 * df["niche_relevance"] + 0.5 * df["kw_rate"].fillna(0)
        # consistency = 1 - dispersion of STEPPS profile across a creator's posts
        disp = prof.groupby("creator_id")[stepps_cols].std().mean(axis=1)
        df = df.merge((1 - disp.clip(0, 1)).rename("cc"), left_on="creator_id", right_index=True, how="left")
        df["content_consistency"] = df["cc"].fillna(0.5)
    if centrality is not None:
        df = df.merge(centrality, on="creator_id", how="left")
    df["community_centrality"] = df.get("community_centrality", pd.Series(np.nan, index=df.index))
    df["bridge_centrality"] = df.get("bridge_centrality", pd.Series(np.nan, index=df.index))
    size_pen = np.clip(np.log10(df["followers"].clip(lower=1)) - 5, 0, 2)  # >100k: trust/relevance dilution prior
    eng = df["engagement_rate"].fillna(df["engagement_rate"].median())
    df["trust_proxy"] = (0.45 * _minmax(eng) + 0.25 * df["audience_quality"] + 0.30 * df["content_consistency"]
                         - 0.12 * size_pen).clip(0, 1)
    df["brand_fit"] = (0.6 * df["niche_relevance"] + 0.4 * df["content_consistency"]).clip(0, 1)
    netpos = df[["community_centrality", "bridge_centrality"]].apply(_minmax).mean(axis=1).fillna(0.5)
    df["estimated_influence"] = (0.20 * _minmax(np.log1p(df["followers"])) + 0.20 * _minmax(eng)
                                 + 0.20 * df["niche_relevance"] + 0.25 * df["trust_proxy"] + 0.15 * netpos)
    reach_rate = df["platform"].map(a.reach_rate).astype(float)
    df["expected_reach"] = df["followers"] * df["audience_quality"] * reach_rate * a.posts
    conv_per_reach = a.click_rate * (0.5 + eng.clip(0, 0.2) * 5) * a.base_conversion_rate * \
        (0.4 + 0.6 * df["trust_proxy"]) * (0.3 + 0.7 * df["niche_relevance"])
    attributed = df["expected_reach"] * conv_per_reach
    df["estimated_cost"] = df["followers"].apply(lambda f: costs.creator_fee(f) * a.posts / 2)
    inc = np.clip(rng.normal(a.incrementality, a.incrementality_sd, (draws, 1)), 0.05, 1)
    ltv = np.clip(rng.normal(a.ltv, a.ltv_sd, (draws, 1)), 0, None)
    conv_draw = rng.poisson(np.maximum(attributed.to_numpy()[None, :], 1e-9) * np.ones((draws, 1))) * inc
    cost_draw = df["estimated_cost"].to_numpy()[None, :] * rng.lognormal(0, 0.25, (draws, 1))
    roi = conv_draw * ltv / cost_draw
    df["expected_incremental_conversions"] = attributed * a.incrementality
    df["expected_ltv"] = a.ltv
    df["estimated_cac"] = df["estimated_cost"] / df["expected_incremental_conversions"].clip(lower=1e-6)
    df["roi_score"] = df["expected_incremental_conversions"] * a.ltv / df["estimated_cost"]
    df["roi_score_low"], df["roi_score_high"] = np.quantile(roi, 0.05, axis=0), np.quantile(roi, 0.95, axis=0)
    df["rank_by_roi"] = df["roi_score"].rank(ascending=False, method="first").astype(int)
    df["rank_by_followers"] = df["followers"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("roi_score", ascending=False).reset_index(drop=True)


def audience_overlap(follower_sets: dict[str, set]) -> pd.DataFrame:
    """Pairwise Jaccard overlap of (authorised/aggregate) audience sets."""
    ids = list(follower_sets)
    M = np.zeros((len(ids), len(ids)))
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            A, B = follower_sets[a], follower_sets[b]
            M[i, j] = len(A & B) / max(1, len(A | B))
    return pd.DataFrame(M, index=ids, columns=ids)
