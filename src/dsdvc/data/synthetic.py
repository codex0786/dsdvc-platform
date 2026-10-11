"""SYNTHETIC data generators with known ground truth.

Used for (a) tests that verify estimators recover planted parameters, (b) demos when no
authorised data exists and the user explicitly selects synthetic mode (spec §13).
Every frame carries ``data_status = "synthetic"``; never present these as observations.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from dsdvc.content.stepps import DIMENSIONS

PHRASES = {
    "social_currency": ["exclusive first look", "the secret nobody tells you", "limited drop for insiders",
                        "only a few know this", "early access unlocked", "rare find"],
    "triggers": ["your morning routine", "every monday", "this monsoon", "before bedtime", "diwali prep",
                 "for the daily commute", "weekend reset"],
    "emotion": ["omg this is insane!", "honestly stunning", "I was so anxious", "mind-blowing results!",
                "furious about fake reviews", "absolutely thrilled", "can't believe it"],
    "public": ["tag a friend", "join the challenge", "#glowchallenge", "show us yours", "repost if you agree",
               "everyone is wearing this"],
    "practical_value": ["5 tips under ₹500", "step by step guide", "save this routine", "honest review vs dupes",
                        "how to layer spf", "20% off ingredients list"],
    "stories": ["my journey with acne", "when I finally found it", "before and after 3 years",
                "I used to struggle", "my mom's remedy story", "the day I realised"],
}
FILLER = ["new launch", "skincare", "glow", "today", "product", "look", "vibes", "serum", "for you", "check"]
CATEGORIES = ["skincare", "haircare", "makeup", "wellness"]
FORMATS = ["reel", "image", "carousel", "video", "text"]
PLATFORM_LIST = ["instagram", "facebook", "whatsapp"]

# Ground-truth generative coefficients (documented so tests can check recovery).
TRUE_SHARE_BETA = np.array([0.6, 0.3, 1.1, 0.5, 0.7, 0.6])     # S, T, E, P, PV, St
TRUE_EMOTION_PUBLIC_INTERACTION = 1.2
TRUE_CONVERSION_BETA = {"practical_value": 1.2, "stories": 0.4, "emotion": 0.1, "cta": 0.8, "trust": 1.5}


def _sig(x):
    return 1 / (1 + np.exp(-x))


def synthetic_creators(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    followers = np.round(np.exp(rng.normal(9.5, 1.6, n))).astype(int) + 200
    platform = rng.choice(PLATFORM_LIST[:2], n, p=[0.7, 0.3])
    size_penalty = np.clip(np.log10(followers) - 4, 0, 3)
    eng = np.clip(rng.lognormal(np.log(0.045), 0.5, n) * (1 - 0.18 * size_penalty), 0.002, 0.3)
    trust = np.clip(rng.beta(4, 3, n) - 0.08 * size_penalty, 0.05, 0.98)
    audience_quality = np.clip(rng.beta(6, 2, n), 0.1, 1)  # 1 - bot/fake share
    niche = rng.choice(CATEGORIES, n)
    posts = rng.poisson(14, n) + 1
    return pd.DataFrame({
        "creator_id": [f"cr_{i:04d}" for i in range(n)],
        "handle": [f"synthetic_creator_{i:04d}" for i in range(n)],
        "platform": platform, "followers": followers,
        "following": np.round(followers * rng.uniform(0.002, 0.6, n)).astype(int) + 50,
        "true_engagement_rate": eng, "true_trust": trust, "audience_quality": audience_quality,
        "niche": niche, "posts_last_30d": posts, "community": rng.integers(0, 8, n),
        "data_status": "synthetic",
    })


def synthetic_content(n: int = 3000, creators: pd.DataFrame | None = None, seed: int = 0,
                      start: datetime | None = None) -> pd.DataFrame:
    """Content corpus with planted STEPPS -> sharing and practical/trust -> conversion effects."""
    rng = np.random.default_rng(seed)
    creators = creators if creators is not None else synthetic_creators(300, seed + 1)
    start = start or datetime(2025, 1, 1, tzinfo=timezone.utc)
    c_idx = rng.integers(0, len(creators), n)
    cr = creators.iloc[c_idx].reset_index(drop=True)
    z = rng.beta(1.6, 2.2, (n, 6))
    captions = []
    for i in range(n):
        parts = []
        for d, dim in enumerate(DIMENSIONS):
            k = rng.binomial(3, z[i, d])
            parts += list(rng.choice(PHRASES[dim], size=k, replace=True))
        parts += list(rng.choice(FILLER, size=rng.integers(2, 6)))
        rng.shuffle(parts)
        cta = rng.random() < 0.35
        if cta:
            parts.append(rng.choice(["shop now link in bio", "use code GLOW10", "dm to order"]))
        captions.append(" ".join(parts))
    fmt = rng.choice(FORMATS, n, p=[0.4, 0.2, 0.2, 0.1, 0.1])
    platform = np.where(rng.random(n) < 0.85, cr.platform.values, "whatsapp")
    ts = [start + timedelta(hours=float(h)) for h in np.sort(rng.uniform(0, 24 * 540, n))]
    fmt_boost = pd.Series(fmt).map({"reel": 0.6, "video": 0.3, "carousel": 0.2, "image": 0.0, "text": -0.4}).values
    plat_reach = pd.Series(platform).map({"instagram": 0.35, "facebook": 0.18, "whatsapp": 0.05}).values
    algo = np.exp(0.8 * z[:, 2] * z[:, 3] + fmt_boost * (platform == "instagram") + rng.normal(0, 0.6, n))
    views = np.maximum(10, cr.followers.values * plat_reach * algo).round().astype(int)
    lin = (-4.6 + z @ TRUE_SHARE_BETA + TRUE_EMOTION_PUBLIC_INTERACTION * z[:, 2] * z[:, 3]
           + 1.0 * (cr.true_trust.values - 0.5) + pd.Series(platform).map(
               {"instagram": 0.0, "facebook": 0.2, "whatsapp": 0.9}).values + rng.normal(0, 0.45, n))
    share_rate = _sig(lin)
    shares = rng.binomial(views, np.clip(share_rate, 0, 1))
    likes = rng.binomial(views, np.clip(cr.true_engagement_rate.values * (0.7 + 0.6 * z[:, 2]), 0, 0.9))
    comments = rng.binomial(views, np.clip(0.004 + 0.01 * z[:, 5] + 0.01 * z[:, 3], 0, 1))
    saves = rng.binomial(views, np.clip(0.002 + 0.03 * z[:, 4], 0, 1))
    cta_flag = np.array(["shop now" in c or "code" in c or "dm to" in c for c in captions], dtype=float)
    conv_lin = (-6.2 + TRUE_CONVERSION_BETA["practical_value"] * z[:, 4] + TRUE_CONVERSION_BETA["stories"] * z[:, 5]
                + TRUE_CONVERSION_BETA["emotion"] * z[:, 2] + TRUE_CONVERSION_BETA["cta"] * cta_flag
                + TRUE_CONVERSION_BETA["trust"] * cr.true_trust.values
                + pd.Series(platform).map({"instagram": 0.0, "facebook": 0.1, "whatsapp": 0.9}).values
                + rng.normal(0, 0.3, n))
    conv_rate = _sig(conv_lin)
    conversions = rng.binomial(views, conv_rate)
    retained_90d = rng.binomial(np.maximum(conversions, 0), np.clip(_sig(0.3 + 1.2 * cr.true_trust.values + 0.8 * z[:, 4]), 0, 1))
    # early dynamics: cumulative shares at 1, 3, 6, 12, 24h - a heterogeneous growth curve
    tau = np.exp(rng.normal(1.8, 0.5, n))  # hours to ~63% of day-1 shares
    hours = np.array([1, 3, 6, 12, 24])
    frac = 1 - np.exp(-hours[None, :] / tau[:, None])
    frac /= frac[:, -1:]
    day1 = rng.binomial(shares, 0.55)
    early = (day1[:, None] * frac).round()
    df = pd.DataFrame({
        "content_id": [f"ct_{i:05d}" for i in range(n)],
        "platform": platform, "creator_id": cr.creator_id.values, "followers": cr.followers.values,
        "creator_trust": cr.true_trust.values, "creator_engagement_rate": cr.true_engagement_rate.values,
        "timestamp": pd.to_datetime(ts), "text": captions, "media_type": fmt,
        "category": cr.niche.values, "country": "IN", "language": "en",
        "views": views, "likes": likes, "comments": comments, "shares": shares, "saves": saves,
        "conversions": conversions, "retained_90d": retained_90d, "cta": cta_flag,
        "data_status": "synthetic",
    })
    for j, h in enumerate(hours):
        df[f"shares_{h}h"] = early[:, j]
    for d, dim in enumerate(DIMENSIONS):
        df[f"true_{dim}"] = z[:, d]
    df["engagement"] = df[["likes", "comments", "shares", "saves"]].sum(axis=1)
    return df


CHANNELS = ["ig_organic", "ig_creator", "ig_paid", "fb_group", "fb_paid", "wa_referral", "search", "email"]
TRUE_CHANNEL_LIFT = {"ig_organic": 0.010, "ig_creator": 0.030, "ig_paid": 0.015, "fb_group": 0.020,
                     "fb_paid": 0.008, "wa_referral": 0.060, "search": 0.025, "email": 0.005}


def synthetic_journeys(n: int = 20000, seed: int = 0, baseline: float = 0.01) -> pd.DataFrame:
    """Touchpoint paths with known per-channel incremental effects (for attribution tests)."""
    rng = np.random.default_rng(seed)
    p_touch = np.array([0.5, 0.25, 0.35, 0.2, 0.25, 0.12, 0.3, 0.2])
    rows = []
    for u in range(n):
        k = rng.integers(1, 6)
        path = list(rng.choice(CHANNELS, size=k, p=p_touch / p_touch.sum()))
        p_no = (1 - baseline) * np.prod([1 - TRUE_CHANNEL_LIFT[c] for c in set(path)])
        conv = rng.random() < 1 - p_no
        rows.append({"user_id": u, "path": path, "converted": bool(conv),
                     "revenue": float(rng.gamma(4, 300)) if conv else 0.0})
    df = pd.DataFrame(rows)
    df["data_status"] = "synthetic"
    return df


def synthetic_customer_panel(n: int = 8000, seed: int = 0) -> pd.DataFrame:
    """Prospect/customer panel with platform-specific conversion and churn (spec §25, §28)."""
    rng = np.random.default_rng(seed)
    platform = rng.choice(PLATFORM_LIST, n, p=[0.5, 0.3, 0.2])
    X = {
        "awareness": rng.beta(2, 2, n), "social_proof": rng.beta(2, 3, n), "trust": rng.beta(3, 2, n),
        "price_fit": rng.beta(2, 2, n), "stepps": rng.beta(2, 2, n), "community_fit": rng.beta(2.5, 2, n),
        "product_fit": rng.beta(3, 2, n),
    }
    plat_trust_beta = pd.Series(platform).map({"instagram": 1.0, "facebook": 1.4, "whatsapp": 2.6}).values
    lin = (-4.0 + 1.2 * X["awareness"] + 1.5 * X["social_proof"] + plat_trust_beta * X["trust"]
           + 1.3 * X["price_fit"] + 0.6 * X["stepps"] + 0.8 * X["community_fit"] + 1.6 * X["product_fit"])
    converted = rng.random(n) < _sig(lin)
    acq_channel = np.where(converted, platform, None)
    competition = rng.beta(2, 4, n)
    price_pressure = rng.beta(2, 4, n)
    engagement = np.clip(rng.beta(2, 2, n) + 0.2 * X["community_fit"], 0, 1)
    dissatisfaction = np.clip(1 - X["product_fit"] + rng.normal(0, 0.15, n), 0, 1)
    churn_lin = (-2.2 + 2.0 * dissatisfaction + 1.0 * competition + 1.2 * price_pressure
                 - 1.4 * engagement - 1.0 * X["community_fit"]
                 + pd.Series(platform).map({"instagram": 0.3, "facebook": 0.0, "whatsapp": -0.4}).values)
    churned = converted & (rng.random(n) < _sig(churn_lin))
    months = np.where(converted, np.where(churned, rng.integers(1, 6, n), 12), 0)
    arpu = rng.gamma(5, 120, n)
    df = pd.DataFrame({**X, "platform": platform, "converted": converted, "acquisition_channel": acq_channel,
                       "competition": competition, "price_pressure": price_pressure, "engagement": engagement,
                       "dissatisfaction": dissatisfaction, "churned": churned, "months_active": months,
                       "monthly_revenue": np.where(converted, arpu, 0.0),
                       "community": rng.integers(0, 8, n),
                       "influencer": np.where(converted, rng.choice([f"cr_{i:04d}" for i in range(20)], n), None),
                       "data_status": "synthetic"})
    return df


def synthetic_daily_metrics(days: int = 180, seed: int = 0, p: float = 0.0015, q: float = 0.045,
                            m: float = 50000, breakpoint: int | None = 120) -> pd.DataFrame:
    """Daily adoption/engagement/share/revenue series following Bass with an optional regime shift."""
    from dsdvc.diffusion.bass import bass_cdf
    rng = np.random.default_rng(seed)
    t = np.arange(1, days + 1)
    mean = m * (bass_cdf(t, p, q) - bass_cdf(t - 1, p, q))
    if breakpoint:
        mean[breakpoint:] *= 0.55  # e.g. competitor entry
    weekly = 1 + 0.12 * np.sin(2 * np.pi * t / 7)
    adoptions = rng.poisson(mean * weekly)
    shares = rng.poisson(adoptions * 2.5 + 30)
    engagement = rng.poisson(adoptions * 18 + 400)
    churn = rng.poisson(np.cumsum(adoptions) * 0.004)
    revenue = adoptions * 899 + np.cumsum(adoptions - churn).clip(0) * 12
    return pd.DataFrame({"day": t, "date": pd.date_range("2026-01-01", periods=days, freq="D"),
                         "adoptions": adoptions, "shares": shares, "engagement": engagement,
                         "churn": churn, "conversions": adoptions, "revenue": revenue.astype(float),
                         "data_status": "synthetic"})
