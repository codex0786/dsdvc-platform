"""Estimated recommendation / exposure model (spec §10).

We do NOT have, and do not claim, access to Instagram's or Facebook's proprietary ranking
systems. This module is an *estimated exposure model*: a transparent logistic score whose
inputs mirror publicly described ranking signals (relevance, predicted engagement, social
proof, creator affinity, similarity, freshness, prior engagement). Its coefficients should
be calibrated to observed reach (e.g. non-follower reach share from authorised Insights).

    P(algorithmic exposure_ic) = reach_p * sigmoid(b0 + b1 Relevance + b2 PredEngagement
                                   + b3 SocialProof + b4 CreatorAffinity + b5 Similarity
                                   + b6 Freshness + b7 PreviousEngagement)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class ExposureCoefficients:
    intercept: float = -4.0
    relevance: float = 2.2
    predicted_engagement: float = 1.6
    social_proof: float = 0.9
    creator_affinity: float = 0.8
    similarity: float = 0.6
    freshness: float = 1.0
    previous_engagement: float = 0.7


PLATFORM_DEFAULTS = {
    # algorithmic share of distribution (prior, to be calibrated)
    "instagram": ExposureCoefficients(),
    "facebook": ExposureCoefficients(intercept=-4.6, relevance=1.8, social_proof=1.2, freshness=0.8),
    "whatsapp": None,  # no algorithmic feed: exposure is purely social / forwarded
}


class EstimatedExposureModel:
    label = "Estimated recommendation/exposure model (not the platform's proprietary algorithm)"

    def __init__(self, platform: str, coefs: ExposureCoefficients | None = None, reach: float = 1.0):
        self.platform = platform
        self.coefs = coefs if coefs is not None else PLATFORM_DEFAULTS.get(platform)
        self.reach = reach

    @property
    def enabled(self) -> bool:
        return self.coefs is not None and self.reach > 0

    def probability(self, relevance, predicted_engagement, social_proof, creator_affinity, similarity,
                    freshness, previous_engagement) -> np.ndarray:
        if not self.enabled:
            return np.zeros_like(np.asarray(relevance, dtype=float))
        c = self.coefs
        z = (c.intercept + c.relevance * relevance + c.predicted_engagement * predicted_engagement
             + c.social_proof * social_proof + c.creator_affinity * creator_affinity + c.similarity * similarity
             + c.freshness * freshness + c.previous_engagement * previous_engagement)
        return self.reach * _sig(z)

    def calibrate_intercept(self, target_rate: float, **inputs) -> float:
        """Adjust the intercept so mean exposure probability matches an observed rate
        (e.g. non-follower reach / audience size from authorised Insights)."""
        from scipy.optimize import brentq
        if not self.enabled:
            return float("nan")
        base = self.coefs.intercept

        def f(b):
            self.coefs.intercept = b
            return float(np.mean(self.probability(**inputs))) - target_rate
        try:
            self.coefs.intercept = brentq(f, base - 15, base + 15)
        except ValueError:
            self.coefs.intercept = base
        return self.coefs.intercept
