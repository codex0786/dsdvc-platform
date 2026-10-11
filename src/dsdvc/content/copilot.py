"""Viral Content Copilot and content optimisation (spec §40, §41).

Input: concept / caption / script / transcript / brief. Output: STEPPS profile, calibrated
probability of top-percentile performance, conversion estimate, risk factors, recommended
platform / creator type / community / CTA, and alternative concepts that strengthen weak
STEPPS dimensions - each re-scored, with an A/B test plan.

Language rule (spec §40): never "this will go viral"; always a conditional probability.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dsdvc.attribution.models import required_sample_size
from dsdvc.content.features import LEX
from dsdvc.content.stepps import DIMENSIONS, STEPPSAnalyzer
from dsdvc.content.virality import ViralityPredictor

# Edits that add evidence for a STEPPS dimension. They are *creative prompts* for a human
# team, not auto-published copy.
DIMENSION_EDITS = {
    "social_currency": ["Frame it as an insider reveal: 'the secret step most people skip'",
                        "Offer limited early access for the first community members"],
    "triggers": ["Tie the product to a daily cue: 'your 7am routine'", "Anchor to a seasonal moment (monsoon, Diwali)"],
    "emotion": ["Open with a high-arousal moment (awe or relief) in the first 2 seconds",
                "Show a genuine before/after reaction rather than a product shot"],
    "public": ["Add a visible participation mechanic: tag a friend / show-us-yours challenge",
               "Use a recognisable visual signature people can recreate"],
    "practical_value": ["Add 3 concrete tips with a price point ('under ₹500')", "Turn it into a step-by-step routine people save"],
    "stories": ["Tell it as a first-person journey with a turning point", "Use a before -> struggle -> discovery -> after arc"],
}
EDIT_TEXT = {
    "social_currency": "exclusive first look - the secret insiders know",
    "triggers": "perfect for your morning routine this monsoon",
    "emotion": "honestly stunning, I was so thrilled!",
    "public": "tag a friend and join the challenge #glowchallenge",
    "practical_value": "5 tips under ₹500, step by step guide - save this",
    "stories": "my journey: I used to struggle, then I finally found it",
}
RISK_TERMS = {"guarantee", "cure", "permanent", "100%", "miracle", "fairness", "whitening", "instant results",
              "doctor recommended", "clinically proven"}


@dataclass
class CopilotReport:
    stepps: dict
    viral: dict | None
    conversion: dict | None
    risks: list[str]
    recommendations: dict
    variants: list[dict] = field(default_factory=list)
    ab_test: dict | None = None
    language_note: str = ("Probabilities are conditional on the historical benchmark and current data; "
                          "they are not guarantees of virality.")


class ContentCopilot:
    def __init__(self, predictor: ViralityPredictor | None = None, analyzer: STEPPSAnalyzer | None = None,
                 brand_constraints: dict | None = None):
        self.predictor = predictor
        self.analyzer = analyzer or STEPPSAnalyzer()
        self.constraints = brand_constraints or {}

    def _risks(self, text: str, profile) -> list[str]:
        tl = text.lower()
        risks = [f"Claim risk: '{t}' may breach advertising standards (e.g. ASCI) or platform policy"
                 for t in RISK_TERMS if t in tl]
        risks += [f"Brand constraint: banned term '{t}'" for t in self.constraints.get("banned_terms", []) if t.lower() in tl]
        if profile.features.controversy > 0.5:
            risks.append("Controversy cues: may raise reach but also negative sentiment and brand-safety risk")
        if profile.features.sentiment < -0.3:
            risks.append("Predominantly negative framing")
        if profile.features.cta_strength == 0:
            risks.append("No call to action: engagement may not translate into conversion")
        return risks

    def _recommend(self, prof) -> dict:
        s = prof.scores()
        if s["public"] > 0.5 and s["emotion"] > 0.4:
            platform, creator = "instagram (Reels)", "micro-creators with high engagement in the niche"
        elif s["practical_value"] > 0.5 and s["stories"] > 0.4:
            platform, creator = "facebook groups + whatsapp forwarding", "trusted community members / expert educators"
        elif s["practical_value"] > 0.5:
            platform, creator = "whatsapp (high-trust forwarding) + instagram carousel", "expert / educator creators"
        else:
            platform, creator = "instagram", "mid-tier creators; test before scaling"
        cta = "save + share with a friend" if s["practical_value"] > 0.5 else ("tag a friend" if s["public"] > 0.5 else "link in bio with a trackable code")
        community = "communities whose interests match the strongest STEPPS dimensions (use the Communities tab)"
        return {"platform": platform, "creator_type": creator, "cta": cta, "community": community,
                "weak_dimensions": prof.weakest(2)}

    def _score(self, text: str, media_type: str, meta: dict) -> tuple[dict | None, dict | None]:
        if self.predictor is None or not self.predictor.fitted:
            return None, None
        row = pd.DataFrame([{"content_id": "draft", "text": text, "media_type": media_type,
                             "platform": meta.get("platform", "instagram"), "followers": meta.get("followers", 20000),
                             "creator_engagement_rate": meta.get("creator_engagement_rate", 0.04),
                             "timestamp": pd.Timestamp.now(tz="UTC")}])
        out = self.predictor.predict(row, data_quality=meta.get("data_quality", 0.6))[0]
        conv = out.get("expected_conversion_rate")
        return out["viral"], conv

    def analyze(self, concept: str = "", caption: str = "", script: str = "", transcript: str = "", brief: str = "",
                media_type: str = "reel", meta: dict | None = None, n_variants: int = 3) -> CopilotReport:
        meta = meta or {}
        text = " ".join(x for x in (concept, caption, script, brief) if x)
        prof = self.analyzer.analyze(text, media_type=media_type, transcript=transcript or None)
        viral, conv = self._score(" ".join([text, transcript]), media_type, meta)
        variants = []
        for d in prof.weakest(n_variants):
            if any(t in EDIT_TEXT[d].lower() for t in self.constraints.get("banned_terms", [])):
                continue
            vt = f"{text} {EDIT_TEXT[d]}"
            vp = self.analyzer.analyze(vt, media_type=media_type)
            v_viral, v_conv = self._score(vt, media_type, meta)
            variants.append({"strengthen": d, "creative_directions": DIMENSION_EDITS[d], "example_copy": vt,
                             "stepps": vp.scores(), "p_viral": v_viral["prediction"] if v_viral else None,
                             "expected_conversion_rate": v_conv["prediction"] if v_conv else None,
                             "risks": self._risks(vt, vp)})
        # constrained choice: maximise expected conversions (rate) subject to no risk flags
        ok = [v for v in variants if not v["risks"] and v["expected_conversion_rate"] is not None]
        best = max(ok, key=lambda v: v["expected_conversion_rate"]) if ok else None
        ab = None
        if best and conv:
            base_rate = max(conv["prediction"], 1e-4)
            lift = max(best["expected_conversion_rate"] / base_rate - 1, 0.05)
            ab = {"control": "original", "treatment": best["strengthen"], "primary_metric": "conversion rate",
                  "secondary_metrics": ["share rate", "save rate", "90-day retention"],
                  "expected_relative_lift": lift,
                  "required_sample_per_arm": required_sample_size(base_rate, lift),
                  "note": "Randomise at audience level; analyse incrementality, not attributed conversions."}
        return CopilotReport(stepps=prof.to_dict(), viral=viral, conversion=conv, risks=self._risks(text, prof),
                             recommendations=self._recommend(prof), variants=variants, ab_test=ab)
