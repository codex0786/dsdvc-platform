"""STEPPSAnalyzer (spec §5, §18).

Scores content on Berger's six STEPPS dimensions. STEPPS is a *hypothesised* mechanism
(Berger 2013): the scores are measurement of content attributes, not a virality
guarantee. Whether and how they relate to sharing is estimated in ``stepps_models``.

Two coders:
* ``lexicon`` (default): transparent rule-based coder; low-to-medium confidence.
* ``supervised``: per-dimension ridge models trained on human-coded examples
  (``fit``), with cross-validated R^2 driving the reported confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dsdvc.content.features import TextFeatures, extract_text_features
from dsdvc.core.literature import cites

DIMENSIONS = ("social_currency", "triggers", "emotion", "public", "practical_value", "stories")
_LEX_KEY = {"social_currency": "social_currency", "triggers": "triggers", "emotion": None,
            "public": "public", "practical_value": "practical", "stories": "stories"}

MEDIA_PUBLIC_BONUS = {"reel": 0.15, "video": 0.10, "story": 0.05, "carousel": 0.05, "image": 0.0, "text": -0.05}


def _sat(x: float, k: float = 1.0) -> float:
    return float(1 - np.exp(-k * max(x, 0.0)))


@dataclass
class DimensionScore:
    score: float
    confidence: float
    evidence: list[str]
    contributions: dict[str, float]
    explanation: str


@dataclass
class STEPPSProfile:
    dimensions: dict[str, DimensionScore]
    coder: str
    features: TextFeatures | None = None
    notes: list[str] = field(default_factory=list)

    def vector(self) -> np.ndarray:
        return np.array([self.dimensions[d].score for d in DIMENSIONS])

    def scores(self) -> dict[str, float]:
        return {d: round(self.dimensions[d].score, 3) for d in DIMENSIONS}

    def weakest(self, k: int = 2) -> list[str]:
        return sorted(DIMENSIONS, key=lambda d: self.dimensions[d].score)[:k]

    def to_dict(self) -> dict:
        return {"coder": self.coder, "scores": self.scores(),
                "detail": {d: {"score": s.score, "confidence": s.confidence, "evidence": s.evidence,
                               "contributions": s.contributions, "explanation": s.explanation}
                           for d, s in self.dimensions.items()},
                "notes": self.notes,
                "disclaimer": "STEPPS scores measure content attributes hypothesised to raise word-of-mouth "
                              "(Berger 2013); they do not guarantee sharing or virality."}


@cites("berger2013", "berger_milkman2012")
class STEPPSAnalyzer:
    def __init__(self, brand_terms: list[str] | None = None, product_terms: list[str] | None = None):
        self.brand_terms = brand_terms or []
        self.product_terms = product_terms or []
        self._models = None
        self._cv_r2: dict[str, float] = {}
        self._vec = None

    # ---------- lexicon coder ----------
    def _lexicon(self, f: TextFeatures, media_type: str | None) -> dict[str, DimensionScore]:
        n = max(f.n_tokens, 1)
        length_conf = min(1.0, n / 25)  # short captions carry little evidence
        out = {}
        for d in DIMENSIONS:
            contrib: dict[str, float] = {}
            if d == "emotion":
                contrib["arousal"] = 0.55 * f.arousal
                contrib["sentiment_intensity"] = 0.20 * abs(f.sentiment)
                contrib["discrete_emotions"] = 0.25 * _sat(sum(f.emotion.values()) / 10)
                ev = f.lexicon_hits["high_arousal"] + [e for k in ("awe", "anger", "anxiety", "joy") for e in f.lexicon_hits[k]]
                expl = "High-arousal and discrete-emotion vocabulary (Berger & Milkman 2012 found arousal, not valence alone, predicts sharing of news)."
            else:
                key = _LEX_KEY[d]
                hits = f.lexicon_hits[key]
                ev = list(hits)
                contrib["lexicon"] = 0.75 * _sat(len(hits), 0.55)
                if d == "practical_value":
                    contrib["informativeness"] = 0.25 * f.informativeness
                    expl = "Useful-information cues: how-to, tips, prices, numbers."
                elif d == "public":
                    contrib["hashtags"] = 0.10 * _sat(len(f.hashtags), 0.5)
                    contrib["media_visibility"] = MEDIA_PUBLIC_BONUS.get((media_type or "").lower(), 0.0)
                    expl = "Observability cues: challenges, tagging, hashtags, visual/Reel formats."
                elif d == "stories":
                    contrib["narrative_length"] = 0.20 * _sat(n / 40)
                    expl = "Narrative cues: first person, temporal sequence, before/after."
                elif d == "social_currency":
                    contrib["question_or_reveal"] = 0.1 * float(f.question)
                    expl = "Insider/remarkability cues: exclusive, secret, first, limited."
                else:
                    expl = "Environmental cue references: routines, times, seasons, festivals."
            score = float(np.clip(sum(contrib.values()), 0, 1))
            conf = float(np.clip(0.25 + 0.35 * length_conf + 0.05 * min(len(ev), 4), 0.2, 0.7))
            out[d] = DimensionScore(score=score, confidence=conf, evidence=ev[:8],
                                    contributions={k: round(v, 4) for k, v in contrib.items()}, explanation=expl)
        return out

    # ---------- supervised coder ----------
    def fit(self, texts: list[str], labels: dict[str, np.ndarray], seed: int = 0) -> dict[str, float]:
        """Train per-dimension ridge regressions on human-coded STEPPS labels in [0,1]."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import cross_val_score
        from scipy import sparse
        self._vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(texts)
        X = sparse.hstack([self._vec.transform(texts), self._dense(texts)]).tocsr()
        self._models = {}
        for d in DIMENSIONS:
            y = np.asarray(labels[d], dtype=float)
            m = RidgeCV(alphas=np.logspace(-2, 3, 12))
            self._cv_r2[d] = float(np.mean(cross_val_score(m, X, y, cv=5, scoring="r2")))
            self._models[d] = m.fit(X, y)
        return dict(self._cv_r2)

    def _dense(self, texts):
        from scipy import sparse
        rows = []
        for t in texts:
            f = extract_text_features(t)
            rows.append([f.arousal, f.sentiment, f.informativeness, f.cta_strength] +
                        [len(f.lexicon_hits[k]) for k in ("social_currency", "triggers", "public", "practical", "stories")])
        return sparse.csr_matrix(np.array(rows, dtype=float))

    # ---------- public API ----------
    def analyze(self, text: str, media_type: str | None = None, transcript: str | None = None,
                coder: str = "auto") -> STEPPSProfile:
        full = " ".join(x for x in (text, transcript) if x)
        f = extract_text_features(full, self.brand_terms, self.product_terms)
        notes = []
        if transcript:
            notes.append("Audio transcript included in analysis.")
        if coder == "supervised" and self._models is None:
            raise RuntimeError("supervised coder requested but STEPPSAnalyzer.fit() has not been called")
        if coder == "lexicon" or (coder == "auto" and self._models is None):
            dims = self._lexicon(f, media_type)
            used = "lexicon"
            notes.append("Lexicon coder: transparent baseline; confidence capped at 0.7. Train a supervised coder "
                         "on human-coded examples for higher validity.")
        else:
            from scipy import sparse
            X = sparse.hstack([self._vec.transform([full]), self._dense([full])]).tocsr()
            lex = self._lexicon(f, media_type)
            dims = {}
            for d in DIMENSIONS:
                s = float(np.clip(self._models[d].predict(X)[0], 0, 1))
                conf = float(np.clip(0.3 + 0.6 * max(self._cv_r2.get(d, 0), 0), 0.2, 0.95))
                dims[d] = DimensionScore(score=s, confidence=conf, evidence=lex[d].evidence,
                                         contributions={"supervised_model": s},
                                         explanation=f"Supervised ridge coder (CV R^2={self._cv_r2.get(d, 0):.2f}). " + lex[d].explanation)
            used = "supervised"
        return STEPPSProfile(dimensions=dims, coder=used, features=f, notes=notes)
