"""Transparent NLP features for content (spec §17).

Lexicon/rule-based by default so every feature is inspectable and works offline. The
lexicons are deliberately small and English/Hinglish-oriented; they are a baseline, not
a claim of state-of-the-art NLP. Swap in a supervised or LLM coder via
``STEPPSAnalyzer(coder=...)`` when labelled data exists.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

LEX = {
    "positive": {"love", "amazing", "best", "great", "beautiful", "glow", "happy", "perfect", "wow", "favourite",
                 "favorite", "obsessed", "incredible", "awesome", "fresh", "gorgeous", "win", "proud", "fun", "soft"},
    "negative": {"hate", "worst", "bad", "terrible", "awful", "problem", "acne", "breakout", "damage", "fake",
                 "scam", "angry", "sad", "fail", "waste", "dull", "irritation", "ugly", "toxic", "disappointed"},
    "high_arousal": {"shocking", "unbelievable", "insane", "omg", "wow", "amazing", "furious", "angry", "anxious",
                     "scared", "thrilled", "excited", "obsessed", "awe", "mind-blowing", "crazy", "urgent", "finally",
                     "can't", "never", "must", "incredible", "stunning"},
    "low_arousal": {"calm", "relax", "gentle", "soothing", "content", "quiet", "slow", "simple", "easy", "sad"},
    "awe": {"awe", "stunning", "breathtaking", "incredible", "mind-blowing", "wonder", "magical"},
    "anger": {"angry", "furious", "outrage", "scam", "fake", "lied", "unfair"},
    "anxiety": {"worried", "anxious", "scared", "risk", "danger", "warning", "harmful"},
    "joy": {"happy", "joy", "love", "fun", "delight", "smile", "celebrate"},
    "social_currency": {"exclusive", "secret", "insider", "first", "limited", "only", "rare", "vip", "early",
                        "hidden", "unlock", "members", "launch", "drop", "nobody", "few", "elite"},
    "triggers": {"daily", "morning", "night", "routine", "everyday", "weekend", "monday", "coffee", "summer",
                 "monsoon", "winter", "diwali", "holi", "festive", "wedding", "office", "commute", "gym", "bedtime"},
    "public": {"share", "tag", "challenge", "duet", "remix", "trend", "show", "wear", "visible", "spotted",
               "repost", "everyone", "join", "together", "community", "#"},
    "practical": {"how", "tips", "tip", "guide", "step", "steps", "save", "deal", "discount", "price", "under",
                  "cheap", "affordable", "hack", "learn", "tutorial", "routine", "ingredients", "spf", "review",
                  "compare", "vs", "budget", "%", "₹", "rs", "off", "free"},
    "stories": {"i", "my", "me", "when", "then", "finally", "journey", "story", "before", "after", "used",
                "remember", "day", "years", "transformation", "struggled", "realised", "realized", "mom", "friend"},
    "controversy": {"controversial", "unpopular", "debate", "myth", "lie", "truth", "exposed", "vs", "wrong",
                    "overrated", "boycott", "honest"},
    "cta": {"shop now", "link in bio", "buy now", "order now", "sign up", "dm", "click", "swipe up", "use code",
            "book now", "download", "subscribe", "follow for", "comment below", "tap", "grab"},
}

WORD_RE = re.compile(r"[#@]?[\w'₹%-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")]


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups) - (1 if w.endswith("e") and len(groups) > 1 else 0)
    return max(1, n)


def flesch_reading_ease(text: str) -> float:
    sentences = max(1, len(re.findall(r"[.!?]+", text or "")) or 1)
    words = [w for w in tokenize(text) if w.isalpha()]
    if not words:
        return 0.0
    syl = sum(_syllables(w) for w in words)
    return float(206.835 - 1.015 * len(words) / sentences - 84.6 * syl / len(words))


@dataclass
class TextFeatures:
    n_tokens: int
    sentiment: float          # [-1, 1]
    arousal: float            # [0, 1]
    emotion: dict[str, float]  # awe / anger / anxiety / joy counts per 100 tokens
    readability: float        # Flesch reading ease (higher = easier)
    informativeness: float    # numbers / specifics density
    cta_strength: float       # [0, 1]
    controversy: float        # [0, 1]
    question: bool
    exclamations: int
    hashtags: list[str]
    mentions: list[str]
    lexicon_hits: dict[str, list[str]] = field(default_factory=dict)
    brand_presence: float = 0.0
    product_presence: float = 0.0
    novelty: float | None = None

    def as_vector(self) -> dict[str, float]:
        d = {"n_tokens": self.n_tokens, "sentiment": self.sentiment, "arousal": self.arousal,
             "readability": self.readability, "informativeness": self.informativeness,
             "cta_strength": self.cta_strength, "controversy": self.controversy,
             "question": float(self.question), "exclamations": float(self.exclamations),
             "n_hashtags": float(len(self.hashtags)), "brand_presence": self.brand_presence,
             "product_presence": self.product_presence}
        for k, v in self.emotion.items():
            d[f"emo_{k}"] = v
        if self.novelty is not None:
            d["novelty"] = self.novelty
        return d


def lexicon_hits(tokens: list[str], text_lower: str) -> dict[str, list[str]]:
    toks = set(tokens)
    hits = {}
    for k, words in LEX.items():
        found = sorted(w for w in words if (w in toks if " " not in w and w not in {"#", "%", "₹"} else w in text_lower))
        if k == "public" and any(t.startswith("#") for t in tokens):
            found = sorted(set(found) | {"#hashtag"})
        hits[k] = found
    return hits


def extract_text_features(text: str, brand_terms: list[str] | None = None,
                          product_terms: list[str] | None = None) -> TextFeatures:
    text = text or ""
    tl = text.lower()
    toks = tokenize(text)
    n = max(1, len(toks))
    hits = lexicon_hits(toks, tl)
    pos = sum(t in LEX["positive"] for t in toks)
    neg = sum(t in LEX["negative"] for t in toks)
    sentiment = (pos - neg) / max(1, pos + neg) if (pos + neg) else 0.0
    hi = sum(t in LEX["high_arousal"] for t in toks)
    lo = sum(t in LEX["low_arousal"] for t in toks)
    excl = text.count("!")
    caps = sum(1 for w in re.findall(r"\b[A-Z]{3,}\b", text))
    arousal = 1 - math.exp(-(hi + 0.5 * excl + 0.5 * caps) / 3) if (hi + excl + caps) else 0.0
    arousal = float(np.clip(arousal - 0.1 * lo, 0, 1))
    emotion = {k: 100 * sum(t in LEX[k] for t in toks) / n for k in ("awe", "anger", "anxiety", "joy")}
    nums = len(re.findall(r"\d+", text))
    informativeness = float(np.clip((nums + len(hits["practical"])) / (0.15 * n + 1), 0, 1))
    cta = float(np.clip(len(hits["cta"]) / 2, 0, 1))
    controversy = float(np.clip((len(hits["controversy"]) + (1 if pos and neg else 0)) / 3, 0, 1))
    brand = float(any(b.lower() in tl for b in (brand_terms or [])))
    product = float(any(p.lower() in tl for p in (product_terms or [])))
    return TextFeatures(
        n_tokens=len(toks), sentiment=float(sentiment), arousal=arousal, emotion=emotion,
        readability=flesch_reading_ease(text), informativeness=informativeness, cta_strength=cta,
        controversy=controversy, question="?" in text, exclamations=excl,
        hashtags=[t for t in toks if t.startswith("#")], mentions=[t for t in toks if t.startswith("@")],
        lexicon_hits=hits, brand_presence=brand, product_presence=product,
    )


def novelty_scores(texts: list[str], reference: list[str] | None = None) -> np.ndarray:
    """Novelty = 1 - max cosine similarity to the reference corpus (TF-IDF).

    A text is never compared with its own occurrence in the reference (leave-self-out),
    so training items are scored on the same footing as new items."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    ref = list(reference) if reference else list(texts)
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2)).fit(ref)
    R, X = vec.transform(ref), vec.transform(texts)
    sims = (X @ R.T).toarray()
    if reference is None:
        np.fill_diagonal(sims, 0)
    else:
        pos: dict[str, list[int]] = {}
        for j, t in enumerate(ref):
            pos.setdefault(t, []).append(j)
        for i, t in enumerate(texts):
            if t in pos:
                sims[i, pos[t][0]] = 0.0  # drop one self-match; true duplicates still count
    return 1 - sims.max(axis=1)


def topic_model(texts: list[str], n_topics: int = 6, seed: int = 0, top_words: int = 6) -> dict:
    """NMF topics on TF-IDF; returns per-document topic id and topic keywords."""
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.9)
    X = vec.fit_transform(texts)
    k = max(1, min(n_topics, X.shape[1] - 1, X.shape[0] - 1))
    nmf = NMF(n_components=k, random_state=seed, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)
    vocab = np.array(vec.get_feature_names_out())
    topics = {i: vocab[np.argsort(-comp)[:top_words]].tolist() for i, comp in enumerate(nmf.components_)}
    return {"doc_topic": W.argmax(axis=1), "topics": topics, "weights": W}
