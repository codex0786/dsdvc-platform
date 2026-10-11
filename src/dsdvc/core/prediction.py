"""Uniform prediction envelope with uncertainty, explanation and versioning (spec §35, §58, §66)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field

from dsdvc import MODEL_VERSION
from dsdvc.core.provenance import EpistemicStatus, utcnow


class Driver(BaseModel):
    feature: str
    contribution: float
    direction: str  # "+" or "-"


class Explanation(BaseModel):
    method: str
    positive_drivers: list[Driver] = Field(default_factory=list)
    negative_drivers: list[Driver] = Field(default_factory=list)

    @classmethod
    def from_contributions(cls, contributions: dict[str, float], method: str, top_k: int = 5) -> "Explanation":
        items = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        pos = [Driver(feature=k, contribution=float(v), direction="+") for k, v in items if v > 0][:top_k]
        neg = [Driver(feature=k, contribution=float(v), direction="-") for k, v in items if v < 0][:top_k]
        return cls(method=method, positive_drivers=pos, negative_drivers=neg)

    def text(self) -> str:
        lines = ["Top positive drivers:"] + [f"  + {d.feature} ({d.contribution:+.3f})" for d in self.positive_drivers]
        lines += ["Top negative drivers:"] + [f"  - {d.feature} ({d.contribution:+.3f})" for d in self.negative_drivers]
        return "\n".join(lines)


def confidence_label(width: float, data_quality: float, agreement: float) -> str:
    """Map interval width, data quality and model agreement to a coarse label."""
    score = (1 - min(width, 1.0)) * 0.4 + data_quality * 0.3 + agreement * 0.3
    if score >= 0.72:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


class Prediction(BaseModel):
    target: str
    prediction: float
    lower: float
    upper: float
    interval_level: float = 0.90
    confidence: str
    data_quality: float = Field(ge=0, le=1)
    model_agreement: float = Field(ge=0, le=1)
    status: EpistemicStatus = EpistemicStatus.PREDICTED
    explanation: Optional[Explanation] = None
    main_uncertainty: Optional[str] = None
    conditional_on: Optional[str] = None
    model_version: str = MODEL_VERSION
    data_version: str = "unversioned"
    timestamp: datetime = Field(default_factory=utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        pct = self.target.startswith("p_") or 0 <= self.prediction <= 1
        fmt = (lambda v: f"{v:.0%}") if pct else (lambda v: f"{v:,.2f}")
        s = (f"{self.target}: {fmt(self.prediction)} "
             f"({int(self.interval_level * 100)}% interval {fmt(self.lower)} – {fmt(self.upper)}); "
             f"confidence {self.confidence}")
        if self.conditional_on:
            s += f"; conditional on {self.conditional_on}"
        return s


def interval_from_samples(samples: np.ndarray, level: float = 0.90) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))
