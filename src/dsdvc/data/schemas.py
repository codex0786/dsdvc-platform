"""Typed domain models (spec §51). All carry provenance where they hold measured values."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from dsdvc.core.provenance import EpistemicStatus, Provenance, utcnow


class Platform(str, Enum):
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    WHATSAPP = "whatsapp"


PLATFORMS = [p.value for p in Platform]


class JourneyStage(str, Enum):
    UNAWARE = "UNAWARE"
    AWARE = "AWARE"
    INTERESTED = "INTERESTED"
    CONSIDERING = "CONSIDERING"
    TRIAL = "TRIAL"
    ADOPTED = "ADOPTED"
    ENGAGED = "ENGAGED"
    ADVOCATE = "ADVOCATE"
    DISSATISFIED = "DISSATISFIED"
    CHURNED = "CHURNED"
    POSSIBLE_RECOVERY = "POSSIBLE_RECOVERY"


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    REEL = "reel"
    CAROUSEL = "carousel"
    TEXT = "text"
    STORY = "story"


class Source(BaseModel):
    source_id: str
    name: str
    url: Optional[str] = None
    provenance: Provenance


class Market(BaseModel):
    market_id: str
    name: str
    country: str
    category: str
    language: Optional[str] = None
    population_estimate: Optional[float] = None
    provenance: Optional[Provenance] = None


class Brand(BaseModel):
    brand_id: str
    name: str
    is_focal: bool = False
    quality: float = 0.5
    price_index: float = 1.0
    brand_equity: float = 0.5


class Product(BaseModel):
    product_id: str
    brand_id: str
    name: str
    price: float
    unit_margin: float
    category: str


class Consumer(BaseModel):
    """Synthetic or aggregated consumer. Never a real identifiable individual (spec §70)."""
    consumer_id: str
    segment: Optional[str] = None
    community_id: Optional[int] = None
    stage: JourneyStage = JourneyStage.UNAWARE
    adoption_threshold: float = 0.5
    price_sensitivity: float = 0.5
    trust_propensity: float = 0.5
    is_synthetic: bool = True


class Creator(BaseModel):
    creator_id: str
    handle: str
    platform: Platform
    followers: Optional[int] = None
    following: Optional[int] = None
    posts_last_30d: Optional[int] = None
    avg_engagement_rate: Optional[float] = None
    niche: Optional[str] = None
    is_public_professional_account: bool = True
    provenance: Provenance


class Community(BaseModel):
    community_id: int
    platform: Optional[Platform] = None
    size: int
    density: float
    modularity_contribution: Optional[float] = None
    conductance: Optional[float] = None
    homophily: Optional[float] = None
    trend: Optional[str] = None  # emerging / stable / declining


class Content(BaseModel):
    content_id: str
    platform: Platform
    creator_id: Optional[str] = None
    timestamp: datetime
    text: str = ""
    hashtags: list[str] = Field(default_factory=list)
    media_type: MediaType = MediaType.IMAGE
    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    saves: Optional[int] = None
    category: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None
    provenance: Provenance

    @property
    def engagement(self) -> Optional[int]:
        parts = [self.likes, self.comments, self.shares, self.saves]
        if all(p is None for p in parts):
            return None
        return int(sum(p or 0 for p in parts))


class PlatformObservation(BaseModel):
    platform: Platform
    metric: str
    value: float
    period_start: datetime
    period_end: datetime
    entity_id: Optional[str] = None
    provenance: Provenance


class Campaign(BaseModel):
    campaign_id: str
    name: str
    brand_id: str
    budget: float
    start: datetime
    end: Optional[datetime] = None
    channels: list[str] = Field(default_factory=list)


class AdExposure(BaseModel):
    consumer_id: str
    campaign_id: Optional[str] = None
    content_id: Optional[str] = None
    platform: Platform
    timestamp: datetime
    exposure_type: str = "organic"  # organic / social / algorithmic / paid
    provenance: Provenance


class Engagement(BaseModel):
    consumer_id: str
    content_id: str
    platform: Platform
    kind: str  # like / comment / save / click
    timestamp: datetime
    provenance: Provenance


class ShareEvent(BaseModel):
    share_id: str
    content_id: str
    sharer_id: str
    parent_share_id: Optional[str] = None
    platform: Platform
    target_platform: Optional[Platform] = None
    timestamp: datetime
    provenance: Provenance


class ConversionEvent(BaseModel):
    consumer_id: str
    product_id: Optional[str] = None
    revenue: float
    channel_path: list[str] = Field(default_factory=list)
    timestamp: datetime
    provenance: Provenance


class ChurnEvent(BaseModel):
    consumer_id: str
    timestamp: datetime
    reason: Optional[str] = None
    provenance: Provenance


class SimulationRun(BaseModel):
    run_id: str
    kind: str
    parameters: dict[str, Any]
    random_seed: int
    network_seed: int
    data_version: str
    model_version: str
    software_version: dict[str, str]
    created_at: datetime = Field(default_factory=utcnow)
    summary: dict[str, Any] = Field(default_factory=dict)
    status: EpistemicStatus = EpistemicStatus.SIMULATED


class Experiment(BaseModel):
    experiment_id: str
    question: str
    design: str  # ab / sweep / monte_carlo / sobol / morris / lhs / counterfactual
    baseline: dict[str, Any]
    treatment: dict[str, Any] = Field(default_factory=dict)
    results: dict[str, Any] = Field(default_factory=dict)
    run_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
