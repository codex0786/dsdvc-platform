"""Data provenance and epistemic status labelling (spec §13, §14, §72).

Every value that leaves the platform is wrapped in a :class:`ProvenancedValue` or
attached to a :class:`Provenance` record. Observed, estimated, simulated, predicted,
recommended and synthetic values are never mixed without a label.

When a requested datum cannot be legally/technically obtained the platform returns
:data:`DATA_UNAVAILABLE` (a :class:`DataUnavailable` instance) instead of fabricating it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field


class EpistemicStatus(str, Enum):
    OBSERVED = "observed"          # measured from an authorised source
    ESTIMATED = "estimated"        # statistically estimated from observed data
    SIMULATED = "simulated"        # output of the agent-based / Monte Carlo simulator
    PREDICTED = "predicted"        # output of a fitted predictive model
    RECOMMENDED = "recommended"    # output of an optimiser / decision rule
    SYNTHETIC = "synthetic"        # generated data; never represents real people


class SourceType(str, Enum):
    AUTHORIZED_API = "authorized_api"
    USER_PROVIDED = "user_provided"
    PUBLIC_DATASET = "public_dataset"
    THIRD_PARTY_API = "third_party_api"
    SURVEY = "survey"
    MODEL = "model"
    SIMULATION = "simulation"
    SYNTHETIC_GENERATOR = "synthetic_generator"
    LITERATURE_PRIOR = "literature_prior"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Provenance(BaseModel):
    source: str
    source_type: SourceType
    timestamp: datetime = Field(default_factory=utcnow)
    collection_method: str
    authorization_scope: str = "none"
    confidence: float = Field(ge=0.0, le=1.0)
    observed_or_estimated: EpistemicStatus
    data_version: Optional[str] = None
    model_version: Optional[str] = None
    notes: Optional[str] = None

    @classmethod
    def synthetic(cls, generator: str, seed: int | None = None) -> "Provenance":
        return cls(
            source=generator,
            source_type=SourceType.SYNTHETIC_GENERATOR,
            collection_method=f"synthetic generation (seed={seed})",
            authorization_scope="n/a - synthetic",
            confidence=1.0,
            observed_or_estimated=EpistemicStatus.SYNTHETIC,
            notes="SYNTHETIC data: does not describe real people or accounts.",
        )

    @classmethod
    def model_output(cls, model: str, status: EpistemicStatus, confidence: float,
                     model_version: str | None = None, data_version: str | None = None) -> "Provenance":
        return cls(
            source=model,
            source_type=SourceType.SIMULATION if status == EpistemicStatus.SIMULATED else SourceType.MODEL,
            collection_method="computed",
            confidence=float(min(max(confidence, 0.0), 1.0)),
            observed_or_estimated=status,
            model_version=model_version,
            data_version=data_version,
        )


T = TypeVar("T")


class ProvenancedValue(BaseModel, Generic[T]):
    metric: str
    value: T
    provenance: Provenance


class DataUnavailable(BaseModel):
    """Explicit marker returned instead of fabricated data."""

    status: str = "DATA_UNAVAILABLE"
    requested: str
    reason: str
    remedy: Optional[str] = None
    timestamp: datetime = Field(default_factory=utcnow)

    def __bool__(self) -> bool:  # lets callers write ``if result:``
        return False


DATA_UNAVAILABLE = "DATA_UNAVAILABLE"


def unavailable(requested: str, reason: str, remedy: str | None = None) -> DataUnavailable:
    return DataUnavailable(requested=requested, reason=reason, remedy=remedy)


def is_unavailable(x: Any) -> bool:
    return isinstance(x, DataUnavailable)


def assert_not_mixed(records: list[Provenance], allow_synthetic: bool = False) -> None:
    """Raise if observed and synthetic records are combined without explicit opt-in."""
    statuses = {r.observed_or_estimated for r in records}
    if EpistemicStatus.SYNTHETIC in statuses and EpistemicStatus.OBSERVED in statuses and not allow_synthetic:
        raise ValueError(
            "Refusing to combine OBSERVED and SYNTHETIC data without allow_synthetic=True "
            "(spec §14: never mix synthetic and observed data without labelling)."
        )
