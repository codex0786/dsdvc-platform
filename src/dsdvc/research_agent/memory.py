"""Research memory (spec §39): every conclusion is traceable to source, data, model, experiment."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from dsdvc.core.provenance import utcnow


class Trace(BaseModel):
    source: str
    data: str
    model: str
    experiment: str | None = None


class Conclusion(BaseModel):
    statement: str
    status: str            # observed / estimated / simulated / predicted / recommended / DATA_UNAVAILABLE
    confidence: str
    trace: Trace
    created_at: datetime = Field(default_factory=utcnow)


class ResearchMemory(BaseModel):
    project_id: str
    request: str
    market_definition: dict[str, Any] = Field(default_factory=dict)
    target_audience: dict[str, Any] = Field(default_factory=dict)
    competitors: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)          # dataset id -> description/provenance
    assumptions: list[dict[str, Any]] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    experiments: list[dict[str, Any]] = Field(default_factory=list)
    results: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[Conclusion] = Field(default_factory=list)
    conclusions: list[Conclusion] = Field(default_factory=list)
    unavailable: list[dict[str, Any]] = Field(default_factory=list)
    plan: list[dict[str, Any]] = Field(default_factory=list)

    def conclude(self, statement: str, status: str, confidence: str, source: str, data: str, model: str,
                 experiment: str | None = None, recommendation: bool = False) -> Conclusion:
        c = Conclusion(statement=statement, status=status, confidence=confidence,
                       trace=Trace(source=source, data=data, model=model, experiment=experiment))
        (self.recommendations if recommendation else self.conclusions).append(c)
        return c

    def assume(self, name: str, value: Any, rationale: str) -> None:
        self.assumptions.append({"name": name, "value": value, "rationale": rationale})

    def save(self, directory: str | Path) -> Path:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        f = p / f"{self.project_id}.json"
        f.write_text(json.dumps(self.model_dump(mode="json"), indent=2, default=str))
        return f

    @classmethod
    def load(cls, path: str | Path) -> "ResearchMemory":
        return cls.model_validate_json(Path(path).read_text())
