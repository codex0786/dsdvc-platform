"""Reproducibility records (spec §67): every simulation stores seeds, parameters and versions."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from dsdvc import MODEL_VERSION, SIMULATION_VERSION, __version__
from dsdvc.core.provenance import utcnow


def _jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, BaseModel):
        return x.model_dump(mode="json")
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def stable_hash(obj: Any) -> str:
    payload = json.dumps(_jsonable(obj), sort_keys=True, default=_jsonable)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@lru_cache(maxsize=1)
def _software_versions() -> tuple:
    return tuple(_sv().items())


def software_versions() -> dict[str, str]:
    return dict(_software_versions())


def _sv() -> dict[str, str]:
    import networkx, pandas, scipy, sklearn  # noqa: E401
    return {
        "dsdvc": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "networkx": networkx.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
    }


class RunRecord(BaseModel):
    run_id: str
    kind: str
    random_seed: int
    network_seed: int
    parameters: dict[str, Any]
    data_version: str = "synthetic"
    model_version: str = MODEL_VERSION
    simulation_version: str = SIMULATION_VERSION
    software_version: dict[str, str] = Field(default_factory=software_versions)
    created_at: datetime = Field(default_factory=utcnow)
    result_hash: str | None = None

    @classmethod
    def create(cls, kind: str, parameters: Any, random_seed: int, network_seed: int,
               data_version: str = "synthetic") -> "RunRecord":
        params = _jsonable(parameters)
        rid = stable_hash({"kind": kind, "p": params, "s": random_seed, "n": network_seed, "d": data_version})
        return cls(run_id=rid, kind=kind, random_seed=random_seed, network_seed=network_seed,
                   parameters=params, data_version=data_version)


def json_safe(x: Any) -> Any:
    """Recursively convert NaN/inf to None and numpy scalars to Python (strict JSON)."""
    import math
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return [json_safe(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x
