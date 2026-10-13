"""Persist onboarded datasets (content-addressed) and their mapping/validation/attestation."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import pandas as pd

from dsdvc.core.reproducibility import json_safe
from dsdvc.database.models import AuditLog, DatasetVersion
from dsdvc.onboarding.mapping import Mapping
from dsdvc.onboarding.validate import Scorecard


def data_dir() -> Path:
    d = Path(os.environ.get("DSDVC_DATA_DIR", "./data_store"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def canonical_hash(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return hashlib.sha256(buf.getvalue().encode()).hexdigest()


def save_dataset(db, *, df: pd.DataFrame, dataset: str, filename: str, mapping: Mapping, scorecard: Scorecard,
                 attestation: str, project_id: int | None = None, user_id: int | None = None) -> DatasetVersion:
    if not attestation or not attestation.strip():
        raise ValueError("An authorisation attestation is required before data is stored.")
    if scorecard.status == "BLOCKED":
        raise ValueError("Dataset failed validation (BLOCKED); fix the failing checks first.")
    sha = canonical_hash(df)
    path = data_dir() / f"{dataset}_{sha[:16]}.csv"
    if not path.exists():
        df.to_csv(path, index=False)
    row = DatasetVersion(project_id=project_id, dataset=dataset, filename=filename, sha256=sha, rows=len(df),
                         mapping=mapping.as_dict(), scorecard=json_safe(scorecard.table().to_dict(orient="records")),
                         status=scorecard.status, attestation=attestation.strip(), storage_path=str(path), uploaded_by=user_id)
    db.add(row)
    db.add(AuditLog(user_id=user_id, action="dataset_onboarded",
                    detail={"dataset": dataset, "sha256": sha, "rows": len(df), "status": scorecard.status}))
    db.commit()
    return row


def load_dataset(row: DatasetVersion) -> pd.DataFrame:
    df = pd.read_csv(row.storage_path)
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"])
    return df
