"""Persist onboarded datasets (content-addressed) and their mapping/validation/attestation."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd

from dsdvc import storage
from dsdvc.core.reproducibility import json_safe
from dsdvc.database.models import AuditLog, DatasetVersion
from dsdvc.onboarding.mapping import Mapping
from dsdvc.onboarding.validate import Scorecard


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode()


def canonical_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(_csv_bytes(df)).hexdigest()


def workspace_dir(project_id: int) -> Path:
    """Local-disk workspace directory (used when object storage is not configured)."""
    return storage.local_workspace_dir(project_id)


def save_dataset(db, *, df: pd.DataFrame, dataset: str, filename: str, mapping: Mapping, scorecard: Scorecard,
                 attestation: str, project_id: int, user_id: int | None = None) -> DatasetVersion:
    """Store a validated dataset inside its client workspace (objects/files live under ws_<id>/)."""
    if project_id is None:
        raise ValueError("A client workspace is required: every dataset belongs to exactly one workspace.")
    if not attestation or not attestation.strip():
        raise ValueError("An authorisation attestation is required before data is stored.")
    if scorecard.status == "BLOCKED":
        raise ValueError("Dataset failed validation (BLOCKED); fix the failing checks first.")
    data = _csv_bytes(df)
    sha = hashlib.sha256(data).hexdigest()
    location = storage.put(project_id, f"{dataset}_{sha[:16]}.csv", data, sha)
    row = DatasetVersion(project_id=project_id, dataset=dataset, filename=filename, sha256=sha, rows=len(df),
                         mapping=mapping.as_dict(), scorecard=json_safe(scorecard.table().to_dict(orient="records")),
                         status=scorecard.status, attestation=attestation.strip(), storage_path=location, uploaded_by=user_id)
    db.add(row)
    db.add(AuditLog(user_id=user_id, project_id=project_id, action="dataset_onboarded",
                    detail={"dataset": dataset, "sha256": sha, "rows": len(df), "status": scorecard.status}))
    db.commit()
    return row


def load_dataset(row: DatasetVersion) -> pd.DataFrame:
    """Read a stored dataset; refuses locations outside the row's own workspace and bytes that fail the hash."""
    df = pd.read_csv(io.BytesIO(storage.get(row.storage_path, row.project_id, row.sha256)))
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"])
    return df


def list_datasets(db, project_id: int, dataset: str = "marketing_weekly") -> list[DatasetVersion]:
    from sqlalchemy import select
    return list(db.scalars(select(DatasetVersion).where(DatasetVersion.project_id == project_id,
                                                        DatasetVersion.dataset == dataset)
                           .order_by(DatasetVersion.created_at.desc())).all())
