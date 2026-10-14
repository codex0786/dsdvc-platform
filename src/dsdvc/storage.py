"""Where onboarded client datasets live: S3-compatible object storage, or the local disk.

* **Object storage** when ``S3_BUCKET`` is set (environment variable or Streamlit secret). Works with AWS S3,
  Cloudflare R2, Google Cloud Storage (S3 interoperability), MinIO, … via ``S3_ENDPOINT_URL``.
  Objects are private and server-side encrypted (``S3_SSE``: ``AES256`` (default), ``aws:kms`` with
  ``S3_KMS_KEY_ID``, or ``none`` for providers that always encrypt and reject the header).
* **Local disk** otherwise (``DSDVC_DATA_DIR``, default ``./data_store``): fine for development, but not durable
  on Streamlit Community Cloud or any container without a volume.

Keys are ``[S3_PREFIX]ws_<workspace id>/<dataset>_<sha256[:16]>.csv``. The stored location is recorded on the
``DatasetVersion`` row as ``s3://bucket/key`` (object storage) or an absolute path (local). Reads refuse anything
outside the row's own workspace and verify the bytes against the SHA-256 recorded at onboarding.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


class StorageError(RuntimeError):
    """Object storage is misconfigured or unreachable. Never silently falls back to the local disk."""


class IntegrityError(RuntimeError):
    """Stored bytes no longer match the SHA-256 recorded when the dataset was onboarded."""


def _setting(name: str) -> str | None:
    v = os.environ.get(name)
    if v:
        return v
    try:  # Streamlit secrets (root-level keys); absent outside Streamlit
        import streamlit as st
        v = st.secrets.get(name)  # type: ignore[attr-defined]
        return str(v) if v else None
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class S3Config:
    bucket: str
    endpoint_url: str | None = None
    region: str | None = None
    prefix: str = ""
    sse: str = "AES256"
    kms_key_id: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    addressing_style: str | None = None


def s3_config() -> S3Config | None:
    bucket = _setting("S3_BUCKET")
    if not bucket:
        return None
    prefix = (_setting("S3_PREFIX") or "").strip("/")
    endpoint = _setting("S3_ENDPOINT_URL")
    return S3Config(bucket=bucket, endpoint_url=endpoint,
                    region=_setting("S3_REGION") or _setting("AWS_DEFAULT_REGION") or _setting("AWS_REGION"),
                    prefix=f"{prefix}/" if prefix else "",
                    sse=_setting("S3_SSE") or "AES256", kms_key_id=_setting("S3_KMS_KEY_ID"),
                    access_key_id=_setting("S3_ACCESS_KEY_ID") or _setting("AWS_ACCESS_KEY_ID"),
                    secret_access_key=_setting("S3_SECRET_ACCESS_KEY") or _setting("AWS_SECRET_ACCESS_KEY"),
                    # MinIO and most self-hosted endpoints need path-style URLs
                    addressing_style=_setting("S3_ADDRESSING_STYLE") or ("path" if endpoint else None))


@lru_cache(maxsize=4)
def _client(cfg: S3Config):
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:  # pragma: no cover - boto3 is in every requirements file
        raise StorageError("S3_BUCKET is set but boto3 is not installed") from e
    kw = {}
    if cfg.access_key_id and cfg.secret_access_key:  # otherwise boto3's default chain (IAM role, profile, …)
        kw = {"aws_access_key_id": cfg.access_key_id, "aws_secret_access_key": cfg.secret_access_key}
    conf = Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"},
                  s3={"addressing_style": cfg.addressing_style} if cfg.addressing_style else None,
                  connect_timeout=10, read_timeout=60)
    return boto3.client("s3", endpoint_url=cfg.endpoint_url, region_name=cfg.region, config=conf, **kw)


def reset() -> None:
    """Forget cached clients (after changing configuration, and in tests)."""
    _client.cache_clear()


def _data_dir() -> Path:
    d = Path(os.environ.get("DSDVC_DATA_DIR", "./data_store"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def local_workspace_dir(project_id: int) -> Path:
    d = _data_dir() / f"ws_{int(project_id)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def describe() -> dict:
    """Non-secret summary of the active backend, for status pages and /health."""
    cfg = s3_config()
    if cfg is None:
        return {"backend": "local", "location": str(_data_dir().resolve()), "durable": False}
    return {"backend": "s3", "location": f"s3://{cfg.bucket}/{cfg.prefix}", "endpoint": cfg.endpoint_url or "aws",
            "encryption": cfg.sse if cfg.sse.lower() != "none" else "provider default", "durable": True}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_s3(uri: str) -> tuple[str, str]:
    bucket, _, key = uri[len("s3://"):].partition("/")
    if not bucket or not key:
        raise PermissionError("malformed object location")
    return bucket, key


def _check_workspace_key(key: str, project_id: int) -> None:
    parts = key.split("/")
    if len(parts) < 2 or parts[-2] != f"ws_{int(project_id)}" or ".." in parts or "" in parts:
        raise PermissionError("dataset object is outside its workspace prefix")


def put(project_id: int, name: str, data: bytes, sha256: str) -> str:
    """Store bytes for a workspace (content-addressed: identical uploads are written once). Returns the location."""
    if project_id is None:
        raise ValueError("A client workspace is required")
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("invalid object name")
    if _sha(data) != sha256:
        raise IntegrityError("bytes do not match the dataset hash")
    cfg = s3_config()
    if cfg is None:
        path = local_workspace_dir(project_id) / name
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(path)  # atomic: readers never see a half-written file
        return str(path)
    key = f"{cfg.prefix}ws_{int(project_id)}/{name}"
    from botocore.exceptions import BotoCoreError, ClientError
    s3 = _client(cfg)
    try:
        try:
            head = s3.head_object(Bucket=cfg.bucket, Key=key)
            if head.get("Metadata", {}).get("sha256") == sha256:
                return f"s3://{cfg.bucket}/{key}"
        except ClientError:  # 404, or 403 for a missing key when the role has no ListBucket: write it
            pass
        extra = {"ContentType": "text/csv", "Metadata": {"sha256": sha256, "workspace": str(int(project_id))}}
        if cfg.sse.lower() != "none":
            extra["ServerSideEncryption"] = cfg.sse
            if cfg.sse == "aws:kms" and cfg.kms_key_id:
                extra["SSEKMSKeyId"] = cfg.kms_key_id
        s3.put_object(Bucket=cfg.bucket, Key=key, Body=data, **extra)
    except (BotoCoreError, ClientError) as e:
        raise StorageError(f"could not write to object storage s3://{cfg.bucket}: {e}") from e
    return f"s3://{cfg.bucket}/{key}"


def get(location: str, project_id: int | None, sha256: str | None = None) -> bytes:
    """Read bytes, refusing locations outside the workspace and (when given) bytes that fail the hash check."""
    if project_id is None:
        raise PermissionError("dataset has no workspace")
    if location.startswith("s3://"):
        bucket, key = _parse_s3(location)
        _check_workspace_key(key, project_id)
        cfg = s3_config()
        if cfg is None:
            raise StorageError("this dataset is in object storage, but S3_BUCKET is not configured")
        if bucket != cfg.bucket:
            raise PermissionError("dataset object is in a bucket this deployment does not use")
        from botocore.exceptions import BotoCoreError, ClientError
        try:
            data = _client(cfg).get_object(Bucket=bucket, Key=key)["Body"].read()
        except (BotoCoreError, ClientError) as e:
            raise StorageError(f"could not read {location}: {e}") from e
    else:
        path = Path(location.removeprefix("file://")).resolve()
        if local_workspace_dir(project_id).resolve() not in path.parents:
            raise PermissionError("dataset file is outside its workspace directory")
        if not path.exists():
            raise StorageError("dataset file is missing from local storage (the disk may have been reset; "
                               "configure object storage for durable datasets)")
        data = path.read_bytes()
    if sha256 and _sha(data) != sha256:
        raise IntegrityError("stored dataset does not match the SHA-256 recorded at onboarding")
    return data


def migrate_local_datasets(db, dry_run: bool = False) -> list[dict]:
    """Copy every locally stored dataset into the configured bucket and repoint its row. Local files are kept.

    Each copy is hash-verified before the row changes, and recorded in the workspace's audit log.
    """
    from sqlalchemy import select

    from dsdvc.database.models import AuditLog, DatasetVersion
    if s3_config() is None:
        raise StorageError("set S3_BUCKET (and credentials) before migrating")
    out = []
    for row in db.scalars(select(DatasetVersion).order_by(DatasetVersion.id)).all():
        if row.storage_path.startswith("s3://"):
            continue
        rec = {"id": row.id, "workspace": row.project_id, "from": row.storage_path}
        try:
            data = get(row.storage_path, row.project_id, row.sha256)
            if not dry_run:
                row.storage_path = put(row.project_id, Path(row.storage_path).name, data, row.sha256)
                db.add(AuditLog(user_id=None, project_id=row.project_id, action="dataset_moved_to_object_storage",
                                detail={"dataset_version": row.id, "sha256": row.sha256, "to": row.storage_path}))
                db.commit()
            rec.update(to=row.storage_path, status="dry run" if dry_run else "moved")
        except (PermissionError, StorageError, IntegrityError) as e:
            db.rollback()
            rec.update(status=f"skipped: {e}")
        out.append(rec)
    return out


if __name__ == "__main__":  # python -m dsdvc.storage migrate [--dry-run]
    import argparse
    import json

    p = argparse.ArgumentParser(prog="python -m dsdvc.storage")
    p.add_argument("command", choices=["status", "migrate"])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.command == "status":
        print(json.dumps(describe(), indent=2))
    else:
        from dsdvc.database.models import SessionLocal, init_db
        init_db()
        with SessionLocal() as db:
            for r in migrate_local_datasets(db, dry_run=a.dry_run):
                print(json.dumps(r))
