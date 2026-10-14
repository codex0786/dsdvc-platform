"""Dataset object storage: workspace isolation, encryption, integrity, legacy local files, migration."""
from __future__ import annotations

import time

import pytest

moto = pytest.importorskip("moto")
boto3 = pytest.importorskip("boto3")

from dsdvc import storage  # noqa: E402
from dsdvc.database.models import DatasetVersion, Organization, Project, SessionLocal, init_db  # noqa: E402
from dsdvc.onboarding.mapping import apply_mapping, suggest_mapping  # noqa: E402
from dsdvc.onboarding.templates import MARKETING_WEEKLY  # noqa: E402
from dsdvc.onboarding.validate import validate  # noqa: E402
from dsdvc.onboarding.workspace import load_dataset, save_dataset  # noqa: E402

BUCKET = "dsdvc-test-datasets"


@pytest.fixture()
def canon():
    from dsdvc.mmm.synthetic import messy_client_export
    raw = messy_client_export()
    m = suggest_mapping(raw, MARKETING_WEEKLY)
    df, _ = apply_mapping(raw, m, MARKETING_WEEKLY)
    return df, m, validate(df, MARKETING_WEEKLY)


@pytest.fixture()
def two_workspaces():
    init_db()
    with SessionLocal() as db:
        org = Organization(name=f"Storage org {time.time_ns()}", allowed_email_domains=[])
        db.add(org)
        db.flush()
        ids = []
        for n in ("a", "b"):
            ws = Project(organization_id=org.id, name=f"{n}-{time.time_ns()}", slug=f"st_{n}_{time.time_ns()}", archived=False)
            db.add(ws)
            db.flush()
            ids.append(ws.id)
        db.commit()
    return ids


@pytest.fixture()
def s3(monkeypatch):
    for k, v in {"S3_BUCKET": BUCKET, "AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
                 "S3_REGION": "us-east-1", "S3_PREFIX": "clients"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    storage.reset()
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client
    storage.reset()


def _save(db, canon, ws_id, name="x.csv"):
    df, m, sc = canon
    return save_dataset(db, df=df, dataset="marketing_weekly", filename=name, mapping=m, scorecard=sc,
                        attestation="test data", project_id=ws_id)


def test_datasets_go_to_object_storage_encrypted_and_private(s3, canon, two_workspaces):
    a, _ = two_workspaces
    with SessionLocal() as db:
        row = _save(db, canon, a)
        assert row.storage_path.startswith(f"s3://{BUCKET}/clients/ws_{a}/marketing_weekly_")
        key = row.storage_path.split(f"s3://{BUCKET}/", 1)[1]
        head = s3.head_object(Bucket=BUCKET, Key=key)
        assert head["ServerSideEncryption"] == "AES256"
        assert head["Metadata"]["sha256"] == row.sha256
        grants = s3.get_object_acl(Bucket=BUCKET, Key=key)["Grants"]
        assert all("AllUsers" not in str(g.get("Grantee", {}).get("URI", "")) for g in grants)
        back = load_dataset(row)
    assert len(back) == len(canon[0]) and str(back["date"].dtype).startswith("datetime64")
    assert storage.describe()["backend"] == "s3" and storage.describe()["durable"]


def test_identical_upload_is_stored_once(s3, canon, two_workspaces):
    a, _ = two_workspaces
    with SessionLocal() as db:
        r1, r2 = _save(db, canon, a, "one.csv"), _save(db, canon, a, "two.csv")
    assert r1.storage_path == r2.storage_path
    assert s3.list_objects_v2(Bucket=BUCKET, Prefix=f"clients/ws_{a}/")["KeyCount"] == 1


def test_object_reads_are_confined_to_the_workspace(s3, canon, two_workspaces):
    a, b = two_workspaces
    with SessionLocal() as db:
        row = _save(db, canon, a)
        row.project_id = b                                     # row tampered to claim another workspace
        with pytest.raises(PermissionError):
            load_dataset(row)
        row.project_id = a
        row.storage_path = f"s3://{BUCKET}/clients/ws_{b}/anything.csv"
        with pytest.raises(PermissionError):
            load_dataset(row)
        row.storage_path = f"s3://other-bucket/clients/ws_{a}/x.csv"
        with pytest.raises(PermissionError):
            load_dataset(row)
        row.storage_path = f"s3://{BUCKET}/clients/ws_{a}/../ws_{b}/x.csv"
        with pytest.raises(PermissionError):
            load_dataset(row)
        db.rollback()


def test_tampered_object_fails_integrity_check(s3, canon, two_workspaces):
    a, _ = two_workspaces
    with SessionLocal() as db:
        row = _save(db, canon, a)
    key = row.storage_path.split(f"s3://{BUCKET}/", 1)[1]
    s3.put_object(Bucket=BUCKET, Key=key, Body=b"date,revenue\n2024-01-01,999999999\n")
    with pytest.raises(storage.IntegrityError):
        load_dataset(row)


def test_storage_outage_raises_instead_of_falling_back_to_disk(s3, canon, two_workspaces, monkeypatch):
    a, _ = two_workspaces
    monkeypatch.setenv("S3_BUCKET", "bucket-that-does-not-exist")
    storage.reset()
    with SessionLocal() as db:
        n = db.query(DatasetVersion).count()
        with pytest.raises(storage.StorageError):
            _save(db, canon, a)
        assert db.query(DatasetVersion).count() == n          # nothing recorded for a dataset that wasn't stored


def test_local_datasets_still_load_and_migrate_to_object_storage(canon, two_workspaces, monkeypatch):
    a, _ = two_workspaces
    monkeypatch.delenv("S3_BUCKET", raising=False)
    storage.reset()
    with SessionLocal() as db:
        local = _save(db, canon, a)
        local_id = local.id
    assert not local.storage_path.startswith("s3://")
    for k, v in {"S3_BUCKET": BUCKET, "AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
                 "S3_REGION": "us-east-1"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("S3_PREFIX", raising=False)
    storage.reset()
    with moto.mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        with SessionLocal() as db:
            assert len(load_dataset(db.get(DatasetVersion, local_id))) == len(canon[0])   # legacy path readable
            assert any(r["id"] == local_id and r["status"] == "dry run" for r in storage.migrate_local_datasets(db, dry_run=True))
            res = {r["id"]: r for r in storage.migrate_local_datasets(db)}
            assert res[local_id]["status"] == "moved"
            row = db.get(DatasetVersion, local_id)
            assert row.storage_path == f"s3://{BUCKET}/ws_{a}/{local.storage_path.rsplit('/', 1)[1]}"
            assert len(load_dataset(row)) == len(canon[0])
            assert local_id not in {r["id"] for r in storage.migrate_local_datasets(db)}   # idempotent
    storage.reset()


def test_object_names_cannot_escape_the_workspace():
    with pytest.raises(ValueError):
        storage.put(1, "../ws_2/x.csv", b"x", storage._sha(b"x"))
    with pytest.raises(ValueError):
        storage.put(None, "x.csv", b"x", storage._sha(b"x"))
    with pytest.raises(storage.IntegrityError):
        storage.put(1, "x.csv", b"x", "0" * 64)
