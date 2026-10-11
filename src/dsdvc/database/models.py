"""Relational schema (PostgreSQL in production, SQLite locally) - spec §49, §65, §66, §67."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from dsdvc.config.settings import get_settings


def _now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Organization(Base):
    __tablename__ = "organizations"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"))
    role: Mapped[str] = mapped_column(String(20), default="viewer")   # admin / analyst / viewer
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    organization = relationship("Organization")


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[int] = mapped_column(primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"))
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    __table_args__ = (UniqueConstraint("organization_id", "name"),)


class ApiKey(Base):
    """Only a salted PBKDF2 hash of the secret is stored; the plaintext is shown once at creation."""
    __tablename__ = "api_keys"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    prefix: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    salt: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    user = relationship("User")


class PlatformCredential(Base):
    """Reference to a platform credential held in a secret manager (never the token itself)."""
    __tablename__ = "platform_credentials"
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    platform: Mapped[str] = mapped_column(String(30))
    secret_ref: Mapped[str] = mapped_column(String(300))   # e.g. "env:META_ACCESS_TOKEN" or "aws-sm:arn:..."
    authorization_scope: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Observation(Base):
    __tablename__ = "observations"
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"), nullable=True)
    platform: Mapped[str | None] = mapped_column(String(30), nullable=True)
    entity_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    metric: Mapped[str] = mapped_column(String(100))
    value: Mapped[float] = mapped_column(Float)
    period_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # provenance (spec §14)
    source: Mapped[str] = mapped_column(String(200))
    source_type: Mapped[str] = mapped_column(String(40))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    collection_method: Mapped[str] = mapped_column(String(200))
    authorization_scope: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    observed_or_estimated: Mapped[str] = mapped_column(String(20))
    data_version: Mapped[str | None] = mapped_column(String(64), nullable=True)


class SimulationRunRow(Base):
    __tablename__ = "simulation_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(32), index=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"), nullable=True)
    kind: Mapped[str] = mapped_column(String(40))
    random_seed: Mapped[int] = mapped_column(Integer)
    network_seed: Mapped[int] = mapped_column(Integer)
    parameters: Mapped[dict] = mapped_column(JSON)
    data_version: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64))
    simulation_version: Mapped[str] = mapped_column(String(64))
    software_version: Mapped[dict] = mapped_column(JSON)
    result_hash: Mapped[str | None] = mapped_column(String(32), nullable=True)
    summary: Mapped[dict] = mapped_column(JSON)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class PredictionRow(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"), nullable=True)
    target: Mapped[str] = mapped_column(String(100))
    payload: Mapped[dict] = mapped_column(JSON)
    model_version: Mapped[str] = mapped_column(String(64))
    data_version: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ExperimentRow(Base):
    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"), nullable=True)
    design: Mapped[str] = mapped_column(String(40))
    question: Mapped[str] = mapped_column(Text)
    results: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ResearchProjectMemory(Base):
    __tablename__ = "research_memory"
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"), nullable=True)
    memory_id: Mapped[str] = mapped_column(String(40), unique=True)
    memory: Mapped[dict] = mapped_column(JSON)
    report_markdown: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(100))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


_engine = None
_Session = None


def get_engine(url: str | None = None):
    global _engine, _Session
    if _engine is None or url is not None:
        url = url or get_settings().database_url
        kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
        _engine = create_engine(url, **kw)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def SessionLocal():
    get_engine()
    return _Session()


def init_db(url: str | None = None):
    eng = get_engine(url)
    Base.metadata.create_all(eng)
    return eng
