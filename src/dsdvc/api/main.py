"""DSDVC REST API (spec §49, §65, §66).

Auth: ``X-API-Key: dsdvc_<prefix>_<secret>``. Roles: viewer (read), analyst (run models),
admin (manage orgs/users/projects/keys). Every model output carries model_version,
data_version and timestamp; every simulation is stored with its full RunRecord.
"""
from __future__ import annotations

import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from dsdvc import MODEL_VERSION, SIMULATION_VERSION, __version__
from dsdvc.config.settings import get_settings
from dsdvc.core.literature import LITERATURE
from dsdvc.core.observability import METRICS, Cache, log_to_mlflow
from dsdvc.core.reproducibility import json_safe
from dsdvc.database import auth as A
from dsdvc.database.models import (AuditLog, ExperimentRow, Organization, PredictionRow, Project,
                                   ResearchProjectMemory, SessionLocal, SimulationRunRow, User, init_db)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    with SessionLocal() as db:
        A.bootstrap_admin(db, get_settings().bootstrap_admin_key)
    yield


app = FastAPI(title="DSDVC Marketing Intelligence API", version=__version__, lifespan=lifespan)
CACHE = Cache()
JOBS: dict[str, dict] = {}
VIRALITY_MODELS: dict[str, Any] = {}


def _now():
    return datetime.now(timezone.utc).isoformat()


def T_audit(db, user, workspace_id, action, **detail):
    from dsdvc.tenancy.access import audit
    audit(db, user, workspace_id, action, **detail)


def meta(data_version: str = "request") -> dict:
    return {"model_version": MODEL_VERSION, "simulation_version": SIMULATION_VERSION, "data_version": data_version,
            "timestamp": _now()}


@app.middleware("http")
async def observe(request: Request, call_next):
    t = time.perf_counter()
    try:
        resp = await call_next(request)
    except Exception:
        METRICS.inc("dsdvc_http_errors", route=request.url.path)
        raise
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    METRICS.observe("dsdvc_http_latency", (time.perf_counter() - t) * 1000, route=path, method=request.method)
    METRICS.inc("dsdvc_http_requests", route=path, status=str(resp.status_code))
    return resp


# ------------------------------------------------------------------ auth
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def current_user(x_api_key: str | None = Header(default=None), authorization: str | None = Header(default=None),
                 db=Depends(get_db)) -> User:
    """API keys (service accounts) via X-API-Key, or SSO session tokens via Authorization: Bearer."""
    from dsdvc.tenancy import sessions
    if x_api_key:
        u = A.verify_api_key(db, x_api_key)
        if u is None:
            raise HTTPException(401, "invalid API key")
        return u
    if authorization and authorization.lower().startswith("bearer "):
        u = sessions.verify(db, authorization.split(" ", 1)[1].strip())
        if u is None:
            raise HTTPException(401, "invalid or expired session")
        return u
    raise HTTPException(401, "missing credentials (X-API-Key or Authorization: Bearer)")


def require(role: str):
    """Organisation-level role, for endpoints that never touch client data (synthetic simulation, text tools)."""
    def dep(u: User = Depends(current_user)) -> User:
        if not A.has_role(u, role):
            raise HTTPException(403, f"requires role '{role}'")
        return u
    return dep


def in_workspace(min_role: str = "viewer"):
    """Client-data endpoints: the X-Workspace-Id header selects the client workspace; membership and role
    are checked on every request. Unknown and forbidden workspaces both return 404."""
    from dsdvc.tenancy import access as T

    def dep(x_workspace_id: int | None = Header(default=None), u: User = Depends(current_user), db=Depends(get_db)):
        try:
            ws, role = T.require(db, u, x_workspace_id, min_role)
        except T.NotFound:
            raise HTTPException(404, "workspace not found (send X-Workspace-Id for a workspace you belong to)")
        except T.Forbidden as e:
            raise HTTPException(403, str(e))
        return u, ws, role
    return dep


# ------------------------------------------------------------------ public
@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, **meta()}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return METRICS.prometheus()


@app.get("/literature")
def literature():
    return LITERATURE


# ------------------------------------------------------------------ single sign-on (OIDC)
def _oidc(provider: str):
    from dsdvc.tenancy.oidc import OIDCClient, configured_providers
    from dsdvc.connectors.credentials import lookup
    provs = configured_providers()
    if provider not in provs:
        raise HTTPException(404, "unknown sign-in provider")
    base = (lookup("DSDVC_PUBLIC_URL") or "http://localhost:8000").rstrip("/")
    return OIDCClient(provs[provider], f"{base}/auth/oidc/{provider}/callback"), provs[provider]


@app.get("/auth/providers")
def auth_providers():
    from dsdvc.tenancy.oidc import configured_providers
    return {"providers": sorted(configured_providers())}


@app.get("/auth/oidc/{provider}/login")
def auth_login(provider: str):
    from fastapi.responses import RedirectResponse
    client, _ = _oidc(provider)
    url, _state = client.authorization_url()
    return RedirectResponse(url, status_code=302)


@app.get("/auth/oidc/{provider}/callback")
def auth_callback(provider: str, code: str | None = None, state: str | None = None, error: str | None = None,
                  db=Depends(get_db)):
    from dsdvc.tenancy import sessions
    from dsdvc.tenancy.identity import resolve_user
    from dsdvc.tenancy.oidc import claims_identity
    if error or not code or not state:
        raise HTTPException(400, f"sign-in was not completed: {error or 'missing code/state'}")
    client, prov = _oidc(provider)
    try:
        claims = client.exchange(code, state)
    except Exception as e:  # noqa: BLE001 - never leak tokens; message only
        METRICS.inc("dsdvc_login_failures", provider=provider)
        raise HTTPException(401, f"sign-in failed: {e}")
    res = resolve_user(db, provider=provider, trust_email=prov.trust_email, **claims_identity(claims))
    if res.user is None:
        METRICS.inc("dsdvc_login_denied", provider=provider)
        raise HTTPException(403, res.reason)
    token = sessions.issue(res.user)
    return {"token_type": "Bearer", "access_token": token, "expires_in": int(sessions.DEFAULT_TTL.total_seconds()),
            "user": {"email": res.user.email, "new_account": res.created}}


@app.post("/auth/logout")
def auth_logout(u: User = Depends(current_user), db=Depends(get_db)):
    from dsdvc.tenancy import sessions
    sessions.revoke_all(db, u)
    return {"signed_out": True, "note": "all session tokens for this user are now invalid"}


@app.get("/me")
def me(u: User = Depends(current_user), db=Depends(get_db)):
    from dsdvc.tenancy import access as T
    return {"id": u.id, "email": u.email, "role": u.role, "organization_id": u.organization_id,
            "workspaces": [{"id": w.id, "name": w.name, "slug": w.slug, "role": r} for w, r in T.visible_workspaces(db, u)]}


# ------------------------------------------------------------------ organisation admin (own organisation only)
class UserIn(BaseModel):
    email: str
    role: Literal["admin", "analyst", "viewer", "member"] = "member"


class DomainsIn(BaseModel):
    allowed_email_domains: list[str]


@app.post("/admin/users")
def create_user(body: UserIn, u: User = Depends(require("admin")), db=Depends(get_db)):
    """Create a user (typically an API service account) in the caller's own organisation."""
    org = db.get(Organization, u.organization_id)
    if db.scalar(select(User).where(User.email == body.email.strip().lower())):
        raise HTTPException(409, "a user with this email already exists")
    nu = A.create_user(db, body.email.strip().lower(), org, body.role)
    key = A.issue_api_key(db, nu)
    db.add(AuditLog(user_id=u.id, action="user_created", detail={"email": nu.email, "role": body.role}))
    db.commit()
    return {"id": nu.id, "email": nu.email, "role": nu.role, "api_key": key,
            "note": "Store this key now; only its hash is kept. Grant workspace access with /workspaces/{id}/members."}


@app.put("/admin/organization/sso")
def set_sso_domains(body: DomainsIn, u: User = Depends(require("admin")), db=Depends(get_db)):
    from dsdvc.tenancy import access as T
    try:
        return {"allowed_email_domains": T.set_allowed_domains(db, u, body.allowed_email_domains)}
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.delete("/admin/api-keys/{prefix}")
def revoke(prefix: str, u: User = Depends(require("admin")), db=Depends(get_db)):
    from dsdvc.database.models import ApiKey
    key = db.scalar(select(ApiKey).where(ApiKey.prefix == prefix))
    owner = db.get(User, key.user_id) if key else None
    if owner is None or owner.organization_id != u.organization_id:
        raise HTTPException(404, "key not found")
    return {"revoked": A.revoke_api_key(db, prefix)}


# ------------------------------------------------------------------ workspaces (client separation)
class WorkspaceIn(BaseModel):
    name: str
    description: str | None = None
    slug: str | None = None


class MemberIn(BaseModel):
    email: str
    role: Literal["admin", "analyst", "viewer"]


@app.get("/workspaces")
def list_workspaces(u: User = Depends(current_user), db=Depends(get_db)):
    from dsdvc.tenancy import access as T
    return [{"id": w.id, "name": w.name, "slug": w.slug, "role": r} for w, r in T.visible_workspaces(db, u)]


@app.post("/workspaces")
def create_workspace(body: WorkspaceIn, u: User = Depends(current_user), db=Depends(get_db)):
    from dsdvc.tenancy import access as T
    try:
        w = T.create_workspace(db, u, body.name, body.description, body.slug)
    except T.Forbidden as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"id": w.id, "name": w.name, "slug": w.slug,
            "credentials_namespace": f"WS_{w.slug.upper()}_* environment variables or [workspaces.{w.slug}] secrets"}


@app.get("/workspaces/{workspace_id}/members")
def get_members(workspace_id: int, u: User = Depends(current_user), db=Depends(get_db)):
    from dsdvc.tenancy import access as T
    try:
        return T.members(db, u, workspace_id)
    except T.NotFound:
        raise HTTPException(404, "workspace not found")


@app.put("/workspaces/{workspace_id}/members")
def put_member(workspace_id: int, body: MemberIn, u: User = Depends(current_user), db=Depends(get_db)):
    from dsdvc.tenancy import access as T
    try:
        m = T.set_member(db, u, workspace_id, body.email, body.role)
    except T.NotFound:
        raise HTTPException(404, "workspace not found")
    except T.Forbidden as e:
        raise HTTPException(403, str(e))
    return {"email": body.email.lower(), "role": m.role}


@app.delete("/workspaces/{workspace_id}/members/{email}")
def delete_member(workspace_id: int, email: str, u: User = Depends(current_user), db=Depends(get_db)):
    from dsdvc.tenancy import access as T
    try:
        return {"removed": T.remove_member(db, u, workspace_id, email)}
    except T.NotFound:
        raise HTTPException(404, "workspace not found")
    except T.Forbidden as e:
        raise HTTPException(403, str(e))


@app.get("/workspace/audit")
def workspace_audit(limit: int = 200, ctx=Depends(in_workspace("admin")), db=Depends(get_db)):
    _, ws, _ = ctx
    rows = db.scalars(select(AuditLog).where(AuditLog.project_id == ws.id).order_by(AuditLog.created_at.desc()).limit(limit)).all()
    emails = {x.id: x.email for x in db.scalars(select(User).where(User.id.in_({r.user_id for r in rows if r.user_id}))).all()}
    return json_safe([{"at": r.created_at.isoformat(), "user": emails.get(r.user_id), "action": r.action, "detail": r.detail}
                      for r in rows])


@app.get("/workspace/datasets")
def workspace_datasets(ctx=Depends(in_workspace("viewer")), db=Depends(get_db)):
    from dsdvc.onboarding.workspace import list_datasets
    _, ws, _ = ctx
    return [{"id": d.id, "filename": d.filename, "rows": d.rows, "status": d.status, "sha256": d.sha256,
             "created_at": d.created_at.isoformat()} for d in list_datasets(db, ws.id)]


# ------------------------------------------------------------------ simulation
class SimIn(BaseModel):
    n_agents: int = Field(1000, ge=100, le=20000)
    steps: int = Field(90, ge=10, le=365)
    seed: int = 0
    network_seed: int = 0
    homophily_scale: float = 1.0
    seeding_strategy: Literal["celebrity_first", "micro_network", "community_seeding", "hybrid", "random"] = "micro_network"
    seeding_budget: float = 200_000.0
    stepps: dict[str, float] | None = None
    stepps_form: Literal["additive", "multiplicative", "interaction"] = "multiplicative"
    complex_share: float = Field(0.5, ge=0, le=1)
    paid_daily_budget: dict[str, float] | None = None
    retention_investment: float = 0.2
    population_scale: float = 100.0


def _cfg(body: SimIn):
    from dsdvc.diffusion.abm import SimConfig
    kw = body.model_dump(exclude={"stepps", "paid_daily_budget"})
    cfg = SimConfig(**kw)
    if body.stepps:
        cfg.stepps.update(body.stepps)
    if body.paid_daily_budget:
        cfg.paid_daily_budget.update(body.paid_daily_budget)
    cfg.cost_model.population_scale = cfg.population_scale
    return cfg


@app.post("/simulations")
def simulate(body: SimIn, u: User = Depends(require("analyst")), db=Depends(get_db)):
    from dsdvc.diffusion.abm import run_simulation
    key = Cache.key({"sim": body.model_dump(), "v": SIMULATION_VERSION})
    hit = CACHE.get(key)
    if hit:
        METRICS.inc("dsdvc_sim_cache_hits")
        return hit
    with METRICS.timer("dsdvc_model_latency", model="abm"):
        res = run_simulation(_cfg(body))
    rr = res.run_record
    summary = json_safe(res.summary())
    db.add(SimulationRunRow(run_id=rr.run_id, project_id=None, kind="abm", random_seed=rr.random_seed,
                            network_seed=rr.network_seed, parameters=rr.parameters, data_version=rr.data_version,
                            model_version=rr.model_version, simulation_version=rr.simulation_version,
                            software_version=rr.software_version, result_hash=rr.result_hash, summary=summary,
                            created_by=u.id))
    db.commit()
    log_to_mlflow(f"abm-{rr.run_id}", rr.parameters, {k: v for k, v in summary.items() if isinstance(v, (int, float))})
    out = json_safe({"status": "SIMULATED", "summary": summary, "run_record": rr.model_dump(mode="json"),
                     "timeseries": res.timeseries.to_dict(orient="list"), **meta(rr.data_version)})
    CACHE.set(key, out)
    return out


@app.get("/simulations/{run_id}")
def get_sim(run_id: str, u: User = Depends(require("viewer")), db=Depends(get_db)):
    row = db.scalar(select(SimulationRunRow).where(SimulationRunRow.run_id == run_id))
    creator = db.get(User, row.created_by) if row is not None and row.created_by else None
    if row is None or creator is None or creator.organization_id != u.organization_id:
        raise HTTPException(404, "run not found")
    return {"run_id": row.run_id, "parameters": row.parameters, "summary": row.summary, "random_seed": row.random_seed,
            "network_seed": row.network_seed, "software_version": row.software_version, "result_hash": row.result_hash,
            "model_version": row.model_version, "data_version": row.data_version}


class CounterfactualIn(BaseModel):
    simulation: SimIn = SimIn()
    scenario: Literal["instagram_reach_minus_30", "celebrity_to_micro", "whatsapp_trust_plus_15",
                      "competitor_price_minus_10", "stepps_emotion_up", "more_segregated"]
    n_reps: int = Field(6, ge=2, le=50)


@app.post("/counterfactuals")
def counterfactual(body: CounterfactualIn, u: User = Depends(require("analyst")), db=Depends(get_db)):
    from dsdvc.experiments.counterfactual import run_counterfactual
    r = run_counterfactual(_cfg(body.simulation), body.scenario, n_reps=body.n_reps)
    table = r["table"].to_dict(orient="records")
    db.add(ExperimentRow(project_id=None, design="counterfactual", question=r["question"],
                         results={"table": table, "interventions": r["interventions"]}))
    db.commit()
    return json_safe({**{k: v for k, v in r.items() if k != "table"}, "table": table, **meta()})


class StrategiesIn(BaseModel):
    simulation: SimIn = SimIn()
    n_reps: int = Field(5, ge=2, le=50)


@app.post("/strategies/compare")
def strategies(body: StrategiesIn, u: User = Depends(require("analyst"))):
    from dsdvc.experiments.counterfactual import compare_seeding_strategies
    t = compare_seeding_strategies(_cfg(body.simulation), n_reps=body.n_reps)
    return json_safe({"status": "SIMULATED", "table": t.reset_index().to_dict(orient="records"), **meta()})


# ------------------------------------------------------------------ models
class SeriesIn(BaseModel):
    values: list[float]


@app.post("/bass/fit")
def bass_fit(body: SeriesIn, u: User = Depends(require("analyst"))):
    from dsdvc.calibration.engine import CalibrationEngine
    y = np.asarray(body.values, float)
    if len(y) < 6:
        raise HTTPException(422, "need at least 6 periods")
    e = CalibrationEngine("request")
    return json_safe({"nls": e.bass_nls(y).table().to_dict(orient="records"),
                      "mle": e.bass_mle(y).table().to_dict(orient="records"), "status": "CALIBRATED", **meta()})


class TextIn(BaseModel):
    text: str
    media_type: str | None = "reel"
    transcript: str | None = None


@app.post("/stepps/analyze")
def stepps(body: TextIn, u: User = Depends(require("viewer"))):
    from dsdvc.content.stepps import STEPPSAnalyzer
    return {**STEPPSAnalyzer().analyze(body.text, body.media_type, body.transcript).to_dict(), **meta()}


class CopilotIn(BaseModel):
    concept: str = ""
    caption: str = ""
    script: str = ""
    transcript: str = ""
    brief: str = ""
    media_type: str = "reel"
    model_id: str | None = None
    banned_terms: list[str] = []


@app.post("/copilot")
def copilot(body: CopilotIn, u: User = Depends(require("viewer"))):
    from dsdvc.content.copilot import ContentCopilot
    pred = VIRALITY_MODELS.get(body.model_id) if body.model_id else None
    if pred is not None and getattr(pred, "owner_org", None) != u.organization_id:
        pred = None
    rep = ContentCopilot(pred, brand_constraints={"banned_terms": body.banned_terms}).analyze(
        body.concept, body.caption, body.script, body.transcript, body.brief, body.media_type)
    return json_safe({**rep.__dict__, **meta()})


class ContentIn(BaseModel):
    rows: list[dict] | None = None
    synthetic: bool = False
    top_pct: float = 0.05
    horizon_h: int = 3


@app.post("/virality/train")
def virality_train(body: ContentIn, u: User = Depends(require("analyst"))):
    from dsdvc.content.virality import ViralityPredictor
    if body.rows:
        df = pd.DataFrame(body.rows)
        dv = "user_rows"
    elif body.synthetic:
        from dsdvc.data.synthetic import synthetic_content
        df, dv = synthetic_content(3000), "synthetic"
    else:
        raise HTTPException(422, "provide rows or set synthetic=true (DATA_UNAVAILABLE otherwise)")
    vp = ViralityPredictor(top_pct=body.top_pct, horizon_h=body.horizon_h)
    with METRICS.timer("dsdvc_model_latency", model="virality_train"):
        rep = vp.fit(df, data_version=dv)
    mid = uuid.uuid4().hex[:10]
    vp.owner_org = u.organization_id
    VIRALITY_MODELS[mid] = vp
    return json_safe({"model_id": mid, "data_version": dv, "definition": rep["viral_definition"], "weights": rep["weights"],
                      "benchmarks": rep["benchmarks"].to_dict(orient="records"), **meta(dv)})


class PredictIn(BaseModel):
    model_id: str
    rows: list[dict]


@app.post("/virality/predict")
def virality_predict(body: PredictIn, u: User = Depends(require("viewer")), db=Depends(get_db)):
    vp = VIRALITY_MODELS.get(body.model_id)
    if vp is None or getattr(vp, "owner_org", None) != u.organization_id:
        raise HTTPException(404, "unknown model_id (models are held in memory; retrain after restart)")
    t = time.perf_counter()
    out = vp.predict(pd.DataFrame(body.rows))
    ms = (time.perf_counter() - t) * 1000
    METRICS.observe("dsdvc_model_latency", ms, model="virality_predict")
    out = json_safe(out)
    for o in out:
        db.add(PredictionRow(target="p_viral_top_pct", payload=o, model_version=MODEL_VERSION,
                             data_version=vp.data_version, latency_ms=ms))
    db.commit()
    return {"predictions": out, **meta(vp.data_version)}


class AttributionIn(BaseModel):
    paths: list[list[str]]
    converted: list[bool]


@app.post("/attribution")
def attribution(body: AttributionIn, u: User = Depends(require("viewer"))):
    from dsdvc.attribution.models import all_attributions
    if len(body.paths) != len(body.converted):
        raise HTTPException(422, "paths and converted must have equal length")
    t = all_attributions(pd.DataFrame({"path": body.paths, "converted": body.converted}))
    return json_safe({"attributed_conversions": t.reset_index(names="channel").to_dict(orient="records"),
            "note": "Attributed, not incremental. Use /attribution/incrementality with experiment data.", **meta()})


class IncrIn(BaseModel):
    treat_conv: int
    treat_n: int
    ctrl_conv: int
    ctrl_n: int


@app.post("/attribution/incrementality")
def incrementality(body: IncrIn, u: User = Depends(require("viewer"))):
    from dsdvc.attribution.models import incrementality_test
    return json_safe({**incrementality_test(**body.model_dump()), **meta()})


class DailyIn(BaseModel):
    rows: list[dict]


@app.post("/forecast")
def forecast(body: DailyIn, u: User = Depends(require("analyst"))):
    from dsdvc.forecasting.forecast import Forecaster
    df = pd.DataFrame(body.rows)
    if len(df) < 28:
        raise HTTPException(422, "DATA_UNAVAILABLE: need >= 28 daily rows")
    return json_safe({"forecast": Forecaster().forecast_frame(df).to_dict(orient="records"), "status": "PREDICTED", **meta()})


@app.post("/regime")
def regime(body: DailyIn, u: User = Depends(require("viewer"))):
    from dsdvc.forecasting.early_warning import scan
    from dsdvc.forecasting.regime import detect_regime
    df = pd.DataFrame(body.rows)
    return json_safe({"regime": detect_regime(df), "alerts": scan(df), **meta()})


@app.post("/data-quality")
def data_quality(body: DailyIn, u: User = Depends(require("viewer"))):
    from dsdvc.data.quality import DataQualityEngine
    r = DataQualityEngine().evaluate(pd.DataFrame(body.rows))
    return json_safe({"score": r.score, "checks": r.table().to_dict(orient="records"), **meta()})


@app.get("/connectors/status")
def connectors(ctx=Depends(in_workspace("viewer"))):
    from dsdvc.connectors.credentials import settings_for_workspace
    from dsdvc.connectors.credentials import status as conn_status
    _, ws, _ = ctx
    s = settings_for_workspace(ws.slug)
    return {"workspace": ws.slug, **{f"{k}_connector": v for k, v in conn_status(s).items()},
            "instagram_graph_api": bool(s.meta_access_token and s.instagram_business_account_id),
            "facebook_pages_api": bool(s.meta_access_token and s.facebook_page_id),
            "whatsapp_business_platform": bool(s.whatsapp_business_token and s.whatsapp_business_account_id),
            "meta_content_library": "approved-researcher exports only",
            "note": "Unconfigured connectors return DATA_UNAVAILABLE; DSDVC never scrapes or bypasses access controls."}


# ------------------------------------------------------------------ onboarding → MMM → exports
MMM_RUNS: dict[int, dict] = {}


def _read_upload(raw: bytes, filename: str) -> pd.DataFrame:
    import io
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw), dtype=str)
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), dtype=str, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise HTTPException(422, "could not decode file")


@app.get("/onboarding/templates/{key}")
def onboarding_template(key: str, u: User = Depends(require("viewer"))):
    from dsdvc.onboarding.templates import TEMPLATES, template_frame
    if key not in TEMPLATES:
        raise HTTPException(404, f"unknown template; choose from {list(TEMPLATES)}")
    spec = TEMPLATES[key]
    return {"key": key, "title": spec.title, "grain": spec.grain, "purpose": spec.purpose,
            "dictionary": spec.dictionary().to_dict(orient="records"),
            "example_csv": template_frame(key).to_csv(index=False)}


@app.post("/onboarding/preview")
async def onboarding_preview(file: UploadFile = File(...), mapping: str | None = Form(None),
                             ctx=Depends(in_workspace("analyst"))):
    """Suggest (or apply a supplied) column mapping and return the validation scorecard. Stores nothing."""
    import json
    from dsdvc.onboarding.mapping import ColumnMatch, Mapping, apply_mapping, suggest_mapping
    from dsdvc.onboarding.templates import MARKETING_WEEKLY
    from dsdvc.onboarding.validate import validate
    raw = _read_upload(await file.read(), file.filename)
    m = suggest_mapping(raw, MARKETING_WEEKLY)
    if mapping:
        user_map = json.loads(mapping)
        m = Mapping("marketing_weekly", [ColumnMatch(c, user_map.get(c, "(ignore)"), 1.0, "supplied") for c in raw.columns])
    canon, notes = apply_mapping(raw, m, MARKETING_WEEKLY)
    sc = validate(canon, MARKETING_WEEKLY)
    return json_safe({"mapping": m.table().to_dict(orient="records"), "parse_notes": notes, "status": sc.status,
                      "score": sc.score, "checks": sc.table().to_dict(orient="records"),
                      "preview": canon.head(10).astype(str).to_dict(orient="records")})


@app.post("/onboarding/datasets")
async def onboarding_save(file: UploadFile = File(...), attestation: str = Form(...), mapping: str | None = Form(None),
                          ctx=Depends(in_workspace("analyst")), db=Depends(get_db)):
    u, ws, _ = ctx
    import json
    from dsdvc.onboarding.mapping import ColumnMatch, Mapping, apply_mapping, suggest_mapping
    from dsdvc.onboarding.templates import MARKETING_WEEKLY
    from dsdvc.onboarding.validate import validate
    from dsdvc.onboarding.workspace import save_dataset
    raw = _read_upload(await file.read(), file.filename)
    m = suggest_mapping(raw, MARKETING_WEEKLY)
    if mapping:
        user_map = json.loads(mapping)
        m = Mapping("marketing_weekly", [ColumnMatch(c, user_map.get(c, "(ignore)"), 1.0, "supplied") for c in raw.columns])
    canon, _ = apply_mapping(raw, m, MARKETING_WEEKLY)
    sc = validate(canon, MARKETING_WEEKLY)
    try:
        row = save_dataset(db, df=canon, dataset="marketing_weekly", filename=file.filename, mapping=m, scorecard=sc,
                           attestation=attestation, project_id=ws.id, user_id=u.id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"dataset_id": row.id, "sha256": row.sha256, "rows": row.rows, "status": row.status, "mapping": row.mapping}


class MMMFitIn(BaseModel):
    dataset_id: int
    target: Literal["revenue", "conversions"] = "revenue"
    channels: list[str] | None = None
    controls: list[str] | None = None
    holdout_weeks: int = Field(12, ge=0, le=52)
    lift_tests: list[dict] = []


@app.post("/mmm/fit")
def mmm_fit(body: MMMFitIn, ctx=Depends(in_workspace("analyst")), db=Depends(get_db)):
    u, ws, _ = ctx
    from dsdvc.database.models import DatasetVersion, MMMRun
    from dsdvc.mmm import analysis as AN
    from dsdvc.mmm.model import LiftTest, MarketingMixModel, MMMSpec
    from dsdvc.onboarding.validate import spend_columns
    from dsdvc.onboarding.workspace import load_dataset
    row = db.get(DatasetVersion, body.dataset_id)
    if row is None or row.project_id != ws.id:
        raise HTTPException(404, "dataset not found in this workspace")
    df = load_dataset(row)
    if body.target not in df:
        raise HTTPException(422, f"dataset has no '{body.target}' column")
    chans = body.channels or spend_columns(df)
    ctrls = body.controls if body.controls is not None else [c for c in ("price_index", "promo", "holiday", "competitor_spend") if c in df]
    lifts = [LiftTest(l["channel"] if l["channel"].startswith("spend_") else f"spend_{l['channel']}", pd.Timestamp(l["start"]),
                      pd.Timestamp(l["end"]), float(l["lift"]), float(l["se"])) for l in body.lift_tests]
    spec = MMMSpec(target_col=body.target, channels=chans, controls=ctrls)
    with METRICS.timer("dsdvc_model_latency", model="mmm_fit"):
        fit = MarketingMixModel(spec).fit(df, lifts, data_version=row.sha256[:16])
        bt = AN.backtest(MarketingMixModel(spec), df, body.holdout_weeks, lifts) if body.holdout_weeks else None
    summ = AN.channel_summary(fit)
    run = MMMRun(project_id=row.project_id, dataset_version_id=row.id, spec=json_safe(spec.__dict__),
                 diagnostics=json_safe(fit.diagnostics), channel_summary=json_safe(summ.to_dict(orient="records")),
                 backtest=json_safe({"verdict": bt.verdict, **bt.metrics}) if bt else None,
                 model_version=fit.model_version, data_version=fit.data_version, created_by=u.id)
    db.add(run)
    db.commit()
    MMM_RUNS[run.id] = {"fit": fit, "bt": bt, "summary": summ, "dataset": row, "workspace_id": ws.id}
    T_audit(db, u, ws.id, "mmm_fit", run_id=run.id, dataset_id=row.id, verdict=bt.verdict if bt else None)
    return json_safe({"run_id": run.id, "status": "ESTIMATED", "diagnostics": fit.diagnostics,
                      "backtest": {"verdict": bt.verdict, **bt.metrics} if bt else None,
                      "channels": summ.to_dict(orient="records"), **meta(fit.data_version)})


class PlanIn(BaseModel):
    weekly_budget: float | None = None
    objective: Literal["outcome", "profit"] = "outcome"
    max_change: float = Field(0.5, gt=0, le=1)
    margin: float = Field(1.0, gt=0, le=1)


def _run(run_id: int, ws_id: int) -> dict:
    if run_id not in MMM_RUNS or MMM_RUNS[run_id]["workspace_id"] != ws_id:
        raise HTTPException(404, "run not in memory (fitted models are kept in memory; refit after a restart)")
    return MMM_RUNS[run_id]


@app.post("/mmm/{run_id}/plan")
def mmm_plan(run_id: int, body: PlanIn, ctx=Depends(in_workspace("analyst")), db=Depends(get_db)):
    u, ws, _ = ctx
    from dsdvc.database.models import MMMRun
    from dsdvc.mmm import analysis as AN
    r = _run(run_id, ws.id)
    plan = AN.optimize_budget(r["fit"], body.weekly_budget, max_change=body.max_change, objective=body.objective, margin=body.margin)
    r["plan"] = plan
    out = {"weekly_budget": plan.weekly_budget, "objective": plan.objective, "uplift": plan.uplift, "uplift_low": plan.uplift_low,
           "uplift_high": plan.uplift_high, "p_better": plan.p_better, "notes": plan.notes,
           "table": plan.table.to_dict(orient="records")}
    row = db.get(MMMRun, run_id)
    if row is not None:
        row.plan = json_safe(out)
        db.commit()
    return json_safe({"status": "RECOMMENDED", **out, **meta(r["fit"].data_version)})


@app.get("/mmm/{run_id}/export/{fmt}")
def mmm_export(run_id: int, fmt: Literal["xlsx", "pptx"], client: str = "Client", horizon_weeks: int = 13,
               start: str | None = None, currency: str = "₹", ctx=Depends(in_workspace("viewer")), db=Depends(get_db)):
    u, ws, _ = ctx
    import datetime as dt
    from fastapi.responses import Response
    from dsdvc.mmm import analysis as AN
    r = _run(run_id, ws.id)
    T_audit(db, u, ws.id, "export_downloaded", run_id=run_id, format=fmt)
    if "plan" not in r:
        raise HTTPException(409, "create a plan first: POST /mmm/{run_id}/plan")
    fit, ds = r["fit"], r["dataset"]
    prov = {"filename": ds.filename, "rows": ds.rows, "sha256": ds.sha256, "validation": ds.status, "attestation": ds.attestation,
            "data_version": fit.data_version, "lift_tests": len(fit.lift_tests)}
    start_d = dt.date.fromisoformat(start) if start else dt.date.today()
    if fmt == "xlsx":
        from dsdvc.exports.excel import media_plan_workbook
        data = media_plan_workbook(client=client, plan=r["plan"], summary=r["summary"], curves=AN.response_curves(fit),
                                   backtest=r["bt"], params=AN.channel_params(fit), horizon_weeks=horizon_weeks, start=start_d,
                                   diagnostics=fit.diagnostics, provenance=prov, currency=currency)
        mt = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        from dsdvc.exports.deck import executive_deck
        data = executive_deck(client=client, plan=r["plan"], summary=r["summary"], decomposition=AN.decomposition(fit),
                              backtest=r["bt"], diagnostics=fit.diagnostics, provenance=prov, horizon_weeks=horizon_weeks,
                              outcome_label=fit.spec.target_col, currency=currency)
        mt = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    return Response(data, media_type=mt, headers={"Content-Disposition": f'attachment; filename="mmm_run_{run_id}.{fmt}"'})


class PullIn(BaseModel):
    start: str
    end: str
    sources: list[Literal["meta", "ga4", "shopify"]] = ["meta", "ga4", "shopify"]
    meta_mode: Literal["mmm", "insights"] = "mmm"
    campaign_rules: list[tuple[str, str]] = []
    split_platforms: bool = False
    outcome_source: Literal["auto", "shopify", "ga4"] = "auto"
    demo: bool = False
    save: bool = False
    attestation: str | None = None


@app.post("/connectors/pull")
def connectors_pull(body: PullIn, ctx=Depends(in_workspace("analyst")), db=Depends(get_db)):
    u, ws, _ = ctx
    """Pull Meta Ads / GA4 / Shopify for a date range and assemble the weekly MMM dataset.
    ``demo=true`` uses the labelled SYNTHETIC API simulators. ``save=true`` stores it as a dataset version."""
    import datetime as dt
    from dsdvc.connectors import credentials as C
    from dsdvc.connectors.meta_ads import ChannelRules
    from dsdvc.connectors.pull import pull_weekly
    from dsdvc.onboarding.mapping import suggest_mapping
    from dsdvc.onboarding.templates import MARKETING_WEEKLY
    from dsdvc.onboarding.validate import validate
    start, end = dt.date.fromisoformat(body.start), dt.date.fromisoformat(body.end)
    if start >= end:
        raise HTTPException(422, "start must be before end")
    with METRICS.timer("dsdvc_connector_latency", source="+".join(body.sources)):
        res, errors = pull_weekly(C.connectors(C.settings_for_workspace(ws.slug), demo=body.demo), start, end, list(body.sources), meta_mode=body.meta_mode,
                                  rules=ChannelRules(campaign_contains=list(body.campaign_rules), split_platforms=body.split_platforms),
                                  outcome_source=body.outcome_source)
    for k in errors:
        METRICS.inc("dsdvc_ingestion_failures", source=k)
    T_audit(db, u, ws.id, "connector_pull", sources=list(body.sources), start=body.start, end=body.end, demo=body.demo,
            errors=sorted(errors))
    if res is None:
        return json_safe({"status": "DATA_UNAVAILABLE", "errors": errors})
    sc = validate(res.data, MARKETING_WEEKLY)
    out = {"status": sc.status, "rows": len(res.data), "columns": res.column_sources, "warnings": res.warnings,
           "errors": errors, "checks": sc.table().to_dict(orient="records"), "data_status": "SYNTHETIC" if body.demo else "OBSERVED",
           "reconciliation": res.reconciliation.to_dict(orient="records") if res.reconciliation is not None else None,
           "data": res.data.assign(date=res.data.date.dt.strftime("%Y-%m-%d")).to_dict(orient="records")}
    if body.save:
        from dsdvc.onboarding.workspace import save_dataset
        att = (body.attestation or res.attestation_text()) + (" [SYNTHETIC demo accounts]" if body.demo else "")
        try:
            row = save_dataset(db, df=res.data, dataset="marketing_weekly",
                               filename=f"api_pull_{'_'.join(body.sources)}_{start}_{end}",
                               mapping=suggest_mapping(res.data, MARKETING_WEEKLY), scorecard=sc, attestation=att,
                               project_id=ws.id, user_id=u.id)
            out["dataset_id"] = row.id
        except ValueError as e:
            raise HTTPException(422, str(e))
    return json_safe(out)


# ------------------------------------------------------------------ research agent (async job)
class ResearchIn(BaseModel):
    request: str
    allow_synthetic: bool = False
    budget: float = 600_000.0
    n_agents: int = Field(1000, ge=200, le=5000)
    reps: int = Field(3, ge=2, le=20)


def _run_research(job_id: str, body: ResearchIn, workspace_id: int):
    from dsdvc.research_agent.agent import MarketingResearchAgent, ResearchInputs
    from dsdvc.research_agent.report import render_markdown
    try:
        JOBS[job_id]["status"] = "running"
        mem, S = MarketingResearchAgent(allow_synthetic=body.allow_synthetic, n_agents=body.n_agents,
                                        reps=body.reps).run(body.request, ResearchInputs(budget=body.budget))
        md = render_markdown(mem, S)
        with SessionLocal() as db:
            db.add(ResearchProjectMemory(project_id=workspace_id, memory_id=mem.project_id,
                                         memory=json_safe(mem.model_dump(mode="json")), report_markdown=md))
            db.commit()
        JOBS[job_id].update(json_safe(dict(status="done", memory_id=mem.project_id, report_markdown=md,
                            plan=mem.plan, recommendations=[c.model_dump(mode="json") for c in mem.recommendations])))
    except Exception as e:  # noqa: BLE001
        METRICS.inc("dsdvc_research_failures")
        JOBS[job_id].update(status="failed", error=f"{type(e).__name__}: {e}")


@app.post("/research")
def research(body: ResearchIn, ctx=Depends(in_workspace("analyst"))):
    u, ws, _ = ctx
    job = uuid.uuid4().hex[:12]
    JOBS[job] = {"status": "queued", "created_at": _now(), "user": u.id, "workspace_id": ws.id}
    threading.Thread(target=_run_research, args=(job, body, ws.id), daemon=True).start()
    return {"job_id": job, "status": "queued"}


@app.get("/research/{job_id}")
def research_status(job_id: str, ctx=Depends(in_workspace("viewer"))):
    _, ws, _ = ctx
    if job_id not in JOBS or JOBS[job_id].get("workspace_id") != ws.id:
        raise HTTPException(404, "unknown job")
    return JOBS[job_id]
