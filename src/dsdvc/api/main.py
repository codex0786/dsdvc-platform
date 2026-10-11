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
from fastapi import Depends, FastAPI, Header, HTTPException, Request
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


def current_user(x_api_key: str | None = Header(default=None), db=Depends(get_db)) -> User:
    if not x_api_key:
        raise HTTPException(401, "missing X-API-Key")
    u = A.verify_api_key(db, x_api_key)
    if u is None:
        raise HTTPException(401, "invalid API key")
    return u


def require(role: str):
    def dep(u: User = Depends(current_user)) -> User:
        if not A.has_role(u, role):
            raise HTTPException(403, f"requires role '{role}'")
        return u
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


# ------------------------------------------------------------------ admin
class OrgIn(BaseModel):
    name: str


class UserIn(BaseModel):
    email: str
    organization: str
    role: Literal["admin", "analyst", "viewer"] = "viewer"


class ProjectIn(BaseModel):
    organization: str
    name: str
    description: str | None = None


@app.post("/admin/organizations")
def create_org(body: OrgIn, u: User = Depends(require("admin")), db=Depends(get_db)):
    o = A.create_org(db, body.name)
    return {"id": o.id, "name": o.name}


@app.post("/admin/users")
def create_user(body: UserIn, u: User = Depends(require("admin")), db=Depends(get_db)):
    org = A.create_org(db, body.organization)
    nu = A.create_user(db, body.email, org, body.role)
    key = A.issue_api_key(db, nu)
    db.add(AuditLog(user_id=u.id, action="user_created", detail={"email": body.email, "role": body.role}))
    db.commit()
    return {"id": nu.id, "email": nu.email, "role": nu.role, "api_key": key,
            "note": "Store this key now; only its hash is kept."}


@app.post("/admin/projects")
def create_project(body: ProjectIn, u: User = Depends(require("admin")), db=Depends(get_db)):
    org = A.create_org(db, body.organization)
    p = A.create_project(db, org, body.name, body.description)
    return {"id": p.id, "name": p.name}


@app.delete("/admin/api-keys/{prefix}")
def revoke(prefix: str, u: User = Depends(require("admin")), db=Depends(get_db)):
    return {"revoked": A.revoke_api_key(db, prefix)}


@app.get("/me")
def me(u: User = Depends(current_user)):
    return {"id": u.id, "email": u.email, "role": u.role, "organization_id": u.organization_id}


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
    project_id: int | None = None


def _cfg(body: SimIn):
    from dsdvc.diffusion.abm import SimConfig
    kw = body.model_dump(exclude={"project_id", "stepps", "paid_daily_budget"})
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
    key = Cache.key({"sim": body.model_dump(exclude={"project_id"}), "v": SIMULATION_VERSION})
    hit = CACHE.get(key)
    if hit:
        METRICS.inc("dsdvc_sim_cache_hits")
        return hit
    with METRICS.timer("dsdvc_model_latency", model="abm"):
        res = run_simulation(_cfg(body))
    rr = res.run_record
    summary = json_safe(res.summary())
    db.add(SimulationRunRow(run_id=rr.run_id, project_id=body.project_id, kind="abm", random_seed=rr.random_seed,
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
    if row is None:
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
    db.add(ExperimentRow(project_id=body.simulation.project_id, design="counterfactual", question=r["question"],
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
    VIRALITY_MODELS[mid] = vp
    return json_safe({"model_id": mid, "data_version": dv, "definition": rep["viral_definition"], "weights": rep["weights"],
                      "benchmarks": rep["benchmarks"].to_dict(orient="records"), **meta(dv)})


class PredictIn(BaseModel):
    model_id: str
    rows: list[dict]


@app.post("/virality/predict")
def virality_predict(body: PredictIn, u: User = Depends(require("viewer")), db=Depends(get_db)):
    vp = VIRALITY_MODELS.get(body.model_id)
    if vp is None:
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
def connectors(u: User = Depends(require("viewer"))):
    s = get_settings()
    return {"instagram_graph_api": bool(s.meta_access_token and s.instagram_business_account_id),
            "facebook_pages_api": bool(s.meta_access_token and s.facebook_page_id),
            "whatsapp_business_platform": bool(s.whatsapp_business_token and s.whatsapp_business_account_id),
            "meta_content_library": "approved-researcher exports only",
            "note": "Unconfigured connectors return DATA_UNAVAILABLE; DSDVC never scrapes or bypasses access controls."}


# ------------------------------------------------------------------ research agent (async job)
class ResearchIn(BaseModel):
    request: str
    allow_synthetic: bool = False
    budget: float = 600_000.0
    n_agents: int = Field(1000, ge=200, le=5000)
    reps: int = Field(3, ge=2, le=20)
    project_id: int | None = None


def _run_research(job_id: str, body: ResearchIn):
    from dsdvc.research_agent.agent import MarketingResearchAgent, ResearchInputs
    from dsdvc.research_agent.report import render_markdown
    try:
        JOBS[job_id]["status"] = "running"
        mem, S = MarketingResearchAgent(allow_synthetic=body.allow_synthetic, n_agents=body.n_agents,
                                        reps=body.reps).run(body.request, ResearchInputs(budget=body.budget))
        md = render_markdown(mem, S)
        with SessionLocal() as db:
            db.add(ResearchProjectMemory(project_id=body.project_id, memory_id=mem.project_id,
                                         memory=json_safe(mem.model_dump(mode="json")), report_markdown=md))
            db.commit()
        JOBS[job_id].update(json_safe(dict(status="done", memory_id=mem.project_id, report_markdown=md,
                            plan=mem.plan, recommendations=[c.model_dump(mode="json") for c in mem.recommendations])))
    except Exception as e:  # noqa: BLE001
        METRICS.inc("dsdvc_research_failures")
        JOBS[job_id].update(status="failed", error=f"{type(e).__name__}: {e}")


@app.post("/research")
def research(body: ResearchIn, u: User = Depends(require("analyst"))):
    job = uuid.uuid4().hex[:12]
    JOBS[job] = {"status": "queued", "created_at": _now(), "user": u.id}
    threading.Thread(target=_run_research, args=(job, body), daemon=True).start()
    return {"job_id": job, "status": "queued"}


@app.get("/research/{job_id}")
def research_status(job_id: str, u: User = Depends(require("viewer"))):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    return JOBS[job_id]
