# Architecture

```
Observed world (authorised data)                Synthetic world (agent-based twin)
───────────────────────────────                 ──────────────────────────────────
platforms/ connectors ─┐                        networks/multiplex  (IG directed follow graph,
user exports / surveys ├─► data/ (schemas,         FB friends + Groups, WA strong ties + chats)
market tables ─────────┘    DataQualityEngine)  agents/seeding      (celebrity/micro/community/hybrid)
          │                                     diffusion/abm       (event-driven MarketSimulator)
          ▼                                              ▲
   calibration/engine ───── parameters ─────────────────┘   (RealWorldData → Calibration →
          │                                                  SyntheticPopulation → Simulation)
          ▼
content/ (STEPPS, virality, velocity, cascades, copilot)   conversion/ churn/ competition/ attribution/
          │                                                          │
          └───────────────► experiments/ (MC, LHS, Sobol, Morris, A/B, counterfactuals)
                                      │            forecasting/ (forecasts, regimes, early warning)
                                      ▼            optimization/ (budget)
                            research_agent/ (14-step plan, tools, memory, report)
                                      │
                         api/ (FastAPI) · dashboard/ (Streamlit) · database/ (SQLAlchemy)
```

## Package map (`src/dsdvc/`)

| Package | Responsibility |
|---|---|
| `core` | provenance & `DATA_UNAVAILABLE`, prediction envelope, reproducibility (`RunRecord`, `json_safe`), literature registry, statistics (`fit_logit`), observability |
| `config` | environment-driven settings (no secrets in code) |
| `data` | Pydantic domain models, data-quality engine, synthetic generators |
| `platforms` | connector contract (`base`), Meta connectors (`meta`), per-platform packages, transmission matrix |
| `networks` | multiplex construction, metrics, community detection, Schelling-inspired rewiring, influencer scoring |
| `contagion` | simple/complex contagion, threshold & independent-cascade models |
| `recommendation` | estimated exposure model (explicitly *not* a proprietary algorithm) |
| `content` | features, STEPPS, STEPPS form comparison, virality, velocity, cascades, copilot |
| `diffusion` | Bass family, event bus, agent-based simulator |
| `agents` | seeding strategies & creator cost model |
| `conversion` / `churn` / `competition` / `attribution` | outcome models |
| `calibration` / `validation` / `experiments` / `forecasting` / `optimization` | scientific workflow |
| `research_agent` | orchestration, tools, memory, report, optional LLM narrator |
| `database` / `api` / `dashboard` | serving |

## Event-driven simulation

One tick = one day. Within a tick the simulator emits typed events through an `EventBus`
(`diffusion/events.py`): `CAMPAIGN_LAUNCHED, CONTENT_CREATED, CONTENT_EXPOSED, CONTENT_ENGAGED,
CONTENT_SHARED, CROSS_PLATFORM_TRANSFER, AWARENESS_CREATED, TRIAL_STARTED, ADOPTION_OCCURRED,
ADVOCACY_OCCURRED, CHURN_OCCURRED, RECOMMENDATION_UPDATED, COMMUNITY_REWIRED, COMPETITOR_ACTION`.
New mechanisms subscribe handlers (`sim.bus.subscribe(EventType.X, fn)`) rather than editing the
loop. High-volume events (exposure, engagement, recommendation, awareness) are aggregated per tick.

Order of operations per tick: competitor actions → broadcasting (sharers, creator/community posts)
→ exposure (social, group/chat, estimated algorithmic, paid, external Bass *p*) → awareness →
engagement & sharing (+ cross-platform transfer, cascade parent assignment) → funnel transitions
(interest → simple/complex-contagion consideration → category purchase × MNL brand choice → trial →
adoption) → lifecycle (satisfaction, engagement, advocacy, dissatisfaction, churn, recovery,
competitor switching) → economics → periodic Schelling rewiring → record.

## Journey state machine

`UNAWARE → AWARE → INTERESTED → CONSIDERING → TRIAL → ADOPTED → ENGAGED → ADVOCATE`, with
`ADOPTED/ENGAGED/ADVOCATE → DISSATISFIED → CHURNED → POSSIBLE_RECOVERY → (ADOPTED | CHURNED)`.
The simulator adds an internal `COMPETITOR` state for consumers who chose a rival brand.
`conversion/journey.py` fits a separate logistic model per allowed transition from panel data and
rejects data containing disallowed transitions.

## Reproducibility

`RunRecord` stores `random_seed`, `network_seed`, all parameters, `data_version`,
`model_version`, `simulation_version`, software versions and a hash of the output time series.
Identical inputs give identical hashes (`tests/test_simulation.py::test_reproducible_exactly`).
Simulator configs are deep-copied so runs cannot leak state into each other.

## Serving

* **API** — FastAPI, API-key auth (PBKDF2-hashed keys), roles admin > analyst > viewer, every
  output carries `model_version`, `data_version` and `timestamp`; simulations persisted with their
  full `RunRecord`; research runs as background jobs.
* **Dashboard** — Streamlit, 22 tabs, epistemic badges on every panel; authorised-only by default.
* **Storage** — PostgreSQL (SQLite locally); Redis cache for deterministic simulation results;
  MLflow run logging; S3-compatible object storage via MinIO in `docker-compose.yml`.
