# API

Interactive docs: `http://localhost:8000/docs` (OpenAPI).

## Authentication & roles
* Header `X-API-Key: dsdvc_<prefix>_<secret>`. Only a salted PBKDF2-SHA256 hash is stored.
* First admin: set `DSDVC_BOOTSTRAP_ADMIN_KEY` (generate with `make admin-key`) before first start;
  it is used only when the user table is empty.
* Roles: **viewer** (read, analyse text) < **analyst** (run models/simulations) < **admin** (orgs, users, projects, keys).

## Endpoints
| Method & path | Role | Purpose |
|---|---|---|
| `GET /health`, `GET /metrics`, `GET /literature` | public | liveness + versions; Prometheus metrics; literature registry |
| `POST /admin/organizations` · `/admin/users` · `/admin/projects` · `DELETE /admin/api-keys/{prefix}` | admin | tenancy & keys (new user responses include their key once) |
| `GET /me` | any | caller identity |
| `POST /simulations` · `GET /simulations/{run_id}` | analyst / viewer | run & retrieve reproducible ABM runs (cached by config hash) |
| `POST /counterfactuals` | analyst | library scenarios → baseline/treatment/difference/CI |
| `POST /strategies/compare` | analyst | celebrity vs micro vs community vs hybrid seeding |
| `POST /bass/fit` | analyst | Bass NLS + MLE |
| `POST /stepps/analyze` | viewer | STEPPS profile with evidence/confidence/contributions |
| `POST /copilot` | viewer | Viral Content Copilot |
| `POST /virality/train` · `POST /virality/predict` | analyst / viewer | train on supplied rows (or explicit synthetic) and predict |
| `POST /attribution` · `POST /attribution/incrementality` | viewer | attributed vs incremental conversions |
| `POST /forecast` · `POST /regime` · `POST /data-quality` | analyst / viewer | forecasts, regime + alerts, quality score |
| `GET /connectors/status` | viewer | which official connectors are configured |
| `POST /research` · `GET /research/{job_id}` | analyst / viewer | async research-agent job → plan, traced recommendations, Markdown report |

Every model response includes `model_version`, `simulation_version`, `data_version`, `timestamp`.
NaN/inf values are returned as `null`.

## Example
```bash
curl -s -X POST localhost:8000/simulations -H "X-API-Key: $KEY" -H 'content-type: application/json' \
  -d '{"n_agents":1000,"steps":90,"seeding_strategy":"community_seeding"}'
curl -s -X POST localhost:8000/counterfactuals -H "X-API-Key: $KEY" -H 'content-type: application/json' \
  -d '{"scenario":"instagram_reach_minus_30","n_reps":6}'
```
