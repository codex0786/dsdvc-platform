# Deployment

## Local
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env            # then set DSDVC_BOOTSTRAP_ADMIN_KEY (make admin-key)
pytest                          # must pass before deployment
uvicorn api.main:app --reload
streamlit run dashboard/app.py
```
Python 3.11–3.12 recommended (LightGBM/SHAP/igraph wheels). Without the optional ML extras the
code falls back to scikit-learn gradient boosting, networkx Leiden and logistic-coefficient
explanations, and reports which backend it used.

## Production (Docker)
```bash
cp .env.example .env            # fill secrets; never commit .env
docker compose up --build -d    # db (PostgreSQL 16), redis, mlflow, minio, api :8000, dashboard :8501
```
* Database URL is injected by compose; tables are created at API start (`init_db`).
* For managed infrastructure, point `DATABASE_URL`, `REDIS_URL`, `MLFLOW_TRACKING_URI`,
  `S3_ENDPOINT_URL` at managed services and run only the `api` and `dashboard` images.
* Put the API behind TLS (reverse proxy / load balancer). Rotate API keys via `/admin/api-keys`.
* Platform tokens belong in a secret manager; the `platform_credentials` table stores references
  only (`secret_ref`), never tokens.

## Dataset storage
Onboarded client datasets go to **S3-compatible object storage** when `S3_BUCKET` is set, and to the local disk
(`DSDVC_DATA_DIR`) otherwise. Local disk is fine for development but is wiped when a cloud container restarts.

| Provider | Settings |
|---|---|
| AWS S3 | `S3_BUCKET`, `S3_REGION` (e.g. `ap-south-1`), keys or an IAM role. Default `S3_SSE=AES256`; for your own key `S3_SSE=aws:kms` + `S3_KMS_KEY_ID`. |
| Cloudflare R2 | `S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com`, `S3_REGION=auto`, R2 API token keys. R2 encrypts every object; set `S3_SSE=none` if it rejects the header. |
| Google Cloud Storage | `S3_ENDPOINT_URL=https://storage.googleapis.com`, HMAC keys, `S3_SSE=none` (GCS encrypts by default). |
| MinIO (docker compose) | Preconfigured: bucket `dsdvc-datasets`, created privately by `minio-init`. |

Credentials: `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (or `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY`); leave
them out to use an instance/IAM role. Optional `S3_PREFIX` (e.g. `prod`) prefixes every key.

Bucket set-up: block all public access; give the platform's key only `s3:GetObject` and `s3:PutObject`
on the bucket (it never lists or deletes); turn on versioning if you want point-in-time recovery.

* Keys are `[S3_PREFIX/]ws_<workspace id>/marketing_weekly_<sha256[:16]>.csv`. Identical uploads are written once.
* If the bucket is unreachable, saving fails with a clear error (API: HTTP 503). The platform never quietly falls
  back to the local disk.
* `make storage-status` shows the active backend; the Administration page and `GET /health` show it too.
* **Moving existing datasets:** set the S3 variables, then run `make migrate-storage` (or
  `python -m dsdvc.storage migrate --dry-run` first). Each file is hash-checked, copied, and its database row is
  repointed and audit-logged. Local files are kept until you delete them.

## GitHub
* **CI** (`.github/workflows/ci.yml`) — installs `requirements.txt` and runs `pytest` on Python 3.11 and 3.12.
* **Container image** (`.github/workflows/docker.yml`) — builds the Dockerfile, pushes
  `ghcr.io/codex0786/dsdvc-platform:{main, sha-…, vX.Y.Z}` and smoke-tests `GET /health`.
  Pull with `docker pull ghcr.io/codex0786/dsdvc-platform:main` (log in to ghcr.io first while the package is private).

## Streamlit Community Cloud (dashboard only)
1. Sign in at https://share.streamlit.io with GitHub and authorise access to the repository.
2. *Create app* → repository `codex0786/dsdvc-platform`, branch `main`, main file `dashboard/app.py`;
   under *Advanced settings* choose Python 3.12.
3. Optional secrets (*Settings → Secrets*, TOML): `META_ACCESS_TOKEN`, `INSTAGRAM_BUSINESS_ACCOUNT_ID`, …
   — Streamlit exposes them as environment variables. Never commit them.
4. **For real client data**, the app's disk is wiped on every restart, so add durable storage in the same secrets:
   ```toml
   DATABASE_URL = "postgresql+psycopg://user:password@host:5432/dsdvc"   # managed PostgreSQL (Neon, Supabase, RDS, …)
   S3_BUCKET = "your-dataset-bucket"
   S3_REGION = "ap-south-1"
   AWS_ACCESS_KEY_ID = "…"
   AWS_SECRET_ACCESS_KEY = "…"
   ```
   Object storage keeps the files; the database keeps the records that point to them, so you need both.

Cloud installs `dashboard/requirements.txt` (lean: no PostgreSQL/Redis/MLflow/SHAP) and `packages.txt`
(`libgomp1` for LightGBM). The free tier has ~1 GB RAM: keep *Agents* ≤ 2,000 and replications ≤ 6.
The dashboard starts in authorised-data-only mode; viewers switch on the labelled synthetic demo in the sidebar.

## Operations
* `GET /metrics` — Prometheus: request latency/counts by route, model latency, cache hits,
  research failures, MLflow failures, build info (model/simulation versions).
* Every simulation row stores its `RunRecord`; `GET /simulations/{run_id}` returns seeds,
  parameters and software versions to reproduce it exactly.
* Virality models are held in API memory (`model_id`) and must be retrained after restart; log them
  to MLflow for durable artefacts.

## Pre-deployment checklist
`pytest` green · `.env` populated from secret store · admin key bootstrapped then removed from env ·
connectors status checked (`/connectors/status`) · data-quality score reviewed for uploaded data ·
calibration status reviewed (PRIOR vs CALIBRATED) before sharing simulated numbers.
