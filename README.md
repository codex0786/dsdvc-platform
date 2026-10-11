# DSDVC — Dynamic Social Diffusion, Virality, Conversion & Marketing Intelligence Platform

[![CI](https://github.com/codex0786/dsdvc-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/codex0786/dsdvc-platform/actions/workflows/ci.yml)
[![Docker image](https://github.com/codex0786/dsdvc-platform/actions/workflows/docker.yml/badge.svg)](https://github.com/codex0786/dsdvc-platform/actions/workflows/docker.yml)

A research-grade decision-support system that goes from data to network, content,
diffusion, conversion, retention, economics and finally strategy. It is the
production successor to the DSDVC-MP prototypes, which are preserved and still run
under [`legacy/`](legacy/README.md).

> **Core principle.** Reach ≠ Engagement ≠ Sharing ≠ Adoption ≠ Conversion ≠ Retention.
> The objective is **expected incremental contribution** (incremental conversions × LTV − cost),
> not raw views. Every number is labelled **OBSERVED, ESTIMATED, SIMULATED, PREDICTED,
> RECOMMENDED or SYNTHETIC**. Data that cannot be legitimately obtained comes back as
> **`DATA_UNAVAILABLE`** and is never fabricated.

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

pytest                                   # 81 tests: unit, integration, simulation, API, calibration, forecast, legacy regression

uvicorn api.main:app --reload            # REST API  -> http://localhost:8000/docs
streamlit run dashboard/app.py           # dashboard -> http://localhost:8501
python scripts/demo_research.py --synthetic   # end-to-end research report (labelled SYNTHETIC)
```

The API needs a first admin key. Generate one with `make admin-key` and set it as
`DSDVC_BOOTSTRAP_ADMIN_KEY` in `.env` before the first start (see [docs/API.md](docs/API.md)).

## Deploy

| Target | How |
|---|---|
| **Dashboard on Streamlit Community Cloud** | share.streamlit.io → *Create app* → repo `codex0786/dsdvc-platform`, branch `main`, main file `dashboard/app.py`, Python 3.12. Uses `dashboard/requirements.txt` + `packages.txt`. |
| **API + dashboard as containers** | every push to `main` publishes `ghcr.io/codex0786/dsdvc-platform` (GitHub Actions). Run with `docker compose up -d` or on any container host. |
| **CI** | `pytest` on Python 3.11 and 3.12 for every push and pull request. |

Details: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## What's inside

| Layer | Module | Highlights |
|---|---|---|
| Provenance & uncertainty | `dsdvc.core` | `Provenance`, `DataUnavailable`, `Prediction` (interval, confidence, model agreement, drivers, versions), `RunRecord`, literature registry (`@cites`) |
| Data | `dsdvc.data` | Pydantic models (§51), `DataQualityEngine` (9 checks → score), SYNTHETIC generators with known ground truth |
| Platforms | `dsdvc.platforms` | Instagram Graph API, Facebook Pages, WhatsApp Business analytics, Meta Content Library exports, user exports; rate-limit compliant; Bayesian cross-platform transmission matrix |
| Networks | `dsdvc.networks` | Structurally distinct multiplex IG/FB/WA layers, Louvain/Leiden/label-propagation/spectral/HDBSCAN, modularity, conductance, homophily, assortativity, bridging centrality, Schelling-inspired rewiring, influencer scoring |
| Contagion | `dsdvc.contagion` | Simple vs complex contagion, thresholds, independent cascade, Centola–Macy long-ties experiment, simple/complex mixture estimation |
| Content | `dsdvc.content` | NLP features, `STEPPSAnalyzer` (evidence, confidence, contributions; lexicon or supervised), additive/multiplicative/interaction model comparison, `ViralityPredictor`, early velocity, `CascadeEngine`, Viral Content Copilot |
| Diffusion | `dsdvc.diffusion` | Bass (NLS/MLE/generalised), event-driven multiplex ABM (`MarketSimulator`) with journey state machine, MNL competition, churn, LTV |
| Economics | `conversion`, `churn`, `competition`, `attribution` | Platform-specific conversion β, churn model + breakdowns, CLV, MNL with log/power network effects, 7 attribution models + incrementality tests |
| Science workflow | `calibration`, `validation`, `experiments`, `forecasting`, `optimization` | MLE/NLS/MCMC/ABC/SBC, rolling-origin benchmarks, ablation report, Monte Carlo/LHS/Sobol/Morris/A-B, counterfactuals, 7–90-day forecasts, PELT regimes, early warnings, budget optimiser |
| Agent | `dsdvc.research_agent` | 14-step `MarketingResearchAgent`, 12 tools, traceable `ResearchMemory`, 24-section report, optional Claude narrator with a numeric anti-fabrication guard |
| Serving | `dsdvc.api`, `dsdvc.dashboard`, `dsdvc.database` | FastAPI (roles: admin/analyst/viewer), Streamlit (22 tabs), SQLAlchemy (PostgreSQL/SQLite), Prometheus metrics, Redis cache, MLflow logging |

## Documentation

[Architecture](docs/ARCHITECTURE.md) · [Model card](docs/MODEL_CARD.md) · [Data card](docs/DATA_CARD.md) ·
[API](docs/API.md) · [Calibration](docs/CALIBRATION.md) · [Validation](docs/VALIDATION.md) ·
[Literature](docs/LITERATURE.md) ([per-paper summaries](docs/literature/)) · [Deployment](docs/DEPLOYMENT.md) · [Ethics](docs/ETHICS.md)

## Status against the deployment success criteria (§71)

Verified on 2026-10-09 (macOS, Python 3.12):

| Criterion | Status |
|---|---|
| clean installation works | ✅ fresh venv from `requirements.txt`, full suite passes |
| tests pass | ✅ 81 passed locally; CI on Python 3.11 + 3.12: 79 passed, 2 skipped (legacy golden values recorded on a different numpy build; checksum and mechanics tests still run) |
| baseline (legacy) model runs | ✅ v2 + v3 byte-identical to originals (SHA-256), golden-output regression tests |
| calibration works | ✅ Bass NLS/MLE/MCMC recover planted p, q, m; ABC recovers identifiable parameters and flags weakly identified ones |
| real/authorised data connectors work where credentials exist | ⚠️ implemented against Meta's documented Graph API and tested with mocked HTTP; **not exercised against live Meta accounts** (no credentials here) |
| unavailable data handled gracefully | ✅ `DATA_UNAVAILABLE` end to end (connectors → tools → agent → report → API) |
| network / influencer / content / virality / conversion / churn / forecasting / counterfactual | ✅ each covered by tests that check recovery of planted effects |
| dashboard works | ✅ all 22 tabs render with 0 exceptions (verified in browser) |
| API works | ✅ verified live under uvicorn plus TestClient tests (auth, roles, revocation, versioned outputs) |
| Docker works | ✅ image built, pushed to `ghcr.io/codex0786/dsdvc-platform` and smoke-tested (`/health`) by GitHub Actions; compose file validates |
| documentation, model card, provenance, uncertainty intervals, reproducibility | ✅ (simulations are bit-for-bit reproducible: identical result hashes) |

## Honest limitations

* Simulator parameters are **priors** until calibrated to your data. Absolute numbers from an
  uncalibrated simulation (e.g. CAC, ROI) are illustrative, not forecasts.
* The STEPPS lexicon coder is a transparent baseline (Spearman ≈ 0.55–0.6 with planted truth on
  synthetic data). For decisions, train the supervised coder on human-coded examples.
* Ablation on synthetic data shows that networks, STEPPS and recommendation exposure earn their
  complexity out of sample, while cross-platform transfer, competition and churn did not at that data
  scale. Re-run `ablation_report` on your own observed series before trusting the full model.
* The "celebrity" tier in a 1–2k-agent simulation is a relative top tier, not real
  multi-million-follower celebrities. Use `population_scale` and calibrated creator data.
