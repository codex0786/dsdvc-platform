# DSDVC-MP — Multi-Platform Social Diffusion, Virality & Churn

A Python agent-based research prototype integrating the original E4/E5 NetLogo mechanics with Bass diffusion, Schelling-style network evolution, Jonah Berger's STEPPS framework, customer churn, firm strategy, and three multiplex social platforms.

## Platform representation

- **Instagram:** scale-free discovery network; algorithmic/recommendation exposure; high reach, lower tie trust.
- **Facebook:** clustered small-world network; groups/community diffusion; medium reach and trust.
- **WhatsApp:** strong-tie small-world network; private forwarding; lower reach but high trust/conversion potential.

These are **synthetic network priors**, not claims about private platform graphs. The model is designed so empirical estimates can replace the defaults.

## Core equations

Adoption hazard combines:

`innovation + multiplex imitation + content exposure + community fit + market conditions + firm push - price pressure`

Sharing probability combines:

`STEPPS + content sensitivity + influence + platform trust + exposure`

Churn combines:

`base churn + competition + price pressure + dissatisfaction - community fit - engagement - retention investment`

Schelling-style rewiring occurs when community fit falls below a threshold.

## Data-driven upgrade path

The model accepts empirical parameters for:

- platform reach/exposure distributions
- network degree and clustering distributions
- tie strength and trust
- engagement/share probabilities
- STEPPS coefficients
- adoption and churn coefficients
- platform cross-posting rates

Recommended empirical sources include authorized Meta research/public datasets, platform/business analytics where access is permitted, and consumer survey/panel data. Do not attempt to reconstruct private consumer social graphs.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Batch experiments

```bash
python experiments/run_baseline.py
```

## Tests

```bash
pytest -q
```

## Important scientific note

The platform topology choices and coefficient values in this prototype are modeling priors, not measured estimates. For research claims, calibrate them using observed data and conduct sensitivity/robustness analysis.
