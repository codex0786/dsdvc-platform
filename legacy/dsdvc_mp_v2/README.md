# DSDVC — Dynamic Social Diffusion, Virality & Churn

A Python agent-based model integrating the mechanics of the supplied NetLogo E4/E5 models with:

- Bass diffusion: innovation (`p`) and imitation (`q`)
- Schelling-style community segregation/homophily and rewiring
- Jonah Berger's STEPPS framework for content sharing
- Customer engagement, satisfaction, churn and recovery
- Market conditions, competition, push and retention strategy

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Windows PowerShell:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

The repository is Streamlit-ready. Push this folder to GitHub and deploy it with Streamlit Community Cloud using `app.py` as the entry point.

## Model architecture

`model.py` contains the simulation engine. `app.py` is the interactive dashboard. The first scenario experiment sweeps homophily × content quality to expose the interaction between community segregation and virality.

## Important modeling note

The STEPPS layer is implemented as a behavioral sharing mechanism, not as a claim that STEPPS dimensions guarantee virality. The Schelling layer is a generalized social-network adaptation of local preference/rewiring, not a literal residential-segregation reproduction.
