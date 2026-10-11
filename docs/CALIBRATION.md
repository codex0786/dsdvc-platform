# Calibration

> Rule: a parameter is **CALIBRATED** only if it was fitted to observed data by one of the methods
> below. Hand-set values are **PRIORS** and are labelled as such everywhere.

## Methods (`calibration/engine.py`)
| Method | Use | Diagnostics reported |
|---|---|---|
| Nonlinear least squares | Bass p, q, m from per-period adoptions | SE (Jacobian), SSE, AIC |
| Maximum likelihood | Bass (multinomial likelihood incl. non-adopter cell) | Hessian SE, log-lik, AIC |
| Bayesian MCMC | Bass posterior (weakly informative priors documented in code) | acceptance, ESS, low-ESS warnings, 90% credible intervals |
| Approximate Bayesian computation | any simulator parameter via summary statistics | posterior shrinkage, **weak-identification warnings**, tolerance |
| Simulation-based calibration | checks the Bayesian sampler | KS test of rank uniformity |
| Beta-Binomial | platform engagement/share/adoption/churn rates | 90% intervals, CALIBRATED vs PRIOR |
| Exposure-model intercept | match observed non-follower reach rate | — |
| Transmission matrix | Dirichlet prior + observed transfer counts | posterior intervals, data weight per row |

Verified on planted synthetic truth (tests): Bass p, q, m recovered within 10%/5%/5%; MCMC intervals
cover truth; ABC recovers the category-purchase intercept (−3.2 vs −3.0) and reports that the
complex-contagion share is **not identified** by the default summary statistics.

## Parameters and their default status
| Parameter | Default | Status | How to calibrate |
|---|---|---|---|
| Bass p (`p_innovation`) | 0.0008/day | PRIOR | `bass_nls/mle/bayesian` on observed adoptions (agent uses p when available) |
| purchase intercept, contagion thresholds | −4.0, 2.0 | PRIOR | ABC on adoption level/timing summaries |
| STEPPS β / form | (0.6, 0.3, 1.1, 0.5, 0.7, 0.6), multiplicative | PRIOR | `compare_stepps_forms` on observed share rates |
| homophily θ | layer specs | PRIOR | match observed assortativity/modularity of authorised aggregate graphs |
| churn θ | `churn_coefs` | PRIOR | `ChurnModel.fit` on customer panel |
| platform β | `conversion_mult`, trust | PRIOR | `ConversionModel` (platform-specific) on panel |
| cross-platform matrix | Dirichlet prior (20 pseudo-counts/row) | PRIOR | `TransmissionMatrix.update` with observed transfers |
| creator costs | rate card | ESTIMATED | replace with quotes |

The transmission prior assumes same-platform resharing dominates and that IG→WA forwarding is
plausible; it is deliberately weak (20 pseudo-counts) so modest data overrides it.

## Digital-twin loop
`RealWorldData → CalibrationEngine → SimConfig (calibrated fields) → MarketSimulator → compare
simulated vs observed summaries (validation) → iterate`. Run `validation.ablation_report` after
each calibration to confirm that added mechanisms still earn their complexity.
