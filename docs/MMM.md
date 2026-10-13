# Marketing mix model (MMM) — methodology

The MMM is the measurement backbone for budget decisions. The agent-based simulator stays
alongside it for questions an MMM cannot answer (creator seeding, network spread, virality).

## Model

```
y_t = a + b·trend_t + Σ_k Fourier_k(t) + Σ_j γ_j·control_jt + Σ_c β_c · Hill(Adstock_c,t / scale_c; K_c, S_c) + ε_t
```

| Component | Specification | Why |
|---|---|---|
| Adstock (carryover) | normalised geometric, `A_t = (1-θ)x_t + θA_{t-1}`, θ ∈ (0, 0.95) | Advertising keeps working after the week it runs. Normalisation makes steady-state adstock equal weekly spend, so response curves read directly in "spend per week". |
| Saturation | Hill, `a^S / (a^S + K^S)`, on spend scaled by the channel's mean non-zero spend | Diminishing returns; S > 1 allows an S-shape (threshold before response). |
| Trend, seasonality | linear trend; 2 yearly Fourier pairs if ≥ 52 weeks | Separates organic growth and seasonal demand from media. |
| Controls | standardised price index, promo, holiday, competitor spend (if supplied) | Avoids crediting media with price/promo effects. |

**Priors** (weakly informative, on the unconstrained scale; `mmm/model.py::PRIORS`): carryover logit
N(−0.4, 1) (median θ ≈ 0.38); log K N(0, 0.6) (half-saturation near average spend); log S
N(log 1.5, 0.35); log β N(log 0.1, 1.5); controls N(0, 0.5); log σ N(log 0.1, 1).

**Estimation.** Maximum a posteriori (L-BFGS-B, 6 restarts, then a high-budget polish so
convergence is genuine), followed by a **Laplace approximation**: a Gaussian centred at the mode
with covariance from a numerical Hessian. 400 posterior draws propagate to every reported number.
This is faster than full MCMC and accurate when the posterior is near-Gaussian. Where a channel is
weakly identified the posterior is wider and may be skewed, so treat its interval as approximate and
prefer a lift test.

**Calibration with experiments.** A lift test (channel, start, end, incremental outcome, standard
error) adds a Gaussian term on the channel's modelled incremental contribution: the difference
between the fitted contribution and a counterfactual with that channel's spend set to zero during the
test window, summed from the test start to the end of the data so carryover counts. This is the
mechanism used by open-source MMMs (Google Meridian, Meta Robyn) to anchor correlational estimates to
causal evidence.

## Outputs

| Output | Definition |
|---|---|
| Contribution | Σ_t β·Hill(adstock) in outcome units, with 90% interval |
| ROI | contribution ÷ spend over the modelled period; also P(ROI > 1) |
| Marginal ROI | Δcontribution ÷ Δspend for +1% spend on that channel: the number that should drive reallocation |
| Response curve | steady-state weekly effect against sustained weekly spend, with 90% band |
| Plan | SLSQP over weekly spend per channel. *Outcome* objective spends the full budget; *profit* objective maximises margin·outcome − spend. Default bounds ±50% of current spend, because large moves extrapolate. Uplift vs the current mix at the same budget is computed for every posterior draw → interval and probability of improvement. |
| Back-test | refit without the last *h* weeks (default 12); predict them from actual spend and controls only. MAPE, total error, R², and coverage of the 90% predictive interval (parameter uncertainty + noise). |

### Back-test verdicts

| Verdict | Rule | Meaning |
|---|---|---|
| PASS | MAPE ≤ 10%, total error ≤ 5%, ≥ 70% interval coverage | accurate enough to support budget decisions |
| CAUTION | MAPE ≤ 20%, total error ≤ 12% | directionally useful; treat reallocations as hypotheses to test |
| FAIL | otherwise | do not base budget decisions on it; improve data or run lift tests |

The verdict is shown before any plan in the dashboard, and printed in both exports.

## Diagnostics and warnings

* R², NRMSE, Durbin-Watson, optimiser convergence message.
* **Identifiability warnings**: < 52 weeks of history; a channel whose spend barely varies (CV < 0.15);
  channel pairs with |r| > 0.85; channels whose posterior is barely narrower than the prior. Each
  warning names the fix (vary spend, stagger flighting, or run a lift test).

## Validation on planted data (tests)

On a 156-week synthetic dataset with known parameters (`mmm/synthetic.py`):
* ROI of well-identified channels recovered (Meta 1.64 vs 1.61 true, Search 1.52 vs 1.53, Influencers 2.45 vs 2.49).
* A slow-building channel (YouTube, high carryover) is under-estimated without experiments
  (0.55 vs 1.15); a single 8-week lift test moves it to 0.99 with the truth inside the interval.
* Back-test: 1.8–1.9% weekly error, < 1% total error → PASS.

These results validate the estimator. Accuracy on a real client is established by that client's own
back-test.

## Limitations

* Correlational model of historical variation; unobserved confounders (e.g. distribution changes)
  can bias channel effects. Lift tests are the remedy.
* National-level weekly model; geo-hierarchical pooling is a planned extension.
* Laplace posterior is an approximation; for audit-grade intervals, run MCMC (planned optional backend).
* Plans assume the next quarter resembles the modelled history (no structural break).
