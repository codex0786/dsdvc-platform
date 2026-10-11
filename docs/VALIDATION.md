# Validation

## Protocols
* **Temporal holdout** — virality: train on the earliest 60%, fit ensemble weights + isotonic
  calibration on the next 20%, report on the latest 20%.
* **Rolling origin** — forecasts and adoption benchmarks (Hyndman & Athanasopoulos 2021).
* **Cross-validation** — STEPPS functional forms, conversion models (5-fold).
* **Ablation** — `validation.benchmarks.ablation_report`: Bass → +Networks → +STEPPS →
  +Recommendation → +Cross-platform → +Competition → +Churn; each variant re-calibrated on the
  training window and scored on the holdout; `earns_complexity` flags positive increments.

## Metrics
MAE, RMSE, MAPE, R² · log loss, Brier, ECE, calibration curve · AUC, PR-AUC, precision/recall at
the viral cut-off · lift (top decile) · cumulative gain.

## Benchmarks (every major model vs simpler baselines)
| Task | Models compared |
|---|---|
| Virality | naive base rate · logistic · LightGBM (HGB fallback) · temporal (early velocity) · calibrated ensemble |
| Adoption | naive · Bass · logistic growth · ML (GBM on lags) · DSDVC ABM (shape-calibrated) |
| Conversion | pooled logistic · platform-specific logistic (DSDVC) · gradient boosting |
| STEPPS | additive · multiplicative · interaction |
| Network effects | none · log · power (AIC/BIC) |

## Results on synthetic ground truth (illustrative; rerun on your data)
* Virality (2,500–3,000 items, test set): AUC naive 0.50 → logistic ≈ 0.74 → LightGBM ≈ 0.85 →
  temporal ≈ 0.98–0.99; the calibrated ensemble ≈ 0.98 with ECE ≈ 0.01.
* Conversion (8,000 rows, 5-fold): platform-specific logit beats pooled logit and GBM on log loss;
  recovers trust β ordering WA > FB > IG.
* Adoption on a Bass-generated series: Bass wins (as it should); the uncalibrated ABM loses —
  evidence that the ABM must be calibrated before it is used for forecasting.
* Ablation on an ABM-generated series (600 agents): networks, STEPPS and recommendation reduce
  holdout RMSE; cross-platform, competition and churn did not at that scale.

These results are on synthetic data with known structure. They validate the *estimators and the
pipeline*, not the platform's predictive accuracy for any real market.
