# Model card — DSDVC 1.0 (`dsdvc-1.0.0`, simulator `abm-event-1.0.0`)

## Intended use
* Decision support for marketing research teams: comparing go-to-market strategies, content
  concepts, creator portfolios, channel budgets and scenarios **at the aggregate/community level**.
* Hypothesis testing about diffusion mechanisms (STEPPS forms, complex vs simple contagion,
  seeding strategies, network effects) with explicit uncertainty.
* Forecasting and monitoring of a brand's *own* authorised metrics.

## Non-intended use
* Targeting, profiling or identifying individual private people; inferring sensitive attributes
  (health, religion, caste, sexuality, political views, etc.); targeting vulnerable groups.
* Accessing private communications or private accounts, or circumventing platform controls.
* Treating any output as a guarantee ("this will go viral"). Outputs are conditional probabilities.
* Making financial commitments from uncalibrated simulations.

## Components and what they output
| Component | Output | Status label |
|---|---|---|
| Marketing mix model | channel contribution, ROI, marginal ROI, response curves (90% intervals); back-test verdict | ESTIMATED |
| MMM budget optimiser | weekly mix, uplift interval, P(better than current) | RECOMMENDED |
| Bass NLS / MLE / MCMC | p, q, m with intervals | ESTIMATED / CALIBRATED |
| `MarketSimulator` | adoption, churn, revenue, cost, CAC, ROI paths | SIMULATED |
| `ViralityPredictor` | P(top-x% within benchmark group), expected views/shares/conversions/retention | PREDICTED |
| `ConversionModel`, `ChurnModel` | coefficients by platform, probabilities | ESTIMATED / PREDICTED |
| `score_creators` | expected incremental conversions × LTV / cost with MC interval | PREDICTED (inputs: OBSERVED/ESTIMATED) |
| Budget optimiser, research agent | allocations, recommendations | RECOMMENDED (traced to experiments) |

## Key assumptions
* Classical Bass assumes a homogeneous, fully mixed population; DSDVC extends it (heterogeneity,
  networks, marketing mix, churn, competition) and does **not** claim the original assumptions hold.
* Platform roles (IG discovery, FB community, WA high-trust) are **priors** encoded as parameters,
  to be estimated, not findings.
* The algorithmic-exposure model is an *estimated* exposure model, not the platforms' algorithms.
* STEPPS is treated as a hypothesis; its functional form is chosen by out-of-sample comparison.
* Schelling's mechanism is used as an inspiration for tie rewiring, not as a literal model of
  online communities.
* "Viral" = top x% (default 5%) of comparable content within benchmark groups.

## Limitations
* Uncalibrated simulation magnitudes (CAC, ROI, adoption share) are illustrative.
* Agent populations of 10³–10⁴ approximate markets through `population_scale`; tail phenomena
  (true celebrity reach) are compressed.
* Lexicon STEPPS coder: English/Hinglish-oriented, low-to-medium confidence (capped at 0.7).
* Virality ensemble weights can collapse onto the temporal model when early-velocity data exist;
  pre-launch predictions (no early data) are materially less accurate (see benchmarks).
* ABC with the default summary statistics does not identify every parameter (e.g. complex-contagion
  share); the engine reports weakly identified parameters explicitly.
* Attribution models distribute *attributed* conversions; only experiments measure incrementality.

## Known biases
* Survivorship bias in platform exports (deleted / low-performing posts missing) — checked by the
  DataQualityEngine but not correctable without complete data.
* Hashtag/keyword discovery over-represents creators who use the searched terms.
* Engagement-based trust proxies favour creators with engaged but possibly non-representative audiences.
* Follower-size penalties in trust/relevance are priors motivated by De Veirman et al. (2017), not
  estimated for each market.

## Data limitations
See [DATA_CARD.md](DATA_CARD.md). WhatsApp is observable only through aggregate business analytics,
user-provided data or surveys. Meta Content Library data are available only to approved researchers.

## Uncertainty
Every major prediction reports a point estimate, lower/upper bound, a confidence label (from
interval width, data quality and model agreement), data quality, model agreement and the main
uncertainty. Simulation intervals reflect stochastic replication; they include parameter uncertainty
only when parameters were calibrated (posterior draws).

## Privacy constraints
Aggregate/community-level analysis by default; no storage of platform tokens in the database (only
secret references); synthetic agents never represent real people. See [ETHICS.md](ETHICS.md).
