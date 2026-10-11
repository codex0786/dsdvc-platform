# Literature traceability

Generated from `dsdvc.core.literature.LITERATURE` and the `@cites` decorators in code (regenerate with `python scripts/gen_literature.py`). *Supports* states narrowly what the work established; *Does not establish* guards against over-claiming.

Topic summaries: [docs/literature/](literature/).

### `bass1969`
Bass, F. M. (1969). A New Product Growth Model for Consumer Durables. Management Science, 15(5), 215-227.

- **Supports:** Aggregate adoption hazard h(t)=p+qF(t) for a homogeneous, fully mixed population.
- **Does not establish:** Network structure, heterogeneity, platform effects, churn, or multi-brand competition.
- **Used by:** `dsdvc.diffusion.abm.MarketSimulator`, `dsdvc.diffusion.bass.fit_bass_mle`, `dsdvc.diffusion.bass.fit_bass_nls`, `dsdvc.forecasting.forecast.bass_forecast`

### `bass_krishnan_jain1994`
Bass, F. M., Krishnan, T. V., & Jain, D. C. (1994). Why the Bass Model Fits without Decision Variables. Marketing Science, 13(3), 203-223.

- **Supports:** Generalized Bass model with a marketing-mix multiplier x(t) on the hazard.
- **Does not establish:** Social-media platform effects.
- **Used by:** _documentation only_

### `mahajan_muller_bass1990`
Mahajan, V., Muller, E., & Bass, F. M. (1990). New Product Diffusion Models in Marketing: A Review and Directions for Research. Journal of Marketing, 54(1), 1-26.

- **Supports:** Survey of Bass extensions (price, advertising, heterogeneity, repeat purchase).
- **Does not establish:** Any specific coefficient values for a new market.
- **Used by:** `dsdvc.diffusion.bass.fit_bass_nls`

### `rogers2003`
Rogers, E. M. (2003). Diffusion of Innovations (5th ed.). Free Press.

- **Supports:** Adoption-stage framing (knowledge, persuasion, decision, implementation, confirmation).
- **Does not establish:** Quantitative transition probabilities.
- **Used by:** `dsdvc.conversion.journey.JourneyModel`, `dsdvc.diffusion.abm.MarketSimulator`

### `schelling1971`
Schelling, T. C. (1971). Dynamic Models of Segregation. Journal of Mathematical Sociology, 1(2), 143-186.

- **Supports:** Mild individual similarity preferences + local relocation can produce strong aggregate separation.
- **Does not establish:** That online communities follow residential relocation dynamics. DSDVC uses a Schelling-INSPIRED tie-rewiring mechanism.
- **Used by:** `dsdvc.diffusion.abm.MarketSimulator`, `dsdvc.networks.schelling.SchellingRewirer`

### `mcpherson2001`
McPherson, M., Smith-Lovin, L., & Cook, J. M. (2001). Birds of a Feather: Homophily in Social Networks. Annual Review of Sociology, 27, 415-444.

- **Supports:** Homophily is pervasive across many tie types.
- **Does not establish:** Causal direction between similarity and tie formation in a given dataset.
- **Used by:** `dsdvc.networks.multiplex._layer`, `dsdvc.networks.multiplex.synthetic_multiplex`, `dsdvc.networks.schelling.SchellingRewirer`

### `aral2009`
Aral, S., Muchnik, L., & Sundararajan, A. (2009). Distinguishing influence-based contagion from homophily-driven diffusion in dynamic networks. PNAS, 106(51), 21544-21549.

- **Supports:** Naive estimates conflate homophily with influence; observed contagion can be substantially overstated.
- **Does not establish:** Influence magnitudes for other products/platforms.
- **Used by:** `dsdvc.networks.schelling.SchellingRewirer`

### `berger2013`
Berger, J. (2013). Contagious: Why Things Catch On. Simon & Schuster.

- **Supports:** STEPPS (Social currency, Triggers, Emotion, Public, Practical value, Stories) as drivers that can raise word-of-mouth likelihood.
- **Does not establish:** A functional form, weights, or guaranteed virality. STEPPS is treated as a hypothesis to test.
- **Used by:** `dsdvc.content.stepps.STEPPSAnalyzer`, `dsdvc.content.stepps_models.compare_stepps_forms`, `dsdvc.diffusion.abm.MarketSimulator`

### `berger_milkman2012`
Berger, J., & Milkman, K. L. (2012). What Makes Online Content Viral? Journal of Marketing Research, 49(2), 192-205.

- **Supports:** High-arousal emotions (awe, anger, anxiety) associated with more sharing of news articles.
- **Does not establish:** Effects for brand/product content on Instagram/Facebook/WhatsApp.
- **Used by:** `dsdvc.content.stepps.STEPPSAnalyzer`, `dsdvc.content.stepps_models.compare_stepps_forms`, `dsdvc.content.virality.ViralityPredictor`

### `centola_macy2007`
Centola, D., & Macy, M. (2007). Complex Contagions and the Weakness of Long Ties. American Journal of Sociology, 113(3), 702-734.

- **Supports:** When adoption needs multiple independent exposures, clustered wide bridges spread better than random long ties.
- **Does not establish:** Which real products are complex vs simple contagions; DSDVC estimates thresholds from data where possible.
- **Used by:** `dsdvc.contagion.models.complex_contagion_prob`, `dsdvc.contagion.models.complex_contagion_signal`, `dsdvc.contagion.models.long_ties_experiment`, `dsdvc.contagion.models.threshold_cascade`, `dsdvc.diffusion.abm.MarketSimulator`

### `granovetter1978`
Granovetter, M. (1978). Threshold Models of Collective Behavior. American Journal of Sociology, 83(6), 1420-1443.

- **Supports:** Heterogeneous individual thresholds drive aggregate tipping behaviour.
- **Does not establish:** Threshold distributions for a given market.
- **Used by:** `dsdvc.contagion.models.complex_contagion_prob`, `dsdvc.contagion.models.complex_contagion_signal`, `dsdvc.contagion.models.threshold_cascade`, `dsdvc.diffusion.abm.MarketSimulator`

### `watts_dodds2007`
Watts, D. J., & Dodds, P. S. (2007). Influentials, Networks, and Public Opinion Formation. Journal of Consumer Research, 34(4), 441-458.

- **Supports:** In simulated threshold models, large cascades are driven mainly by a critical mass of easily influenced individuals, not by a few influentials.
- **Does not establish:** That celebrity seeding never works; results are model-based.
- **Used by:** `dsdvc.agents.seeding.select_seeds`, `dsdvc.contagion.models.threshold_cascade`, `dsdvc.diffusion.abm.MarketSimulator`, `dsdvc.experiments.counterfactual.compare_seeding_strategies`, `dsdvc.experiments.counterfactual.run_counterfactual`, `dsdvc.networks.influencers.score_creators`

### `deveirman2017`
De Veirman, M., Cauberghe, V., & Hudders, L. (2017). Marketing through Instagram influencers: the impact of number of followers and product divergence on brand attitude. International Journal of Advertising, 36(5), 798-828.

- **Supports:** Follower count raises perceived popularity/likeability, but very high counts with divergent products can reduce brand attitude; influence is not proportional to followers.
- **Does not establish:** Conversion or ROI effects of influencer size.
- **Used by:** `dsdvc.agents.seeding.select_seeds`, `dsdvc.experiments.counterfactual.compare_seeding_strategies`, `dsdvc.networks.influencers.score_creators`

### `aral_walker2012`
Aral, S., & Walker, D. (2012). Identifying Influential and Susceptible Members of Social Networks. Science, 337(6092), 337-341.

- **Supports:** Influence and susceptibility vary systematically and can be estimated experimentally.
- **Does not establish:** Generalization beyond the studied app adoption context.
- **Used by:** `dsdvc.networks.influencers.score_creators`

### `goel2016`
Goel, S., Anderson, A., Hofman, J., & Watts, D. J. (2016). The Structural Virality of Online Diffusion. Management Science, 62(1), 180-196.

- **Supports:** Cascade size and structural virality (broadcast vs. viral spread) are distinct; large cascades are rare.
- **Does not establish:** Predictability of individual cascades.
- **Used by:** `dsdvc.content.cascade.CascadeEngine`, `dsdvc.content.virality.ViralityPredictor`

### `cheng2014`
Cheng, J., Adamic, L., Dow, P. A., Kleinberg, J., & Leskovec, J. (2014). Can Cascades be Predicted? Proceedings of WWW 2014, 925-936.

- **Supports:** Cascade growth is partially predictable from early temporal and structural features; framing as 'will it double' relative prediction.
- **Does not establish:** Prediction at time zero from content alone.
- **Used by:** `dsdvc.content.velocity.early_features`, `dsdvc.content.virality.ViralityPredictor`

### `kempe2003`
Kempe, D., Kleinberg, J., & Tardos, E. (2003). Maximizing the Spread of Influence through a Social Network. KDD 2003, 137-146.

- **Supports:** Independent cascade and linear threshold models; greedy seeding approximation.
- **Does not establish:** Real-world parameter values.
- **Used by:** `dsdvc.agents.seeding.select_seeds`, `dsdvc.contagion.models.independent_cascade`

### `goldenberg2001`
Goldenberg, J., Libai, B., & Muller, E. (2001). Talk of the Network: A Complex Systems Look at the Underlying Process of Word-of-Mouth. Marketing Letters, 12(3), 211-223.

- **Supports:** Agent-based/cellular-automata approach to WOM with strong and weak ties.
- **Does not establish:** Platform-specific behaviour.
- **Used by:** `dsdvc.diffusion.abm.MarketSimulator`, `dsdvc.networks.multiplex._layer`

### `rand_rust2011`
Rand, W., & Rust, R. T. (2011). Agent-based modeling in marketing: Guidelines for rigor. International Journal of Research in Marketing, 28(3), 181-193.

- **Supports:** Verification and validation guidelines for marketing ABMs (docking, empirical validation, sensitivity analysis).
- **Does not establish:** Any particular mechanism.
- **Used by:** `dsdvc.diffusion.abm.MarketSimulator`, `dsdvc.experiments.counterfactual.run_counterfactual`, `dsdvc.validation.benchmarks.ablation_report`, `dsdvc.validation.benchmarks.adoption_benchmark`

### `newman2006`
Newman, M. E. J. (2006). Modularity and community structure in networks. PNAS, 103(23), 8577-8582.

- **Supports:** Modularity Q as a community-quality measure.
- **Does not establish:** That modular communities are meaningful consumer segments.
- **Used by:** `dsdvc.networks.communities.detect_communities`, `dsdvc.networks.metrics.modularity`

### `blondel2008`
Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. J. Stat. Mech., P10008.

- **Supports:** Louvain modularity optimisation.
- **Does not establish:** -
- **Used by:** `dsdvc.networks.communities.detect_communities`

### `traag2019`
Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. Scientific Reports, 9, 5233.

- **Supports:** Leiden algorithm; Louvain can yield badly connected communities.
- **Does not establish:** -
- **Used by:** `dsdvc.networks.communities.detect_communities`

### `shao_li2011`
Shao, X., & Li, L. (2011). Data-driven Multi-touch Attribution Models. KDD 2011, 258-264.

- **Supports:** Data-driven (incl. Shapley-like) multi-touch attribution.
- **Does not establish:** Causal incrementality.
- **Used by:** `dsdvc.attribution.models.shapley_attribution`

### `anderl2016`
Anderl, E., Becker, I., von Wangenheim, F., & Schumann, J. H. (2016). Mapping the customer journey: Lessons learned from graph-based online attribution modeling. IJRM, 33(3), 457-474.

- **Supports:** Markov-chain removal-effect attribution.
- **Does not establish:** Causal incrementality.
- **Used by:** `dsdvc.attribution.models.markov_attribution`

### `lewis_rao2015`
Lewis, R. A., & Rao, J. M. (2015). The Unfavorable Economics of Measuring the Returns to Advertising. Quarterly Journal of Economics, 130(4), 1941-1973.

- **Supports:** Ad effects are small relative to outcome noise; experiments need very large samples.
- **Does not establish:** -
- **Used by:** `dsdvc.attribution.models.incrementality_test`

### `gordon2019`
Gordon, B. R., Zettelmeyer, F., Bhargava, N., & Chapsky, D. (2019). A Comparison of Approaches to Advertising Measurement: Evidence from Big Field Experiments at Facebook. Marketing Science, 38(2), 193-225.

- **Supports:** Observational methods often fail to recover experimental ad lift; RCTs remain the benchmark.
- **Does not establish:** -
- **Used by:** `dsdvc.attribution.models.incrementality_test`

### `gupta2006`
Gupta, S., Hanssens, D., Hardie, B., et al. (2006). Modeling Customer Lifetime Value. Journal of Service Research, 9(2), 139-155.

- **Supports:** Discounted margin formulation of CLV and model families.
- **Does not establish:** -
- **Used by:** `dsdvc.churn.ltv.discounted_ltv`, `dsdvc.churn.ltv.retention_ltv`

### `fader2005`
Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). 'Counting Your Customers' the Easy Way: An Alternative to the Pareto/NBD Model. Marketing Science, 24(2), 275-284.

- **Supports:** BG/NBD model for non-contractual repeat purchase.
- **Does not establish:** -
- **Used by:** _documentation only_

### `mcfadden1974`
McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In P. Zarembka (Ed.), Frontiers in Econometrics (pp. 105-142). Academic Press.

- **Supports:** Multinomial/conditional logit choice probabilities.
- **Does not establish:** -
- **Used by:** `dsdvc.competition.mnl.choice_probabilities`, `dsdvc.competition.mnl.fit_mnl`, `dsdvc.diffusion.abm.MarketSimulator`

### `katz_shapiro1985`
Katz, M. L., & Shapiro, C. (1985). Network Externalities, Competition, and Compatibility. American Economic Review, 75(3), 424-440.

- **Supports:** Consumption value increasing in the number of other users.
- **Does not establish:** The functional form (log vs power); DSDVC compares specifications empirically.
- **Used by:** `dsdvc.competition.mnl.choice_probabilities`, `dsdvc.competition.mnl.fit_mnl`, `dsdvc.diffusion.abm.MarketSimulator`

### `morris1991`
Morris, M. D. (1991). Factorial Sampling Plans for Preliminary Computational Experiments. Technometrics, 33(2), 161-174.

- **Supports:** Elementary-effects screening.
- **Does not establish:** -
- **Used by:** `dsdvc.experiments.design.morris_screening`

### `saltelli2010`
Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., & Tarantola, S. (2010). Variance based sensitivity analysis of model output. Computer Physics Communications, 181(2), 259-270.

- **Supports:** Estimators for first-order and total Sobol indices.
- **Does not establish:** -
- **Used by:** `dsdvc.experiments.design.sobol_indices`

### `beaumont2002`
Beaumont, M. A., Zhang, W., & Balding, D. J. (2002). Approximate Bayesian Computation in Population Genetics. Genetics, 162(4), 2025-2035.

- **Supports:** Rejection ABC with summary statistics.
- **Does not establish:** -
- **Used by:** _documentation only_

### `talts2018`
Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018). Validating Bayesian Inference Algorithms with Simulation-Based Calibration. arXiv:1804.06788.

- **Supports:** Rank-uniformity check for posterior inference.
- **Does not establish:** -
- **Used by:** _documentation only_

### `killick2012`
Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal Detection of Changepoints With a Linear Computational Cost. JASA, 107(500), 1590-1598.

- **Supports:** PELT change-point detection with penalised cost.
- **Does not establish:** -
- **Used by:** `dsdvc.forecasting.regime.pelt`

### `hyndman2021`
Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.

- **Supports:** Rolling-origin (time-series cross-validation) evaluation.
- **Does not establish:** -
- **Used by:** `dsdvc.forecasting.forecast.rolling_origin_errors`, `dsdvc.validation.benchmarks.adoption_benchmark`

### `lundberg2017`
Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.

- **Supports:** SHAP additive feature attributions.
- **Does not establish:** Causal effects of features.
- **Used by:** `dsdvc.content.virality.ViralityPredictor`
