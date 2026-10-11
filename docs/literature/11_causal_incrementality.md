# Causal inference & incrementality

**Claim.** Advertising effects are small relative to outcome noise, so experiments need very large samples (Lewis & Rao 2015); observational methods often fail to recover experimentally measured lift (Gordon et al. 2019); homophily confounds influence (Aral et al. 2009).

**In DSDVC.** `attribution.incrementality_test` and `required_sample_size`; Copilot A/B plans; the counterfactual engine labels its outputs *model-based* and recommends field experiments to confirm them.

**Not established.** That simulated counterfactuals equal real causal effects.

## References
- Lewis, R. A., & Rao, J. M. (2015). The Unfavorable Economics of Measuring the Returns to Advertising. Quarterly Journal of Economics, 130(4), 1941-1973.
- Gordon, B. R., Zettelmeyer, F., Bhargava, N., & Chapsky, D. (2019). A Comparison of Approaches to Advertising Measurement: Evidence from Big Field Experiments at Facebook. Marketing Science, 38(2), 193-225.
- Aral, S., Muchnik, L., & Sundararajan, A. (2009). Distinguishing influence-based contagion from homophily-driven diffusion in dynamic networks. PNAS, 106(51), 21544-21549.
