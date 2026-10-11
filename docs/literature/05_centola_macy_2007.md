# Centola & Macy (2007) — Complex Contagions and the Weakness of Long Ties

**Claim.** When adoption requires reinforcement from multiple contacts (complex contagion), random long ties that speed up simple contagion can impede spread; wide, clustered bridges matter. Granovetter (1978) formalised heterogeneous thresholds.

**In DSDVC.** `contagion/models.py` (simple vs complex probabilities, threshold cascades, `long_ties_experiment` reproducing the qualitative result: complex reach 100% → ~6% → ~2% as lattice rewiring rises from 0 to 0.1 to 0.6), heterogeneous thresholds and a configurable complex-adopter share in the simulator, and `fit_threshold_mixture` to estimate that share from data.

**Not established.** Which products are complex contagions — an empirical question per market.

## References
- Centola, D., & Macy, M. (2007). Complex Contagions and the Weakness of Long Ties. American Journal of Sociology, 113(3), 702-734.
- Granovetter, M. (1978). Threshold Models of Collective Behavior. American Journal of Sociology, 83(6), 1420-1443.
