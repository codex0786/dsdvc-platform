# Marketing attribution

**Covered.** Data-driven multi-touch attribution (Shao & Li 2011) and graph/Markov-chain attribution with removal effects (Anderl et al. 2016), alongside rule-based models.

**In DSDVC.** `attribution/models.py`: last click, first touch, linear, time decay, position based, Markov removal effect, Shapley value — all conserving total conversions.

**Not established.** Causality. Attributed conversions ≠ incremental conversions; see the causal-inference summary.

## References
- Shao, X., & Li, L. (2011). Data-driven Multi-touch Attribution Models. KDD 2011, 258-264.
- Anderl, E., Becker, I., von Wangenheim, F., & Schumann, J. H. (2016). Mapping the customer journey: Lessons learned from graph-based online attribution modeling. IJRM, 33(3), 457-474.
