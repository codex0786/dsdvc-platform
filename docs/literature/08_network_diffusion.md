# Network diffusion & community structure

**Covered.** Independent-cascade/linear-threshold models (Kempe et al. 2003); agent-based word of mouth with strong and weak ties (Goldenberg et al. 2001); modularity (Newman 2006); Louvain (Blondel et al. 2008) and Leiden, which guarantees well-connected communities (Traag et al. 2019); structural virality (Goel et al. 2016); cascade prediction from early features (Cheng et al. 2014).

**In DSDVC.** `networks/communities.py`, `networks/metrics.py`, `content/cascade.py` (generations, reproduction ratio, structural virality), `content/velocity.py` and the temporal virality model.

**Not established.** That algorithmic communities correspond to marketing segments, or that cascades are predictable from content alone (DSDVC's own benchmarks show pre-launch prediction is much weaker than early-velocity prediction).

## References
- Kempe, D., Kleinberg, J., & Tardos, E. (2003). Maximizing the Spread of Influence through a Social Network. KDD 2003, 137-146.
- Goldenberg, J., Libai, B., & Muller, E. (2001). Talk of the Network: A Complex Systems Look at the Underlying Process of Word-of-Mouth. Marketing Letters, 12(3), 211-223.
- Newman, M. E. J. (2006). Modularity and community structure in networks. PNAS, 103(23), 8577-8582.
- Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. J. Stat. Mech., P10008.
- Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. Scientific Reports, 9, 5233.
- Goel, S., Anderson, A., Hofman, J., & Watts, D. J. (2016). The Structural Virality of Online Diffusion. Management Science, 62(1), 180-196.
- Cheng, J., Adamic, L., Dow, P. A., Kleinberg, J., & Leskovec, J. (2014). Can Cascades be Predicted? Proceedings of WWW 2014, 925-936.
- McPherson, M., Smith-Lovin, L., & Cook, J. M. (2001). Birds of a Feather: Homophily in Social Networks. Annual Review of Sociology, 27, 415-444.
