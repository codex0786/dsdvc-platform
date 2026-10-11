# Schelling (1971) — Dynamic Models of Segregation

**Claim.** Agents with only mild preferences for similar neighbours, relocating locally, can generate strongly segregated aggregate patterns: individual preference ≠ aggregate structure.

**In DSDVC.** `networks/schelling.py` implements a Schelling-*inspired* tie-rewiring mechanism: agents whose neighbourhood similarity falls below an individual threshold drop their least similar tie (probability reduced by switching cost) and connect to a more similar candidate. Aggregate outcomes (homophily, modularity, assortativity, cross-community exposure) are measured, not assumed. McPherson et al. (2001) document homophily empirically; Aral et al. (2009) warn that homophily can masquerade as influence.

**Not established.** That social-media communities follow residential relocation dynamics.

## References
- Schelling, T. C. (1971). Dynamic Models of Segregation. Journal of Mathematical Sociology, 1(2), 143-186.
- McPherson, M., Smith-Lovin, L., & Cook, J. M. (2001). Birds of a Feather: Homophily in Social Networks. Annual Review of Sociology, 27, 415-444.
- Aral, S., Muchnik, L., & Sundararajan, A. (2009). Distinguishing influence-based contagion from homophily-driven diffusion in dynamic networks. PNAS, 106(51), 21544-21549.
