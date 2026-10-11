"""Literature registry (spec §59). Mechanisms in code reference these keys via ``cites``.

``supports`` states narrowly what the work established; ``does_not_establish`` guards
against over-claiming (spec: "Do not claim that a paper validates a mechanism it did not study").
"""
from __future__ import annotations

from typing import Callable, TypeVar

LITERATURE: dict[str, dict[str, str]] = {
    "bass1969": {
        "citation": "Bass, F. M. (1969). A New Product Growth Model for Consumer Durables. Management Science, 15(5), 215-227.",
        "supports": "Aggregate adoption hazard h(t)=p+qF(t) for a homogeneous, fully mixed population.",
        "does_not_establish": "Network structure, heterogeneity, platform effects, churn, or multi-brand competition.",
    },
    "bass_krishnan_jain1994": {
        "citation": "Bass, F. M., Krishnan, T. V., & Jain, D. C. (1994). Why the Bass Model Fits without Decision Variables. Marketing Science, 13(3), 203-223.",
        "supports": "Generalized Bass model with a marketing-mix multiplier x(t) on the hazard.",
        "does_not_establish": "Social-media platform effects.",
    },
    "mahajan_muller_bass1990": {
        "citation": "Mahajan, V., Muller, E., & Bass, F. M. (1990). New Product Diffusion Models in Marketing: A Review and Directions for Research. Journal of Marketing, 54(1), 1-26.",
        "supports": "Survey of Bass extensions (price, advertising, heterogeneity, repeat purchase).",
        "does_not_establish": "Any specific coefficient values for a new market.",
    },
    "rogers2003": {
        "citation": "Rogers, E. M. (2003). Diffusion of Innovations (5th ed.). Free Press.",
        "supports": "Adoption-stage framing (knowledge, persuasion, decision, implementation, confirmation).",
        "does_not_establish": "Quantitative transition probabilities.",
    },
    "schelling1971": {
        "citation": "Schelling, T. C. (1971). Dynamic Models of Segregation. Journal of Mathematical Sociology, 1(2), 143-186.",
        "supports": "Mild individual similarity preferences + local relocation can produce strong aggregate separation.",
        "does_not_establish": "That online communities follow residential relocation dynamics. DSDVC uses a Schelling-INSPIRED tie-rewiring mechanism.",
    },
    "mcpherson2001": {
        "citation": "McPherson, M., Smith-Lovin, L., & Cook, J. M. (2001). Birds of a Feather: Homophily in Social Networks. Annual Review of Sociology, 27, 415-444.",
        "supports": "Homophily is pervasive across many tie types.",
        "does_not_establish": "Causal direction between similarity and tie formation in a given dataset.",
    },
    "aral2009": {
        "citation": "Aral, S., Muchnik, L., & Sundararajan, A. (2009). Distinguishing influence-based contagion from homophily-driven diffusion in dynamic networks. PNAS, 106(51), 21544-21549.",
        "supports": "Naive estimates conflate homophily with influence; observed contagion can be substantially overstated.",
        "does_not_establish": "Influence magnitudes for other products/platforms.",
    },
    "berger2013": {
        "citation": "Berger, J. (2013). Contagious: Why Things Catch On. Simon & Schuster.",
        "supports": "STEPPS (Social currency, Triggers, Emotion, Public, Practical value, Stories) as drivers that can raise word-of-mouth likelihood.",
        "does_not_establish": "A functional form, weights, or guaranteed virality. STEPPS is treated as a hypothesis to test.",
    },
    "berger_milkman2012": {
        "citation": "Berger, J., & Milkman, K. L. (2012). What Makes Online Content Viral? Journal of Marketing Research, 49(2), 192-205.",
        "supports": "High-arousal emotions (awe, anger, anxiety) associated with more sharing of news articles.",
        "does_not_establish": "Effects for brand/product content on Instagram/Facebook/WhatsApp.",
    },
    "centola_macy2007": {
        "citation": "Centola, D., & Macy, M. (2007). Complex Contagions and the Weakness of Long Ties. American Journal of Sociology, 113(3), 702-734.",
        "supports": "When adoption needs multiple independent exposures, clustered wide bridges spread better than random long ties.",
        "does_not_establish": "Which real products are complex vs simple contagions; DSDVC estimates thresholds from data where possible.",
    },
    "granovetter1978": {
        "citation": "Granovetter, M. (1978). Threshold Models of Collective Behavior. American Journal of Sociology, 83(6), 1420-1443.",
        "supports": "Heterogeneous individual thresholds drive aggregate tipping behaviour.",
        "does_not_establish": "Threshold distributions for a given market.",
    },
    "watts_dodds2007": {
        "citation": "Watts, D. J., & Dodds, P. S. (2007). Influentials, Networks, and Public Opinion Formation. Journal of Consumer Research, 34(4), 441-458.",
        "supports": "In simulated threshold models, large cascades are driven mainly by a critical mass of easily influenced individuals, not by a few influentials.",
        "does_not_establish": "That celebrity seeding never works; results are model-based.",
    },
    "deveirman2017": {
        "citation": "De Veirman, M., Cauberghe, V., & Hudders, L. (2017). Marketing through Instagram influencers: the impact of number of followers and product divergence on brand attitude. International Journal of Advertising, 36(5), 798-828.",
        "supports": "Follower count raises perceived popularity/likeability, but very high counts with divergent products can reduce brand attitude; influence is not proportional to followers.",
        "does_not_establish": "Conversion or ROI effects of influencer size.",
    },
    "aral_walker2012": {
        "citation": "Aral, S., & Walker, D. (2012). Identifying Influential and Susceptible Members of Social Networks. Science, 337(6092), 337-341.",
        "supports": "Influence and susceptibility vary systematically and can be estimated experimentally.",
        "does_not_establish": "Generalization beyond the studied app adoption context.",
    },
    "goel2016": {
        "citation": "Goel, S., Anderson, A., Hofman, J., & Watts, D. J. (2016). The Structural Virality of Online Diffusion. Management Science, 62(1), 180-196.",
        "supports": "Cascade size and structural virality (broadcast vs. viral spread) are distinct; large cascades are rare.",
        "does_not_establish": "Predictability of individual cascades.",
    },
    "cheng2014": {
        "citation": "Cheng, J., Adamic, L., Dow, P. A., Kleinberg, J., & Leskovec, J. (2014). Can Cascades be Predicted? Proceedings of WWW 2014, 925-936.",
        "supports": "Cascade growth is partially predictable from early temporal and structural features; framing as 'will it double' relative prediction.",
        "does_not_establish": "Prediction at time zero from content alone.",
    },
    "kempe2003": {
        "citation": "Kempe, D., Kleinberg, J., & Tardos, E. (2003). Maximizing the Spread of Influence through a Social Network. KDD 2003, 137-146.",
        "supports": "Independent cascade and linear threshold models; greedy seeding approximation.",
        "does_not_establish": "Real-world parameter values.",
    },
    "goldenberg2001": {
        "citation": "Goldenberg, J., Libai, B., & Muller, E. (2001). Talk of the Network: A Complex Systems Look at the Underlying Process of Word-of-Mouth. Marketing Letters, 12(3), 211-223.",
        "supports": "Agent-based/cellular-automata approach to WOM with strong and weak ties.",
        "does_not_establish": "Platform-specific behaviour.",
    },
    "rand_rust2011": {
        "citation": "Rand, W., & Rust, R. T. (2011). Agent-based modeling in marketing: Guidelines for rigor. International Journal of Research in Marketing, 28(3), 181-193.",
        "supports": "Verification and validation guidelines for marketing ABMs (docking, empirical validation, sensitivity analysis).",
        "does_not_establish": "Any particular mechanism.",
    },
    "newman2006": {
        "citation": "Newman, M. E. J. (2006). Modularity and community structure in networks. PNAS, 103(23), 8577-8582.",
        "supports": "Modularity Q as a community-quality measure.",
        "does_not_establish": "That modular communities are meaningful consumer segments.",
    },
    "blondel2008": {
        "citation": "Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. J. Stat. Mech., P10008.",
        "supports": "Louvain modularity optimisation.",
        "does_not_establish": "-",
    },
    "traag2019": {
        "citation": "Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. Scientific Reports, 9, 5233.",
        "supports": "Leiden algorithm; Louvain can yield badly connected communities.",
        "does_not_establish": "-",
    },
    "shao_li2011": {
        "citation": "Shao, X., & Li, L. (2011). Data-driven Multi-touch Attribution Models. KDD 2011, 258-264.",
        "supports": "Data-driven (incl. Shapley-like) multi-touch attribution.",
        "does_not_establish": "Causal incrementality.",
    },
    "anderl2016": {
        "citation": "Anderl, E., Becker, I., von Wangenheim, F., & Schumann, J. H. (2016). Mapping the customer journey: Lessons learned from graph-based online attribution modeling. IJRM, 33(3), 457-474.",
        "supports": "Markov-chain removal-effect attribution.",
        "does_not_establish": "Causal incrementality.",
    },
    "lewis_rao2015": {
        "citation": "Lewis, R. A., & Rao, J. M. (2015). The Unfavorable Economics of Measuring the Returns to Advertising. Quarterly Journal of Economics, 130(4), 1941-1973.",
        "supports": "Ad effects are small relative to outcome noise; experiments need very large samples.",
        "does_not_establish": "-",
    },
    "gordon2019": {
        "citation": "Gordon, B. R., Zettelmeyer, F., Bhargava, N., & Chapsky, D. (2019). A Comparison of Approaches to Advertising Measurement: Evidence from Big Field Experiments at Facebook. Marketing Science, 38(2), 193-225.",
        "supports": "Observational methods often fail to recover experimental ad lift; RCTs remain the benchmark.",
        "does_not_establish": "-",
    },
    "gupta2006": {
        "citation": "Gupta, S., Hanssens, D., Hardie, B., et al. (2006). Modeling Customer Lifetime Value. Journal of Service Research, 9(2), 139-155.",
        "supports": "Discounted margin formulation of CLV and model families.",
        "does_not_establish": "-",
    },
    "fader2005": {
        "citation": "Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). 'Counting Your Customers' the Easy Way: An Alternative to the Pareto/NBD Model. Marketing Science, 24(2), 275-284.",
        "supports": "BG/NBD model for non-contractual repeat purchase.",
        "does_not_establish": "-",
    },
    "mcfadden1974": {
        "citation": "McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In P. Zarembka (Ed.), Frontiers in Econometrics (pp. 105-142). Academic Press.",
        "supports": "Multinomial/conditional logit choice probabilities.",
        "does_not_establish": "-",
    },
    "katz_shapiro1985": {
        "citation": "Katz, M. L., & Shapiro, C. (1985). Network Externalities, Competition, and Compatibility. American Economic Review, 75(3), 424-440.",
        "supports": "Consumption value increasing in the number of other users.",
        "does_not_establish": "The functional form (log vs power); DSDVC compares specifications empirically.",
    },
    "morris1991": {
        "citation": "Morris, M. D. (1991). Factorial Sampling Plans for Preliminary Computational Experiments. Technometrics, 33(2), 161-174.",
        "supports": "Elementary-effects screening.",
        "does_not_establish": "-",
    },
    "saltelli2010": {
        "citation": "Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., & Tarantola, S. (2010). Variance based sensitivity analysis of model output. Computer Physics Communications, 181(2), 259-270.",
        "supports": "Estimators for first-order and total Sobol indices.",
        "does_not_establish": "-",
    },
    "beaumont2002": {
        "citation": "Beaumont, M. A., Zhang, W., & Balding, D. J. (2002). Approximate Bayesian Computation in Population Genetics. Genetics, 162(4), 2025-2035.",
        "supports": "Rejection ABC with summary statistics.",
        "does_not_establish": "-",
    },
    "talts2018": {
        "citation": "Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018). Validating Bayesian Inference Algorithms with Simulation-Based Calibration. arXiv:1804.06788.",
        "supports": "Rank-uniformity check for posterior inference.",
        "does_not_establish": "-",
    },
    "killick2012": {
        "citation": "Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal Detection of Changepoints With a Linear Computational Cost. JASA, 107(500), 1590-1598.",
        "supports": "PELT change-point detection with penalised cost.",
        "does_not_establish": "-",
    },
    "hyndman2021": {
        "citation": "Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.",
        "supports": "Rolling-origin (time-series cross-validation) evaluation.",
        "does_not_establish": "-",
    },
    "lundberg2017": {
        "citation": "Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.",
        "supports": "SHAP additive feature attributions.",
        "does_not_establish": "Causal effects of features.",
    },
}

F = TypeVar("F", bound=Callable)


def cites(*keys: str) -> Callable[[F], F]:
    """Decorator attaching literature keys to a function/class; validated at import time."""
    for k in keys:
        if k not in LITERATURE:
            raise KeyError(f"Unknown literature key: {k}")

    def deco(obj: F) -> F:
        existing = list(getattr(obj, "__literature__", []))
        obj.__literature__ = existing + list(keys)  # type: ignore[attr-defined]
        return obj

    return deco


def references_for(obj) -> list[str]:
    return [LITERATURE[k]["citation"] for k in getattr(obj, "__literature__", [])]
