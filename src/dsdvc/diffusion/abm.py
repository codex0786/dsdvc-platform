"""DSDVC market simulator: event-driven multiplex agent-based model (spec §9-§12, §24-§29, §52, §53).

The SYNTHETIC world of the market digital twin. One tick = one day.

Mechanisms (each documented with its literature motivation, none claimed as validated):
  * Bass-style external influence p (awareness from mass media)               [bass1969]
  * Multiplex social exposure: IG follow graph, FB friends + Groups, WA ties + chats
  * Estimated algorithmic exposure (IG/FB only), paid exposure, creator seeding
  * STEPPS content appeal with configurable functional form                   [berger2013]
  * Simple vs complex contagion with heterogeneous thresholds                  [centola_macy2007, granovetter1978]
  * Journey state machine with transition-specific models                     [rogers2003]
  * Multi-brand multinomial-logit choice with network effects                 [mcfadden1974, katz_shapiro1985]
  * Satisfaction, dissatisfaction, churn and recovery; revenue and LTV         [gupta2006]
  * Schelling-inspired tie rewiring                                            [schelling1971]
  * Cross-platform transmission matrix                                         (estimated, §23)

All outputs are SIMULATED and carry a RunRecord for exact reproducibility (§67).
"""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import asdict, dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from dsdvc import SIMULATION_VERSION
from dsdvc.agents.seeding import CostModel, select_seeds
from dsdvc.competition.mnl import ATTRS, MNLParams, network_utility
from dsdvc.contagion.models import complex_contagion_prob, complex_contagion_signal
from dsdvc.core.literature import cites
from dsdvc.core.reproducibility import RunRecord, stable_hash
from dsdvc.diffusion.events import Event, EventBus, EventType
from dsdvc.networks.multiplex import LAYERS, MultiplexNetwork, synthetic_multiplex
from dsdvc.networks.schelling import SchellingParams, SchellingRewirer
from dsdvc.platforms.transmission import TransmissionMatrix
from dsdvc.recommendation.exposure import EstimatedExposureModel

STAGES = ["UNAWARE", "AWARE", "INTERESTED", "CONSIDERING", "TRIAL", "ADOPTED", "ENGAGED", "ADVOCATE",
          "DISSATISFIED", "CHURNED", "POSSIBLE_RECOVERY", "COMPETITOR"]
S = {s: i for i, s in enumerate(STAGES)}
ACTIVE = np.array([S["ADOPTED"], S["ENGAGED"], S["ADVOCATE"], S["DISSATISFIED"]])
CHANNELS = ["ig_social", "ig_algo", "ig_paid", "ig_creator", "fb_social", "fb_group", "fb_algo", "fb_paid",
            "wa_social", "wa_group", "external"]
CH = {c: i for i, c in enumerate(CHANNELS)}
PLAT_OF_CHANNEL = {c: ("instagram" if c.startswith("ig") else "facebook" if c.startswith("fb")
                       else "whatsapp" if c.startswith("wa") else "external") for c in CHANNELS}
PFX = {"instagram": "ig", "facebook": "fb", "whatsapp": "wa"}
STEPPS_KEYS = ("social_currency", "triggers", "emotion", "public", "practical_value", "stories")


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class PlatformParams:
    social_reach: float      # P(see a contact's share) per unit tie strength
    trust: float             # platform-level trust multiplier for persuasion
    share_base: float        # logit intercept of sharing given exposure
    algo_reach: float        # scale of estimated algorithmic exposure (0 = none)
    group_reach: float       # P(member sees a share posted in a group/chat)
    conversion_mult: float   # logit shift on conversion when this platform is the dominant touch
    cpm: float               # cost per 1000 paid impressions (ESTIMATED rate-card prior)


DEFAULT_PLATFORMS = {
    # Priors encoding the *hypotheses* in spec §10-12 (IG: reach/discovery; FB: community;
    # WA: low reach, high trust). They are parameters to calibrate, not findings.
    "instagram": PlatformParams(social_reach=0.30, trust=0.35, share_base=-3.8, algo_reach=0.025,
                                group_reach=0.0, conversion_mult=0.0, cpm=180.0),
    "facebook": PlatformParams(social_reach=0.35, trust=0.55, share_base=-3.8, algo_reach=0.01,
                               group_reach=0.25, conversion_mult=0.15, cpm=120.0),
    "whatsapp": PlatformParams(social_reach=0.75, trust=0.90, share_base=-2.8, algo_reach=0.0,
                               group_reach=0.45, conversion_mult=0.50, cpm=0.0),
}


@dataclass
class BrandSpec:
    name: str
    quality: float
    price: float            # per-period price (currency)
    brand_equity: float
    marketing: float = 0.3  # baseline awareness pressure for competitors
    is_focal: bool = False


def default_brands() -> list[BrandSpec]:
    return [BrandSpec("focal", 0.70, 30.0, 0.30, is_focal=True),
            BrandSpec("competitor_b", 0.60, 25.0, 0.60, 0.4),
            BrandSpec("competitor_c", 0.55, 18.0, 0.40, 0.3)]


@dataclass
class SimConfig:
    n_agents: int = 2000
    n_communities: int = 8
    steps: int = 90
    seed: int = 0
    network_seed: int = 0
    homophily_scale: float = 1.0
    population_scale: float = 100.0          # real consumers per agent (market sizing)
    p_innovation: float = 0.0008             # Bass external awareness hazard / day
    content_decay: float = 0.02
    stepps: dict = field(default_factory=lambda: {"social_currency": 0.6, "triggers": 0.5, "emotion": 0.7,
                                                  "public": 0.6, "practical_value": 0.5, "stories": 0.6})
    stepps_form: str = "multiplicative"      # additive | multiplicative | interaction
    stepps_weights: tuple = (0.6, 0.3, 1.1, 0.5, 0.7, 0.6)
    share_appeal_coef: float = 2.5
    complex_share: float = 0.5               # share of agents needing social confirmation
    threshold_mean: float = 2.0              # independent confirmations needed (complex agents)
    threshold_sd: float = 0.8
    kappa: float = 3.0
    platforms: dict = field(default_factory=lambda: {k: PlatformParams(**asdict(v)) for k, v in DEFAULT_PLATFORMS.items()})
    transmission: list | None = None         # 3x3; default = posterior mean of TransmissionMatrix prior
    brands: list = field(default_factory=default_brands)
    mnl: MNLParams = field(default_factory=MNLParams)
    category_purchase_intercept: float = -4.0
    seeding_strategy: str = "micro_network"
    seeding_budget: float = 200_000.0
    cost_model: CostModel = field(default_factory=CostModel)
    creator_post_every: int = 7              # seeded creators re-post on this cadence
    creator_posts: int = 3
    paid_daily_budget: dict = field(default_factory=lambda: {"instagram": 0.0, "facebook": 0.0})
    paid_targeting: float = 2.0              # relevance exponent for paid targeting
    retention_investment: float = 0.2        # 0..1
    retention_cost_per_active: float = 0.5   # currency / active customer / day at investment 1.0
    advocacy_share_rate: float = 0.03
    margin_rate: float = 0.6
    discount_rate_daily: float = 0.0003
    churn_coefs: dict = field(default_factory=lambda: {"intercept": -4.2, "dissatisfaction": 3.0, "competition": 1.0,
                                                       "price_pressure": 1.2, "engagement": 1.5, "community_fit": 1.0,
                                                       "retention": 1.2})
    recovery_rate: float = 0.02
    rewire_every: int = 15
    schelling: SchellingParams = field(default_factory=lambda: SchellingParams(rewiring_rate=0.03))
    competitor_actions: list = field(default_factory=list)  # [(t, brand_name, attr, value)]
    # budget levers (total campaign spend, currency); diminishing returns via 1 - exp(-spend/scale)
    referral_budget: float = 0.0              # referral incentives -> more advocacy sharing (esp. WhatsApp)
    retention_budget: float | None = None     # if set, overrides retention_investment and per-active costing
    content_budget: float = 0.0               # creative production -> higher content appeal
    lever_scales: dict = field(default_factory=lambda: {"referral": 100_000.0, "retention": 250_000.0,
                                                        "content": 150_000.0})
    content_taste: list | None = None         # content position in taste space (None = random)
    content_relevance_center: int | None = 0  # community whose taste the content targets

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class SimulationResult:
    config: SimConfig
    timeseries: pd.DataFrame
    agents: pd.DataFrame
    shares: pd.DataFrame
    paths: pd.DataFrame
    events: pd.DataFrame
    event_counts: pd.DataFrame
    seeding: dict
    run_record: RunRecord
    status: str = "simulated"

    def summary(self) -> dict:
        ts, ag, c = self.timeseries, self.agents, self.config
        sc = c.population_scale
        last = ts.iloc[-1]
        ever_adopted = int(ag.ever_adopted.sum())
        cost = float(last.cum_cost)            # already real currency
        rev = float(last.cum_revenue) * sc     # per-agent revenue scaled to market
        gross = rev * c.margin_rate
        acq = ag[ag.ever_adopted]
        by_plat = acq.acquisition_platform.value_counts()
        tot = max(by_plat.sum(), 1)
        ltv_obs = float(acq.discounted_margin.mean()) if len(acq) else 0.0
        return {
            "status": "SIMULATED",
            "agents": c.n_agents, "population_scale": sc,
            "reach": int(ag.ever_exposed.sum() * sc),
            "aware": int((ag.stage != "UNAWARE").sum() * sc),
            "total_shares": int(len(self.shares) * sc),
            "ever_adopted": int(ever_adopted * sc),
            "active_customers": int(last.active * sc),
            "churned": int(last.churned * sc),
            "lost_to_competitors": int(last.competitor * sc),
            "adoption_rate": ever_adopted / c.n_agents,
            "active_rate": float(last.active) / c.n_agents,
            "conversion_rate_of_reached": ever_adopted / max(1, int(ag.ever_exposed.sum())),
            "revenue": rev, "gross_margin": gross, "cost": cost,
            "profit": gross - cost, "roi": (gross - cost) / cost if cost > 0 else float("nan"),
            "cac": cost / (ever_adopted * sc) if ever_adopted else float("nan"),
            "realised_ltv_per_customer": ltv_obs,
            "acquisition_share_by_platform": (by_plat / tot).round(4).to_dict(),
            "platform_dependency": float((by_plat / tot).max()) if len(by_plat) else 0.0,
            "peak_daily_adoptions": int(ts.new_adoptions.max() * sc),
            "time_to_peak": int(ts.loc[ts.new_adoptions.idxmax(), "t"]) if ts.new_adoptions.max() > 0 else None,
            "max_cascade_generation": int(self.shares.generation.max()) if len(self.shares) else 0,
            "focal_market_share": float(last.focal_share),
            "seeding_strategy": self.seeding.get("strategy"), "seeds": len(self.seeding.get("seeds", [])),
            "run_id": self.run_record.run_id,
        }


@cites("bass1969", "berger2013", "centola_macy2007", "granovetter1978", "schelling1971", "mcfadden1974",
       "katz_shapiro1985", "rogers2003", "goldenberg2001", "rand_rust2011", "watts_dodds2007")
class MarketSimulator:
    def __init__(self, cfg: SimConfig | None = None, network: MultiplexNetwork | None = None):
        # deep copy: competitor actions mutate brand specs during a run
        self.cfg = cfg = copy.deepcopy(cfg) if cfg is not None else SimConfig()
        self.rng = np.random.default_rng(cfg.seed)
        self.mx = network if network is not None else synthetic_multiplex(
            cfg.n_agents, cfg.n_communities, cfg.homophily_scale, seed=cfg.network_seed)
        # copy layers that may be rewired so the caller's network is not mutated
        self.mx = MultiplexNetwork(self.mx.n, {k: g.copy() for k, g in self.mx.layers.items()},
                                   self.mx.community, self.mx.taste, self.mx.groups, self.mx.provenance,
                                   self.mx.network_seed)
        self.n = self.mx.n
        self.bus = EventBus()
        self.t = 0
        self._init_agents()
        self._build_matrices()
        self._init_content()
        self._init_seeding()
        self.paths: list[deque] = [deque(maxlen=12) for _ in range(self.n)]
        self.share_rows: list[tuple] = []
        self.records: list[dict] = []
        self.cum_cost = self.seeding["cost"] + self.lever_cost
        self.cum_revenue = 0.0
        self.T = np.array(cfg.transmission) if cfg.transmission is not None else TransmissionMatrix().transfer_matrix()
        self.bus.emit(Event(0, EventType.CAMPAIGN_LAUNCHED, data={"strategy": cfg.seeding_strategy,
                                                                  "seeds": len(self.seeding["seeds"]),
                                                                  "cost": self.seeding["cost"]}))

    # ------------------------------------------------------------------ setup
    def _init_agents(self):
        n, r = self.n, self.rng
        c = self.cfg
        self.stage = np.zeros(n, dtype=np.int8)
        self.trust = r.beta(3, 2, n)
        self.price_sens = r.beta(2.2, 2.5, n)
        self.susceptibility = r.lognormal(0, 0.5, n)
        self.susceptibility /= self.susceptibility.mean()
        self.is_complex = r.random(n) < c.complex_share
        self.threshold = np.clip(r.normal(c.threshold_mean, c.threshold_sd, n), 1.0, 6.0)
        self.product_fit_noise = r.normal(0, 0.12, n)
        self.satisfaction = np.full(n, 0.7)
        self.engagement = np.zeros(n)
        self.ever_exposed = np.zeros(n, dtype=bool)
        self.exposure_count = np.zeros(n)
        self.has_shared = np.zeros((n, 3), dtype=bool)
        self.generation = np.full((n, 3), -1, dtype=int)
        self.ever_adopted = np.zeros(n, dtype=bool)
        self.acq_channel = np.full(n, -1, dtype=int)
        self.acq_t = np.full(n, -1, dtype=int)
        self.churn_t = np.full(n, -1, dtype=int)
        self.seed_credit = np.full(n, -1, dtype=int)  # which seed's cascade reached the agent first
        self.revenue = np.zeros(n)
        self.disc_margin = np.zeros(n)
        self.brand_choice = np.full(n, -1, dtype=int)
        self.last_touch_platform = np.full(n, -1, dtype=int)

    def _build_matrices(self):
        self.A, self.W = {}, {}
        for name in LAYERS:
            g = self.mx.layers[name]
            # receiver x sender: for IG an edge (follower -> creator) means follower receives creator content.
            M = nx.to_scipy_sparse_array(g, nodelist=range(self.n), weight="weight", format="csr")
            if not g.is_directed():
                M = M.maximum(M.T)
            self.W[name] = sparse.csr_matrix(M)
            B = self.W[name].copy()
            B.data[:] = 1.0
            self.A[name] = B
        U = self.A["instagram"] + self.A["instagram"].T + self.A["facebook"] + self.A["whatsapp"]
        U.data[:] = 1.0
        self.A_union = sparse.csr_matrix(U)
        self.deg_union = np.asarray(self.A_union.sum(axis=1)).ravel()
        self.G = {}
        for p in ("facebook", "whatsapp"):
            groups = self.mx.groups.get(p, [])
            rows = np.concatenate([np.full(len(g), k) for k, g in enumerate(groups)]) if groups else np.array([], int)
            cols = np.concatenate(groups) if groups else np.array([], int)
            self.G[p] = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(max(len(groups), 1), self.n))
        # community fit = share of union-neighbours in own community (structural)
        same = np.zeros(self.n)
        Ucoo = self.A_union.tocoo()
        np.add.at(same, Ucoo.row, (self.mx.community[Ucoo.row] == self.mx.community[Ucoo.col]).astype(float))
        self.community_fit = np.where(self.deg_union > 0, same / np.maximum(self.deg_union, 1), 0.5)

    def _init_content(self):
        c = self.cfg
        d = self.mx.taste.shape[1]
        if c.content_taste is not None:
            v = np.asarray(c.content_taste, dtype=float)
        elif c.content_relevance_center is not None and c.content_relevance_center in set(self.mx.community):
            v = self.mx.taste[self.mx.community == c.content_relevance_center].mean(axis=0)
        else:
            v = self.rng.normal(0, 1, d)
        v = v / (np.linalg.norm(v) + 1e-9)
        self.relevance = np.clip((self.mx.taste @ v + 1) / 2, 0, 1)  # cosine -> [0,1]
        self.appeal = self.stepps_appeal(c.stepps, c.stepps_form, c.stepps_weights)
        sc = c.lever_scales
        if c.content_budget > 0:
            self.appeal += (1 - self.appeal) * 0.3 * (1 - math.exp(-c.content_budget / sc["content"]))
        self.advocacy_rate = c.advocacy_share_rate * (1 + 2 * (1 - math.exp(-c.referral_budget / sc["referral"])))
        if c.retention_budget is not None:
            c.retention_investment = 1 - math.exp(-c.retention_budget / sc["retention"])
        self.lever_cost = c.content_budget + c.referral_budget + (c.retention_budget or 0.0)
        self.exposure_models = {p: EstimatedExposureModel(p, reach=self.cfg.platforms[p].algo_reach) for p in LAYERS}
        self.total_shares = np.zeros(3)

    @staticmethod
    @cites("berger2013")
    def stepps_appeal(stepps: dict, form: str, weights) -> float:
        """Content appeal in [0,1] from STEPPS under the chosen functional form (spec §5)."""
        x = np.clip(np.array([stepps[k] for k in STEPPS_KEYS], dtype=float), 0.01, 1.0)
        w = np.asarray(weights, dtype=float)
        if form == "additive":
            return float(np.clip((w @ x) / w.sum(), 0, 1))
        if form == "multiplicative":
            return float(np.exp((w @ np.log(x)) / w.sum()))  # weighted geometric mean
        if form == "interaction":
            base = (w @ x) / w.sum()
            return float(np.clip(base + 0.3 * x[2] * x[3] - 0.15, 0, 1))  # emotion x public
        raise ValueError(form)

    def _init_seeding(self):
        c = self.cfg
        self.seeding = select_seeds(self.mx, c.seeding_strategy, c.seeding_budget, c.cost_model,
                                    relevance=self.relevance, seed=c.seed)
        self.seeds = np.array(self.seeding["seeds"], dtype=int)
        self.is_community_seed = np.zeros(self.n, dtype=bool)
        self.is_community_seed[[s for s in self.seeds if self.seeding["roles"][s] == "community"]] = True
        for s in self.seeds:
            self.stage[s] = S["ADOPTED"]  # seeded creators/ambassadors have the product
            self.ever_adopted[s] = True
            self.acq_t[s] = 0
            self.acq_channel[s] = CH["ig_creator"] if self.seeding["roles"][s] != "community" else CH["wa_social"]
            self.satisfaction[s] = 0.85
            self.engagement[s] = 0.8
            self.seed_credit[s] = s

    # ------------------------------------------------------------------ dynamics
    def _touch(self, mask: np.ndarray, channel: str):
        idx = np.flatnonzero(mask)
        ci = CH[channel]
        for i in idx:
            self.paths[i].append((self.t, ci))
        self.exposure_count[idx] += 1
        self.ever_exposed[idx] = True
        plat = PLAT_OF_CHANNEL[channel]
        if plat != "external":
            self.last_touch_platform[idx] = LAYERS.index(plat)

    def _assign_parents(self, new_sharers: np.ndarray, p: int, sharers_prev: np.ndarray):
        """Link each new sharer to a sharing contact (cascade tree) or mark as a root."""
        name = LAYERS[p]
        A = self.A[name]
        for i in new_sharers:
            row = A.indices[A.indptr[i]:A.indptr[i + 1]]
            cand = row[sharers_prev[row]] if len(row) else row
            if len(cand):
                par = int(self.rng.choice(cand))
                gen = self.generation[par, p] + 1 if self.generation[par, p] >= 0 else 1
                if self.seed_credit[i] < 0:
                    self.seed_credit[i] = self.seed_credit[par]
            else:
                par, gen = -1, 0
            self.generation[i, p] = gen
            self.share_rows.append((self.t, int(i), name, par, int(gen), int(self.mx.community[i]),
                                    int(self.seed_credit[i])))
            self.bus.emit(Event(self.t, EventType.CONTENT_SHARED, int(i), name, {"parent": par, "generation": int(gen)}))

    def step(self):
        c, r, n = self.cfg, self.rng, self.n
        self.t += 1
        t = self.t
        freshness = math.exp(-c.content_decay * t)
        self._apply_competitor_actions()

        # ---- 1. who is broadcasting this tick
        sharers = self._sharers_now  # (n,3) bool from previous tick's sharing decisions
        creator_post = np.zeros(n, dtype=bool)
        if len(self.seeds) and (t - 1) % c.creator_post_every == 0 and (t - 1) // c.creator_post_every < c.creator_posts:
            creator_post[self.seeds] = True
            self.bus.emit(Event(t, EventType.CONTENT_CREATED, data={"posts": int(len(self.seeds))}))

        exposed = np.zeros((n, 3), dtype=bool)
        social_any = np.zeros(n, dtype=bool)
        # ---- 2. exposure
        for p, name in enumerate(LAYERS):
            pp = c.platforms[name]
            # creators post on Instagram; community seeds post into FB groups / WA chats
            seed_post = creator_post & (~self.is_community_seed if name == "instagram" else self.is_community_seed)
            src = sharers[:, p] | seed_post
            pressure = self.W[name] @ src.astype(float)
            p_soc = 1 - np.exp(-pp.social_reach * pressure * freshness ** 0.5)
            soc = r.random(n) < p_soc
            if name == "instagram":
                from_creator = (self.A[name] @ seed_post.astype(float)) > 0
                self._touch(soc & from_creator, "ig_creator")
                self._touch(soc & ~from_creator, "ig_social")
                newly = soc & (self.seed_credit < 0) & from_creator
                if newly.any():
                    # credit the first creator followed
                    A = self.A[name]
                    for i in np.flatnonzero(newly):
                        row = A.indices[A.indptr[i]:A.indptr[i + 1]]
                        cr = row[seed_post[row]]
                        if len(cr):
                            self.seed_credit[i] = int(cr[0])
            else:
                self._touch(soc, f"{PFX[name]}_social")
            exposed[:, p] |= soc
            social_any |= soc
            # groups / chats
            if name in self.G and pp.group_reach > 0:
                active_groups = (self.G[name] @ src.astype(float)) > 0
                members = (self.G[name].T @ active_groups.astype(float)) > 0
                grp = members & (r.random(n) < pp.group_reach * freshness ** 0.5)
                self._touch(grp & ~soc, f"{PFX[name]}_group")
                exposed[:, p] |= grp
                social_any |= grp
            # estimated algorithmic exposure (not the proprietary algorithm)
            em = self.exposure_models[name]
            if em.enabled:
                proof = math.log1p(self.total_shares[p]) / math.log1p(n)
                prob = em.probability(relevance=self.relevance, predicted_engagement=self.appeal,
                                      social_proof=proof, creator_affinity=0.0, similarity=self.community_fit,
                                      freshness=freshness, previous_engagement=self.engagement)
                algo = r.random(n) < prob
                self._touch(algo & ~exposed[:, p], f"{PFX[name]}_algo")
                exposed[:, p] |= algo
                self.bus.emit_count(t, EventType.RECOMMENDATION_UPDATED, int(algo.sum()), name)
            # paid
            budget = c.paid_daily_budget.get(name, 0.0)
            if budget > 0 and pp.cpm > 0:
                impressions = budget / pp.cpm * 1000 / c.population_scale
                w = self.relevance ** c.paid_targeting
                prob = np.clip(impressions * w / w.sum(), 0, 1)
                paid = r.random(n) < prob
                self._touch(paid, f"{PFX[name]}_paid")
                exposed[:, p] |= paid
                self.cum_cost += budget  # real currency
            self.bus.emit_count(t, EventType.CONTENT_EXPOSED, int(exposed[:, p].sum()), name)
        # external (Bass p): mass-media / search / offline awareness
        ext = (self.stage == S["UNAWARE"]) & (r.random(n) < c.p_innovation)
        self._touch(ext, "external")
        any_exp = exposed.any(axis=1) | ext

        # ---- 3. awareness
        newly_aware = (self.stage == S["UNAWARE"]) & any_exp
        self.stage[newly_aware] = S["AWARE"]
        self.bus.emit_count(t, EventType.AWARENESS_CREATED, int(newly_aware.sum()))

        # ---- 4. engagement & sharing
        new_sharers = np.zeros((n, 3), dtype=bool)
        advocates = self.stage == S["ADVOCATE"]
        for p, name in enumerate(LAYERS):
            pp = c.platforms[name]
            logit = (pp.share_base + c.share_appeal_coef * (self.appeal - 0.5) + 1.2 * (self.relevance - 0.5)
                     + 1.0 * pp.trust * (self.trust - 0.5) + 0.5 * np.log(self.susceptibility)
                     + 1.5 * np.isin(self.stage, ACTIVE))
            p_share = _sig(logit)
            engaged = exposed[:, p] & (r.random(n) < np.clip(3 * p_share, 0, 1))
            self.bus.emit_count(t, EventType.CONTENT_ENGAGED, int(engaged.sum()), name)
            sh = exposed[:, p] & ~self.has_shared[:, p] & (r.random(n) < p_share)
            adv = advocates & ~self.has_shared[:, p] & (r.random(n) < self.advocacy_rate) & (self.deg_union > 0)
            for a in np.flatnonzero(adv):
                self.bus.emit(Event(t, EventType.ADVOCACY_OCCURRED, int(a), name))
            new_sharers[:, p] = sh | adv
        # cross-platform transfer
        for p in range(3):
            for q in range(3):
                if p == q or self.T[p, q] <= 0:
                    continue
                move = new_sharers[:, p] & ~self.has_shared[:, q] & ~new_sharers[:, q] & (r.random(n) < self.T[p, q] * 3)
                for a in np.flatnonzero(move):
                    self.bus.emit(Event(t, EventType.CROSS_PLATFORM_TRANSFER, int(a), LAYERS[p], {"to": LAYERS[q]}))
                new_sharers[:, q] |= move
        prev = sharers | creator_post[:, None]
        for p in range(3):
            idx = np.flatnonzero(new_sharers[:, p])
            if len(idx):
                self._assign_parents(idx, p, prev[:, p])
        self.has_shared |= new_sharers
        self.total_shares += new_sharers.sum(axis=0)
        self._sharers_now = new_sharers

        # ---- 5. funnel transitions
        adopters_or_sharers = (np.isin(self.stage, ACTIVE) | self.has_shared.any(axis=1)).astype(float)
        n_indep = self.A_union @ adopters_or_sharers
        frac_adopt = np.where(self.deg_union > 0, (self.A_union @ np.isin(self.stage, ACTIVE).astype(float)) / np.maximum(self.deg_union, 1), 0)
        dom = np.where(self.last_touch_platform >= 0, self.last_touch_platform, 0)
        plat_trust = np.array([c.platforms[x].trust for x in LAYERS])[dom]

        st = self.stage
        aw = (st == S["AWARE"]) & any_exp
        p_int = _sig(-2.2 + 2.2 * self.appeal + 2.0 * (self.relevance - 0.5) + 0.4 * np.log1p(self.exposure_count))
        st[aw & (r.random(n) < p_int)] = S["INTERESTED"]

        it = st == S["INTERESTED"]
        signal = complex_contagion_signal(n_indep, tie_strength=plat_trust, trust=self.trust, social_proof=frac_adopt * 3)
        p_cx = complex_contagion_prob(signal, self.threshold, c.kappa) * 0.35
        p_simple = _sig(-2.6 + 1.5 * (self.relevance - 0.5) + 2.0 * plat_trust + 0.8 * (n_indep > 0) + 0.6 * any_exp)
        p_cons = np.where(self.is_complex, p_cx, p_simple)
        st[it & (r.random(n) < p_cons)] = S["CONSIDERING"]

        cons = np.flatnonzero(st == S["CONSIDERING"])
        if len(cons):
            self._choose(cons, frac_adopt, plat_trust, dom, any_exp)

        # trial -> adopted / failed
        tri = np.flatnonzero((st == S["TRIAL"]) & (self.acq_t < t))
        if len(tri):
            q = self._focal().quality
            fit = np.clip(q + self.product_fit_noise[tri] + 0.1 * self.community_fit[tri], 0, 1)
            ok = r.random(len(tri)) < _sig(8 * (fit - 0.5))
            st[tri[ok]] = S["ADOPTED"]
            st[tri[~ok]] = S["CHURNED"]
            self.churn_t[tri[~ok]] = t
            for a in tri[ok]:
                self.bus.emit(Event(t, EventType.ADOPTION_OCCURRED, int(a), data={"channel": CHANNELS[self.acq_channel[a]]}))
            for a in tri[~ok]:
                self.bus.emit(Event(t, EventType.CHURN_OCCURRED, int(a), data={"reason": "failed_trial"}))

        self._lifecycle(any_exp, frac_adopt)
        self._economics()
        if c.rewire_every and t % c.rewire_every == 0:
            self._rewire()
        self._record(new_sharers, any_exp)

    def _focal(self) -> BrandSpec:
        return next(b for b in self.cfg.brands if b.is_focal)

    def _choose(self, cons: np.ndarray, frac_adopt, plat_trust, dom, any_exp):
        """Category purchase decision then multinomial-logit brand choice (spec §25-§27)."""
        c, r, t = self.cfg, self.rng, self.t
        brands = c.brands
        focal_idx = next(k for k, b in enumerate(brands) if b.is_focal)
        conv_mult = np.array([c.platforms[x].conversion_mult for x in LAYERS])[dom[cons]]
        p_buy = _sig(c.category_purchase_intercept + 1.2 * frac_adopt[cons] * 3 + 1.5 * (plat_trust[cons] - 0.5)
                     + 1.0 * (self.appeal - 0.5) + 0.8 * (self.community_fit[cons] - 0.5) + conv_mult)
        buy = cons[r.random(len(cons)) < p_buy]
        if not len(buy):
            return
        prices = np.array([b.price for b in brands])
        price_norm = prices / prices.mean() - 1
        N = np.array([self._installed_base(k) for k in range(len(brands))], dtype=float)
        X = np.zeros((len(buy), len(brands), len(ATTRS)))
        X[:, :, 0] = np.array([b.quality for b in brands])[None, :] - 0.5
        X[:, :, 1] = -(price_norm[None, :] * (0.5 + self.price_sens[buy, None]))  # beta_price < 0 => use magnitude
        X[:, :, 2] = np.array([b.brand_equity for b in brands])[None, :]
        X[:, :, 3] = 0.0
        X[:, focal_idx, 3] = frac_adopt[buy]
        X[:, :, 4] = 0.0
        X[:, focal_idx, 4] = self.community_fit[buy] - 0.5
        X[:, :, 5] = np.array([b.marketing for b in brands])[None, :]
        X[:, focal_idx, 5] = 1.0
        beta = c.mnl.beta.copy()
        b = np.array([abs(beta["price"]) if a == "price" else beta[a] for a in ATTRS])
        U = X @ b + network_utility(N / self.n * 100, c.mnl.network_spec, c.mnl.alpha, c.mnl.gamma)[None, :]
        U -= U.max(axis=1, keepdims=True)
        P = np.exp(U)
        P /= P.sum(axis=1, keepdims=True)
        choice = (r.random((len(buy), 1)) > P.cumsum(axis=1)).sum(axis=1)
        choice = np.minimum(choice, len(brands) - 1)
        st = self.stage
        for i, ch in zip(buy, choice):
            self.brand_choice[i] = ch
            if ch == focal_idx:
                st[i] = S["TRIAL"]
                self.acq_t[i] = t
                self.ever_adopted[i] = True
                last = self.paths[i][-1][1] if self.paths[i] else CH["external"]
                self.acq_channel[i] = last
                self.satisfaction[i] = 0.7
                self.engagement[i] = 0.4
                self.bus.emit(Event(t, EventType.TRIAL_STARTED, int(i), data={"channel": CHANNELS[last]}))
            else:
                st[i] = S["COMPETITOR"]

    def _installed_base(self, k: int) -> int:
        focal_idx = next(j for j, b in enumerate(self.cfg.brands) if b.is_focal)
        if k == focal_idx:
            return int(np.isin(self.stage, ACTIVE).sum())
        return int(((self.stage == S["COMPETITOR"]) & (self.brand_choice == k)).sum()) + int(self.n * 0.05 * self.cfg.brands[k].marketing)

    def _lifecycle(self, any_exp, frac_adopt):
        c, r, t, st = self.cfg, self.rng, self.t, self.stage
        focal = self._focal()
        prices = np.array([b.price for b in c.brands])
        comp_prices = np.array([b.price for b in c.brands if not b.is_focal])
        price_pressure = float(np.clip((focal.price - comp_prices.min()) / focal.price, 0, 1)) if len(comp_prices) else 0.0
        competition = float(np.mean([b.marketing * b.brand_equity for b in c.brands if not b.is_focal])) if len(comp_prices) else 0.0
        act = np.isin(st, ACTIVE)
        fit = np.clip(focal.quality + self.product_fit_noise, 0, 1)
        target_sat = np.clip(0.35 + 0.6 * fit + 0.15 * (self.community_fit - 0.5) - 0.35 * price_pressure * self.price_sens
                             + 0.1 * c.retention_investment, 0, 1)
        self.satisfaction[act] = 0.93 * self.satisfaction[act] + 0.07 * target_sat[act] + r.normal(0, 0.02, act.sum())
        self.engagement[act] = np.clip(0.95 * self.engagement[act] + 0.05 * (0.4 + 0.4 * self.satisfaction[act]
                                       + 0.3 * frac_adopt[act] + 0.2 * c.retention_investment), 0, 1)
        self.engagement[~act] *= 0.9

        ad = st == S["ADOPTED"]
        st[ad & (self.engagement > 0.55)] = S["ENGAGED"]
        en = st == S["ENGAGED"]
        st[en & (self.satisfaction > 0.78) & (self.engagement > 0.6) & (r.random(self.n) < 0.08)] = S["ADVOCATE"]
        dis = np.isin(st, [S["ADOPTED"], S["ENGAGED"], S["ADVOCATE"]]) & (self.satisfaction < 0.5)
        st[dis] = S["DISSATISFIED"]
        rec = (st == S["DISSATISFIED"]) & (self.satisfaction > 0.62)
        st[rec] = S["ENGAGED"]

        cc = c.churn_coefs
        churn_logit = (cc["intercept"] + cc["dissatisfaction"] * (1 - self.satisfaction) + cc["competition"] * competition
                       + cc["price_pressure"] * price_pressure * self.price_sens - cc["engagement"] * self.engagement
                       - cc["community_fit"] * self.community_fit - cc["retention"] * c.retention_investment)
        boost = np.where(st == S["DISSATISFIED"], 1.0, 0.0)
        churn = np.isin(st, ACTIVE) & (r.random(self.n) < _sig(churn_logit + 1.5 * boost))
        st[churn] = S["CHURNED"]
        self.churn_t[churn] = t
        for a in np.flatnonzero(churn):
            self.bus.emit(Event(t, EventType.CHURN_OCCURRED, int(a), data={"reason": "hazard"}))
        ch = (st == S["CHURNED"]) & any_exp & (r.random(self.n) < c.recovery_rate * 5)
        st[ch] = S["POSSIBLE_RECOVERY"]
        pr = np.flatnonzero(st == S["POSSIBLE_RECOVERY"])
        if len(pr):
            win = r.random(len(pr)) < _sig(-1.5 + 2 * (fit[pr] - 0.5) + c.retention_investment)
            st[pr[win]] = S["ADOPTED"]
            self.satisfaction[pr[win]] = 0.65
            st[pr[~win]] = S["CHURNED"]
        # competitor customers may reconsider (switching)
        comp = (st == S["COMPETITOR"]) & any_exp & (r.random(self.n) < 0.01)
        st[comp] = S["CONSIDERING"]

    def _economics(self):
        c, t = self.cfg, self.t
        focal = self._focal()
        act = np.isin(self.stage, ACTIVE)
        daily_price = focal.price / 30.0
        self.revenue[act] += daily_price
        self.disc_margin[act] += daily_price * c.margin_rate / (1 + c.discount_rate_daily) ** (t - np.maximum(self.acq_t[act], 0))
        self.cum_revenue += daily_price * act.sum()
        if c.retention_budget is None:  # otherwise retention is paid from the fixed retention budget
            self.cum_cost += c.retention_investment * c.retention_cost_per_active * act.sum() * c.population_scale

    def _rewire(self):
        moved = 0
        for name in ("facebook", "whatsapp"):
            rw = SchellingRewirer(self.mx.layers[name], self.mx.taste, self.cfg.schelling, seed=self.cfg.seed + self.t)
            moved += rw.step()
        if moved:
            self._build_matrices()
        self.bus.emit(Event(self.t, EventType.COMMUNITY_REWIRED, data={"moves": moved}))

    def _apply_competitor_actions(self):
        for (t, brand, attr, value) in self.cfg.competitor_actions:
            if t == self.t:
                for b in self.cfg.brands:
                    if b.name == brand:
                        setattr(b, attr, value)
                self.bus.emit(Event(self.t, EventType.COMPETITOR_ACTION, data={"brand": brand, "attr": attr, "value": value}))

    def _record(self, new_sharers, any_exp):
        st = self.stage
        counts = {s: int((st == i).sum()) for i, s in enumerate(STAGES)}
        active = int(np.isin(st, ACTIVE).sum())
        comp = counts["COMPETITOR"]
        prev_adopt = self.records[-1]["cum_adopted"] if self.records else int(self.ever_adopted[self.seeds].sum()) if len(self.seeds) else 0
        cum_adopted = int(self.ever_adopted.sum())
        from dsdvc.networks.metrics import edge_homophily
        rec = {"t": self.t, **{f"n_{k.lower()}": v for k, v in counts.items()},
               "active": active, "churned": counts["CHURNED"], "competitor": comp,
               "cum_adopted": cum_adopted, "new_adoptions": cum_adopted - prev_adopt,
               "reach": int(self.ever_exposed.sum()), "exposed_today": int(any_exp.sum()),
               "shares_ig": int(new_sharers[:, 0].sum()), "shares_fb": int(new_sharers[:, 1].sum()),
               "shares_wa": int(new_sharers[:, 2].sum()),
               "cum_revenue": self.cum_revenue, "cum_cost": self.cum_cost,
               "focal_share": active / max(1, active + comp)}
        if self.cfg.rewire_every and self.t % self.cfg.rewire_every == 0:
            rec["fb_homophily"] = edge_homophily(self.mx.layers["facebook"], self.mx.community)
            rec["wa_homophily"] = edge_homophily(self.mx.layers["whatsapp"], self.mx.community)
        self.records.append(rec)

    # ------------------------------------------------------------------ run
    def run(self) -> SimulationResult:
        self._sharers_now = np.zeros((self.n, 3), dtype=bool)
        for _ in range(self.cfg.steps):
            self.step()
        ts = pd.DataFrame(self.records)
        for col in ("fb_homophily", "wa_homophily"):
            if col in ts:
                ts[col] = ts[col].ffill()
        stage_names = np.array(STAGES)[self.stage]
        acq_plat = [PLAT_OF_CHANNEL[CHANNELS[c]] if c >= 0 else None for c in self.acq_channel]
        agents = pd.DataFrame({
            "agent": np.arange(self.n), "community": self.mx.community, "stage": stage_names,
            "ever_exposed": self.ever_exposed, "ever_adopted": self.ever_adopted,
            "acquisition_channel": [CHANNELS[c] if c >= 0 else None for c in self.acq_channel],
            "acquisition_platform": acq_plat, "acquired_t": self.acq_t, "churn_t": self.churn_t,
            "revenue": self.revenue, "discounted_margin": self.disc_margin, "satisfaction": self.satisfaction,
            "engagement": self.engagement, "relevance": self.relevance, "is_complex": self.is_complex,
            "is_seed": np.isin(np.arange(self.n), self.seeds), "seed_credit": self.seed_credit,
            "community_fit": self.community_fit, "price_sensitivity": self.price_sens,
        })
        shares = pd.DataFrame(self.share_rows, columns=["t", "agent", "platform", "parent", "generation", "community", "seed_credit"])
        paths = pd.DataFrame({"agent": np.arange(self.n),
                              "path": [[CHANNELS[c] for _, c in p] for p in self.paths],
                              "converted": self.ever_adopted & ~np.isin(np.arange(self.n), self.seeds)})
        paths = paths[paths.path.map(len) > 0].reset_index(drop=True)
        rr = RunRecord.create("abm", self.cfg.to_dict(), self.cfg.seed, self.cfg.network_seed,
                              data_version=(self.mx.provenance.observed_or_estimated.value if self.mx.provenance else "unknown"))
        rr.result_hash = stable_hash(ts.round(8).to_dict(orient="list"))
        return SimulationResult(self.cfg, ts, agents, shares, paths, self.bus.frame(), self.bus.count_frame(),
                                self.seeding, rr)


def run_simulation(cfg: SimConfig | None = None, network: MultiplexNetwork | None = None) -> SimulationResult:
    return MarketSimulator(cfg, network).run()
