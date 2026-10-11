from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import math
import numpy as np
import pandas as pd
import networkx as nx

PLATFORMS = ("instagram", "facebook", "whatsapp")
STATES = ("unaware", "aware", "adopter", "advocate", "dissatisfied", "churned")


def sigmoid(x: float) -> float:
    x = np.clip(x, -30, 30)
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass
class ModelConfig:
    n_agents: int = 800
    steps: int = 90
    seed: int = 42
    initial_adoption: float = 0.05
    p_innovation: float = 0.015
    q_imitation: float = 0.55
    market_condition: float = 0.80
    competition: float = 0.25
    price_pressure: float = 0.20
    homophily: float = 0.60
    rewiring_rate: float = 0.035
    community_threshold: float = 0.55
    instagram_reach: float = 0.90
    facebook_reach: float = 0.55
    whatsapp_reach: float = 0.25
    instagram_trust: float = 0.25
    facebook_trust: float = 0.45
    whatsapp_trust: float = 0.85
    instagram_algorithm: float = 0.80
    facebook_algorithm: float = 0.45
    whatsapp_forwarding: float = 0.70
    content_social_currency: float = 0.70
    content_triggers: float = 0.60
    content_emotion: float = 0.75
    content_public: float = 0.75
    content_practical: float = 0.55
    content_story: float = 0.70
    content_decay: float = 0.055
    push_intensity: float = 0.25
    retention_investment: float = 0.20
    content_investment: float = 0.50
    base_churn: float = 0.018
    recovery_rate: float = 0.03
    cross_platform_rate: float = 0.12
    simulation_noise: float = 0.05


class DSDVCMP:
    """Dynamic Social Diffusion, Virality & Churn — Multi-Platform ABM.

    Three-layer multiplex network:
      Instagram: discovery + algorithmic exposure + creator-like reach
      Facebook: community/group diffusion + public discussion
      WhatsApp: strong-tie/private forwarding + high trust

    The model is deliberately modular so empirical estimates can replace defaults.
    """

    def __init__(self, config: ModelConfig | None = None):
        self.cfg = config or ModelConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.t = 0
        self._build_agents()
        self._build_networks()
        self._seed_adoption()
        self.content_age = 0
        self.content_reach = 0.0
        self.total_shares = 0
        self.prev_adopters = int(np.sum(self.state == 2))
        self.metrics: List[dict] = []

    def _build_agents(self):
        n = self.cfg.n_agents
        r = self.rng
        self.innovation = np.clip(r.beta(2.0, 8.0, n), 0.01, 0.95)
        self.influence = np.clip(r.lognormal(-0.15, 0.65, n), 0.05, 1.0)
        self.price_sensitivity = r.beta(2.2, 3.0, n)
        self.content_sensitivity = r.beta(2.8, 2.0, n)
        self.trust = r.beta(3.2, 2.0, n)
        self.adoption_threshold = np.clip(r.normal(0.48, 0.13, n), 0.10, 0.90)
        self.churn_threshold = np.clip(r.normal(0.42, 0.12, n), 0.10, 0.90)
        self.engagement = np.zeros(n)
        self.satisfaction = np.clip(r.normal(0.60, 0.12, n), 0.10, 0.95)
        self.group = r.integers(0, max(2, int(np.sqrt(n) / 2)), n)
        self.state = np.zeros(n, dtype=np.int8)  # unaware
        self.exposure = {p: np.zeros(n) for p in PLATFORMS}
        self.shares = {p: np.zeros(n) for p in PLATFORMS}

    def _build_networks(self):
        n = self.cfg.n_agents
        r = self.rng
        self.networks: Dict[str, nx.Graph] = {}
        # Different empirical priors for platform structure.
        self.networks["instagram"] = nx.barabasi_albert_graph(n, max(2, min(6, n - 1)), seed=int(r.integers(1e9)))
        self.networks["facebook"] = nx.watts_strogatz_graph(n, k=max(4, min(10, n - 1)), p=0.12, seed=int(r.integers(1e9)))
        self.networks["whatsapp"] = nx.watts_strogatz_graph(n, k=max(2, min(6, n - 1)), p=0.04, seed=int(r.integers(1e9)))
        self.edge_weights: Dict[str, Dict[Tuple[int, int], float]] = {}
        for p, g in self.networks.items():
            weights = {}
            trust_scale = {"instagram": 0.55, "facebook": 0.75, "whatsapp": 1.0}[p]
            for a, b in g.edges():
                base = float(np.clip(r.beta(2.5, 2.5), 0.05, 1.0))
                if p == "whatsapp":
                    base = float(np.clip(0.45 + 0.55 * base, 0.35, 1.0))
                weights[(a, b)] = weights[(b, a)] = base * trust_scale
            self.edge_weights[p] = weights

    def _seed_adoption(self):
        n = self.cfg.n_agents
        k = max(1, int(n * self.cfg.initial_adoption))
        seeds = self.rng.choice(n, size=k, replace=False)
        self.state[seeds] = 2
        self.engagement[seeds] = self.rng.uniform(0.55, 0.9, k)
        self.satisfaction[seeds] = self.rng.uniform(0.60, 0.90, k)

    def _network_signal(self, platform: str) -> np.ndarray:
        g = self.networks[platform]
        signal = np.zeros(self.cfg.n_agents)
        for i in range(self.cfg.n_agents):
            nbrs = list(g.neighbors(i))
            if not nbrs:
                continue
            vals = []
            for j in nbrs:
                w = self.edge_weights[platform].get((i, j), 0.2)
                adopter_signal = 1.0 if self.state[j] in (2, 3) else 0.0
                vals.append(w * adopter_signal * (0.6 + 0.4 * self.influence[j]))
            signal[i] = np.clip(np.mean(vals), 0, 1)
        return signal

    def _community_fit(self, platform: str) -> np.ndarray:
        g = self.networks[platform]
        fit = np.full(self.cfg.n_agents, 0.5)
        for i in range(self.cfg.n_agents):
            nbrs = list(g.neighbors(i))
            if not nbrs:
                continue
            same = sum(self.group[j] == self.group[i] for j in nbrs)
            product_same = sum((self.state[j] in (2, 3)) == (self.state[i] in (2, 3)) for j in nbrs)
            fit[i] = 0.65 * same / len(nbrs) + 0.35 * product_same / len(nbrs)
        return fit

    def _stepps_score(self) -> float:
        c = self.cfg
        vals = [c.content_social_currency, c.content_triggers, c.content_emotion,
                c.content_public, c.content_practical, c.content_story]
        # Geometric mean rewards balanced content instead of allowing one dimension to dominate.
        return float(np.prod(np.clip(vals, 0.01, 1.0)) ** (1 / 6))

    def _platform_exposure(self, platform: str, network_signal: np.ndarray, community_fit: np.ndarray):
        c = self.cfg
        reach = {"instagram": c.instagram_reach, "facebook": c.facebook_reach, "whatsapp": c.whatsapp_reach}[platform]
        algorithm = {"instagram": c.instagram_algorithm, "facebook": c.facebook_algorithm, "whatsapp": c.whatsapp_forwarding}[platform]
        trust = {"instagram": c.instagram_trust, "facebook": c.facebook_trust, "whatsapp": c.whatsapp_trust}[platform]
        stepps = self._stepps_score()
        noise = self.rng.normal(0, c.simulation_noise, self.cfg.n_agents)
        if platform == "instagram":
            discovery = reach * (0.45 * network_signal + 0.55 * algorithm * stepps)
        elif platform == "facebook":
            discovery = reach * (0.60 * network_signal + 0.40 * community_fit * stepps)
        else:
            discovery = reach * (0.72 * network_signal * trust + 0.28 * algorithm * stepps * self.trust)
        return np.clip(discovery + noise, 0, 1)

    def _share_and_cross_post(self, platform: str, exposure: np.ndarray):
        c = self.cfg
        stepps = self._stepps_score()
        platform_bias = {"instagram": 0.25, "facebook": 0.10, "whatsapp": 0.18}[platform]
        trust = {"instagram": c.instagram_trust, "facebook": c.facebook_trust, "whatsapp": c.whatsapp_trust}[platform]
        public = c.content_public
        share_prob = np.array([
            sigmoid(-2.1 + 1.35 * stepps + 0.70 * self.content_sensitivity[i] +
                    0.45 * self.influence[i] + 0.55 * trust * self.trust[i] +
                    platform_bias + 0.60 * exposure[i])
            for i in range(self.cfg.n_agents)
        ])
        eligible = (self.state != 0) | (exposure > 0.15)
        shares = self.rng.random(self.cfg.n_agents) < (share_prob * eligible)
        self.shares[platform] += shares.astype(float)
        count = int(shares.sum())
        self.total_shares += count
        return shares, count

    def _rewire(self, community_fit: Dict[str, np.ndarray]):
        # Generalized Schelling: dissatisfied agents replace a weak tie with a more similar node.
        n = self.cfg.n_agents
        r = self.rng
        for platform, g in self.networks.items():
            fits = community_fit[platform]
            candidates = np.where(fits < self.cfg.community_threshold)[0]
            moves = r.random(len(candidates)) < self.cfg.rewiring_rate
            for i in candidates[moves]:
                nbrs = list(g.neighbors(int(i)))
                if not nbrs:
                    continue
                same_candidates = np.where(self.group == self.group[i])[0]
                same_candidates = same_candidates[same_candidates != i]
                # Homophily controls how strongly an agent prefers same-community replacement.
                if r.random() > self.cfg.homophily:
                    same_candidates = np.where(self.group != self.group[i])[0]
                    same_candidates = same_candidates[same_candidates != i]
                if len(same_candidates) == 0:
                    continue
                j = int(r.choice(same_candidates))
                weakest = min(nbrs, key=lambda x: self.edge_weights[platform].get((i, x), 0.1))
                if j != weakest and not g.has_edge(i, j):
                    g.remove_edge(i, weakest)
                    g.add_edge(i, j)
                    w = float(np.clip(0.45 + 0.55 * r.random(), 0.1, 1.0))
                    self.edge_weights[platform][(i, j)] = self.edge_weights[platform][(j, i)] = w

    def step(self):
        c = self.cfg
        self.t += 1
        self.content_age += 1
        decay = math.exp(-c.content_decay * self.content_age)
        network_signal = {p: self._network_signal(p) for p in PLATFORMS}
        community_fit = {p: self._community_fit(p) for p in PLATFORMS}

        # Platform exposure and sharing.
        for p in PLATFORMS:
            exp = self._platform_exposure(p, network_signal[p], community_fit[p]) * decay
            self.exposure[p] = 0.75 * self.exposure[p] + 0.25 * exp
            self.exposure[p] = np.clip(self.exposure[p], 0, 1)
            self._share_and_cross_post(p, self.exposure[p])

        # Cross-platform cascade: shared content creates exposure elsewhere.
        cross = c.cross_platform_rate * self._stepps_score() * decay
        self.exposure["facebook"] = np.clip(self.exposure["facebook"] + cross * self.shares["instagram"] / (1 + self.shares["instagram"].max()), 0, 1)
        self.exposure["whatsapp"] = np.clip(self.exposure["whatsapp"] + cross * self.shares["facebook"] / (1 + self.shares["facebook"].max()), 0, 1)
        self.exposure["instagram"] = np.clip(self.exposure["instagram"] + 0.5 * cross * self.shares["whatsapp"] / (1 + self.shares["whatsapp"].max()), 0, 1)

        # Adoption hazard: Bass innovation + endogenous multiplex imitation + content + community + market.
        weighted_social = (
            0.35 * network_signal["instagram"] +
            0.25 * network_signal["facebook"] +
            0.40 * network_signal["whatsapp"]
        )
        total_exposure = np.clip(
            0.35 * self.exposure["instagram"] +
            0.25 * self.exposure["facebook"] +
            0.40 * self.exposure["whatsapp"], 0, 1
        )
        avg_fit = 0.35 * community_fit["instagram"] + 0.25 * community_fit["facebook"] + 0.40 * community_fit["whatsapp"]
        market = np.clip(c.market_condition * (1 - 0.55 * c.competition), 0.05, 1)
        step = self._stepps_score()
        adoption_prob = np.zeros(self.cfg.n_agents)
        for i in range(self.cfg.n_agents):
            if self.state[i] in (2, 3):
                continue
            hazard = (
                c.p_innovation * (0.5 + self.innovation[i]) * market +
                c.q_imitation * weighted_social[i] * (0.5 + self.influence[i]) +
                0.55 * total_exposure[i] * self.content_sensitivity[i] * step +
                0.30 * avg_fit[i] +
                0.22 * c.push_intensity * market -
                0.35 * c.price_pressure * self.price_sensitivity[i]
            )
            adoption_prob[i] = np.clip(1 - math.exp(-max(0, hazard)), 0, 0.95)
        adopt = (self.rng.random(self.cfg.n_agents) < adoption_prob) & np.isin(self.state, [0, 1, 4, 5])
        self.state[adopt] = 2
        self.engagement[adopt] = np.maximum(self.engagement[adopt], 0.35 + 0.45 * total_exposure[adopt])

        # Awareness transition for exposed non-adopters.
        aware = (self.state == 0) & (total_exposure > 0.10)
        self.state[aware] = 1

        # Engagement and satisfaction.
        engagement_delta = 0.08 * total_exposure + 0.04 * weighted_social + 0.03 * c.retention_investment - 0.02 * c.competition
        self.engagement = np.clip(0.97 * self.engagement + engagement_delta, 0, 1)
        self.satisfaction = np.clip(
            0.96 * self.satisfaction +
            0.04 * (0.55 + 0.25 * self.engagement + 0.20 * avg_fit - 0.15 * c.price_pressure), 0, 1
        )

        # Advocate state.
        advocates = (self.state == 2) & (self.engagement > 0.72) & (self.satisfaction > 0.70)
        self.state[advocates] = 3

        # Churn hazard.
        churn_hazard = (
            c.base_churn * (1 + 1.2 * c.competition + 0.8 * c.price_pressure) *
            (1.35 - 0.75 * self.satisfaction) *
            (1.25 - 0.50 * avg_fit) *
            (1.15 - 0.35 * self.engagement) -
            0.015 * c.retention_investment
        )
        churn_prob = np.clip(churn_hazard, 0.001, 0.30)
        churn = (self.rng.random(self.cfg.n_agents) < churn_prob) & np.isin(self.state, [2, 3, 4])
        self.state[churn] = 5
        self.engagement[churn] *= 0.2

        # Recovery/re-entry from churn when social/content exposure is strong.
        recovery_prob = c.recovery_rate * (0.5 + total_exposure) * (0.5 + c.retention_investment)
        recover = (self.state == 5) & (self.rng.random(self.cfg.n_agents) < recovery_prob)
        self.state[recover] = 2

        self._rewire(community_fit)
        self._record(network_signal, community_fit, weighted_social, total_exposure)

    def _record(self, network_signal, community_fit, weighted_social, total_exposure):
        adopters = int(np.isin(self.state, [2, 3]).sum())
        advocates = int((self.state == 3).sum())
        churned = int((self.state == 5).sum())
        aware = int((self.state == 1).sum())
        dissatisfied = int((self.state == 4).sum())
        # Reproduction proxy: new shares per currently active advocate/adopter per step.
        prev_shares = 0 if self.t == 1 else sum(self.metrics[-1][k] for k in ["instagram_shares", "facebook_shares", "whatsapp_shares"])
        current_shares = self.total_shares
        new_shares = max(0.0, current_shares - prev_shares)
        rv = new_shares / max(1.0, float(adopters))
        platform_share = {p: float(self.shares[p].sum()) for p in PLATFORMS}
        homophily = {p: self._edge_homophily(p) for p in PLATFORMS}
        self.metrics.append({
            "step": self.t,
            "aware": aware,
            "adopters": adopters,
            "advocates": advocates,
            "dissatisfied": dissatisfied,
            "churned": churned,
            "adoption_rate": adopters / self.cfg.n_agents,
            "churn_rate": churned / self.cfg.n_agents,
            "avg_satisfaction": float(self.satisfaction[np.isin(self.state, [2,3])].mean()) if adopters else 0,
            "avg_engagement": float(self.engagement[np.isin(self.state, [2,3])].mean()) if adopters else 0,
            "viral_R_proxy": float(rv),
            "stepps_score": self._stepps_score(),
            "cross_platform_exposure": float(total_exposure.mean()),
            "network_imitation": float(weighted_social.mean()),
            "instagram_shares": platform_share["instagram"],
            "facebook_shares": platform_share["facebook"],
            "whatsapp_shares": platform_share["whatsapp"],
            "instagram_homophily": homophily["instagram"],
            "facebook_homophily": homophily["facebook"],
            "whatsapp_homophily": homophily["whatsapp"],
            "market_condition": self.cfg.market_condition,
            "competition": self.cfg.competition,
            "push_intensity": self.cfg.push_intensity,
            "retention_investment": self.cfg.retention_investment,
        })

    def _edge_homophily(self, platform):
        g = self.networks[platform]
        if g.number_of_edges() == 0:
            return 0.0
        return float(np.mean([self.group[a] == self.group[b] for a, b in g.edges()]))

    def run(self) -> pd.DataFrame:
        for _ in range(self.cfg.steps):
            self.step()
        return pd.DataFrame(self.metrics)

    def summary(self) -> dict:
        df = pd.DataFrame(self.metrics)
        if df.empty:
            return {}
        last = df.iloc[-1].to_dict()
        last["peak_adoption"] = float(df.adoption_rate.max())
        last["peak_adoption_step"] = int(df.loc[df.adoption_rate.idxmax(), "step"])
        last["net_active_users"] = int(df.iloc[-1].adopters)
        last["total_shares"] = int(self.total_shares)
        return last


def run_scenario(config: ModelConfig, seeds: List[int] | None = None) -> pd.DataFrame:
    seeds = seeds or [config.seed]
    rows = []
    for s in seeds:
        cfg = ModelConfig(**asdict(config))
        cfg.seed = int(s)
        model = DSDVCMP(cfg)
        model.run()
        out = model.summary()
        out["seed"] = s
        rows.append(out)
    return pd.DataFrame(rows)


def sweep(config: ModelConfig, parameter_grid: Dict[str, List], repetitions: int = 3) -> pd.DataFrame:
    import itertools
    keys = list(parameter_grid)
    rows = []
    for values in itertools.product(*[parameter_grid[k] for k in keys]):
        base = asdict(config)
        for k, v in zip(keys, values):
            base[k] = v
        for rep in range(repetitions):
            base["seed"] = config.seed + rep
            model = DSDVCMP(ModelConfig(**base))
            model.run()
            s = model.summary()
            s.update({k: v for k, v in zip(keys, values)})
            s["rep"] = rep
            rows.append(s)
    return pd.DataFrame(rows)
