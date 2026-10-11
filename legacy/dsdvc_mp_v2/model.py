from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd
import networkx as nx


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))

@dataclass
class Params:
    n_agents: int = 500
    periods: int = 120
    p: float = 0.02
    q: float = 0.40
    churn_rate: float = 0.08
    push: float = 0.50
    price_sensitivity: float = 0.45
    market_condition: float = 0.65
    competition: float = 0.30
    initial_adoption: float = 0.02
    prob_to_adopt: float = 0.60
    staying_threshold: float = 0.50
    homophily: float = 0.55
    rewiring_rate: float = 0.06
    content_frequency: float = 0.50
    content_quality: float = 0.60
    social_currency: float = 0.60
    triggers: float = 0.55
    emotion: float = 0.70
    public: float = 0.60
    practical_value: float = 0.55
    stories: float = 0.65
    content_decay: float = 0.08
    retention_investment: float = 0.25
    recovery_rate: float = 0.04
    seed: int = 42

class DSDVCModel:
    """Dynamic Social Diffusion, Virality & Churn model.

    Integrates:
      - Bass external innovation + social imitation
      - Schelling-style homophily/rewiring
      - Berger STEPPS content sharing
      - lifecycle engagement, churn and recovery
      - endogenous firm response to churn
    """
    def __init__(self, params: Params):
        self.p = params
        self.rng = np.random.default_rng(params.seed)
        self.t = 0
        self._setup_agents()
        self._setup_network()
        self._setup_content()
        self.history = []

    def _setup_agents(self):
        n = self.p.n_agents
        rng = self.rng
        self.innovation = np.clip(rng.normal(self.p.p, max(self.p.p * .35, .002), n), .001, 1)
        self.imitation = np.clip(rng.normal(self.p.q, max(self.p.q * .30, .01), n), 0, 1)
        self.influence = np.clip(rng.beta(2, 5, n), .01, .99)
        self.threshold = np.clip(rng.normal(self.p.prob_to_adopt, .12, n), .05, .95)
        self.homophily_pref = np.clip(rng.normal(self.p.homophily, .15, n), 0, 1)
        self.price_sens = np.clip(rng.beta(2.5, 3, n), 0, 1)
        self.content_sens = np.clip(rng.beta(2.5, 2.5, n), 0, 1)
        self.trust = np.clip(rng.beta(4, 2, n), 0, 1)
        self.satisfaction = np.clip(rng.normal(.60, .12, n), 0, 1)
        self.engagement = np.zeros(n)
        self.adopted = np.zeros(n, dtype=bool)
        self.churned = np.zeros(n, dtype=bool)
        self.aware = np.zeros(n, dtype=bool)
        self.advocate = np.zeros(n, dtype=bool)
        self.adoption_time = np.full(n, -1, dtype=int)
        self.churn_time = np.full(n, -1, dtype=int)
        self.shares = np.zeros(n, dtype=int)
        initial = max(1, int(self.p.n_agents * self.p.initial_adoption))
        seeds = rng.choice(n, initial, replace=False)
        self.adopted[seeds] = True
        self.aware[seeds] = True
        self.adoption_time[seeds] = 0
        self.engagement[seeds] = .5
        self.groups = rng.integers(0, 4, n)

    def _setup_network(self):
        n = self.p.n_agents
        # Small-world base gives local neighborhoods while preserving bridges.
        k = min(8, max(2, n - 1 if n < 10 else 8))
        if k % 2: k += 1
        self.G = nx.watts_strogatz_graph(n, k, 0.10, seed=int(self.p.seed))
        self._rewire_by_homophily(initial=True)

    def _rewire_by_homophily(self, initial=False):
        # Probabilistically rewire edges toward similar latent groups/preferences.
        rate = min(1, self.p.rewiring_rate * (2 if initial else 1))
        edges = list(self.G.edges())
        for u, v in edges:
            if self.rng.random() > rate:
                continue
            similarity = 1 - abs(self.groups[u] - self.groups[v]) / 3
            if self.rng.random() < self.p.homophily * (1 - similarity + .25):
                candidates = np.arange(self.p.n_agents)
                self.rng.shuffle(candidates)
                for w in candidates[:30]:
                    if w == u or self.G.has_edge(u, w):
                        continue
                    sim = 1 - abs(self.groups[u] - self.groups[w]) / 3
                    if self.rng.random() < self.p.homophily * sim + (1-self.p.homophily)*.05:
                        self.G.remove_edge(u, v)
                        self.G.add_edge(u, int(w))
                        break

    def _setup_content(self):
        self.content_age = 0
        self.content_alive = True
        self.content_reach = 0.0
        self.total_shares = 0
        self.secondary_shares = 0
        self.content_score = self._stepps_score()

    def _stepps_score(self):
        p = self.p
        x = np.array([p.social_currency, p.triggers, p.emotion, p.public,
                      p.practical_value, p.stories])
        # Geometric mean penalizes weak dimensions while allowing complementarity.
        gm = float(np.prod(np.clip(x, 1e-4, 1)) ** (1/6))
        interaction = 0.15 * math.sqrt(p.emotion * p.public) + 0.10 * math.sqrt(p.stories * p.social_currency)
        return float(np.clip(0.75 * gm + interaction + 0.25 * p.content_quality, 0, 1))

    def _network_metrics(self):
        same = 0; total = self.G.number_of_edges(); cross = 0
        if total:
            for u, v in self.G.edges():
                if self.groups[u] == self.groups[v]: same += 1
                else: cross += 1
        homophily = same / total if total else 0
        cross_exposure = cross / total if total else 0
        return homophily, cross_exposure

    def step(self):
        p = self.p
        n = p.n_agents
        t = self.t
        # Market conditions evolve mildly around the chosen baseline.
        market = np.clip(p.market_condition + 0.06*np.sin(t/18) + self.rng.normal(0, .015), 0, 1)
        competition = np.clip(p.competition + 0.05*np.sin(t/25 + 1), 0, 1)

        # 1) Social exposure and community fit.
        social_exposure = np.zeros(n)
        community_fit = np.zeros(n)
        neighbor_adoption = np.zeros(n)
        for i in range(n):
            nbrs = list(self.G.neighbors(i))
            if nbrs:
                weights = np.array([0.5 + self.influence[j] for j in nbrs])
                adop = np.array([self.adopted[j] and not self.churned[j] for j in nbrs], dtype=float)
                social_exposure[i] = float(np.average(adop, weights=weights))
                neighbor_adoption[i] = social_exposure[i]
                same = sum(self.groups[j] == self.groups[i] for j in nbrs)
                community_fit[i] = same / len(nbrs)

        # 2) Content exposure / sharing cascade.
        self.content_age += 1
        attention = math.exp(-p.content_decay * self.content_age)
        viral_strength = self.content_score * attention * p.content_frequency
        exposures = np.zeros(n, dtype=float)
        for i in range(n):
            nbrs = list(self.G.neighbors(i))
            if not nbrs: continue
            sharer_fraction = sum(self.shares[j] > 0 for j in nbrs) / len(nbrs)
            exposures[i] = np.clip(viral_strength * (.25 + .75*sharer_fraction) * self.content_sens[i], 0, 1)

        # Initial content seeding each time content is active.
        if self.content_age == 1:
            seed_count = max(1, int(n * (.02 + .10*self.content_score)))
            seeds = self.rng.choice(n, seed_count, replace=False)
            exposures[seeds] = np.maximum(exposures[seeds], viral_strength)

        # 3) Adoption hazard: Bass + local imitation + content + community + market.
        active = self.adopted & ~self.churned
        market_factor = .55 + .90*market - .35*competition
        price_factor = 1 - .35*self.price_sens  # normalized placeholder for relative price
        push_factor = .65 + .70*p.push
        adoption_hazard = (
            self.innovation * market_factor * push_factor * .55
            + self.imitation * social_exposure * market_factor
            + .85*exposures
            + .20*community_fit
        ) * price_factor
        adoption_prob = 1 - np.exp(-np.clip(adoption_hazard, 0, 5))
        candidates = (~self.adopted) | self.churned
        draw = self.rng.random(n)
        new_adopters = candidates & (draw < adoption_prob)
        # Don't let every churned user instantly recover; recovery has its own probability.
        new_adopters &= (~self.churned) | (self.rng.random(n) < p.recovery_rate + .10*p.push)
        self.adopted[new_adopters] = True
        self.churned[new_adopters] = False
        self.aware[new_adopters] = True
        self.adoption_time[new_adopters & (self.adoption_time < 0)] = t
        self.engagement[new_adopters] = np.maximum(self.engagement[new_adopters], .35)

        # 4) Engagement and satisfaction.
        self.engagement[active] = np.clip(
            .88*self.engagement[active] + .12*(social_exposure[active] + exposures[active] + community_fit[active])/3, 0, 1)
        experience = np.clip(.35 + .35*market + .25*community_fit + .25*exposures - .30*competition, 0, 1)
        self.satisfaction[active] = np.clip(.90*self.satisfaction[active] + .10*experience[active], 0, 1)

        # 5) Churn hazard.
        churn_hazard = (
            p.churn_rate * (1.25 - self.satisfaction) * (1.15 - self.engagement)
            + .08*competition
            + .06*(1-community_fit)
            - .10*p.retention_investment
            + .025*p.push
        )
        churn_prob = np.clip(churn_hazard, 0, .75)
        churn_draw = self.rng.random(n)
        churn_now = active & (churn_draw < churn_prob)
        self.churned[churn_now] = True
        self.churn_time[churn_now] = t
        self.engagement[churn_now] *= .25

        # 6) Advocacy / content sharing using STEPPS.
        active = self.adopted & ~self.churned
        share_logit = (
            -2.1
            + 2.0*self.content_score
            + 1.1*self.engagement
            + .7*self.influence
            + .5*self.trust
            + .4*community_fit
            + .5*p.push
        )
        share_prob = np.clip(sigmoid(share_logit) * attention * p.content_frequency, 0, .95)
        share_now = active & (self.rng.random(n) < share_prob)
        self.shares[share_now] += 1
        self.advocate = self.shares > 0
        self.total_shares += int(share_now.sum())
        self.secondary_shares += int((share_now & (self.adoption_time >= 0) & (self.adoption_time < t)).sum())

        # 7) Schelling-style rewiring. Unsatisfied agents seek a more compatible local neighborhood.
        unhappy = community_fit < self.p.staying_threshold
        for i in np.where(unhappy)[0]:
            if self.rng.random() > p.rewiring_rate:
                continue
            nbrs = list(self.G.neighbors(int(i)))
            if not nbrs: continue
            old = nbrs[int(self.rng.integers(len(nbrs)))]
            candidates = self.rng.choice(n, size=min(50, n), replace=False)
            best = None
            best_score = community_fit[i]
            for w in candidates:
                if w == i or self.G.has_edge(i, int(w)): continue
                sim = 1 - abs(self.groups[i] - self.groups[w]) / 3
                score = p.homophily*sim + (1-p.homophily)*self.rng.random()
                if score > best_score:
                    best_score = score; best = int(w)
            if best is not None:
                self.G.remove_edge(i, old); self.G.add_edge(i, best)

        homophily, cross_exposure = self._network_metrics()
        active_count = int((self.adopted & ~self.churned).sum())
        cumulative_adopters = int(self.adopted.sum())
        churned_count = int(self.churned.sum())
        rv = self.secondary_shares / max(1, self.total_shares)
        record = dict(
            t=t, market_condition=market, competition=competition,
            active_adopters=active_count, cumulative_adopters=cumulative_adopters,
            churned=churned_count, new_adopters=int(new_adopters.sum()),
            new_churn=int(churn_now.sum()), adoption_rate=active_count/n,
            churn_rate_realized=int(churn_now.sum())/max(1, int(active.sum())),
            stepps_score=self.content_score, content_attention=attention,
            shares=int(self.total_shares), shares_this_period=int(share_now.sum()),
            viral_R=rv, community_fit=float(community_fit.mean()),
            homophily=homophily, cross_community_exposure=cross_exposure,
            average_satisfaction=float(self.satisfaction[active].mean()) if active.any() else 0,
            average_engagement=float(self.engagement[active].mean()) if active.any() else 0,
            push=p.push, retention_investment=p.retention_investment
        )
        self.history.append(record)
        self.t += 1
        return record

    def run(self, periods=None):
        periods = periods or self.p.periods
        for _ in range(periods): self.step()
        return pd.DataFrame(self.history)


def run_simulation(params: Params):
    return DSDVCModel(params).run()


def scenario_grid(base: Params, names, values, periods=80):
    rows = []
    import itertools
    for combo in itertools.product(*values):
        d = asdict(base)
        for k, v in zip(names, combo): d[k] = v
        p = Params(**d); df = run_simulation(p)
        last = df.iloc[-1]
        peak = df.active_adopters.max()
        rows.append({**{k:v for k,v in zip(names, combo)},
                     'final_active': last.active_adopters,
                     'peak_active': peak,
                     'final_churned': last.churned,
                     'viral_R': last.viral_R,
                     'homophily': last.homophily,
                     'cross_community_exposure': last.cross_community_exposure})
    return pd.DataFrame(rows)
