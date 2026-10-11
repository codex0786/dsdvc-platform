import numpy as np
import pandas as pd
import pytest

from dsdvc.content.cascade import CascadeEngine
from dsdvc.diffusion.abm import ACTIVE, STAGES, MarketSimulator, SimConfig, run_simulation
from dsdvc.diffusion.events import EventType
from dsdvc.experiments.counterfactual import compare_seeding_strategies, run_counterfactual
from dsdvc.experiments.design import ab_simulation, latin_hypercube, monte_carlo, morris_screening, sobol_indices

SMALL = dict(n_agents=400, steps=40, rewire_every=20)


@pytest.fixture(scope="module")
def result():
    return run_simulation(SimConfig(**SMALL))


def test_reproducible_exactly():
    a = run_simulation(SimConfig(**SMALL, seed=5))
    b = run_simulation(SimConfig(**SMALL, seed=5))
    assert a.run_record.result_hash == b.run_record.result_hash
    pd.testing.assert_frame_equal(a.timeseries, b.timeseries)
    c = run_simulation(SimConfig(**SMALL, seed=6))
    assert c.run_record.result_hash != a.run_record.result_hash
    rr = a.run_record
    assert {"random_seed", "network_seed", "parameters", "data_version", "model_version", "software_version"} <= set(rr.model_dump())


def test_population_conserved_and_states_valid(result):
    ts = result.timeseries
    stage_cols = [f"n_{s.lower()}" for s in STAGES]
    assert (ts[stage_cols].sum(axis=1) == SMALL["n_agents"]).all()
    assert set(result.agents.stage) <= set(STAGES)
    assert (ts.cum_adopted.diff().dropna() >= 0).all()


def test_all_core_events_fire(result):
    counts = result.event_counts.sum()
    for e in ("CAMPAIGN_LAUNCHED", "CONTENT_CREATED", "CONTENT_EXPOSED", "CONTENT_SHARED", "AWARENESS_CREATED",
              "TRIAL_STARTED", "ADOPTION_OCCURRED", "RECOMMENDATION_UPDATED", "COMMUNITY_REWIRED"):
        assert counts.get(e, 0) > 0, e


def test_event_bus_subscription():
    seen = []
    sim = MarketSimulator(SimConfig(n_agents=300, steps=10, rewire_every=0))
    sim.bus.subscribe(EventType.CONTENT_SHARED, lambda ev: seen.append(ev))
    sim.run()
    assert seen and all(ev.type == EventType.CONTENT_SHARED for ev in seen)


def test_cascade_tree_from_simulation(result):
    sh = result.shares.reset_index().rename(columns={"index": "share_id"})
    key = {(r.agent, r.platform): r.share_id for r in sh.itertuples()}
    sh["parent_share_id"] = [key.get((p, pl)) if p >= 0 else None for p, pl in zip(sh.parent, sh.platform)]
    sh = sh.rename(columns={"agent": "sharer_id"})
    s = CascadeEngine(sh).summary()
    assert s["size"] == len(sh) and s["depth"] >= 1


def test_summary_economics_consistent(result):
    s = result.summary()
    assert s["status"] == "SIMULATED"
    assert s["profit"] == pytest.approx(s["gross_margin"] - s["cost"])
    assert abs(sum(s["acquisition_share_by_platform"].values()) - 1) < 1e-3  # shares rounded to 4 dp
    assert s["cost"] >= result.seeding["cost"]


def test_stepps_appeal_forms():
    st = {"social_currency": 0.5, "triggers": 0.5, "emotion": 0.5, "public": 0.5, "practical_value": 0.5, "stories": 0.5}
    for form in ("additive", "multiplicative"):
        assert MarketSimulator.stepps_appeal(st, form, (1,) * 6) == pytest.approx(0.5)
    st0 = {**st, "emotion": 0.01}
    assert MarketSimulator.stepps_appeal(st0, "multiplicative", (1,) * 6) < MarketSimulator.stepps_appeal(st0, "additive", (1,) * 6)


@pytest.mark.slow
def test_counterfactual_instagram_reach_cut_reduces_reach():
    r = run_counterfactual(SimConfig(**SMALL), "instagram_reach_minus_30", n_reps=4)
    row = r["table"].set_index("metric").loc["reach"]
    assert row.difference < 0 and {"baseline", "treatment", "difference", "ci_low", "ci_high"} <= set(r["table"].columns)


@pytest.mark.slow
def test_competitor_price_cut_is_applied():
    r = run_counterfactual(SimConfig(**SMALL), "competitor_price_minus_10", n_reps=3)
    assert any(k.startswith("brands.") and k.endswith(".price") for k in r["interventions"])


@pytest.mark.slow
def test_strategy_comparison_table():
    t = compare_seeding_strategies(SimConfig(**SMALL), n_reps=2)
    assert set(t.index) == {"celebrity_first", "micro_network", "community_seeding", "hybrid"}
    assert {"reach", "cost", "cascade_probability", "conversion_of_reached", "retention", "roi"} <= set(t.columns)


def test_monte_carlo_and_ab():
    mc = monte_carlo(SimConfig(n_agents=300, steps=20, rewire_every=0), n_seeds=3)
    assert len(mc) == 3 and mc.seed.nunique() == 3
    res, _ = ab_simulation(SimConfig(n_agents=300, steps=20, rewire_every=0), {"seeding_budget": 0.0}, n_reps=3)
    m = {r.metric: r for r in res}
    assert m["ever_adopted"].difference < 0


def test_lhs_design():
    df = latin_hypercube(SimConfig(n_agents=300, steps=15, rewire_every=0), {"complex_share": (0, 1)}, n=4)
    assert len(df) == 4 and df.complex_share.between(0, 1).all()


def test_sobol_and_morris_on_analytic_function():
    f = lambda X: X[:, 0] + 0.1 * X[:, 1] + 0 * X[:, 2]  # noqa: E731
    b = {"a": (0, 1), "b": (0, 1), "c": (0, 1)}
    s = sobol_indices(f, b, n=512).set_index("parameter")
    assert s.loc["a", "S1"] > 0.9 and abs(s.loc["c", "ST"]) < 0.02
    m = morris_screening(f, b, r=8).set_index("parameter")
    assert m.loc["a", "mu_star"] > m.loc["b", "mu_star"] > m.loc["c", "mu_star"] - 1e-12
