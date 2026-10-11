import numpy as np
import pytest

from dsdvc.contagion.models import (complex_contagion_prob, fit_threshold_mixture, independent_cascade,
                                    long_ties_experiment, simple_contagion_prob)
from dsdvc.networks.communities import community_dynamics, detect_communities
from dsdvc.networks.metrics import bridge_centrality, coleman_homophily, structure_report
from dsdvc.networks.multiplex import synthetic_multiplex
from dsdvc.networks.schelling import SchellingParams, SchellingRewirer


@pytest.fixture(scope="module")
def mx():
    return synthetic_multiplex(500, 6, seed=4)


def test_layers_are_structurally_different(mx):
    s = {k: structure_report(g, mx.community) for k, g in mx.layers.items()}
    assert mx.layers["instagram"].is_directed() and not mx.layers["whatsapp"].is_directed()
    assert s["whatsapp"]["edge_homophily"] > s["instagram"]["edge_homophily"]
    deg = dict(mx.layers["instagram"].in_degree())
    assert max(deg.values()) > 5 * np.mean(list(deg.values()))  # heavy-tailed creator reach
    assert mx.provenance.observed_or_estimated.value == "synthetic"


def test_homophily_scale_increases_segregation():
    lo = synthetic_multiplex(400, 6, homophily_scale=0.6, seed=1)
    hi = synthetic_multiplex(400, 6, homophily_scale=1.3, seed=1)
    assert coleman_homophily(hi.layers["facebook"], hi.community) > coleman_homophily(lo.layers["facebook"], lo.community)


@pytest.mark.parametrize("method", ["louvain", "leiden", "label_propagation", "spectral", "hdbscan"])
def test_community_detection_methods(mx, method):
    r = detect_communities(mx.layers["facebook"], method)
    assert r.n_communities >= 2 and len(r.labels) == mx.n and len(r.bridges) > 0


def test_community_dynamics_flags_growth():
    prev = np.array([0] * 10 + [1] * 10)
    curr = np.array([0] * 16 + [1] * 4)
    trends = {d["community"]: d["trend"] for d in community_dynamics(prev, curr)}
    assert trends[0] == "growing" and trends[1] == "declining"


def test_bridge_centrality_peaks_at_bridge():
    import networkx as nx
    g = nx.barbell_graph(6, 1)
    assert int(np.argmax(bridge_centrality(g, k_sample=None))) == 6


def test_schelling_rewiring_increases_neighbour_similarity(mx):
    g = mx.layers["facebook"].copy()
    rw = SchellingRewirer(g, mx.taste, SchellingParams(threshold_mean=0.8, rewiring_rate=0.3), seed=0)
    h = rw.run(20, mx.community, 20)
    assert h[-1]["mean_fit"] > h[0]["mean_fit"]
    assert h[-1]["edge_homophily"] >= h[0]["edge_homophily"]
    assert g.number_of_edges() == mx.layers["facebook"].number_of_edges()  # rewiring preserves tie count


def test_simple_vs_complex_contagion():
    assert simple_contagion_prob(np.array([1.0]), 0.3) == pytest.approx(0.3)
    p1, p3 = complex_contagion_prob(np.array([1.0, 3.0]), np.array([2.0, 2.0]))
    assert p1 < 0.1 < 0.9 < p3


def test_centola_macy_long_ties_hurt_complex_contagion():
    r = {x["rewire_p"]: x for x in long_ties_experiment(n=300, rewire_probs=(0.0, 0.6), reps=3)}
    assert r[0.0]["complex_reach"] > r[0.6]["complex_reach"]
    assert r[0.6]["simple_reach"] >= r[0.0]["complex_reach"]


def test_threshold_mixture_detects_complex_adopters():
    rng = np.random.default_rng(0)
    k = np.concatenate([rng.poisson(0.4, 600), 2 + rng.poisson(1.0, 400)])
    r = fit_threshold_mixture(k)
    assert 0.25 < r["pi_complex"] < 0.6 and r["p_value_vs_simple"] < 0.01


def test_independent_cascade(mx):
    r = independent_cascade(mx.layers["whatsapp"], [0, 1, 2], 0.2, seed=1)
    assert r.size >= 3 and r.size_by_round == sorted(r.size_by_round)
