import numpy as np
import pandas as pd
import pytest

from dsdvc.attribution.models import (all_attributions, incrementality_test, markov_attribution, required_sample_size,
                                      shapley_attribution)
from dsdvc.churn.ltv import discounted_ltv, ltv_by, retention_ltv
from dsdvc.churn.model import ChurnModel, churn_breakdown
from dsdvc.competition.mnl import MNLParams, compare_network_specs, fit_mnl, simulate_choices
from dsdvc.conversion.journey import TRANSITIONS, JourneyModel, funnel_table, is_allowed
from dsdvc.conversion.model import ConversionModel
from dsdvc.data.synthetic import synthetic_customer_panel, synthetic_journeys


@pytest.fixture(scope="module")
def panel():
    return synthetic_customer_panel(8000, seed=1)


def test_conversion_recovers_platform_specific_trust(panel):
    c = ConversionModel().fit(panel).coefficients_by_platform()
    t = c[c.feature == "trust"].set_index("platform").beta
    assert t["whatsapp"] > t["facebook"] > t["instagram"]


def test_churn_signs_match_theory(panel):
    t = ChurnModel().fit(panel[panel.converted]).table()
    assert t.loc[["dissatisfaction", "competition", "price_pressure", "engagement", "community_fit"], "sign_agrees"].all()
    b = churn_breakdown(panel[panel.converted], "platform")
    assert (b.ci_low <= b.churn_rate).all() and (b.churn_rate <= b.ci_high).all()


def test_ltv_formulas(panel):
    assert discounted_ltv(np.array([100, 100]), 20, r=0.1) == pytest.approx(80 + 80 / 1.1)
    assert retention_ltv(10, 1.0, r=0.0, horizon=12) == pytest.approx(120)
    g = ltv_by(panel[panel.converted], "platform", cac={"instagram": 100, "facebook": 100, "whatsapp": 100})
    assert {"ltv_mean", "ltv_to_cac", "roi"} <= set(g.columns)


def test_journey_state_machine_and_fit():
    assert is_allowed("ADOPTED", "DISSATISFIED") and not is_allowed("UNAWARE", "ADOPTED")
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(400):
        x = rng.random()
        rows.append({"stage_from": "AWARE", "stage_to": "INTERESTED" if rng.random() < 0.2 + 0.6 * x else "AWARE", "x": x})
    jm = JourneyModel(["x"])
    tab = jm.fit(pd.DataFrame(rows))
    est = tab[(tab["from"] == "AWARE")].iloc[0]
    assert est.status == "estimated" and est.b_x > 0
    assert (tab[tab["from"] == "CHURNED"].status == "DATA_UNAVAILABLE").all()
    with pytest.raises(ValueError):
        jm.fit(pd.DataFrame([{"stage_from": "UNAWARE", "stage_to": "ADOPTED", "x": 0.1}] * 40))
    f = funnel_table({"UNAWARE": 50, "AWARE": 30, "ADOPTED": 20})
    assert f.reached.is_monotonic_decreasing


def test_mnl_recovers_network_effect():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (4000, 4, 6))
    N = np.array([50, 10, 30, 5.0]) * rng.uniform(0.5, 1.5, (4000, 4))
    ch = simulate_choices(X, N, MNLParams(network_spec="power", alpha=0.4, gamma=0.6), 1)
    r = fit_mnl(X, N, ch, "power")
    assert r["alpha"] == pytest.approx(0.4, abs=0.1) and r["gamma"] == pytest.approx(0.6, abs=0.12)
    assert compare_network_specs(X, N, ch).iloc[-1].spec == "none"


def test_attribution_models_conserve_conversions():
    j = synthetic_journeys(3000, seed=2)
    tab = all_attributions(j)
    total = j.converted.sum()
    for m in tab.columns:
        assert tab[m].sum() == pytest.approx(total, rel=1e-6 if m != "markov" else 1e-3), m
    sh = shapley_attribution(j)
    assert sh["wa_referral"] / (j.path.map(lambda p: "wa_referral" in p).sum()) > sh["email"] / (j.path.map(lambda p: "email" in p).sum())
    assert set(markov_attribution(j)) == set(sh)


def test_incrementality_distinct_from_attribution():
    r = incrementality_test(520, 20000, 430, 20000)
    assert r["incremental_conversions"] == pytest.approx(90) and r["ci_low"] < 90 < r["ci_high"]
    assert required_sample_size(0.02, 0.1) > 10000
