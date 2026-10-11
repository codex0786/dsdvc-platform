import numpy as np
import pandas as pd
import pytest

from dsdvc.calibration.engine import CalibrationEngine, abm_summary_stats, make_abm_simulator
from dsdvc.diffusion.abm import SimConfig, run_simulation
from dsdvc.diffusion.bass import bass_cdf, bass_peak_time, fit_bass_mle, fit_bass_nls, generalized_bass_cdf, simulate_bass_adoptions


def test_bass_closed_form_properties():
    t = np.linspace(0, 200, 400)
    F = bass_cdf(t, 0.01, 0.4)
    assert F[0] == pytest.approx(0) and F[-1] == pytest.approx(1, abs=1e-6) and np.all(np.diff(F) >= 0)
    x = np.ones(400)
    np.testing.assert_allclose(generalized_bass_cdf(t, 0.01, 0.4, x)[-1], 1, atol=1e-3)
    assert bass_peak_time(0.01, 0.4) == pytest.approx(np.log(40) / 0.41)


@pytest.mark.parametrize("fit", [fit_bass_nls, fit_bass_mle])
def test_bass_fit_recovers_parameters(fit):
    y = simulate_bass_adoptions(0.01, 0.35, 20000, 30, seed=2)
    f = fit(y)
    assert f.p == pytest.approx(0.01, rel=0.1)
    assert f.q == pytest.approx(0.35, rel=0.05)
    assert f.m == pytest.approx(20000, rel=0.05)


def test_bayesian_calibration_interval_covers_truth():
    y = simulate_bass_adoptions(0.008, 0.3, 15000, 30, seed=5)
    r = CalibrationEngine(seed=1).bass_bayesian(y, n_iter=4000, burn=1500)
    for k, true in (("p", 0.008), ("q", 0.3)):
        lo, hi = r.intervals[k]
        assert lo * 0.9 <= true <= hi * 1.1
    assert 0.1 < r.diagnostics["acceptance"] < 0.7


def test_platform_rate_posteriors():
    obs = pd.DataFrame({"platform": ["instagram", "whatsapp"], "exposures": [1000, 100], "engagements": [80, 40],
                        "shares": [20, 30], "adoptions": [5, 8], "churns": [1, 1]})
    t = CalibrationEngine().platform_rates(obs)
    wa = t[(t.platform == "whatsapp") & (t.metric == "share_rate")].iloc[0]
    assert wa.ci_low < 0.3 < wa.ci_high and wa.status == "CALIBRATED"


@pytest.mark.slow
def test_abc_identifies_purchase_intercept():
    base = SimConfig(n_agents=300, steps=30, rewire_every=0)
    truth_cfg = SimConfig(n_agents=300, steps=30, rewire_every=0, category_purchase_intercept=-3.0, seed=77)
    obs = abm_summary_stats(run_simulation(truth_cfg))
    r = CalibrationEngine(seed=3).abc(make_abm_simulator(base), obs, {"category_purchase_intercept": (-6.0, -1.5)},
                                      n_sims=50, accept_frac=0.2)
    assert -4.2 < r.parameters["category_purchase_intercept"] < -2.0
    assert r.diagnostics["posterior_shrinkage"]["category_purchase_intercept"] > 0.2
