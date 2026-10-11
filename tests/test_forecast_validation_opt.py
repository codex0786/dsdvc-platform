import numpy as np
import pandas as pd
import pytest

from dsdvc.data.quality import DataQualityEngine
from dsdvc.data.synthetic import synthetic_content, synthetic_daily_metrics
from dsdvc.diffusion.abm import SimConfig, run_simulation
from dsdvc.forecasting.early_warning import platform_dependency, scan
from dsdvc.forecasting.forecast import HORIZONS, Forecaster
from dsdvc.forecasting.regime import detect_regime, pelt
from dsdvc.optimization.budget import ResponseCurve, curves_from_simulator, optimize_allocation
from dsdvc.validation.benchmarks import ablation_report, adoption_benchmark
from dsdvc.validation.metrics import calibration_curve, classification_report, cumulative_gain, regression_report


def test_forecast_horizons_and_intervals():
    d = synthetic_daily_metrics(150, breakpoint=None)
    f = Forecaster().forecast_frame(d, ["adoptions", "revenue"])
    assert set(f.horizon_days) == set(HORIZONS)
    assert ((f.lower <= f.forecast_total) & (f.forecast_total <= f.upper)).all()
    w = f[f.metric == "revenue"].set_index("horizon_days")
    assert (w.upper - w.lower).is_monotonic_increasing


def test_pelt_finds_break():
    rng = np.random.default_rng(0)
    y = np.r_[rng.normal(0, 1, 60), rng.normal(4, 1, 60)]
    cps = pelt(y)
    assert any(abs(c - 60) <= 3 for c in cps)


def test_regime_and_warnings():
    d = synthetic_daily_metrics(180, breakpoint=120)
    assert detect_regime(d.iloc[:20])["regime"] in ("pre-diffusion", "early diffusion", "viral acceleration")
    assert detect_regime(d)["regime"] in ("saturation", "decline", "competitive disruption")
    alerts = scan(d, competitor_events=[120], conversions_by_platform={"instagram": 80, "facebook": 15, "whatsapp": 5})
    kinds = {a["kind"] for a in alerts}
    assert {"competitor_attack", "platform_dependency"} <= kinds
    assert platform_dependency({"a": 1, "b": 1})["flag"] is False


def test_metrics():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 1])
    p = np.array([.1, .2, .8, .7, .3, .9, .2, .1, .4, .6])
    r = classification_report(y, p, top_frac=0.3)
    assert r["auc"] == 1.0 and r["precision_at_top"] == 1.0
    assert cumulative_gain(y, p)[-1][1] == 1.0
    assert calibration_curve(y, p, 3)
    assert regression_report([1, 2, 3], [1, 2, 4])["mae"] == pytest.approx(1 / 3)


@pytest.mark.slow
def test_adoption_benchmark_bass_wins_on_bass_data():
    d = synthetic_daily_metrics(120, breakpoint=None)
    t = adoption_benchmark(d.adoptions.to_numpy(float))
    assert t.iloc[0].model == "bass"


@pytest.mark.slow
def test_ablation_report_structure():
    truth = run_simulation(SimConfig(n_agents=300, steps=40, seed=9, rewire_every=0)).timeseries.new_adoptions.to_numpy()
    r = ablation_report(truth, SimConfig(n_agents=300, rewire_every=0), grid=np.linspace(-5, -3, 3), seeds=(0,))
    assert list(r.variant) == ["bass_only", "+networks", "+stepps", "+recommendation", "+cross_platform",
                               "+competition", "+churn"]
    assert "earns_complexity" in r


def test_allocation_respects_budget_and_equalises_marginal_roi():
    curves = {"a": ResponseCurve("a", 1000, 200), "b": ResponseCurve("b", 600, 100), "c": ResponseCurve("c", 10, 1000)}
    r = optimize_allocation(curves, budget=500, value_per_conversion=1.0, n_boot=20)
    a = r["allocation"].set_index("channel")
    assert a.spend.sum() <= 500 + 1e-6 and a.loc["c", "spend"] < 1
    assert r["status"] == "RECOMMENDED"


@pytest.mark.slow
def test_response_curves_from_simulator_are_increasing_for_creators():
    # low base budget: at 300 agents the mid-tier creator pool is exhausted near the default 200k
    curves, tab = curves_from_simulator(SimConfig(n_agents=300, steps=30, rewire_every=0, seeding_budget=10e3),
                                        ["creators"], [20e3, 80e3], seeds=(0, 1))
    t = tab.set_index("spend").incremental_mean
    assert t.loc[80e3] > t.loc[20e3]


def test_data_quality_engine():
    df = synthetic_content(400, seed=3)
    good = DataQualityEngine().evaluate(df).score
    bad = df.copy()
    bad.loc[:100, "views"] = np.nan
    bad = pd.concat([bad, bad.head(50)])
    bad.loc[bad.index[:30], "likes"] = bad["views"].fillna(0).head(30) * 2 + 10
    r = DataQualityEngine().evaluate(bad)
    assert r.score < good
    assert r.checks["duplicates"]["value"] > 0 and r.checks["missingness"]["value"] > 0 and r.checks["bot_like"]["value"] > 0
