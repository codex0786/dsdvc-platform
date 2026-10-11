import numpy as np
import pandas as pd
import pytest
from scipy import stats

from dsdvc.content.cascade import CascadeEngine
from dsdvc.content.copilot import ContentCopilot
from dsdvc.content.stepps import DIMENSIONS, STEPPSAnalyzer
from dsdvc.content.stepps_models import compare_stepps_forms
from dsdvc.content.velocity import early_features, velocity_profile
from dsdvc.content.virality import ViralityPredictor, label_viral
from dsdvc.data.synthetic import synthetic_content


@pytest.fixture(scope="module")
def corpus():
    return synthetic_content(2500, seed=11)


def test_stepps_output_contract():
    p = STEPPSAnalyzer().analyze("OMG my journey: 5 tips under ₹500 for your morning routine. Tag a friend! "
                                 "exclusive first look", media_type="reel")
    d = p.to_dict()
    assert set(d["scores"]) == set(DIMENSIONS)
    for dim in DIMENSIONS:
        det = d["detail"][dim]
        assert 0 <= det["score"] <= 1 and 0 < det["confidence"] <= 1
        assert {"evidence", "contributions", "explanation"} <= set(det)
    assert "do not guarantee" in d["disclaimer"]


def test_lexicon_coder_tracks_planted_stepps(corpus):
    an = STEPPSAnalyzer()
    S = np.array([an.analyze(t).vector() for t in corpus.text[:800]])
    for k, d in enumerate(DIMENSIONS):
        assert stats.spearmanr(S[:, k], corpus[f"true_{d}"][:800]).statistic > 0.3, d


def test_supervised_stepps_coder(corpus):
    an = STEPPSAnalyzer()
    r2 = an.fit(corpus.text[:1500].tolist(), {d: corpus[f"true_{d}"][:1500].to_numpy() for d in DIMENSIONS})
    assert min(r2.values()) > 0.2
    assert an.analyze(corpus.text.iloc[2000]).coder == "supervised"


def test_stepps_form_comparison_runs(corpus):
    X = corpus[[f"true_{d}" for d in DIMENSIONS]].values
    r = compare_stepps_forms(X, (corpus.shares / corpus.views).values)
    assert set(r["table"].form) == {"additive", "multiplicative", "interaction"}
    assert r["best_form"] in ("additive", "interaction")  # truth has an emotion x public interaction


def test_viral_label_is_relative_top_pct(corpus):
    y = label_viral(corpus, top_pct=0.05)
    assert 0.03 < y.mean() < 0.08


def test_velocity():
    v = velocity_profile(np.array([1, 3, 6, 12]), np.array([10, 40, 100, 400]))
    assert v["trend"] == "accelerating"
    v = velocity_profile(np.array([1, 3, 6, 12]), np.array([100, 180, 220, 240]))
    assert v["trend"] == "decelerating"


@pytest.mark.slow
def test_virality_model_beats_baselines_and_reports_uncertainty(corpus):
    vp = ViralityPredictor(seed=0)
    rep = vp.fit(corpus)
    b = rep["benchmarks"].set_index("model")
    assert b.loc["tree", "auc"] > 0.6 and b.loc["temporal", "auc"] > b.loc["naive", "auc"]
    assert b.loc["ensemble_calibrated", "log_loss"] < b.loc["naive", "log_loss"]
    out = vp.predict(corpus.head(3))
    v = out[0]["viral"]
    assert v["lower"] <= v["prediction"] <= v["upper"]
    assert v["model_version"] and v["data_version"] and v["explanation"]["positive_drivers"] is not None
    assert "will go viral" not in out[0]["summary"].lower()
    assert out[0]["expected_cascade_depth"]["status"] == "DATA_UNAVAILABLE"


def test_cascade_engine():
    df = pd.DataFrame({"share_id": ["a", "b", "c", "d", "e"], "parent_share_id": [None, "a", "a", "b", None],
                       "sharer_id": [1, 2, 3, 4, 5], "platform": ["instagram", "instagram", "whatsapp", "whatsapp", "facebook"],
                       "timestamp": pd.date_range("2026-01-01", periods=5, freq="h"), "community": [0, 0, 1, 1, 2]})
    s = CascadeEngine(df).summary()
    assert s["depth"] == 2 and s["breadth"] == 2 and s["roots"] == 2
    assert s["reproduction_ratio"] == pytest.approx(1 / 2)
    assert 0 < s["cross_platform_spread"] <= 1


def test_copilot_language_and_risks():
    rep = ContentCopilot(brand_constraints={"banned_terms": ["cheap"]}).analyze(
        caption="Guaranteed miracle cure, 100% permanent results")
    assert any("Claim risk" in r for r in rep.risks)
    assert rep.recommendations["platform"] and rep.variants
    assert "not guarantees" in rep.language_note
