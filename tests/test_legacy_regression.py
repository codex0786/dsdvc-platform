"""Regression tests: the preserved v2/v3 models stay byte-identical and keep their mechanics."""
import hashlib
import json
from pathlib import Path

import networkx
import numpy as np
import pandas as pd
import pytest

from legacy.dsdvc_mp_v2 import DSDVCModel, Params
from legacy.dsdvc_mp_v3.model import PLATFORMS, DSDVCMP, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests/fixtures/legacy_golden.json").read_text())


def test_legacy_files_unmodified():
    for line in (ROOT / "legacy/CHECKSUMS.sha256").read_text().splitlines():
        digest, rel = line.split()
        assert hashlib.sha256((ROOT / "legacy" / rel).read_bytes()).hexdigest() == digest, rel


def _same_stack():
    v = GOLDEN["versions"]
    return v["numpy"] == np.__version__ and v["networkx"] == networkx.__version__


@pytest.mark.skipif(not _same_stack(), reason="golden values were recorded on a different numpy/networkx version")
def test_v3_golden_output():
    df = DSDVCMP(ModelConfig(**GOLDEN["v3"]["config"])).run()
    for col, vals in GOLDEN["v3"]["series"].items():
        np.testing.assert_allclose(df[col].to_numpy(), vals, rtol=1e-9, atol=1e-9, err_msg=col)


@pytest.mark.skipif(not _same_stack(), reason="golden values were recorded on a different numpy/networkx version")
def test_v2_golden_output():
    c = GOLDEN["v2"]["config"]
    df = DSDVCModel(Params(n_agents=c["n_agents"], periods=c["periods"], seed=c["seed"])).run()
    for col, vals in GOLDEN["v2"]["series"].items():
        np.testing.assert_allclose(df[col].to_numpy(dtype=float), vals, rtol=1e-9, atol=1e-9, err_msg=col)


def test_v3_mechanics_bounds_and_platforms():
    m = DSDVCMP(ModelConfig(n_agents=150, steps=8, seed=3))
    df = m.run()
    assert set(m.networks) == set(PLATFORMS)
    assert df.adoption_rate.between(0, 1).all() and df.churn_rate.between(0, 1).all()
    assert (df[["instagram_shares", "facebook_shares", "whatsapp_shares"]].diff().dropna() >= 0).all().all()


def test_v3_bass_innovation_raises_adoption():
    lo = np.mean([DSDVCMP(ModelConfig(n_agents=150, steps=6, seed=s, p_innovation=0.001, initial_adoption=0.01)).run()
                  .adoption_rate.iloc[-1] for s in range(3)])
    hi = np.mean([DSDVCMP(ModelConfig(n_agents=150, steps=6, seed=s, p_innovation=0.5, initial_adoption=0.01)).run()
                  .adoption_rate.iloc[-1] for s in range(3)])
    assert hi >= lo


def test_v3_stepps_is_geometric_mean():
    m = DSDVCMP(ModelConfig(n_agents=50, steps=1))
    c = m.cfg
    vals = [c.content_social_currency, c.content_triggers, c.content_emotion, c.content_public,
            c.content_practical, c.content_story]
    assert m._stepps_score() == pytest.approx(float(np.prod(vals) ** (1 / 6)))


def test_v2_runs_and_rewires():
    df = DSDVCModel(Params(n_agents=120, periods=10, seed=1)).run()
    assert len(df) == 10 and df.homophily.between(0, 1).all()
