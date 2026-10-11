import numpy as np
import pytest

from dsdvc.core.literature import LITERATURE, cites, references_for
from dsdvc.core.prediction import Explanation, Prediction
from dsdvc.core.provenance import (DATA_UNAVAILABLE, EpistemicStatus, Provenance, SourceType, assert_not_mixed,
                                   is_unavailable, unavailable)
from dsdvc.core.reproducibility import RunRecord, stable_hash


def test_provenance_fields_required():
    p = Provenance(source="authorized_api", source_type=SourceType.AUTHORIZED_API, collection_method="graph_api",
                   authorization_scope="own account", confidence=0.99, observed_or_estimated=EpistemicStatus.OBSERVED)
    d = p.model_dump()
    for k in ("source", "source_type", "timestamp", "collection_method", "authorization_scope", "confidence",
              "observed_or_estimated"):
        assert k in d


def test_data_unavailable_is_falsy_and_labelled():
    u = unavailable("followers", "no token")
    assert not u and is_unavailable(u) and u.status == DATA_UNAVAILABLE


def test_never_mix_observed_and_synthetic():
    obs = Provenance(source="x", source_type=SourceType.USER_PROVIDED, collection_method="upload", confidence=1,
                     observed_or_estimated=EpistemicStatus.OBSERVED)
    syn = Provenance.synthetic("gen", 1)
    with pytest.raises(ValueError):
        assert_not_mixed([obs, syn])
    assert_not_mixed([obs, syn], allow_synthetic=True)


def test_prediction_envelope_has_versions_and_uncertainty():
    p = Prediction(target="p_viral_top_pct", prediction=0.72, lower=0.58, upper=0.82, confidence="medium",
                   data_quality=0.7, model_agreement=0.8,
                   explanation=Explanation.from_contributions({"emotion": 0.4, "practical": -0.2}, "test"))
    d = p.model_dump()
    assert d["model_version"] and d["data_version"] and d["timestamp"]
    assert "72%" in p.summary() and "58%" in p.summary()
    assert p.explanation.positive_drivers[0].feature == "emotion"


def test_literature_keys_validated():
    with pytest.raises(KeyError):
        cites("not_a_paper")
    @cites("bass1969")
    def f():
        pass
    assert references_for(f)[0].startswith("Bass")
    assert all("does_not_establish" in v for v in LITERATURE.values())


def test_run_record_deterministic_id():
    a = RunRecord.create("abm", {"x": 1}, 1, 2)
    b = RunRecord.create("abm", {"x": 1}, 1, 2)
    assert a.run_id == b.run_id and stable_hash({"a": 1}) == stable_hash({"a": 1})
    assert {"numpy", "python", "dsdvc"} <= set(a.software_version)
