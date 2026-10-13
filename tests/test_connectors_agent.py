import json

import pandas as pd
import pytest

from dsdvc.config.settings import Settings
from dsdvc.core.provenance import DataUnavailable
from dsdvc.platforms.base import FORBIDDEN_ACTIONS
from dsdvc.platforms.meta import (InstagramGraphConnector, MetaContentLibraryConnector, WhatsAppBusinessConnector,
                                  survey_forwarding_rates)
from dsdvc.research_agent import tools as T


class FakeResp:
    def __init__(self, status, payload, headers=None):
        self.status_code, self._p, self.headers, self.text = status, payload, headers or {}, json.dumps(payload)

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def request(self, method, url, params=None, timeout=None, **kw):
        self.calls.append((url, params))
        return self.responses.pop(0)


def test_connectors_without_credentials_return_data_unavailable():
    s = Settings()
    assert isinstance(InstagramGraphConnector(s).own_media(), DataUnavailable)
    assert isinstance(InstagramGraphConnector(s).business_discovery("brand"), DataUnavailable)
    assert isinstance(MetaContentLibraryConnector().search("x"), DataUnavailable)
    import datetime as dt
    assert isinstance(WhatsAppBusinessConnector(s).message_analytics(dt.datetime(2026, 1, 1), dt.datetime(2026, 2, 1)), DataUnavailable)
    assert "bypass rate limits" in FORBIDDEN_ACTIONS


def test_business_discovery_with_mocked_official_api():
    s = Settings(meta_access_token="tok", instagram_business_account_id="123")
    sess = FakeSession([FakeResp(200, {"business_discovery": {"username": "brand", "followers_count": 5000,
                                                              "follows_count": 100, "media": {"data": []}}})])
    r = InstagramGraphConnector(s, session=sess).business_discovery("brand")
    assert r["followers_count"] == 5000 and r["_provenance"]["observed_or_estimated"] == "observed"
    assert "business_discovery.username(brand)" in sess.calls[0][1]["fields"]


def test_unauthorised_is_not_bypassed():
    s = Settings(meta_access_token="tok", instagram_business_account_id="123")
    sess = FakeSession([FakeResp(403, {"error": "no"})])
    r = InstagramGraphConnector(s, session=sess).own_media()
    assert isinstance(r, DataUnavailable) and "not authorised" in r.reason and len(sess.calls) == 1


def test_rate_limit_backoff_is_compliant(monkeypatch):
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    s = Settings(meta_access_token="tok", instagram_business_account_id="123")
    sess = FakeSession([FakeResp(429, {}), FakeResp(200, {"data": [{"id": "1", "caption": "hi", "like_count": 3,
                                                                     "comments_count": 1, "media_type": "IMAGE"}]})])
    df = InstagramGraphConnector(s, session=sess).own_media()
    assert isinstance(df, pd.DataFrame) and slept and slept[0] >= 60
    assert (df.observed_or_estimated == "observed").all()


def test_survey_forwarding_rates():
    sv = pd.DataFrame({"content_type": ["tips"] * 50 + ["promo"] * 50, "would_forward": [1] * 30 + [0] * 20 + [1] * 10 + [0] * 40})
    r = survey_forwarding_rates(sv)
    assert r.loc["tips", "forward_rate"] > r.loc["promo", "forward_rate"] and (r.status == "estimated").all()


def test_tools_never_fabricate():
    assert T.search_web("x").status == "DATA_UNAVAILABLE"
    assert T.discover_influencers("skincare", ["glow"]).status == "DATA_UNAVAILABLE"
    assert T.run_calibration(None).status == "DATA_UNAVAILABLE"
    syn = T.discover_influencers("skincare", ["glow"], allow_synthetic=True)
    assert syn.status == "SYNTHETIC" and "SYNTHETIC" in syn.provenance["note"]


@pytest.mark.slow
def test_research_agent_authorized_only_mode():
    from dsdvc.research_agent.agent import MarketingResearchAgent, parse_brief
    from dsdvc.research_agent.report import SECTION_ORDER, render_markdown
    b = parse_brief("I want to launch a premium skincare brand in India targeting women 18-30.")
    assert b["category"] == "skincare" and b["country"] == "IN" and b["gender"] == "women" and b["age_max"] == 30
    mem, S = MarketingResearchAgent(allow_synthetic=False, n_agents=300, steps=30, reps=2).run(b["raw"])
    status = {p["name"]: p["status"] for p in mem.plan}
    assert status["Define market"] == "DATA_UNAVAILABLE" and status["Discover creators"] == "DATA_UNAVAILABLE"
    assert "SYNTHETIC" not in status.values()
    assert all(c.trace.experiment for c in mem.recommendations)
    md = render_markdown(mem, S)
    for sec in SECTION_ORDER:
        assert f"## {sec}" in md, sec
    assert "will go viral" not in md.lower()
