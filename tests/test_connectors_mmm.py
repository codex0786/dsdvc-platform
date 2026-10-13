"""Meta Ads / GA4 / Shopify connectors against SYNTHETIC API simulators shaped like the real APIs."""
import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

from dsdvc.connectors.fakes import FakeGA4Session, FakeMetaSession, FakeResponse, FakeShopifySession, SyntheticBusiness
from dsdvc.connectors.ga4 import GA4Connector
from dsdvc.connectors.meta_ads import ChannelRules, MetaAdsConnector
from dsdvc.connectors.shopify import ShopifyConnector
from dsdvc.connectors.weekly import build_weekly, complete_weeks
from dsdvc.core.provenance import DataUnavailable

START, END = date(2023, 1, 2), date(2025, 12, 28)


@pytest.fixture(scope="module")
def biz():
    return SyntheticBusiness(0)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)


def test_complete_weeks():
    w = complete_weeks(date(2026, 1, 1), date(2026, 1, 25))  # Thu .. Sun
    assert list(w.strftime("%Y-%m-%d")) == ["2026-01-05", "2026-01-12", "2026-01-19"]
    assert len(complete_weeks(date(2026, 1, 6), date(2026, 1, 10))) == 0


def test_unconfigured_connectors_return_data_unavailable():
    assert isinstance(MetaAdsConnector(None, None).daily_spend(START, END), DataUnavailable)
    assert isinstance(GA4Connector(None).daily_outcomes(START, END), DataUnavailable)
    assert isinstance(ShopifyConnector(None, None).daily_sales(START, END), DataUnavailable)


def test_meta_mmm_export_async_flow_and_rules(biz):
    sess = FakeMetaSession(biz, polls_before_done=2)
    m = MetaAdsConnector("tok", "act_123", session=sess, poll_seconds=0)
    df = m.daily_spend(START, END, "mmm", ChannelRules(campaign_contains=[("influencer", "influencers")]))
    assert set(df.channel) == {"meta_ads", "influencers"}
    assert df.groupby("channel").spend.sum()["influencers"] == pytest.approx(biz.weekly.influencers.sum(), rel=1e-4)
    posts = [u for meth, u in sess.calls if meth == "POST"]
    assert len(posts) == 37 and all(u.endswith("/act_123/insights") for u in posts)  # 30-day chunks over 3 years
    assert any("v26.0" in u for _, u in sess.calls)
    assert df.attrs["provenance"]["observed_or_estimated"] == "observed"


def test_meta_insights_pagination_and_platform_split(biz):
    m = MetaAdsConnector("tok", "123", session=FakeMetaSession(biz), poll_seconds=0)
    df = m.daily_spend(date(2024, 1, 1), date(2024, 3, 31), "insights", ChannelRules(split_platforms=True))
    assert {"instagram_ads", "facebook_ads"} <= set(df.channel)
    d = biz.daily[(biz.daily.date >= "2024-01-01") & (biz.daily.date <= "2024-03-31")]
    assert df.spend.sum() == pytest.approx((d.meta_ads + d.influencers).sum(), rel=1e-4)


def test_meta_job_failure_is_reported(biz):
    sess = FakeMetaSession(biz)
    orig = sess.request

    def failing(method, url, **kw):
        r = orig(method, url, **kw)
        if method == "GET" and "/rr" in url and "insights" not in url and "async_report_url" not in str(kw.get("params")):
            return FakeResponse(200, {"async_status": "Job Failed", "error_message": "too much data"})
        return r
    sess.request = failing
    r = MetaAdsConnector("tok", "123", session=sess, poll_seconds=0).mmm_export(date(2024, 1, 1), date(2024, 1, 31))
    assert isinstance(r, DataUnavailable) and "Job Failed" in r.reason


def test_ga4_outcomes_and_ads_cost(biz):
    sess = FakeGA4Session(biz)
    g = GA4Connector("properties/1", access_token="tok", page_size=400, session=sess)
    df = g.daily_outcomes(START, END)
    assert len(df) == 1092 and {"keyEvents", "purchaseRevenue", "transactions"} <= set(df.columns)
    assert len(sess.calls) == 3  # 1092 rows / 400 per page
    body = sess.calls[0][2]
    assert body["dimensions"] == [{"name": "date"}] and "keyEvents" in [m["name"] for m in body["metrics"]]
    cost = g.daily_google_ads_cost(START, END)
    assert cost.advertiserAdCost.sum() == pytest.approx((biz.weekly.google_search + biz.weekly.youtube).sum(), rel=1e-3)
    unlinked = GA4Connector("1", access_token="tok", session=FakeGA4Session(biz, ads_linked=False))
    assert isinstance(unlinked.daily_google_ads_cost(START, END), DataUnavailable)


def test_ga4_auth_failure_not_bypassed(biz):
    class NoAuth(FakeGA4Session):
        def request(self, method, url, headers=None, **kw):
            return FakeResponse(403, {"error": {"message": "User does not have sufficient permissions"}})
    r = GA4Connector("1", access_token="tok", session=NoAuth(biz)).daily_outcomes(START, END)
    assert isinstance(r, DataUnavailable) and "not authorised" in r.reason


@pytest.mark.parametrize("method", ["paginated", "bulk"])
def test_shopify_daily_sales(biz, method):
    s = ShopifyConnector("demo-store", "shpat", session=FakeShopifySession(biz), poll_seconds=0)
    a, b = (date(2024, 1, 1), date(2024, 1, 31)) if method == "paginated" else (START, END)
    df = s.daily_sales(a, b, method=method)
    d = biz.daily[(biz.daily.date >= pd.Timestamp(a)) & (biz.daily.date <= pd.Timestamp(b))]
    assert df.net_sales.sum() == pytest.approx(d.revenue.sum(), rel=0.01)
    assert "Excluded 1 test and 1 cancelled" in df.attrs["provenance"]["notes"] and "Asia/Kolkata" in df.attrs["provenance"]["notes"]
    assert s.shop == "demo-store.myshopify.com"


def test_shopify_requests_no_customer_fields(biz):
    from dsdvc.connectors.shopify import ORDER_FIELDS
    assert "customer" not in ORDER_FIELDS.lower() and "email" not in ORDER_FIELDS.lower()


def test_build_weekly_reconstructs_planted_data(biz):
    meta = MetaAdsConnector("t", "1", session=FakeMetaSession(biz), poll_seconds=0).daily_spend(
        START, END, "mmm", ChannelRules(campaign_contains=[("influencer", "influencers")]))
    ga = GA4Connector("1", access_token="t", session=FakeGA4Session(biz))
    shop = ShopifyConnector("x", "t", session=FakeShopifySession(biz), poll_seconds=0).daily_sales(START, END)
    res = build_weekly(START, END, meta=meta, ga4=ga.daily_outcomes(START, END), ga4_ads_cost=ga.daily_google_ads_cost(START, END),
                       shopify=shop, extra_spend={"ooh": biz.weekly[["date", "ooh"]].rename(columns={"ooh": "spend"})})
    d, w = res.data, biz.weekly
    assert len(d) == 156
    np.testing.assert_allclose(d.revenue, w.revenue, rtol=0.01)
    np.testing.assert_allclose(d.spend_meta_ads, w.meta_ads, atol=1.0)
    np.testing.assert_allclose(d.spend_influencers, w.influencers, atol=1.0)
    np.testing.assert_allclose(d.spend_google_ads, w.google_search + w.youtube, rtol=1e-3)
    np.testing.assert_allclose(d.spend_ooh, w.ooh)
    assert res.column_sources["revenue"] == "shopify:net_sales" and res.column_sources["spend_ooh"] == "user_upload:spend"
    assert 0.75 < res.reconciliation.ga4_to_shopify.median() < 0.9
    assert "official APIs" in res.attestation_text()


def test_tracking_gap_warning(biz):
    ga = GA4Connector("1", access_token="t", session=FakeGA4Session(biz, tracking_share=0.5))
    shop = ShopifyConnector("x", "t", session=FakeShopifySession(biz), poll_seconds=0)
    a, b = date(2024, 1, 1), date(2024, 3, 31)
    res = build_weekly(a, b, ga4=ga.daily_outcomes(a, b), shopify=shop.daily_sales(a, b))
    assert any("tracking gap" in w for w in res.warnings)


@pytest.mark.slow
def test_pulled_data_feeds_mmm_end_to_end(biz):
    from dsdvc.connectors.credentials import connectors
    from dsdvc.connectors.pull import pull_weekly
    from dsdvc.mmm import analysis as A
    from dsdvc.mmm.model import MarketingMixModel, MMMSpec
    from dsdvc.onboarding.templates import MARKETING_WEEKLY
    from dsdvc.onboarding.validate import validate
    res, errors = pull_weekly(connectors(demo=True), START, END, ["meta", "ga4", "shopify"],
                              rules=ChannelRules(campaign_contains=[("influencer", "influencers")]))
    assert not errors and validate(res.data, MARKETING_WEEKLY).status in ("READY", "READY_WITH_WARNINGS")
    chans = [c for c in res.data.columns if c.startswith("spend_")]
    fit = MarketingMixModel(MMMSpec(target_col="revenue", channels=chans, controls=["promo"]), n_restarts=3).fit(res.data)
    s = A.channel_summary(fit).set_index("channel")
    assert s.loc["spend_influencers", "roi"] > s.loc["spend_meta_ads", "roi"]  # planted: influencers 2.5 vs meta 1.6


def test_api_pull_demo_and_save():
    from fastapi.testclient import TestClient
    from dsdvc.api.main import app
    H = {"X-API-Key": os.environ["DSDVC_BOOTSTRAP_ADMIN_KEY"]}
    with TestClient(app) as c:
        st = c.get("/connectors/status", headers=H).json()
        assert st["meta_connector"] is False and st["shopify_connector"] is False
        r = c.post("/connectors/pull", headers=H, json={"start": "2024-01-01", "end": "2024-12-29", "demo": True,
                                                        "campaign_rules": [["influencer", "influencers"]], "save": True}).json()
        assert r["data_status"] == "SYNTHETIC" and r["rows"] == 52 and r["dataset_id"]
        assert r["columns"]["revenue"] == "shopify:net_sales"
        r2 = c.post("/connectors/pull", headers=H, json={"start": "2024-01-01", "end": "2024-06-30"}).json()
        assert r2["status"] == "DATA_UNAVAILABLE" and set(r2["errors"]) == {"meta", "ga4", "shopify"}


def test_shopify_client_credentials_token_exchange(biz, monkeypatch):
    sess = FakeShopifySession(biz, throttle_first=False)
    s = ShopifyConnector("demo-store", None, client_id="cid", client_secret="secret", session=sess, poll_seconds=0)
    assert s.ready
    df = s.daily_sales(date(2024, 1, 1), date(2024, 1, 14), method="paginated")
    assert len(df) == 14 and s.token == "shpat_cc_1" and sess.issued == 1  # one exchange, token reused
    import time as _t
    monkeypatch.setattr(_t, "time", lambda: 10**12)  # far future: token expired -> refreshed
    s.shop_info()
    assert sess.issued == 2


def test_ga4_service_account_json_secret_is_parsed():
    g = GA4Connector("1", service_account_info='{"type": "service_account", "client_email": "x@y.iam.gserviceaccount.com"}')
    assert g.ready and g.sa_info["client_email"].endswith("gserviceaccount.com")
