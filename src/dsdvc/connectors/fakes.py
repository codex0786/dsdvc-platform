"""SYNTHETIC API simulators for Meta Marketing, GA4 Data and Shopify Admin GraphQL.

They implement ``requests.Session.request`` and return responses shaped like the real APIs
(async report jobs, paging cursors, runReport rows, GraphQL connections, bulk JSONL, throttling),
generated from ``dsdvc.mmm.synthetic`` data with planted effects. Used by the tests and by the
dashboard's labelled demo mode, so the real parsing code paths run without any credentials.
"""
from __future__ import annotations

import io
import json
import re
from datetime import date, timedelta
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from dsdvc.mmm.synthetic import synthetic_mmm_data


class FakeResponse:
    def __init__(self, status: int, payload=None, text: str | None = None, headers: dict | None = None):
        self.status_code = status
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)
        self.headers = headers or {}

    def json(self):
        return self._payload if self._payload is not None else json.loads(self.text)


def _daily_from_weekly(weekly: pd.DataFrame, cols: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in weekly.iterrows():
        w = rng.dirichlet(np.ones(7) * 6)
        for k in range(7):
            rows.append({"date": r.date + pd.Timedelta(days=k), **{c: r[c] * w[k] for c in cols}})
    return pd.DataFrame(rows)


class SyntheticBusiness:
    """One synthetic brand behind all three fake APIs (156 weeks from 2023-01-02)."""

    def __init__(self, seed: int = 0, aov: float = 20_000.0):
        weekly, self.truth = synthetic_mmm_data(156, seed)
        self.weekly = weekly
        self.daily = _daily_from_weekly(weekly, ["revenue", "meta_ads", "influencers", "google_search", "youtube", "ooh"], seed)
        self.aov = aov
        self.seed = seed

    def orders(self, start: date, end: date) -> list[dict]:
        rng = np.random.default_rng(self.seed + 7)
        d = self.daily[(self.daily.date >= pd.Timestamp(start)) & (self.daily.date <= pd.Timestamp(end))]
        out, k = [], 0
        for _, r in d.iterrows():
            n = max(1, int(round(r.revenue / self.aov)))
            vals = rng.dirichlet(np.ones(n) * 3) * r.revenue / 0.97  # gross so that net ≈ revenue after ~3% refunds
            for v in vals:
                k += 1
                ts = (r.date + pd.Timedelta(minutes=int(rng.integers(0, 1440)))).tz_localize("Asia/Kolkata").tz_convert("UTC")
                disc = float(v * rng.choice([0, 0, 0, 0.1, 0.2]))
                out.append({"id": f"gid://shopify/Order/{k}", "createdAt": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "cancelledAt": None, "test": False, "sourceName": "web",
                            "totalPriceSet": {"shopMoney": {"amount": f"{v:.2f}", "currencyCode": "INR"}},
                            "subtotalPriceSet": {"shopMoney": {"amount": f"{v:.2f}"}},
                            "totalDiscountsSet": {"shopMoney": {"amount": f"{disc:.2f}"}},
                            "totalRefundedSet": {"shopMoney": {"amount": f"{v * 0.03:.2f}"}}})
        if out:  # one test and one cancelled order that must be excluded
            out.append({**out[0], "id": "gid://shopify/Order/test", "test": True})
            out.append({**out[1], "id": "gid://shopify/Order/cancel", "cancelledAt": out[1]["createdAt"]})
        return out


class FakeMetaSession:
    CAMPAIGNS = {"101": "Always-on Prospecting", "102": "Retargeting", "201": "Influencer Collab Q-series"}

    def __init__(self, biz: SyntheticBusiness, polls_before_done: int = 1):
        self.biz, self.polls = biz, polls_before_done
        self.jobs: dict[str, dict] = {}
        self.calls = []

    def _rows(self, since: date, until: date, mmm: bool) -> list[dict]:
        d = self.biz.daily[(self.biz.daily.date >= pd.Timestamp(since)) & (self.biz.daily.date <= pd.Timestamp(until))]
        rows = []
        for _, r in d.iterrows():
            ds = r.date.strftime("%Y-%m-%d")
            parts = [("101", "a1", "instagram", 0.55 * r.meta_ads), ("102", "a2", "facebook", 0.45 * r.meta_ads),
                     ("201", "a3", "instagram", r.influencers)]
            for cid, aid, plat, spend in parts:
                if spend <= 0:
                    continue
                base = {"date_start": ds, "date_stop": ds, "campaign_id": cid, "spend": round(spend, 2),
                        "impressions": int(spend * 9), "publisher_platform": plat}
                if mmm:
                    rows.append({"account_id": "123", **base, "adset_id": aid, "country": "IN", "region": "Maharashtra",
                                 "dma": "", "device_platform": "mobile_app", "platform_position": "feed",
                                 "creative_media_type": "video"})
                else:
                    rows.append({**base, "campaign_name": self.CAMPAIGNS[cid], "clicks": int(spend / 40)})
        return rows

    def request(self, method, url, params=None, data=None, json=None, headers=None, timeout=None):
        self.calls.append((method, url))
        hdr = {"x-fb-ads-insights-throttle": '{"app_id_util_pct": 12, "acc_id_util_pct": 8}'}
        path = urlparse(url).path
        if method == "POST" and path.endswith("/insights"):
            tr = __import__("json").loads(data["time_range"])
            rid = f"rr{len(self.jobs) + 1}"
            self.jobs[rid] = {"since": date.fromisoformat(tr["since"]), "until": date.fromisoformat(tr["until"]),
                              "mmm": data.get("breakdowns") == "mmm", "polls": 0}
            return FakeResponse(200, {"report_run_id": rid}, headers=hdr)
        m = re.search(r"/(rr\d+)(/insights)?$", path)
        if m:
            job = self.jobs[m.group(1)]
            fields = (params or {}).get("fields", "")
            if m.group(2):  # paged insights
                rows = self._rows(job["since"], job["until"], False)
                off = int(parse_qs(urlparse(url).query).get("offset", ["0"])[0])
                page = rows[off:off + 500]
                nxt = f"https://graph.facebook.com/v26.0/{m.group(1)}/insights?offset={off + 500}" if off + 500 < len(rows) else None
                return FakeResponse(200, {"data": page, "paging": {"next": nxt} if nxt else {}}, headers=hdr)
            if "async_report_url" in fields:
                return FakeResponse(200, {"async_report_url": f"https://www.facebook.com/ads/export/{m.group(1)}.csv"})
            job["polls"] += 1
            done = job["polls"] > self.polls
            return FakeResponse(200, {"async_status": "Job Completed" if done else "Job Running",
                                      "async_percent_completion": 100 if done else 50}, headers=hdr)
        m = re.search(r"/export/(rr\d+)\.csv$", path)
        if m:
            job = self.jobs[m.group(1)]
            buf = io.StringIO()
            cols = ["account_id", "campaign_id", "adset_id", "date_start", "date_stop", "impressions", "spend", "country",
                    "region", "dma", "device_platform", "platform_position", "publisher_platform", "creative_media_type"]
            pd.DataFrame(self._rows(job["since"], job["until"], True), columns=cols).to_csv(buf, index=False)
            return FakeResponse(200, text=buf.getvalue())
        if path.endswith("/campaigns"):
            return FakeResponse(200, {"data": [{"id": k, "name": v} for k, v in self.CAMPAIGNS.items()], "paging": {}})
        return FakeResponse(404, {"error": {"message": f"unknown path {path}"}})


class FakeGA4Session:
    def __init__(self, biz: SyntheticBusiness, tracking_share: float = 0.82, ads_linked: bool = True):
        self.biz, self.share, self.ads_linked = biz, tracking_share, ads_linked
        self.calls = []

    def request(self, method, url, params=None, data=None, json=None, headers=None, timeout=None):
        self.calls.append((method, url, json))
        if headers is None or not headers.get("Authorization", "").startswith("Bearer "):
            return FakeResponse(401, {"error": {"message": "missing credentials"}})
        dr = json["dateRanges"][0]
        d = self.biz.daily[(self.biz.daily.date >= pd.Timestamp(dr["startDate"])) & (self.biz.daily.date <= pd.Timestamp(dr["endDate"]))]
        rng = np.random.default_rng(1)
        dims, mets = [x["name"] for x in json["dimensions"]], [x["name"] for x in json["metrics"]]
        rows = []
        for _, r in d.iterrows():
            rev = r.revenue * self.share * (1 + rng.normal(0, 0.03))
            vals = {"sessions": rev / 35, "keyEvents": rev / 18000, "transactions": rev / 21000, "purchaseRevenue": rev,
                    "totalRevenue": rev, "advertiserAdCost": (r.google_search + r.youtube) if self.ads_linked else 0.0}
            rows.append({"dimensionValues": [{"value": r.date.strftime("%Y%m%d")}],
                         "metricValues": [{"value": f"{vals[m]:.2f}"} for m in mets]})
        off, lim = json.get("offset", 0), json.get("limit", 100000)
        return FakeResponse(200, {"dimensionHeaders": [{"name": n} for n in dims],
                                  "metricHeaders": [{"name": n, "type": "TYPE_FLOAT"} for n in mets],
                                  "rows": rows[off:off + lim], "rowCount": len(rows)})


class FakeShopifySession:
    def __init__(self, biz: SyntheticBusiness, throttle_first: bool = True):
        self.biz, self.throttle_first = biz, throttle_first
        self.bulk: dict | None = None
        self.calls = []

    def request(self, method, url, params=None, data=None, json=None, headers=None, timeout=None):
        self.calls.append((method, url))
        if url.endswith("/admin/oauth/access_token"):
            if (data or {}).get("grant_type") != "client_credentials" or not (data or {}).get("client_secret"):
                return FakeResponse(400, {"error": "invalid_request"})
            self.issued = getattr(self, "issued", 0) + 1
            return FakeResponse(200, {"access_token": f"shpat_cc_{self.issued}", "scope": "read_orders,read_all_orders",
                                      "expires_in": 86399})
        if url.endswith(".jsonl"):
            lines = "\n".join(__import__("json").dumps(o) for o in self.bulk["orders"])
            return FakeResponse(200, text=lines)
        if not headers or not headers.get("X-Shopify-Access-Token"):
            return FakeResponse(401, {"errors": "[API] Invalid API key or access token"})
        q, v = json["query"], json.get("variables") or {}
        cost = {"cost": {"requestedQueryCost": 50, "throttleStatus": {"maximumAvailable": 2000, "currentlyAvailable": 1900, "restoreRate": 100}}}
        if self.throttle_first:
            self.throttle_first = False
            return FakeResponse(200, {"errors": [{"message": "Throttled", "extensions": {"code": "THROTTLED"}}],
                                      "extensions": {"cost": {"requestedQueryCost": 50, "throttleStatus": {
                                          "maximumAvailable": 2000, "currentlyAvailable": 0, "restoreRate": 100}}}})
        if "shop {" in q:
            return FakeResponse(200, {"data": {"shop": {"name": "Demo Skincare", "ianaTimezone": "Asia/Kolkata",
                                                        "currencyCode": "INR"}}, "extensions": cost})

        def rng_from(search):
            a = re.search(r"created_at:>=(\d{4}-\d{2}-\d{2})", search).group(1)
            b = re.search(r"created_at:<(\d{4}-\d{2}-\d{2})", search).group(1)
            return date.fromisoformat(a), date.fromisoformat(b) - timedelta(days=1)

        if "bulkOperationRunQuery" in q:
            inner = re.search(r'orders\(query: \\"([^\\]+)\\"', q).group(1)
            a, b = rng_from(inner)
            self.bulk = {"id": "gid://shopify/BulkOperation/1", "orders": self.biz.orders(a, b), "polls": 0}
            return FakeResponse(200, {"data": {"bulkOperationRunQuery": {"bulkOperation": {"id": self.bulk["id"], "status": "CREATED"},
                                                                          "userErrors": []}}, "extensions": cost})
        if "node(id:" in q:
            self.bulk["polls"] += 1
            done = self.bulk["polls"] > 1
            return FakeResponse(200, {"data": {"node": {"status": "COMPLETED" if done else "RUNNING", "errorCode": None,
                                                        "objectCount": str(len(self.bulk["orders"])),
                                                        "url": "https://storage.googleapis.com/shopify-bulk/1.jsonl" if done else None}},
                                      "extensions": cost})
        if "orders(first:" in q:
            a, b = rng_from(v["q"])
            allo = self.biz.orders(a, b)
            off = int(v["after"] or 0)
            page = allo[off:off + 250]
            nxt = off + 250 < len(allo)
            return FakeResponse(200, {"data": {"orders": {"pageInfo": {"hasNextPage": nxt, "endCursor": str(off + 250)},
                                                          "edges": [{"node": o} for o in page]}}, "extensions": cost})
        return FakeResponse(200, {"errors": [{"message": "unsupported query in fake"}]})
