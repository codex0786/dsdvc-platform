"""Meta Ads (Marketing API) connector - paid media spend for the MMM.

Two modes:

* ``mmm`` (preferred): Meta's official Marketing Mix Modeling breakdown
  (``breakdowns=mmm``, ``export_format=csv``, ``level=adset``, ``time_increment=1``). Always an
  async CSV export: POST → poll ``async_status`` → GET ``async_report_url``. Returns daily
  spend/impressions by campaign, ad set, publisher platform and geo (country/region/DMA).
* ``insights``: standard Insights at campaign level with ``breakdowns=publisher_platform``,
  run as an async job and read from ``/{report_run_id}/insights``. Gives campaign names, so
  name-based channel rules work.

Long ranges are split into chunks (default 30 days) to avoid timeouts. Requires ``ads_read``
on the ad account. Rate-limit headers (``x-fb-ads-insights-throttle``, ``x-ad-account-usage``,
``x-business-use-case-usage``) are honoured by the shared HTTP layer; nothing is retried with
different credentials.
"""
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from dsdvc.core.provenance import DataUnavailable, SourceType, unavailable
from dsdvc.platforms.base import Connector

GRAPH = "https://graph.facebook.com"
DONE = "Job Completed"
FAILED = ("Job Failed", "Job Skipped")


@dataclass
class ChannelRules:
    """Map Meta spend to MMM channels.

    ``campaign_contains``: ordered (substring → channel) rules on campaign name (case-insensitive),
    e.g. [("influencer", "influencers"), ("brand", "meta_brand")]. Unmatched spend goes to ``default``.
    ``split_platforms``: if True, Instagram and Facebook become separate channels.
    """
    default: str = "meta_ads"
    campaign_contains: list[tuple[str, str]] = field(default_factory=list)
    split_platforms: bool = False

    def channel(self, campaign_name: str | None, publisher_platform: str | None) -> str:
        name = (campaign_name or "").lower()
        for needle, ch in self.campaign_contains:
            if needle.lower() in name:
                return ch
        if self.split_platforms and publisher_platform:
            return {"instagram": "instagram_ads", "facebook": "facebook_ads"}.get(publisher_platform, "meta_other")
        return self.default


def _chunks(start: date, end: date, days: int):
    cur = start
    while cur <= end:
        stop = min(end, cur + timedelta(days=days - 1))
        yield cur, stop
        cur = stop + timedelta(days=1)


class MetaAdsConnector(Connector):
    name = "meta_marketing_api"
    source_type = SourceType.AUTHORIZED_API

    def __init__(self, access_token: str | None, ad_account_id: str | None, api_version: str = "v26.0",
                 poll_seconds: float = 3.0, max_wait_seconds: float = 900, **kw):
        super().__init__(**kw)
        self.token = access_token
        self.account = (ad_account_id or "").removeprefix("act_")
        self.version = api_version
        self.poll_seconds = poll_seconds
        self.max_wait = max_wait_seconds

    @property
    def ready(self) -> bool:
        return bool(self.token and self.account)

    def _url(self, path: str) -> str:
        return f"{GRAPH}/{self.version}/{path}"

    def _missing(self) -> DataUnavailable:
        return unavailable("meta_ads", "Meta Marketing API credentials not configured",
                           "Set META_ACCESS_TOKEN (with ads_read) and META_AD_ACCOUNT_ID for an ad account you manage.")

    # ------------------------------------------------------------------ async job helpers
    def _submit(self, params: dict) -> str | DataUnavailable:
        r = self.post(self._url(f"act_{self.account}/insights"), data={**params, "access_token": self.token})
        if isinstance(r, DataUnavailable):
            return r
        rid = r.get("report_run_id")
        return rid or unavailable("meta_ads.insights", f"no report_run_id returned: {str(r)[:200]}")

    def _wait(self, rid: str) -> dict | DataUnavailable:
        waited = 0.0
        while True:
            st = self.get(self._url(rid), {"fields": "async_status,async_percent_completion,error_message",
                                            "access_token": self.token})
            if isinstance(st, DataUnavailable):
                return st
            status = st.get("async_status")
            if status == DONE:
                return st
            if status in FAILED:
                return unavailable("meta_ads.insights", f"report job {status}: {st.get('error_message', '')}",
                                   "Narrow the date range or retry later.")
            if waited >= self.max_wait:
                return unavailable("meta_ads.insights", f"report job still '{status}' after {self.max_wait:.0f}s")
            time.sleep(self.poll_seconds)
            waited += self.poll_seconds

    # ------------------------------------------------------------------ MMM breakdown export
    def mmm_export(self, start: date, end: date, chunk_days: int = 30) -> pd.DataFrame | DataUnavailable:
        if not self.ready:
            return self._missing()
        frames = []
        for a, b in _chunks(start, end, chunk_days):
            rid = self._submit({"breakdowns": "mmm", "export_format": "csv", "level": "adset", "time_increment": 1,
                                "time_range": json.dumps({"since": a.isoformat(), "until": b.isoformat()})})
            if isinstance(rid, DataUnavailable):
                return rid
            st = self._wait(rid)
            if isinstance(st, DataUnavailable):
                return st
            meta = self.get(self._url(rid), {"fields": "async_report_url", "access_token": self.token})
            if isinstance(meta, DataUnavailable):
                return meta
            url = meta.get("async_report_url")
            if not url:
                return unavailable("meta_ads.mmm", "export finished without a download URL")
            resp = self.get(url, raw=True)
            if isinstance(resp, DataUnavailable) or "isn't available" in getattr(resp, "text", ""):
                resp = self.get(url, {"access_token": self.token}, raw=True)  # documented fallback
                if isinstance(resp, DataUnavailable):
                    return resp
            if resp.text.strip():  # a window with no delivery yields a header-only (or empty) file
                try:
                    frames.append(pd.read_csv(io.StringIO(resp.text)))
                except pd.errors.EmptyDataError:
                    pass
        frames = [f for f in frames if len(f)]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if df.empty:
            return unavailable("meta_ads.mmm", "no spend in the requested range")
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df["date"] = pd.to_datetime(df["date_start"])
        for c in ("spend", "impressions"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df.attrs["provenance"] = self.prov("marketing_api.insights breakdowns=mmm (async CSV)",
                                           f"ads_read on act_{self.account}", confidence=0.95).model_dump(mode="json")
        df.attrs["provenance"]["notes"] = "Meta labels MMM spend as estimated; dates are in the ad account time zone."
        return df

    # ------------------------------------------------------------------ standard insights
    def insights(self, start: date, end: date, chunk_days: int = 30) -> pd.DataFrame | DataUnavailable:
        if not self.ready:
            return self._missing()
        rows = []
        for a, b in _chunks(start, end, chunk_days):
            rid = self._submit({"level": "campaign", "time_increment": 1, "breakdowns": "publisher_platform",
                                "fields": "campaign_id,campaign_name,spend,impressions,clicks",
                                "time_range": json.dumps({"since": a.isoformat(), "until": b.isoformat()})})
            if isinstance(rid, DataUnavailable):
                return rid
            st = self._wait(rid)
            if isinstance(st, DataUnavailable):
                return st
            url, params = self._url(f"{rid}/insights"), {"limit": 500, "access_token": self.token}
            while url:
                page = self.get(url, params)
                if isinstance(page, DataUnavailable):
                    return page
                rows += page.get("data", [])
                url, params = (page.get("paging") or {}).get("next"), None  # next URL already carries the query
        df = pd.DataFrame(rows)
        if df.empty:
            return unavailable("meta_ads.insights", "no spend in the requested range")
        df["date"] = pd.to_datetime(df["date_start"])
        for c in ("spend", "impressions", "clicks"):
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df.attrs["provenance"] = self.prov("marketing_api.insights level=campaign breakdowns=publisher_platform",
                                           f"ads_read on act_{self.account}").model_dump(mode="json")
        return df

    def campaign_names(self) -> dict[str, str] | DataUnavailable:
        if not self.ready:
            return self._missing()
        out, url, params = {}, self._url(f"act_{self.account}/campaigns"), {"fields": "id,name", "limit": 500,
                                                                             "access_token": self.token}
        while url:
            page = self.get(url, params)
            if isinstance(page, DataUnavailable):
                return page
            out.update({c["id"]: c.get("name", "") for c in page.get("data", [])})
            url, params = (page.get("paging") or {}).get("next"), None
        return out

    # ------------------------------------------------------------------ to MMM channels
    def daily_spend(self, start: date, end: date, mode: str = "mmm", rules: ChannelRules | None = None) -> pd.DataFrame | DataUnavailable:
        """Daily spend per MMM channel: columns date, channel, spend, impressions."""
        rules = rules or ChannelRules()
        df = self.mmm_export(start, end) if mode == "mmm" else self.insights(start, end)
        if isinstance(df, DataUnavailable):
            return df
        prov = df.attrs.get("provenance", {})
        if "campaign_name" not in df and rules.campaign_contains:
            names = self.campaign_names()
            if isinstance(names, DataUnavailable):
                return names
            df["campaign_name"] = df["campaign_id"].astype(str).map(names)
        df["channel"] = [rules.channel(c, p) for c, p in zip(df.get("campaign_name", pd.Series([None] * len(df))),
                                                              df.get("publisher_platform", pd.Series([None] * len(df))))]
        out = df.groupby(["date", "channel"], as_index=False)[["spend", "impressions"]].sum()
        out.attrs["provenance"] = prov
        return out
