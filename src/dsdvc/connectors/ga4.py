"""Google Analytics 4 (Data API v1beta) connector - web outcomes and Google Ads cost.

``runReport`` with ``date`` dimension; paginated with ``limit``/``offset`` until ``rowCount``.
Auth: a short-lived OAuth access token (GA4_ACCESS_TOKEN), or a service account - as a key file
(GOOGLE_APPLICATION_CREDENTIALS) or as the key JSON in a secret (GA4_SERVICE_ACCOUNT_JSON); service
accounts need the ``google-auth`` package and the
``analytics.readonly`` scope; the service account must be added as a Viewer on the property.

Metric names follow the current GA4 schema: ``keyEvents`` (formerly "conversions"),
``purchaseRevenue`` and ``totalRevenue`` (net of refunds), ``transactions``,
``advertiserAdCost`` (linked Google Ads cost).

Note: channel-group sessions (Organic, Direct, …) are *outcomes of marketing*, not controls -
they are returned for diagnostics only and must not be fed to the MMM as regressors.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from dsdvc.core.provenance import DataUnavailable, SourceType, unavailable
from dsdvc.platforms.base import Connector

API = "https://analyticsdata.googleapis.com/v1beta"
SCOPE = "https://www.googleapis.com/auth/analytics.readonly"
OUTCOME_METRICS = ["sessions", "keyEvents", "transactions", "purchaseRevenue", "totalRevenue"]


class GA4Connector(Connector):
    name = "ga4_data_api"
    source_type = SourceType.AUTHORIZED_API

    def __init__(self, property_id: str | None, access_token: str | None = None,
                 service_account_file: str | None = None, page_size: int = 100_000,
                 service_account_info: dict | str | None = None, **kw):
        super().__init__(**kw)
        if isinstance(service_account_info, str) and service_account_info.strip():
            import json as _json
            service_account_info = _json.loads(service_account_info)
        self.sa_info = service_account_info or None
        self.property = (property_id or "").removeprefix("properties/")
        self._token = access_token
        self.sa_file = service_account_file
        self.page_size = page_size

    @property
    def ready(self) -> bool:
        return bool(self.property and (self._token or self.sa_file or self.sa_info))

    def _missing(self) -> DataUnavailable:
        return unavailable("ga4", "GA4 credentials not configured",
                           "Set GA4_PROPERTY_ID and either GA4_ACCESS_TOKEN or GOOGLE_APPLICATION_CREDENTIALS "
                           "(service account added as Viewer on the property).")

    def token(self) -> str | DataUnavailable:
        if self._token:
            return self._token
        try:
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account
        except ImportError:
            return unavailable("ga4", "google-auth not installed", "pip install google-auth, or supply GA4_ACCESS_TOKEN.")
        creds = (service_account.Credentials.from_service_account_info(self.sa_info, scopes=[SCOPE]) if self.sa_info
                 else service_account.Credentials.from_service_account_file(self.sa_file, scopes=[SCOPE]))
        creds.refresh(Request())
        return creds.token

    def run_report(self, start: date, end: date, dimensions: list[str], metrics: list[str]) -> pd.DataFrame | DataUnavailable:
        if not self.ready:
            return self._missing()
        tok = self.token()
        if isinstance(tok, DataUnavailable):
            return tok
        url = f"{API}/properties/{self.property}:runReport"
        rows, offset, total = [], 0, None
        while total is None or offset < total:
            body = {"dateRanges": [{"startDate": start.isoformat(), "endDate": end.isoformat()}],
                    "dimensions": [{"name": d} for d in dimensions], "metrics": [{"name": m} for m in metrics],
                    "limit": self.page_size, "offset": offset, "keepEmptyRows": True}
            r = self.post(url, json=body, headers={"Authorization": f"Bearer {tok}"})
            if isinstance(r, DataUnavailable):
                return r
            total = int(r.get("rowCount", 0))
            dh = [h["name"] for h in r.get("dimensionHeaders", [])]
            mh = [h["name"] for h in r.get("metricHeaders", [])]
            for row in r.get("rows", []):
                rec = {k: v["value"] for k, v in zip(dh, row.get("dimensionValues", []))}
                rec.update({k: float(v["value"]) for k, v in zip(mh, row.get("metricValues", []))})
                rows.append(rec)
            offset += self.page_size
            if not r.get("rows"):
                break
        df = pd.DataFrame(rows)
        if df.empty:
            return unavailable("ga4.runReport", "no rows returned for the requested range")
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
            df = df.sort_values("date").reset_index(drop=True)
        df.attrs["provenance"] = self.prov(f"ga4 runReport {dimensions} x {metrics}", f"analytics.readonly on property {self.property}"
                                           ).model_dump(mode="json")
        df.attrs["provenance"]["notes"] = "Dates are in the GA4 property time zone; revenue is net of refunds."
        return df

    def daily_outcomes(self, start: date, end: date) -> pd.DataFrame | DataUnavailable:
        return self.run_report(start, end, ["date"], OUTCOME_METRICS)

    def daily_google_ads_cost(self, start: date, end: date) -> pd.DataFrame | DataUnavailable:
        """Linked Google Ads cost by day (DATA_UNAVAILABLE if Ads is not linked or the metric is all zero)."""
        df = self.run_report(start, end, ["date"], ["advertiserAdCost"])
        if isinstance(df, DataUnavailable):
            return df
        if df["advertiserAdCost"].sum() <= 0:
            return unavailable("ga4.advertiserAdCost", "no Google Ads cost on this property (Ads not linked?)",
                               "Link Google Ads to GA4 or upload Google Ads spend separately.")
        return df

    def daily_channel_sessions(self, start: date, end: date) -> pd.DataFrame | DataUnavailable:
        """Diagnostics only (mediators, not MMM controls)."""
        return self.run_report(start, end, ["date", "sessionDefaultChannelGroup"], ["sessions"])
