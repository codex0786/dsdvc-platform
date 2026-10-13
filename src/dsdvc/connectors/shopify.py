"""Shopify Admin GraphQL connector - orders → daily sales (the MMM outcome of record).

* Short ranges: paginated ``orders(first: 250, after:, query: "created_at:>=… created_at:<…")``.
* Long ranges: a **bulk operation** (``bulkOperationRunQuery``) polled via ``node(id:)`` (works on
  all API versions; ``currentBulkOperation`` is deprecated from 2026-01), then the JSONL result is
  streamed line by line.
* Throttling: GraphQL cost budget from ``extensions.cost.throttleStatus`` and ``THROTTLED`` errors
  are respected with back-off.
* Privacy: no customer fields are requested - only order-level money, timestamps and flags.
* Scope: ``read_orders`` exposes the last 60 days; ``read_all_orders`` is needed for history. The
  connector warns when the earliest order returned is suspiciously close to the 60-day limit.
* Time zone: orders are bucketed into days in the **shop's** time zone (``shop.ianaTimezone``).
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta

import pandas as pd

from dsdvc.core.provenance import DataUnavailable, SourceType, unavailable
from dsdvc.platforms.base import Connector

ORDER_FIELDS = """
  id createdAt cancelledAt test sourceName
  totalPriceSet { shopMoney { amount currencyCode } }
  subtotalPriceSet { shopMoney { amount } }
  totalDiscountsSet { shopMoney { amount } }
  totalRefundedSet { shopMoney { amount } }
"""


class ShopifyConnector(Connector):
    name = "shopify_admin_graphql"
    source_type = SourceType.AUTHORIZED_API

    def __init__(self, shop: str | None, access_token: str | None, api_version: str = "2026-07",
                 bulk_threshold_days: int = 120, poll_seconds: float = 3.0, max_wait_seconds: float = 1800,
                 client_id: str | None = None, client_secret: str | None = None, **kw):
        super().__init__(**kw)
        self.client_id, self.client_secret = client_id, client_secret
        self._token_expires = None
        s = (shop or "").replace("https://", "").replace("http://", "").strip("/")
        self.shop = s if not s or "." in s else f"{s}.myshopify.com"
        self.token = access_token
        self.version = api_version
        self.bulk_threshold_days = bulk_threshold_days
        self.poll_seconds = poll_seconds
        self.max_wait = max_wait_seconds

    @property
    def ready(self) -> bool:
        return bool(self.shop and (self.token or (self.client_id and self.client_secret)))

    def _ensure_token(self) -> DataUnavailable | None:
        """Dev Dashboard apps (the only kind creatable since 2026-01-01) obtain short-lived Admin API tokens
        with the client-credentials grant; legacy custom apps use a static ``shpat_`` token."""
        if self.token and (self._token_expires is None or time.time() < self._token_expires - 300):
            return None
        if not (self.client_id and self.client_secret):
            return None if self.token else self._missing()
        r = self.post(f"https://{self.shop}/admin/oauth/access_token",
                      data={"client_id": self.client_id, "client_secret": self.client_secret, "grant_type": "client_credentials"})
        if isinstance(r, DataUnavailable):
            return r
        self.token = r.get("access_token")
        self._token_expires = time.time() + float(r.get("expires_in", 86400))
        return None if self.token else unavailable("shopify.oauth", "token exchange returned no access_token")

    def _missing(self) -> DataUnavailable:
        return unavailable("shopify", "Shopify credentials not configured",
                           "Set SHOPIFY_SHOP plus SHOPIFY_CLIENT_ID + SHOPIFY_CLIENT_SECRET (Dev Dashboard app) or "
                           "SHOPIFY_ACCESS_TOKEN (legacy custom app); scope read_orders, and read_all_orders for > 60 days.")

    def graphql(self, query: str, variables: dict | None = None, attempts: int = 6) -> dict | DataUnavailable:
        url = f"https://{self.shop}/admin/api/{self.version}/graphql.json"
        if (err := self._ensure_token()) is not None:
            return err
        for i in range(attempts):
            r = self.post(url, json={"query": query, "variables": variables or {}},
                          headers={"X-Shopify-Access-Token": self.token, "Content-Type": "application/json"})
            if isinstance(r, DataUnavailable):
                return r
            errs = r.get("errors") or []
            if any((e.get("extensions") or {}).get("code") == "THROTTLED" for e in errs if isinstance(e, dict)):
                ts = ((r.get("extensions") or {}).get("cost") or {}).get("throttleStatus") or {}
                need = ((r.get("extensions") or {}).get("cost") or {}).get("requestedQueryCost", 100)
                rate = ts.get("restoreRate", 50) or 50
                time.sleep(max(1.0, (need - ts.get("currentlyAvailable", 0)) / rate))
                continue
            if errs:
                return unavailable("shopify.graphql", f"GraphQL error: {str(errs)[:300]}")
            ts = ((r.get("extensions") or {}).get("cost") or {}).get("throttleStatus") or {}
            if ts and ts.get("currentlyAvailable", 1000) < 0.1 * ts.get("maximumAvailable", 1000):
                time.sleep(2.0)  # stay well inside the cost budget
            return r["data"]
        return unavailable("shopify.graphql", "throttled repeatedly; try again later")

    def shop_info(self) -> dict | DataUnavailable:
        if not self.ready:
            return self._missing()
        d = self.graphql("{ shop { name ianaTimezone currencyCode } }")
        return d if isinstance(d, DataUnavailable) else d["shop"]

    @staticmethod
    def _search(start: date, end: date) -> str:
        return f"created_at:>={start.isoformat()} created_at:<{(end + timedelta(days=1)).isoformat()}"

    def _orders_paginated(self, start: date, end: date) -> list[dict] | DataUnavailable:
        q = ("query($q: String!, $after: String) { orders(first: 250, after: $after, query: $q, sortKey: CREATED_AT) "
             "{ pageInfo { hasNextPage endCursor } edges { node {" + ORDER_FIELDS + "} } } }")
        out, after = [], None
        while True:
            d = self.graphql(q, {"q": self._search(start, end), "after": after})
            if isinstance(d, DataUnavailable):
                return d
            con = d["orders"]
            out += [e["node"] for e in con["edges"]]
            if not con["pageInfo"]["hasNextPage"]:
                return out
            after = con["pageInfo"]["endCursor"]

    def _orders_bulk(self, start: date, end: date) -> list[dict] | DataUnavailable:
        inner = '{ orders(query: "%s") { edges { node {%s} } } }' % (self._search(start, end), ORDER_FIELDS)
        m = 'mutation { bulkOperationRunQuery(query: %s) { bulkOperation { id status } userErrors { field message } } }' % json.dumps(inner)
        d = self.graphql(m)
        if isinstance(d, DataUnavailable):
            return d
        res = d["bulkOperationRunQuery"]
        if res.get("userErrors"):
            return unavailable("shopify.bulk", f"bulk query rejected: {res['userErrors']}")
        op_id, waited = res["bulkOperation"]["id"], 0.0
        while True:
            s = self.graphql("query($id: ID!) { node(id: $id) { ... on BulkOperation { status errorCode objectCount url } } }",
                             {"id": op_id})
            if isinstance(s, DataUnavailable):
                return s
            node = s["node"]
            if node["status"] == "COMPLETED":
                break
            if node["status"] in ("FAILED", "CANCELED", "EXPIRED"):
                return unavailable("shopify.bulk", f"bulk operation {node['status']} ({node.get('errorCode')})")
            if waited >= self.max_wait:
                return unavailable("shopify.bulk", "bulk operation did not finish in time")
            time.sleep(self.poll_seconds)
            waited += self.poll_seconds
        if not node.get("url"):
            return []  # documented: url is null when the query matched nothing
        resp = self.get(node["url"], raw=True)  # signed URL, no auth header
        if isinstance(resp, DataUnavailable):
            return resp
        return [json.loads(line) for line in resp.text.splitlines() if line.strip()]

    def orders(self, start: date, end: date, method: str = "auto") -> pd.DataFrame | DataUnavailable:
        if not self.ready:
            return self._missing()
        use_bulk = method == "bulk" or (method == "auto" and (end - start).days > self.bulk_threshold_days)
        nodes = self._orders_bulk(start, end) if use_bulk else self._orders_paginated(start, end)
        if isinstance(nodes, DataUnavailable):
            return nodes

        def money(n, k):
            v = n.get(k) or {}
            return float((v.get("shopMoney") or {}).get("amount") or 0.0)

        df = pd.DataFrame([{"id": n["id"], "created_at": n["createdAt"], "cancelled": n.get("cancelledAt") is not None,
                            "test": bool(n.get("test")), "source": n.get("sourceName"), "total": money(n, "totalPriceSet"),
                            "subtotal": money(n, "subtotalPriceSet"), "discounts": money(n, "totalDiscountsSet"),
                            "refunded": money(n, "totalRefundedSet"),
                            "currency": ((n.get("totalPriceSet") or {}).get("shopMoney") or {}).get("currencyCode")}
                           for n in nodes])
        df.attrs["method"] = "bulk" if use_bulk else "paginated"
        return df

    def daily_sales(self, start: date, end: date, method: str = "auto") -> pd.DataFrame | DataUnavailable:
        """Daily orders, gross/net sales and discount rate in the shop's time zone (test and cancelled orders excluded)."""
        info = self.shop_info()
        if isinstance(info, DataUnavailable):
            return info
        df = self.orders(start, end, method)
        if isinstance(df, DataUnavailable):
            return df
        tz = info.get("ianaTimezone") or "UTC"
        notes = []
        if df.empty:
            return unavailable("shopify.orders", "no orders in the requested range")
        n_test, n_cancel = int(df.test.sum()), int((df.cancelled & ~df.test).sum())
        df = df[~df.test & ~df.cancelled].copy()
        df["date"] = pd.to_datetime(df.created_at, utc=True).dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()
        g = df.groupby("date").agg(orders=("id", "count"), gross_sales=("total", "sum"), refunds=("refunded", "sum"),
                                   discounts=("discounts", "sum"), subtotal=("subtotal", "sum")).reset_index()
        full = pd.DataFrame({"date": pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")})
        g = full.merge(g, on="date", how="left").fillna(0.0)
        g["net_sales"] = g.gross_sales - g.refunds
        g["discount_rate"] = (g.discounts / (g.subtotal + g.discounts).where(lambda s: s > 0)).fillna(0.0)
        first = df.date.min()
        if (end - start).days > 60 and first is not pd.NaT and (first - pd.Timestamp(start)).days > 45:
            notes.append(f"Earliest order returned is {first.date()} - if the store is older, the app may lack the "
                         "read_all_orders scope (read_orders only covers the last 60 days).")
        notes.append(f"Excluded {n_test} test and {n_cancel} cancelled orders; days bucketed in {tz}; currency "
                     f"{info.get('currencyCode')}; fetched via {df.attrs.get('method', 'paginated')} query.")
        g.attrs["provenance"] = self.prov("shopify admin graphql orders → daily aggregation",
                                          f"read_orders on {self.shop}").model_dump(mode="json")
        g.attrs["provenance"]["notes"] = " ".join(notes)
        g.attrs["warnings"] = notes[:-1]
        g.attrs["currency"] = info.get("currencyCode")
        return g
