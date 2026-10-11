"""Platform connector contract and governance rules (spec §13, §14).

Connectors MUST:
  * use official APIs or user-provided/authorised exports only;
  * honour platform rate-limit headers (back off; never rotate tokens/IPs to evade limits);
  * never request passwords, never scrape private accounts or private messages;
  * return ``DataUnavailable`` instead of fabricating data;
  * attach a ``Provenance`` record to every value.
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC
from typing import Any

from dsdvc.core.provenance import (DataUnavailable, EpistemicStatus, Provenance, ProvenancedValue, SourceType,
                                   unavailable)

log = logging.getLogger("dsdvc.platforms")

FORBIDDEN_ACTIONS = (
    "scrape private accounts", "read private WhatsApp conversations", "bypass authentication",
    "bypass rate limits", "collect passwords", "infer sensitive personal attributes",
    "identify private individuals without authorisation",
)


class RateLimitExceeded(RuntimeError):
    pass


class Connector(ABC):
    name: str = "connector"
    source_type: SourceType = SourceType.AUTHORIZED_API

    def __init__(self, session=None, max_usage_pct: float = 80.0, max_retries: int = 3):
        import requests
        self.session = session or requests.Session()
        self.max_usage_pct = max_usage_pct
        self.max_retries = max_retries

    # -- provenance helpers
    def prov(self, method: str, scope: str, confidence: float = 0.99, status=EpistemicStatus.OBSERVED) -> Provenance:
        return Provenance(source=self.name, source_type=self.source_type, collection_method=method,
                          authorization_scope=scope, confidence=confidence, observed_or_estimated=status)

    def value(self, metric: str, v: Any, method: str, scope: str) -> ProvenancedValue:
        return ProvenancedValue(metric=metric, value=v, provenance=self.prov(method, scope))

    # -- polite HTTP with rate-limit compliance
    def _usage_pct(self, headers) -> float:
        pct = 0.0
        for h in ("x-app-usage", "x-business-use-case-usage", "x-ad-account-usage"):
            raw = headers.get(h) or headers.get(h.title())
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except ValueError:
                continue
            vals = []
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, (int, float)):
                        vals.append(v)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                vals += [x for x in item.values() if isinstance(x, (int, float))]
            pct = max([pct] + vals)
        return pct

    def get(self, url: str, params: dict) -> dict | DataUnavailable:
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, params=params, timeout=30)
            except Exception as e:  # noqa: BLE001
                return unavailable(url.split("?")[0], f"network error: {type(e).__name__}")
            usage = self._usage_pct(r.headers)
            if r.status_code == 429 or usage >= 100:
                wait = min(60 * (attempt + 1), 300)
                log.warning("rate limit reached on %s; backing off %ss (compliant, no evasion)", self.name, wait)
                time.sleep(wait)
                continue
            if usage >= self.max_usage_pct:
                log.info("%s usage at %.0f%%; slowing down", self.name, usage)
                time.sleep(5)
            if r.status_code in (401, 403):
                return unavailable(url.split("?")[0], f"not authorised ({r.status_code})",
                                   "Check token permissions / app review; DSDVC will not bypass access controls.")
            if r.status_code >= 400:
                return unavailable(url.split("?")[0], f"HTTP {r.status_code}: {r.text[:200]}")
            return r.json()
        return unavailable(url.split("?")[0], "rate limit persisted after compliant back-off")
