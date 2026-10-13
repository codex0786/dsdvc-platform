"""One call that pulls the selected sources and assembles the weekly MMM dataset."""
from __future__ import annotations

from datetime import date

from dsdvc.connectors.meta_ads import ChannelRules
from dsdvc.connectors.weekly import BuildResult, build_weekly
from dsdvc.core.provenance import DataUnavailable


def pull_weekly(conns: dict, start: date, end: date, sources: list[str], *, meta_mode: str = "mmm",
                rules: ChannelRules | None = None, outcome_source: str = "auto", extra_spend: dict | None = None,
                progress=None) -> tuple[BuildResult | None, dict[str, str]]:
    """Returns (result, errors). ``errors`` maps a source to its DATA_UNAVAILABLE reason; other sources still load."""
    errors: dict[str, str] = {}
    got: dict = {}

    def step(msg):
        if progress:
            progress(msg)

    def take(name, result):
        if isinstance(result, DataUnavailable):
            errors[name] = f"{result.reason}. {result.remedy or ''}".strip()
        else:
            got[name] = result

    if "meta" in sources:
        step("Meta Ads: requesting spend export…")
        take("meta", conns["meta"].daily_spend(start, end, meta_mode, rules))
    if "ga4" in sources:
        step("GA4: running reports…")
        take("ga4", conns["ga4"].daily_outcomes(start, end))
        if "ga4" in got:
            take("ga4_ads_cost", conns["ga4"].daily_google_ads_cost(start, end))
    if "shopify" in sources:
        step("Shopify: exporting orders…")
        take("shopify", conns["shopify"].daily_sales(start, end))
    if not got and not extra_spend:
        return None, errors
    step("Assembling weekly dataset…")
    res = build_weekly(start, end, meta=got.get("meta"), ga4=got.get("ga4"), ga4_ads_cost=got.get("ga4_ads_cost"),
                       shopify=got.get("shopify"), extra_spend=extra_spend, outcome_source=outcome_source)
    return res, errors
