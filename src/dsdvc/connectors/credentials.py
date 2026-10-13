"""Resolve connector credentials from environment variables or Streamlit secrets (never from the database)."""
from __future__ import annotations

import os

from dsdvc.config.settings import Settings

KEYS = {
    "meta": ("META_ACCESS_TOKEN", "META_AD_ACCOUNT_ID"),
    "ga4": ("GA4_PROPERTY_ID",),
    "shopify": ("SHOPIFY_SHOP", "SHOPIFY_ACCESS_TOKEN"),
}


def _streamlit_secret(name: str) -> str | None:
    try:
        import streamlit as st
        return st.secrets.get(name)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - no secrets file / not running under Streamlit
        return None


def lookup(name: str) -> str | None:
    """`secret_ref` style lookup: 'env:NAME' / 'streamlit:NAME' / bare NAME (env, then Streamlit secrets)."""
    if name.startswith("env:"):
        return os.environ.get(name[4:]) or None
    if name.startswith("streamlit:"):
        return _streamlit_secret(name[10:])
    return os.environ.get(name) or _streamlit_secret(name)


def settings_with_secrets() -> Settings:
    """Settings where every connector credential falls back to Streamlit secrets when not in the environment."""
    from dsdvc.config.settings import get_settings
    s = get_settings().model_copy()
    for field, env in (("meta_access_token", "META_ACCESS_TOKEN"), ("meta_ad_account_id", "META_AD_ACCOUNT_ID"),
                       ("ga4_property_id", "GA4_PROPERTY_ID"), ("ga4_access_token", "GA4_ACCESS_TOKEN"),
                       ("google_application_credentials", "GOOGLE_APPLICATION_CREDENTIALS"),
                       ("shopify_shop", "SHOPIFY_SHOP"), ("shopify_access_token", "SHOPIFY_ACCESS_TOKEN"),
                       ("shopify_client_id", "SHOPIFY_CLIENT_ID"), ("shopify_client_secret", "SHOPIFY_CLIENT_SECRET"),
                       ("ga4_service_account_json", "GA4_SERVICE_ACCOUNT_JSON"),
                       ("shopify_api_version", "SHOPIFY_API_VERSION"), ("graph_api_version", "META_GRAPH_API_VERSION")):
        if not getattr(s, field) or field.endswith("_version"):
            v = lookup(env)
            if v:
                setattr(s, field, v)
    return s


def status(s: Settings | None = None) -> dict[str, bool]:
    s = s or settings_with_secrets()
    return {"meta": bool(s.meta_access_token and s.meta_ad_account_id),
            "ga4": bool(s.ga4_property_id and (s.ga4_access_token or s.google_application_credentials or s.ga4_service_account_json)),
            "shopify": bool(s.shopify_shop and (s.shopify_access_token or (s.shopify_client_id and s.shopify_client_secret)))}


def connectors(s: Settings | None = None, demo: bool = False, seed: int = 0) -> dict:
    """Instantiate the three connectors - against the real APIs, or the labelled SYNTHETIC simulators."""
    from dsdvc.connectors.ga4 import GA4Connector
    from dsdvc.connectors.meta_ads import MetaAdsConnector
    from dsdvc.connectors.shopify import ShopifyConnector
    if demo:
        from dsdvc.connectors.fakes import FakeGA4Session, FakeMetaSession, FakeShopifySession, SyntheticBusiness
        biz = SyntheticBusiness(seed)
        return {"meta": MetaAdsConnector("demo", "demo", session=FakeMetaSession(biz), poll_seconds=0),
                "ga4": GA4Connector("demo", access_token="demo", session=FakeGA4Session(biz)),
                "shopify": ShopifyConnector("demo-store", "demo", session=FakeShopifySession(biz, throttle_first=False),
                                            poll_seconds=0)}
    s = s or settings_with_secrets()
    return {"meta": MetaAdsConnector(s.meta_access_token, s.meta_ad_account_id, s.graph_api_version),
            "ga4": GA4Connector(s.ga4_property_id, s.ga4_access_token, s.google_application_credentials,
                                service_account_info=s.ga4_service_account_json),
            "shopify": ShopifyConnector(s.shopify_shop, s.shopify_access_token, s.shopify_api_version,
                                        client_id=s.shopify_client_id, client_secret=s.shopify_client_secret)}
