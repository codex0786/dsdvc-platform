"""Runtime settings from environment variables. Secrets are never hard-coded (spec §64, §65)."""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel


class Settings(BaseModel):
    env: str = "local"
    database_url: str = "sqlite:///./dsdvc.db"
    redis_url: str | None = None
    mlflow_tracking_uri: str | None = None
    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None
    secret_key: str = "change-me-in-production"
    bootstrap_admin_key: str | None = None
    # Platform credentials: only ever read from env/secret store, never logged.
    meta_access_token: str | None = None
    instagram_business_account_id: str | None = None
    facebook_page_id: str | None = None
    meta_content_library_token: str | None = None
    whatsapp_business_token: str | None = None
    whatsapp_business_account_id: str | None = None
    graph_api_version: str = "v26.0"
    meta_ad_account_id: str | None = None          # numeric id, without the act_ prefix
    ga4_property_id: str | None = None
    ga4_access_token: str | None = None            # short-lived OAuth token, or use a service account file
    google_application_credentials: str | None = None
    shopify_shop: str | None = None                # e.g. mybrand.myshopify.com
    shopify_access_token: str | None = None        # legacy custom app (static shpat_ token)
    shopify_client_id: str | None = None           # Dev Dashboard app (client-credentials grant)
    shopify_client_secret: str | None = None
    shopify_api_version: str = "2026-07"
    ga4_service_account_json: str | None = None
    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-5-5"
    synthetic_data_allowed: bool = True

    def redacted(self) -> dict:
        d = self.model_dump()
        for k in list(d):
            if any(s in k for s in ("token", "key", "secret", "password")) and d[k]:
                d[k] = "***"
        return d


def _env(name: str, default=None):
    v = os.environ.get(name)
    return default if v in (None, "") else v


@lru_cache
def get_settings() -> Settings:
    return Settings(
        env=_env("DSDVC_ENV", "local"),
        database_url=_env("DATABASE_URL", "sqlite:///./dsdvc.db"),
        redis_url=_env("REDIS_URL"),
        mlflow_tracking_uri=_env("MLFLOW_TRACKING_URI"),
        s3_endpoint_url=_env("S3_ENDPOINT_URL"),
        s3_bucket=_env("S3_BUCKET"),
        secret_key=_env("DSDVC_SECRET_KEY", "change-me-in-production"),
        bootstrap_admin_key=_env("DSDVC_BOOTSTRAP_ADMIN_KEY"),
        meta_access_token=_env("META_ACCESS_TOKEN"),
        instagram_business_account_id=_env("INSTAGRAM_BUSINESS_ACCOUNT_ID"),
        facebook_page_id=_env("FACEBOOK_PAGE_ID"),
        meta_content_library_token=_env("META_CONTENT_LIBRARY_TOKEN"),
        whatsapp_business_token=_env("WHATSAPP_BUSINESS_TOKEN"),
        whatsapp_business_account_id=_env("WHATSAPP_BUSINESS_ACCOUNT_ID"),
        graph_api_version=_env("META_GRAPH_API_VERSION", "v26.0"),
        meta_ad_account_id=_env("META_AD_ACCOUNT_ID"),
        ga4_property_id=_env("GA4_PROPERTY_ID"),
        ga4_access_token=_env("GA4_ACCESS_TOKEN"),
        google_application_credentials=_env("GOOGLE_APPLICATION_CREDENTIALS"),
        shopify_shop=_env("SHOPIFY_SHOP"),
        shopify_access_token=_env("SHOPIFY_ACCESS_TOKEN"),
        shopify_client_id=_env("SHOPIFY_CLIENT_ID"),
        shopify_client_secret=_env("SHOPIFY_CLIENT_SECRET"),
        ga4_service_account_json=_env("GA4_SERVICE_ACCOUNT_JSON"),
        shopify_api_version=_env("SHOPIFY_API_VERSION", "2026-07"),
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        llm_model=_env("DSDVC_LLM_MODEL", "claude-opus-5-5"),
        synthetic_data_allowed=_env("DSDVC_ALLOW_SYNTHETIC", "true").lower() == "true",
    )
