"""Meta platform connectors: Instagram Graph API, Facebook Pages, Meta Content Library,
WhatsApp Business Platform (spec §10-§13).

What each can legitimately return:
  * Instagram Graph API (professional account the token manages): own profile, own media and
    insights; ``business_discovery`` for *other public business/creator accounts* (basic public
    fields only); hashtag search (rate-limited to 30 unique hashtags / 7 days).
  * Facebook Pages API: posts and insights for Pages the token manages.
  * Meta Content Library: available only to approved researchers inside Meta's research
    environment; DSDVC ingests exports produced there (``import_content_library_export``).
  * WhatsApp Business Platform: aggregated message/conversation analytics for the business's own
    WABA. Never message content, never personal chats.

Endpoint/field names follow Meta's public Graph API documentation; API versions change, so the
version is configurable (META_GRAPH_API_VERSION).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from dsdvc.config.settings import Settings, get_settings
from dsdvc.core.provenance import DataUnavailable, Provenance, SourceType, EpistemicStatus, unavailable
from dsdvc.platforms.base import Connector

GRAPH = "https://graph.facebook.com"


class InstagramGraphConnector(Connector):
    name = "instagram_graph_api"

    def __init__(self, settings: Settings | None = None, **kw):
        super().__init__(**kw)
        self.s = settings or get_settings()

    @property
    def ready(self) -> bool:
        return bool(self.s.meta_access_token and self.s.instagram_business_account_id)

    def _need(self, what: str) -> DataUnavailable | None:
        if not self.ready:
            return unavailable(what, "Instagram Graph API credentials not configured",
                               "Set META_ACCESS_TOKEN and INSTAGRAM_BUSINESS_ACCOUNT_ID for a professional account you manage, "
                               "or upload an authorised export.")
        return None

    def _url(self, path: str) -> str:
        return f"{GRAPH}/{self.s.graph_api_version}/{path}"

    def own_media(self, limit: int = 50) -> pd.DataFrame | DataUnavailable:
        if (u := self._need("instagram.own_media")):
            return u
        fields = "id,caption,media_type,media_product_type,timestamp,like_count,comments_count,permalink"
        data = self.get(self._url(f"{self.s.instagram_business_account_id}/media"),
                        {"fields": fields, "limit": limit, "access_token": self.s.meta_access_token})
        if isinstance(data, DataUnavailable):
            return data
        df = pd.DataFrame(data.get("data", []))
        if df.empty:
            return unavailable("instagram.own_media", "no media returned")
        df = df.rename(columns={"id": "content_id", "caption": "text", "like_count": "likes",
                                "comments_count": "comments", "media_type": "media_type"})
        df["platform"] = "instagram"
        df["source"], df["source_type"] = self.name, SourceType.AUTHORIZED_API.value
        df["authorization_scope"] = "instagram_basic,instagram_manage_insights (own account)"
        df["observed_or_estimated"] = "observed"
        df["collected_at"] = datetime.now(timezone.utc)
        return df

    def media_insights(self, media_id: str, metrics: str = "reach,saved,shares,views") -> dict | DataUnavailable:
        if (u := self._need("instagram.media_insights")):
            return u
        data = self.get(self._url(f"{media_id}/insights"), {"metric": metrics, "access_token": self.s.meta_access_token})
        if isinstance(data, DataUnavailable):
            return data
        return {d["name"]: d["values"][0]["value"] for d in data.get("data", []) if d.get("values")}

    def business_discovery(self, username: str) -> dict | DataUnavailable:
        """Public fields of another *professional* account (official endpoint). Private/personal
        accounts are not accessible and are never attempted."""
        if (u := self._need("instagram.business_discovery")):
            return u
        f = (f"business_discovery.username({username}){{username,followers_count,follows_count,media_count,"
             f"media.limit(25){{caption,like_count,comments_count,timestamp,media_type}}}}")
        data = self.get(self._url(self.s.instagram_business_account_id), {"fields": f, "access_token": self.s.meta_access_token})
        if isinstance(data, DataUnavailable):
            return data
        bd = data.get("business_discovery")
        if not bd:
            return unavailable(f"business_discovery:{username}", "account not found or not a professional account")
        bd["_provenance"] = self.prov("graph_api.business_discovery", "public professional-account fields").model_dump(mode="json")
        return bd

    def hashtag_top_media(self, hashtag: str) -> pd.DataFrame | DataUnavailable:
        if (u := self._need("instagram.hashtag")):
            return u
        h = self.get(self._url("ig_hashtag_search"), {"user_id": self.s.instagram_business_account_id, "q": hashtag,
                                                       "access_token": self.s.meta_access_token})
        if isinstance(h, DataUnavailable) or not h.get("data"):
            return h if isinstance(h, DataUnavailable) else unavailable(f"hashtag:{hashtag}", "hashtag not found")
        hid = h["data"][0]["id"]
        data = self.get(self._url(f"{hid}/top_media"), {"user_id": self.s.instagram_business_account_id,
                                                         "fields": "id,caption,media_type,timestamp,like_count,comments_count",
                                                         "access_token": self.s.meta_access_token})
        if isinstance(data, DataUnavailable):
            return data
        df = pd.DataFrame(data.get("data", [])).rename(columns={"id": "content_id", "caption": "text",
                                                                 "like_count": "likes", "comments_count": "comments"})
        df["platform"], df["source"], df["observed_or_estimated"] = "instagram", self.name, "observed"
        df["note"] = "Hashtag media exposes no author identity via this endpoint; aggregate analysis only."
        return df


class FacebookPageConnector(Connector):
    name = "facebook_pages_api"

    def __init__(self, settings: Settings | None = None, **kw):
        super().__init__(**kw)
        self.s = settings or get_settings()

    def page_posts(self, limit: int = 50) -> pd.DataFrame | DataUnavailable:
        if not (self.s.meta_access_token and self.s.facebook_page_id):
            return unavailable("facebook.page_posts", "Facebook Page credentials not configured",
                               "Set META_ACCESS_TOKEN (page token) and FACEBOOK_PAGE_ID for a Page you manage.")
        data = self.get(f"{GRAPH}/{self.s.graph_api_version}/{self.s.facebook_page_id}/posts",
                        {"fields": "id,message,created_time,shares,reactions.summary(true),comments.summary(true)",
                         "limit": limit, "access_token": self.s.meta_access_token})
        if isinstance(data, DataUnavailable):
            return data
        rows = []
        for p in data.get("data", []):
            rows.append({"content_id": p["id"], "text": p.get("message", ""), "timestamp": p.get("created_time"),
                         "shares": (p.get("shares") or {}).get("count", 0),
                         "likes": ((p.get("reactions") or {}).get("summary") or {}).get("total_count"),
                         "comments": ((p.get("comments") or {}).get("summary") or {}).get("total_count"),
                         "platform": "facebook", "source": self.name, "observed_or_estimated": "observed"})
        return pd.DataFrame(rows)


class MetaContentLibraryConnector(Connector):
    name = "meta_content_library"
    source_type = SourceType.PUBLIC_DATASET

    def search(self, *_a, **_k) -> DataUnavailable:
        return unavailable("meta_content_library.search",
                           "Meta Content Library is accessed only by approved researchers inside Meta's secure research environment",
                           "Apply via Meta's research programme; then export aggregate results and load them with "
                           "import_content_library_export().")

    def import_content_library_export(self, path: str, eligibility_reference: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["source"], df["source_type"] = self.name, self.source_type.value
        df["authorization_scope"] = f"Meta Content Library approved access ({eligibility_reference})"
        df["observed_or_estimated"] = "observed"
        return df


class WhatsAppBusinessConnector(Connector):
    name = "whatsapp_business_platform"

    def __init__(self, settings: Settings | None = None, **kw):
        super().__init__(**kw)
        self.s = settings or get_settings()

    def message_analytics(self, start: datetime, end: datetime, granularity: str = "DAY") -> pd.DataFrame | DataUnavailable:
        """Aggregated sent/delivered counts for the business's own WABA (no message content)."""
        if not (self.s.whatsapp_business_token and self.s.whatsapp_business_account_id):
            return unavailable("whatsapp.analytics", "WhatsApp Business Platform credentials not configured",
                               "Set WHATSAPP_BUSINESS_TOKEN and WHATSAPP_BUSINESS_ACCOUNT_ID, or upload survey-derived "
                               "forwarding data. Private conversations are never accessed.")
        f = f"analytics.start({int(start.timestamp())}).end({int(end.timestamp())}).granularity({granularity})"
        data = self.get(f"{GRAPH}/{self.s.graph_api_version}/{self.s.whatsapp_business_account_id}",
                        {"fields": f, "access_token": self.s.whatsapp_business_token})
        if isinstance(data, DataUnavailable):
            return data
        pts = (data.get("analytics") or {}).get("data_points", [])
        df = pd.DataFrame(pts)
        df["platform"], df["source"], df["observed_or_estimated"] = "whatsapp", self.name, "observed"
        df["authorization_scope"] = "whatsapp_business_management (own WABA, aggregate only)"
        return df


def load_user_export(path: str, platform: str, owner_attestation: str) -> pd.DataFrame:
    """User-provided export (CSV in the DSDVC content schema, or the user's own platform data download).
    ``owner_attestation`` records that the uploader is authorised to use the data."""
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_json(path)
    df["platform"] = df.get("platform", platform)
    df["source"], df["source_type"] = f"user_export:{path.split('/')[-1]}", SourceType.USER_PROVIDED.value
    df["authorization_scope"] = owner_attestation
    df["observed_or_estimated"] = "observed"
    df["collected_at"] = datetime.now(timezone.utc)
    return df


def survey_forwarding_rates(survey: pd.DataFrame) -> pd.DataFrame:
    """WhatsApp forwarding behaviour from survey data (columns: respondent_id, would_forward (0/1),
    group_size, trust_in_sender 1-5, content_type). Returns ESTIMATED rates with Wilson intervals."""
    from scipy import stats
    z = stats.norm.ppf(0.95)
    g = survey.groupby("content_type")["would_forward"].agg(["sum", "count"])
    p = g["sum"] / g["count"]
    n = g["count"]
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return pd.DataFrame({"forward_rate": p, "ci_low": centre - half, "ci_high": centre + half, "n": n,
                         "status": EpistemicStatus.ESTIMATED.value, "source": "survey"})
