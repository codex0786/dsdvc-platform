"""Data templates (data contracts) that clients fill in or that the mapper targets.

Each template lists canonical columns with type, requirement, unit, description and the
synonyms we recognise in client exports. ``spend_*`` style channel columns are pattern-based:
any number of marketing channels can be supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ColumnSpec:
    name: str
    dtype: str                  # date | number | integer | boolean | string | category
    required: bool
    description: str
    unit: str = ""
    synonyms: tuple[str, ...] = ()
    min: float | None = None
    max: float | None = None
    example: object = None


@dataclass
class DatasetSpec:
    key: str
    title: str
    grain: str
    purpose: str
    columns: list[ColumnSpec]
    channel_pattern: str | None = None      # e.g. "spend_{channel}"
    min_rows: int = 1
    recommended_rows: int = 1
    one_of: list[list[str]] = field(default_factory=list)  # at least one column from each group

    def column(self, name: str) -> ColumnSpec | None:
        return next((c for c in self.columns if c.name == name), None)

    def dictionary(self) -> pd.DataFrame:
        rows = [{"column": c.name, "type": c.dtype, "required": "yes" if c.required else "optional", "unit": c.unit,
                 "description": c.description, "example": c.example} for c in self.columns]
        if self.channel_pattern:
            rows.append({"column": self.channel_pattern, "type": "number", "required": "at least one", "unit": "currency",
                         "description": "Spend per marketing channel per period, one column per channel "
                                        "(e.g. spend_meta_ads, spend_google_search, spend_influencers).",
                         "example": 250000})
        out = pd.DataFrame(rows)
        out["example"] = out["example"].map(lambda v: "" if v is None else str(v))
        return out


CHANNEL_ALIASES = {
    "meta_ads": ("meta", "facebook", "fb", "instagram", "ig", "fb_ig", "fbig"),
    "google_search": ("google_search", "search", "sem", "adwords", "google_ads", "paid_search", "ppc"),
    "youtube": ("youtube", "yt", "video"),
    "influencers": ("influencer", "influencers", "creator", "creators", "kol"),
    "ooh": ("ooh", "hoarding", "hoardings", "billboard", "outdoor"),
    "tv": ("tv", "television", "ctv", "ott"),
    "print": ("print", "newspaper", "magazine"),
    "radio": ("radio", "fm"),
    "whatsapp": ("whatsapp", "wa"),
    "email": ("email", "mail", "edm"),
    "affiliate": ("affiliate", "affiliates", "partner"),
    "display": ("display", "programmatic", "dv360", "banner"),
}
SPEND_TOKENS = ("spend", "cost", "budget", "investment", "amount_spent", "payments", "payment", "media", "inr", "rs", "rupees")

MARKETING_WEEKLY = DatasetSpec(
    key="marketing_weekly", title="Weekly marketing & sales", grain="one row per week (optionally per geo)",
    purpose="Marketing mix model: channel ROI, response curves, budget plans, back-tests.",
    columns=[
        ColumnSpec("date", "date", True, "Week start date", synonyms=("week", "week_start", "week_starting", "week_commencing",
                                                                       "period", "date", "wk", "week_beginning"), example="2025-01-06"),
        ColumnSpec("revenue", "number", False, "Sales revenue in the week", "currency",
                   synonyms=("revenue", "sales", "total_sales", "net_sales", "gmv", "turnover", "income"), min=0, example=6200000),
        ColumnSpec("conversions", "number", False, "Orders / sign-ups / units in the week", "count",
                   synonyms=("conversions", "orders", "units", "signups", "sign_ups", "purchases", "transactions"), min=0, example=4100),
        ColumnSpec("price_index", "number", False, "Average price relative to normal (1.0 = list price)", "index",
                   synonyms=("price", "price_index", "avg_price", "average_price", "asp", "selling_price"), min=0, example=1.0),
        ColumnSpec("promo", "number", False, "Promotion running (0/1 or discount depth)", "flag",
                   synonyms=("promo", "promotion", "discount", "sale", "offer", "promo_week"), min=0, example=0),
        ColumnSpec("holiday", "number", False, "Festival / holiday week (0/1)", "flag",
                   synonyms=("holiday", "festival", "festive", "event"), min=0, example=0),
        ColumnSpec("competitor_spend", "number", False, "Estimated competitor media spend", "currency",
                   synonyms=("competitor", "competition", "competitor_spend", "share_of_voice"), min=0),
        ColumnSpec("geo", "string", False, "Region / city (for geo-level data)", synonyms=("geo", "region", "city", "state", "market")),
    ],
    channel_pattern="spend_{channel}", min_rows=26, recommended_rows=104, one_of=[["revenue", "conversions"]],
)

LIFT_TESTS = DatasetSpec(
    key="lift_tests", title="Experiment (lift test) results", grain="one row per experiment",
    purpose="Calibrates the MMM with causal evidence (geo-lift or platform conversion-lift studies).",
    columns=[
        ColumnSpec("channel", "string", True, "Channel tested (must match a spend_ column name without the prefix)",
                   synonyms=("channel", "media", "platform"), example="youtube"),
        ColumnSpec("start", "date", True, "Test start date", synonyms=("start", "start_date", "from"), example="2025-03-03"),
        ColumnSpec("end", "date", True, "Test end date", synonyms=("end", "end_date", "to"), example="2025-04-27"),
        ColumnSpec("lift", "number", True, "Incremental outcome caused by the channel during the test", "outcome units",
                   synonyms=("lift", "incremental", "incremental_revenue", "incremental_sales", "effect"), example=1400000),
        ColumnSpec("se", "number", True, "Standard error of the lift estimate", "outcome units",
                   synonyms=("se", "std_error", "standard_error", "stderr"), min=0, example=210000),
    ],
)

TEMPLATES: dict[str, DatasetSpec] = {s.key: s for s in (MARKETING_WEEKLY, LIFT_TESTS)}


def template_frame(key: str, rows: int = 3) -> pd.DataFrame:
    """Example CSV a client can fill in."""
    spec = TEMPLATES[key]
    if key == "marketing_weekly":
        dates = pd.date_range("2025-01-06", periods=rows, freq="W-MON").strftime("%Y-%m-%d")
        return pd.DataFrame({"date": dates, "revenue": [6200000, 5900000, 6450000][:rows],
                             "conversions": [4100, 3950, 4300][:rows],
                             "spend_meta_ads": [400000, 380000, 420000][:rows], "spend_google_search": [300000, 310000, 295000][:rows],
                             "spend_influencers": [150000, 0, 175000][:rows], "price_index": [1.0, 1.0, 0.95][:rows],
                             "promo": [0, 0, 1][:rows], "holiday": [0, 0, 0][:rows]})
    return pd.DataFrame([{c.name: c.example for c in spec.columns}] * min(rows, 1))
