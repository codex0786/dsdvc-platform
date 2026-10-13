# Connectors: Meta Ads, Google Analytics 4, Shopify

Pull two-plus years of paid-media spend and sales directly from the client's accounts into the
weekly MMM dataset: **Client workflow → 2 · Data → Pull from connected accounts** (or
`POST /connectors/pull`). Pulled data goes through the same validation and authorisation gate as
uploads.

| Source | What we pull | Used as |
|---|---|---|
| **Meta Ads** (Marketing API) | Daily spend + impressions per campaign / ad set / platform / region | `spend_*` channels (campaign-name rules split e.g. influencer campaigns into their own channel) |
| **GA4** (Data API) | Daily sessions, key events, transactions, purchase/total revenue; linked Google Ads cost | Google Ads spend (`spend_google_ads`); fallback outcome; tracking reconciliation |
| **Shopify** (Admin GraphQL) | Order-level totals, discounts, refunds, cancelled/test flags (no customer data) | Outcome of record: weekly net sales, orders; `promo` = discount depth |

Credentials are read from **environment variables or Streamlit secrets**, never typed into the UI
and never stored in the database. Every connector returns `DATA_UNAVAILABLE` (with the fix) instead of
failing silently, and the other sources still load.

## Setup

### Meta Ads
1. In Meta Business Settings, create a **System User** and assign it the client's **ad account**
   (view/analyse permission is enough).
2. Generate a system-user token for your Meta app with the **`ads_read`** permission.
3. Set:
   ```
   META_ACCESS_TOKEN=…
   META_AD_ACCOUNT_ID=1234567890        # the number after act_ in Ads Manager
   META_GRAPH_API_VERSION=v26.0         # current as of July 2026
   ```
**Export mode.** The default `mmm` mode uses Meta's official
[Marketing Mix Modeling breakdown](https://developers.facebook.com/documentation/ads-commerce/marketing-api/insights/marketing-mix-modeling):
an async CSV export at ad-set level with country/region/DMA, the input format Meta designed for
MMM. Meta marks its spend as estimated. `insights` mode uses standard campaign Insights (with campaign
names). Ranges are pulled in 30-day chunks; the `x-fb-ads-insights-throttle` and ad-account usage
headers are honoured.

### Google Analytics 4
1. In Google Cloud, enable the **Google Analytics Data API** and create a **service account**; create a JSON key.
2. In GA4 **Admin → Property access management**, add the service account's email as **Viewer**.
3. Set the property ID (Admin → Property settings) and the key, as a file *or* as a secret string:
   ```
   GA4_PROPERTY_ID=123456789
   GOOGLE_APPLICATION_CREDENTIALS=/path/key.json      # servers / Docker
   GA4_SERVICE_ACCOUNT_JSON='{"type": "service_account", …}'   # Streamlit Cloud secrets
   ```
   (`GA4_ACCESS_TOKEN` with a short-lived OAuth token also works for one-off pulls.)

Metric names follow the current GA4 schema: `keyEvents` (formerly "conversions"), `purchaseRevenue`,
`totalRevenue`, `transactions`, `advertiserAdCost` ([API schema](https://developers.google.com/analytics/devguides/reporting/data/v1/api-schema)).
Google Ads cost appears only if Google Ads is linked to the property. **Channel-group sessions are
returned for diagnostics only**: organic/direct traffic is itself caused by marketing, so it must
not be used as an MMM control.

### Shopify
Since **1 January 2026, new "legacy" custom apps can no longer be created in the store admin**; existing
ones keep working ([Shopify announcement](https://community.shopify.dev/t/starting-january-1-2026-you-will-not-be-able-to-create-new-legacy-custom-apps-this-will-not-impact-any-existing-apps/26798)).
The connector supports both:

* **New app (Dev Dashboard).** Create the app, give it the **`read_orders`** Admin API scope, install it on the
  store, and set the client ID and secret. The connector exchanges them for an Admin API token with
  the client-credentials grant and refreshes it before expiry (community reports say these tokens last about 24 h).
  ```
  SHOPIFY_SHOP=mybrand.myshopify.com
  SHOPIFY_CLIENT_ID=…
  SHOPIFY_CLIENT_SECRET=…
  ```
* **Existing legacy custom app.** Set its static token instead: `SHOPIFY_ACCESS_TOKEN=shpat_…`.

**History beyond 60 days needs `read_all_orders`**, which Shopify must approve via an access
request in the Partner/Dev Dashboard, after which the app must be re-authorised so the scope lands on the
token ([Shopify changelog](https://shopify.dev/changelog/apps-now-need-shopify-approval-to-read-orders-older-than-60-days)).
The connector warns when the earliest order returned suggests that scope is missing.
Developer reports differ on the exact token and approval flow for Dev Dashboard apps, so confirm in the
client's Dev Dashboard during onboarding.

Long ranges use a **bulk operation** (polled via `node(id:)`; `currentBulkOperation` is deprecated
from API 2026-01, see [bulk queries](https://shopify.dev/docs/api/usage/bulk-operations/queries)); short
ranges paginate. The GraphQL cost budget and `THROTTLED` errors are respected. Orders are bucketed into
days in the **shop's time zone**; test and cancelled orders are excluded; net sales = total − refunds.

### Streamlit Cloud secrets example
App → **Settings → Secrets** (TOML):
```toml
META_ACCESS_TOKEN = "…"
META_AD_ACCOUNT_ID = "1234567890"
GA4_PROPERTY_ID = "123456789"
GA4_SERVICE_ACCOUNT_JSON = '''{"type": "service_account", "project_id": "…", "private_key": "…", "client_email": "…"}'''
SHOPIFY_SHOP = "mybrand.myshopify.com"
SHOPIFY_CLIENT_ID = "…"
SHOPIFY_CLIENT_SECRET = "…"
```

## How the weekly dataset is assembled
* Weeks run **Monday–Sunday**; partial weeks at either end are dropped. Each source is bucketed in its
  own time zone (ad account, GA4 property, shop). Record those time zones in the engagement notes.
* **Outcome of record:** Shopify net sales (orders as conversions). GA4 revenue is used only if Shopify
  isn't connected, with a warning that web tracking under-counts sales.
* **Reconciliation:** GA4 revenue ÷ Shopify revenue per week. A median below 70% flags a tracking gap.
  Swings of more than 20% across weeks flag tracking changes. Both mean platform-reported ROAS is biased.
* **Channels without an API** (TV, OOH, print, influencer invoices): upload `date, spend` CSVs, one per
  channel, alongside the pull.

## Demo mode
With no credentials, switch on **Demo accounts (SYNTHETIC)**. Simulators shaped like the three APIs
(async report jobs, paging, bulk JSONL, throttling) run the real connector code on a synthetic brand with
planted effects. Everything pulled in demo mode is labelled SYNTHETIC.

## Verification status
The connectors are tested against those simulators (`tests/test_connectors_mmm.py`): async jobs and
failures, pagination, bulk export, throttling, token exchange and refresh, authorisation errors, time
zones, exclusion of test/cancelled orders, reconstruction of the planted weekly data, and an end-to-end MMM
fit. **They have not yet been run against live client accounts.** Do a supervised first pull with
each new client and compare totals against Ads Manager, GA4 and Shopify reports.
