# Data card

## Sources the platform accepts
| Source | Access path | What it can provide | What it cannot |
|---|---|---|---|
| Instagram Graph API | token for a professional account you manage | own media, insights; `business_discovery` public fields of other professional accounts; hashtag media (30 unique hashtags / 7 days) | private/personal accounts, follower identities, other accounts' insights |
| Facebook Pages API | Page token | posts, reactions, comments, shares for Pages you manage | private profiles, private groups |
| Meta Content Library | approved researchers inside Meta's environment | exports of public content analyses | direct API access from DSDVC |
| WhatsApp Business Platform | WABA token | aggregate message analytics for your own business | message content, personal chats, contact graphs |
| User exports / CSV | upload with authorisation attestation | anything the uploader is entitled to | — |
| Surveys | upload | forwarding intent, trust, group norms | behavioural ground truth |
| Market tables | upload with citations | market size, penetration, pricing | — |
| SYNTHETIC generators | explicit opt-in | test/demo data with known ground truth | anything about real people |

## Provenance schema (every observation)
`source, source_type, timestamp, collection_method, authorization_scope, confidence,
observed_or_estimated` (+ `data_version`, `model_version`). Stored in the `observations` table
and attached to every connector output. Observed and synthetic records cannot be combined
without an explicit opt-in (`assert_not_mixed`).

## Expected schemas
* **Content**: `content_id, platform, creator_id, timestamp, text, hashtags, media_type, views,
  likes, comments, shares, saves` (+ optional `followers, conversions, retained_90d,
  shares_1h…shares_24h` for early-velocity models).
* **Creators**: `creator_id, platform, followers, following, engagement_rate, posts_last_30d,
  niche, audience_quality`.
* **Daily metrics**: `date, adoptions, engagement, shares, conversions, churn, revenue`.
* **Customer panel**: `awareness, social_proof, trust, price_fit, stepps, community_fit,
  product_fit, platform, converted, churned, months_active, monthly_revenue`.
* **Touchpoint paths**: `user_id, path (ordered channels), converted`.

## Quality checks (DataQualityEngine)
missingness · duplicates · bot-like activity · outliers · API gaps · sampling bias (vs a
reference mix) · survivorship · platform measurement differences · temporal mismatch →
`DataQualityScore ∈ [0, 1]`, propagated into prediction confidence.

## Synthetic data
`data/synthetic.py` plants documented ground truth (`TRUE_SHARE_BETA`, emotion×public
interaction, platform-specific conversion/trust effects, channel lifts) so tests can verify that
estimators recover them. Synthetic frames carry `data_status = "synthetic"`, and every UI and
report surface shows a SYNTHETIC badge.
