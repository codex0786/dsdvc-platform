# Ethics & privacy

DSDVC is a marketing-intelligence system for **aggregate, community-level** analysis.

## Hard rules (enforced in code where possible)
| Rule | Enforcement |
|---|---|
| No scraping of private accounts or private WhatsApp conversations | connectors call only official endpoints for authorised assets; there is no scraper |
| No bypassing authentication, rate limits or access controls | 401/403 → `DATA_UNAVAILABLE` (no retries with other credentials); 429/usage headers → compliant back-off (tested) |
| No improper collection of passwords/tokens | tokens read from environment/secret store only; DB stores secret references; settings redact secrets |
| No inference of sensitive personal attributes | no such features or models exist; audience definitions come from the user's brief |
| No identification of private individuals | synthetic agents are not people; creator discovery covers public professional accounts or user-supplied, authorised lists |
| No fabrication of unavailable data | `DATA_UNAVAILABLE` sentinel; synthetic data only on explicit opt-in and always labelled; observed + synthetic mixing guarded |
| No overclaiming | "will go viral" language is tested against; LLM narration cannot introduce numbers absent from computed evidence |

## Do not use DSDVC to
infer sensitive characteristics · target vulnerable individuals · manipulate individuals using
sensitive traits · access private communications · bypass platform controls · deanonymise users ·
scrape protected content.

## Content and claims
The Copilot flags claim risks (e.g. "cure", "guaranteed", "fairness/whitening") that may breach
advertising standards and platform policies. Recommendations favour long-term customer value
over vanity metrics, and churn and retention are first-class outcomes.

## Experimentation
Field experiments (holdouts, A/B tests) should use platform-sanctioned tooling, respect user
consent, and be reviewed by the organisation's ethics/compliance process where required.
