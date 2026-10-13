# Client onboarding guide

For the client's marketing or data team, and the analyst running the engagement.

## The engagement flow

| Step | Who | What happens | Output |
|---|---|---|---|
| 1 · Brief | Analyst + client lead | Agree the decision, outcome metric, horizon, budget and margin | Decision brief (dashboard step 1) |
| 2 · Data | Client data team → analyst | Export weekly data; the platform maps columns, validates, records authorisation | Accepted dataset with SHA-256, scorecard, attestation |
| 3 · Model | Analyst | Fit the MMM; run the back-test on hidden recent weeks | Back-test verdict, ROI with intervals, response curves |
| 4 · Plan | Analyst + client lead | Optimise the weekly budget under limits; review budget scenarios | Recommended mix, expected gain and probability of improvement |
| 5 · Export | Analyst | Download the Excel media plan and executive deck; log the run | `.xlsx` + `.pptx` + run record |
| 6 · Prove | Client + analyst | Geo-lift / conversion-lift test on the biggest recommended shift | Lift result, uploaded as a calibration input |

## What data to send

One row per **week** (Monday week-start preferred), ideally **104+ weeks** (minimum 26).

| Column | Required | Notes |
|---|---|---|
| date | yes | Week start. `YYYY-MM-DD` or `DD/MM/YYYY` both work |
| revenue *or* conversions | yes (one of them) | The outcome to grow |
| one spend column per channel | at least one | e.g. Meta, Google Search, YouTube, influencers, OOH, TV. Use the money actually spent that week |
| price_index | recommended | Average price ÷ list price (1.0 = no discount) |
| promo, holiday | recommended | 1 in promotion / festival weeks |
| competitor_spend | optional | If you have share-of-voice estimates |

Download the template and data dictionary from step 2, or send your existing export: names like
"FB + IG Spend (₹)", "Google Search Cost" or "Total Sales (INR)" are mapped automatically, and
values like `₹1,20,000`, `2.5 Cr` or `12 lakh` are parsed. Long tables (one row per week per channel)
are pivoted automatically.

**Do not send personal data.** Aggregated weekly totals are all the model needs.

## Validation gate

| Status | Meaning |
|---|---|
| READY | All checks pass |
| READY_WITH_WARNINGS | Usable; warnings explain what limits accuracy (e.g. a channel whose spend never varies) |
| BLOCKED | Must be fixed: missing outcome or spend, unreadable dates, duplicate weeks, negative spend, < 26 weeks |

A dataset is stored only with an **authorisation note** (who provided it, under what agreement).
That note is printed in every export.

## Lift tests (optional, strongly recommended)

Experiments correct what history alone cannot separate. Upload one row per test: `channel, start,
end, lift, se` (incremental outcome and its standard error). The model then agrees with the experiment
within its stated uncertainty.

## Deliverables

* **Media plan (Excel)**: summary, channel plan, weekly schedule, ROI with intervals, response curves,
  model accuracy with chart, data & assumptions.
* **Executive deck (PowerPoint, editable native charts)**: summary, drivers, ROI, recommended mix,
  expected impact, accuracy, caveats and next experiments, data & method appendix.

## API

The same flow runs headless (see [API.md](API.md)): `POST /onboarding/preview` → `POST /onboarding/datasets`
→ `POST /mmm/fit` → `POST /mmm/{run_id}/plan` → `GET /mmm/{run_id}/export/xlsx|pptx`.
