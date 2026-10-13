"""Client workflow: Brief → Data → Model → Plan → Export.

The guided path an analyst follows for a client engagement. Each step gates the next:
no model without validated, attested data; no plan without a back-test verdict on screen.
"""
from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dsdvc.dashboard.app import BADGE, SERIES, badge, style
from dsdvc.onboarding.mapping import IGNORE, Mapping, apply_mapping, suggest_mapping
from dsdvc.onboarding.templates import CHANNEL_ALIASES, LIFT_TESTS, MARKETING_WEEKLY, template_frame
from dsdvc.onboarding.validate import spend_columns, validate

STATUS_COLOR = {"READY": BADGE["OBSERVED"], "READY_WITH_WARNINGS": BADGE["RECOMMENDED"], "BLOCKED": BADGE["DATA_UNAVAILABLE"]}
CHECK_ICON = {"pass": "✅", "warn": "⚠️", "fail": "⛔"}


def _state() -> dict:
    return st.session_state.setdefault("wf", {})


def _read_any(f) -> pd.DataFrame:
    name = f.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(f, dtype=str)
    raw = f.getvalue()
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), dtype=str, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw), dtype=str, encoding_errors="replace")


@st.cache_resource(show_spinner=False)
def _fit(sha: str, target: str, channels: tuple, controls: tuple, holdout: int, lift_key: str, _df: pd.DataFrame, _lifts):
    from dsdvc.mmm import analysis as A
    from dsdvc.mmm.model import MarketingMixModel, MMMSpec
    spec = MMMSpec(target_col=target, channels=list(channels), controls=list(controls))
    model = MarketingMixModel(spec)
    fit = model.fit(_df, _lifts, data_version=sha[:16])
    bt = A.backtest(MarketingMixModel(spec), _df, holdout, _lifts) if holdout else None
    return fit, bt


@st.cache_data(show_spinner=False)
def _summaries(run_key: str, _fit):
    from dsdvc.mmm import analysis as A
    return A.channel_summary(_fit), A.channel_params(_fit), A.response_curves(_fit), A.decomposition(_fit)


def _money(v, cur):
    return f"{cur}{v:,.0f}"


def workflow_page():
    st.title("Client workflow")
    st.caption("Brief → Data → Model → Plan → Export. Each step must pass before the next one unlocks.")
    W = _state()
    tabs = st.tabs(["1 · Brief", "2 · Data", "3 · Model", "4 · Plan", "5 · Export"])

    # ------------------------------------------------------------------ 1 brief
    with tabs[0]:
        st.subheader("Decision brief")
        c1, c2 = st.columns(2)
        W["client"] = c1.text_input("Client / brand", W.get("client", "Demo Skincare Co."))
        W["decision"] = c1.text_area("Decision to support", W.get("decision", "How should we split next quarter's media "
                                                                      "budget across channels to maximise incremental revenue?"))
        W["currency"] = c2.selectbox("Currency symbol", ["₹", "$", "€", "£"], index=["₹", "$", "€", "£"].index(W.get("currency", "₹")))
        W["horizon"] = c2.number_input("Planning horizon (weeks)", 4, 52, int(W.get("horizon", 13)))
        W["start"] = c2.date_input("Plan starts", W.get("start", dt.date.today() + dt.timedelta(days=(7 - dt.date.today().weekday()) % 7 or 7)))
        W["margin"] = c2.slider("Gross margin on the outcome (for profit view)", 0.0, 1.0, float(W.get("margin", 0.6)), 0.05,
                                help="Used only when the planning objective is profit.")
        st.info("Success metric: **incremental** outcome versus the current mix at the same budget, confirmed by an "
                "experiment before large shifts are scaled.")

    # ------------------------------------------------------------------ 2 data
    with tabs[1]:
        st.subheader("Weekly marketing & sales data")
        c1, c2 = st.columns([2, 1])
        with c2:
            st.download_button("⬇ Template (CSV)", template_frame("marketing_weekly").to_csv(index=False), "dsdvc_marketing_weekly_template.csv")
            st.download_button("⬇ Data dictionary", MARKETING_WEEKLY.dictionary().to_csv(index=False), "dsdvc_data_dictionary.csv")
            if st.button("Load sample client file (SYNTHETIC)"):
                from dsdvc.mmm.synthetic import messy_client_export
                W["raw"], W["raw_name"], W["raw_synthetic"] = messy_client_export(), "sample_client_export.csv", True
                W.pop("accepted", None)
        with c1:
            f = st.file_uploader("Upload the client's weekly export (CSV or Excel)", type=["csv", "xlsx", "xls"], key="wf_upload")
            if f is not None and f.name != W.get("raw_name"):
                W["raw"], W["raw_name"], W["raw_synthetic"] = _read_any(f), f.name, False
                W.pop("accepted", None)
        if "raw" not in W:
            st.info("Upload a file or load the sample to continue.")
        else:
            raw = W["raw"]
            if W.get("raw_synthetic"):
                badge("SYNTHETIC", "sample file with planted effects — not client data")
            st.write(f"**{W['raw_name']}** · {len(raw)} rows · {raw.shape[1]} columns")
            st.dataframe(raw.head(8), width="stretch")
            st.markdown("**Column mapping** — check the suggestions; change any target that is wrong.")
            if W.get("mapping_for") != W["raw_name"]:
                W["mapping"] = suggest_mapping(raw, MARKETING_WEEKLY).table()
                W["mapping_for"] = W["raw_name"]
            options = sorted({c.name for c in MARKETING_WEEKLY.columns} | {f"spend_{k}" for k in CHANNEL_ALIASES}
                             | set(W["mapping"].target)) + [IGNORE]
            edited = st.data_editor(W["mapping"], width="stretch", hide_index=True, disabled=["source", "confidence", "reason"],
                                    column_config={"target": st.column_config.SelectboxColumn("target", options=options, required=True),
                                                   "confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1)})
            mapping = Mapping.from_table("marketing_weekly", edited)
            canon, notes = apply_mapping(raw, mapping, MARKETING_WEEKLY)
            for n in notes:
                st.caption(n)
            sc = validate(canon, MARKETING_WEEKLY)
            st.markdown(f"<span style='background:{STATUS_COLOR[sc.status]};color:white;padding:3px 10px;border-radius:4px;"
                        f"font-weight:600'>{sc.status}</span> &nbsp; data-quality score {sc.score:.0%}", unsafe_allow_html=True)
            tab = sc.table()
            tab.insert(0, "", tab.status.map(CHECK_ICON))
            st.dataframe(tab.drop(columns="status"), width="stretch", hide_index=True)
            st.dataframe(canon.head(8), width="stretch")
            att = st.text_input("Authorisation", W.get("attestation", ""),
                                placeholder="e.g. 'Provided by client marketing team under MSA dated …; contains no personal data'",
                                help="Required. Recorded with the dataset and printed in every export.")
            if st.button("Accept dataset", type="primary", disabled=sc.status == "BLOCKED" or not att.strip()):
                from dsdvc.database.models import SessionLocal, init_db
                from dsdvc.onboarding.workspace import save_dataset
                init_db()
                with SessionLocal() as db:
                    row = save_dataset(db, df=canon, dataset="marketing_weekly", filename=W["raw_name"], mapping=mapping,
                                       scorecard=sc, attestation=att)
                    W["accepted"] = {"df": canon, "sha": row.sha256, "rows": len(canon), "status": sc.status, "id": row.id,
                                     "filename": W["raw_name"], "synthetic": W.get("raw_synthetic", False)}
                W["attestation"] = att
                for k in ("fit_key", "plan"):
                    W.pop(k, None)
            if sc.status == "BLOCKED":
                st.error("Fix the ⛔ checks (mapping or source file) before the data can be accepted.")
            if "accepted" in W:
                st.success(f"Accepted · {W['accepted']['rows']} weeks · SHA-256 {W['accepted']['sha'][:16]}…")
        with st.expander("Optional: lift-test results (calibrates the model with experiments)"):
            st.dataframe(LIFT_TESTS.dictionary(), width="stretch", hide_index=True)
            lf = st.file_uploader("Lift tests CSV (channel, start, end, lift, se)", type=["csv"], key="wf_lifts")
            if lf is not None:
                lt = pd.read_csv(lf)
                missing = [c for c in ("channel", "start", "end", "lift", "se") if c not in lt]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    W["lifts"] = lt
                    st.dataframe(lt, width="stretch")

    # ------------------------------------------------------------------ 3 model
    with tabs[2]:
        if "accepted" not in W:
            st.info("Accept a dataset in step 2 first.")
        else:
            df = W["accepted"]["df"]
            targets = [t for t in ("revenue", "conversions") if t in df]
            c1, c2, c3 = st.columns(3)
            target = c1.selectbox("Outcome", targets)
            channels = c2.multiselect("Channels", spend_columns(df), default=spend_columns(df))
            ctrl_opts = [c for c in ("price_index", "promo", "holiday", "competitor_spend") if c in df]
            controls = c3.multiselect("Controls", ctrl_opts, default=ctrl_opts)
            holdout = st.slider("Back-test holdout (weeks hidden from the model)", 0, 26, 12 if len(df) >= 78 else 8,
                                help="The model is refitted without these weeks and must predict them from spend alone.")
            lifts = []
            if "lifts" in W:
                from dsdvc.mmm.model import LiftTest
                lifts = [LiftTest(f"spend_{r.channel}" if not str(r.channel).startswith("spend_") else r.channel,
                                  pd.Timestamp(r.start), pd.Timestamp(r.end), float(r.lift), float(r.se)) for r in W["lifts"].itertuples()]
            if st.button("Fit model and back-test", type="primary", disabled=not channels):
                W["fit_key"] = (W["accepted"]["sha"], target, tuple(channels), tuple(controls), holdout,
                                str([(l.channel, l.start, l.end, l.lift, l.se) for l in lifts]))
                W.pop("plan", None)
            if "fit_key" in W:
                with st.spinner("Fitting the marketing mix model (incl. back-test)…"):
                    fit, bt = _fit(*W["fit_key"], df, lifts)
                summ, params, curves, dec = _summaries(str(W["fit_key"]), fit)
                W["fit"], W["bt"], W["summ"], W["params"], W["curves"], W["dec"] = fit, bt, summ, params, curves, dec
                cur = W["currency"]
                if bt:
                    color = BADGE["OBSERVED"] if bt.verdict.startswith("PASS") else (BADGE["RECOMMENDED"] if bt.verdict.startswith("CAUTION") else BADGE["DATA_UNAVAILABLE"])
                    st.markdown(f"<div style='border-left:6px solid {color};padding:8px 14px;background:#f6f6f4'>"
                                f"<b>Back-test: {bt.verdict}</b><br>Weekly error {bt.metrics['mape']:.1%} · total error "
                                f"{bt.metrics['total_error_pct']:+.1%} · {bt.metrics['interval_coverage_90']:.0%} of unseen weeks inside "
                                f"the 90% band</div>", unsafe_allow_html=True)
                    s = bt.series
                    fig = go.Figure([go.Scatter(x=s.date, y=s.high, line=dict(width=0), showlegend=False, hoverinfo="skip"),
                                     go.Scatter(x=s.date, y=s.low, fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
                                                line=dict(width=0), name="90% band"),
                                     go.Scatter(x=s.date, y=s.actual, mode="lines", name="actual", line=dict(color="#52514e", width=2)),
                                     go.Scatter(x=s.date, y=s.predicted, mode="lines", name="model", line=dict(color=SERIES[0], width=2))])
                    fig.add_vrect(x0=s.date[s.holdout].min(), x1=s.date.max(), fillcolor="#eda100", opacity=0.08, line_width=0,
                                  annotation_text="unseen holdout", annotation_position="top left")
                    st.plotly_chart(style(fig.update_layout(title="Back-test: actual vs predicted")), width="stretch")
                else:
                    st.warning("No back-test was run. Do not present recommendations without one.")
                d = fit.diagnostics
                badge("ESTIMATED", f"R² {d['r2']:.2f} · NRMSE {d['nrmse']:.3f} · converged {d['converged']} · {d['posterior']}")
                for w_ in d["warnings"]:
                    st.warning(w_)
                c1, c2 = st.columns(2)
                comp_cols = [c for c in fit.spec.channels]
                base = dec["base"] + dec["trend"] + dec["seasonality"] + dec["controls"]
                fig = go.Figure([go.Scatter(x=dec.date, y=base, stackgroup="one", name="base demand", line=dict(width=0),
                                            fillcolor="#d5d4cf")])
                for k, ch in enumerate(comp_cols):
                    fig.add_trace(go.Scatter(x=dec.date, y=dec[ch], stackgroup="one", name=ch.replace("spend_", ""),
                                             line=dict(width=0), fillcolor=SERIES[k % 8]))
                c1.plotly_chart(style(fig.update_layout(title="What drives the outcome (stacked)")), width="stretch")
                lab = summ.channel.str.replace("spend_", "")
                fig = go.Figure(go.Bar(x=lab, y=summ.roi, marker_color=SERIES[0], name="ROI",
                                       error_y=dict(type="data", array=summ.roi_high - summ.roi, arrayminus=summ.roi - summ.roi_low)))
                fig.add_hline(y=1, line_dash="dot", line_color="#8a8985", annotation_text="break-even")
                c2.plotly_chart(style(fig.update_layout(title="ROI by channel (90% interval)", showlegend=False)), width="stretch")
                show = summ.assign(channel=lab)[["channel", "spend", "contribution", "roi", "roi_low", "roi_high", "mroi",
                                                 "p_roi_above_1", "share_of_spend", "share_of_effect"]]
                st.dataframe(show.style.format({"spend": cur + "{:,.0f}", "contribution": cur + "{:,.0f}", "roi": "{:.2f}",
                                                "roi_low": "{:.2f}", "roi_high": "{:.2f}", "mroi": "{:.2f}",
                                                "p_roi_above_1": "{:.0%}", "share_of_spend": "{:.0%}", "share_of_effect": "{:.0%}"}),
                             width="stretch", hide_index=True)
                fig = go.Figure()
                for k, ch in enumerate(curves.channel.unique()):
                    cc = curves[curves.channel == ch]
                    fig.add_trace(go.Scatter(x=cc.weekly_spend, y=cc.weekly_effect, mode="lines", name=ch.replace("spend_", ""),
                                             line=dict(color=SERIES[k % 8], width=2)))
                    fig.add_trace(go.Scatter(x=[cc.current_weekly_spend.iloc[0]],
                                             y=[np.interp(cc.current_weekly_spend.iloc[0], cc.weekly_spend, cc.weekly_effect)],
                                             mode="markers", marker=dict(color=SERIES[k % 8], size=10, line=dict(color="white", width=2)),
                                             showlegend=False, hovertemplate="current spend<extra></extra>"))
                st.plotly_chart(style(fig.update_layout(title="Response curves (dot = current weekly spend)",
                                                        xaxis_title="weekly spend", yaxis_title="weekly incremental outcome"), 420),
                                width="stretch")
                with st.expander("Channel parameters (carryover, saturation)"):
                    st.dataframe(params.round(3), width="stretch", hide_index=True)

    # ------------------------------------------------------------------ 4 plan
    with tabs[3]:
        if "fit" not in W:
            st.info("Fit the model in step 3 first.")
        else:
            from dsdvc.mmm import analysis as A
            fit, cur = W["fit"], W["currency"]
            current = float(sum(fit.data[c].mean() for c in fit.spec.channels))
            c1, c2, c3 = st.columns(3)
            budget = c1.number_input("Weekly budget", 0.0, current * 10, float(W.get("budget", current)), max(current * 0.01, 1.0))
            objective = c2.radio("Objective", ["outcome", "profit"], horizontal=True,
                                 help="outcome: spend the full budget for maximum incremental outcome. profit: margin × outcome − spend; may under-spend.")
            max_change = c3.slider("Max change per channel vs today", 0.1, 1.0, 0.5, 0.05,
                                   help="Larger moves leave the spend range the model has evidence for.")
            if st.button("Optimise plan", type="primary"):
                W["budget"] = budget
                W["plan"] = A.optimize_budget(fit, budget, max_change=max_change, objective=objective, margin=W["margin"])
            if "plan" in W:
                plan = W["plan"]
                t = plan.table.assign(label=plan.table.channel.str.replace("spend_", ""))
                m1, m2, m3 = st.columns(3)
                m1.metric("Expected gain / week", _money(plan.uplift, cur), help=f"90% interval {_money(plan.uplift_low, cur)} to {_money(plan.uplift_high, cur)}")
                m1.caption(f"{_money(plan.uplift_low, cur)} – {_money(plan.uplift_high, cur)}")
                m2.metric(f"Over {W['horizon']} weeks", _money(plan.uplift * W["horizon"], cur))
                m3.metric("Probability plan beats current mix", f"{plan.p_better:.0%}")
                badge("RECOMMENDED", "based on the ESTIMATED model; validate large shifts with a lift test")
                fig = go.Figure([go.Bar(x=t.label, y=t.current_weekly, name="current", marker_color="#b5b4ae"),
                                 go.Bar(x=t.label, y=t.recommended_weekly, name="recommended", marker_color=SERIES[0])])
                st.plotly_chart(style(fig.update_layout(barmode="group", title="Weekly spend: current vs recommended")), width="stretch")
                st.dataframe(t[["label", "current_weekly", "recommended_weekly", "change_pct", "current_weekly_effect",
                                "recommended_weekly_effect"]].rename(columns={"label": "channel"}).style.format(
                    {"current_weekly": cur + "{:,.0f}", "recommended_weekly": cur + "{:,.0f}", "change_pct": "{:+.0%}",
                     "current_weekly_effect": cur + "{:,.0f}", "recommended_weekly_effect": cur + "{:,.0f}"}),
                    width="stretch", hide_index=True)
                for n in plan.notes:
                    st.warning(n)
                with st.expander("What if the budget changes? (optimal mix at each budget level)"):
                    rows = []
                    for mult in (0.7, 0.85, 1.0, 1.15, 1.3):
                        p = A.optimize_budget(fit, budget * mult, max_change=max_change, objective="outcome", n_draws=60)
                        rows.append({"weekly budget": budget * mult, "expected weekly effect": float(p.table.recommended_weekly_effect.sum()),
                                     "gain vs current mix": p.uplift})
                    sw = pd.DataFrame(rows)
                    sw["marginal return of extra budget"] = sw["expected weekly effect"].diff() / sw["weekly budget"].diff()
                    st.dataframe(sw.style.format({"weekly budget": cur + "{:,.0f}", "expected weekly effect": cur + "{:,.0f}",
                                                  "gain vs current mix": cur + "{:,.0f}", "marginal return of extra budget": "{:.2f}"}),
                                 width="stretch", hide_index=True)

    # ------------------------------------------------------------------ 5 export
    with tabs[4]:
        if "plan" not in W:
            st.info("Optimise a plan in step 4 first.")
        else:
            from dsdvc.exports.deck import executive_deck
            from dsdvc.exports.excel import media_plan_workbook
            acc = W["accepted"]
            prov = {"filename": acc["filename"], "rows": acc["rows"], "sha256": acc["sha"], "validation": acc["status"],
                    "attestation": W.get("attestation", ""), "data_version": W["fit"].data_version,
                    "lift_tests": len(W["fit"].lift_tests)}
            if acc.get("synthetic"):
                st.warning("These exports are built from the SYNTHETIC sample file, not client data.")
            if W["bt"] is None or not W["bt"].verdict.startswith(("PASS", "CAUTION")):
                st.error("The back-test did not pass. Exports are marked accordingly; do not present them as recommendations.")
            target = W["fit"].spec.target_col
            xlsx = media_plan_workbook(client=W["client"], plan=W["plan"], summary=W["summ"], curves=W["curves"], backtest=W["bt"],
                                       params=W["params"], horizon_weeks=int(W["horizon"]), start=W["start"],
                                       diagnostics=W["fit"].diagnostics, provenance=prov, decision=W["decision"])
            pptx = executive_deck(client=W["client"], plan=W["plan"], summary=W["summ"], decomposition=W["dec"], backtest=W["bt"],
                                  diagnostics=W["fit"].diagnostics, provenance=prov, horizon_weeks=int(W["horizon"]),
                                  outcome_label=target, currency=W["currency"])
            slug = "".join(ch if ch.isalnum() else "_" for ch in W["client"]).strip("_").lower()
            c1, c2, c3 = st.columns(3)
            c1.download_button("⬇ Media plan (Excel)", xlsx, f"{slug}_media_plan.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
            c2.download_button("⬇ Executive deck (PowerPoint)", pptx, f"{slug}_mmm_deck.pptx",
                               "application/vnd.openxmlformats-officedocument.presentationml.presentation", type="primary")
            c3.download_button("⬇ Canonical data (CSV)", acc["df"].to_csv(index=False), f"{slug}_canonical_data.csv")
            if st.button("Record this run in the project log"):
                from dsdvc.core.reproducibility import json_safe
                from dsdvc.database.models import MMMRun, SessionLocal
                f = W["fit"]
                with SessionLocal() as db:
                    db.add(MMMRun(dataset_version_id=acc["id"], spec=json_safe(f.spec.__dict__), diagnostics=json_safe(f.diagnostics),
                                  channel_summary=json_safe(W["summ"].to_dict(orient="records")),
                                  backtest=json_safe({"verdict": W["bt"].verdict, **W["bt"].metrics}) if W["bt"] else None,
                                  plan=json_safe({"table": W["plan"].table.to_dict(orient="records"), "uplift": W["plan"].uplift,
                                                  "uplift_low": W["plan"].uplift_low, "uplift_high": W["plan"].uplift_high,
                                                  "p_better": W["plan"].p_better}),
                                  model_version=f.model_version, data_version=f.data_version))
                    db.commit()
                st.success("Run recorded with data hash, model version and back-test.")
