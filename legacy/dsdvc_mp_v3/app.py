import streamlit as st
import pandas as pd
import plotly.express as px
from dsdvc_mp.model import ModelConfig, DSDVCMP, sweep

st.set_page_config(page_title="DSDVC-MP", page_icon="🌐", layout="wide")
st.title("DSDVC-MP — Multi-Platform Social Diffusion, Virality & Churn")
st.caption("Bass diffusion × Schelling-style network evolution × Berger STEPPS × Instagram/Facebook/WhatsApp multiplex diffusion")

with st.sidebar:
    st.header("Simulation")
    n_agents = st.slider("Agents", 200, 2500, 800, 100)
    steps = st.slider("Time steps", 20, 240, 90, 10)
    seed = st.number_input("Seed", 1, 999999, 42)
    st.header("Bass / Market")
    p = st.slider("Innovation p", 0.001, 0.10, 0.015, 0.001)
    q = st.slider("Imitation q", 0.05, 1.50, 0.55, 0.05)
    market = st.slider("Market condition", 0.10, 1.00, 0.80, 0.05)
    competition = st.slider("Competition", 0.0, 1.0, 0.25, 0.05)
    price = st.slider("Price pressure", 0.0, 1.0, 0.20, 0.05)
    st.header("Schelling / Network")
    homophily = st.slider("Homophily", 0.0, 1.0, 0.60, 0.05)
    rewiring = st.slider("Rewiring rate", 0.0, 0.20, 0.035, 0.005)
    st.header("STEPPS")
    stepps = {}
    for name in ["Social currency", "Triggers", "Emotion", "Public", "Practical value", "Story"]:
        stepps[name] = st.slider(name, 0.0, 1.0, {"Social currency":0.70,"Triggers":0.60,"Emotion":0.75,"Public":0.75,"Practical value":0.55,"Story":0.70}[name], 0.05)
    st.header("Platform strategy")
    ig = st.slider("Instagram reach", 0.0, 1.0, 0.90, 0.05)
    fb = st.slider("Facebook reach", 0.0, 1.0, 0.55, 0.05)
    wa = st.slider("WhatsApp reach", 0.0, 1.0, 0.25, 0.05)
    st.header("Firm")
    push = st.slider("Acquisition push", 0.0, 1.0, 0.25, 0.05)
    retention = st.slider("Retention investment", 0.0, 1.0, 0.20, 0.05)
    content = st.slider("Content investment", 0.0, 1.0, 0.50, 0.05)

cfg = ModelConfig(
    n_agents=n_agents, steps=steps, seed=int(seed), p_innovation=p, q_imitation=q,
    market_condition=market, competition=competition, price_pressure=price,
    homophily=homophily, rewiring_rate=rewiring,
    content_social_currency=stepps["Social currency"], content_triggers=stepps["Triggers"],
    content_emotion=stepps["Emotion"], content_public=stepps["Public"],
    content_practical=stepps["Practical value"], content_story=stepps["Story"],
    instagram_reach=ig, facebook_reach=fb, whatsapp_reach=wa,
    push_intensity=push, retention_investment=retention, content_investment=content
)

if st.button("▶ Run simulation", type="primary", use_container_width=True):
    model = DSDVCMP(cfg)
    df = model.run()
    st.session_state["df"] = df
    st.session_state["summary"] = model.summary()

if "df" not in st.session_state:
    st.info("Set parameters in the sidebar and run the model.")
    st.stop()

df = st.session_state["df"]
s = st.session_state["summary"]

cols = st.columns(6)
for col, title, value in zip(cols, ["Final adoption", "Peak adoption", "Churn", "Total shares", "STEPPS", "Viral R proxy"],
                            [f"{s['adoption_rate']:.1%}", f"{s['peak_adoption']:.1%}", f"{s['churn_rate']:.1%}", f"{s['total_shares']:,}", f"{s['stepps_score']:.2f}", f"{s['viral_R_proxy']:.3f}"]):
    col.metric(title, value)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Adoption & churn")
    plot = df.melt(id_vars="step", value_vars=["adoption_rate", "churn_rate"], var_name="metric", value_name="value")
    st.plotly_chart(px.line(plot, x="step", y="value", color="metric"), use_container_width=True)
with c2:
    st.subheader("Platform sharing")
    plot = df.melt(id_vars="step", value_vars=["instagram_shares", "facebook_shares", "whatsapp_shares"], var_name="platform", value_name="shares")
    st.plotly_chart(px.line(plot, x="step", y="shares", color="platform"), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.subheader("Network homophily")
    plot = df.melt(id_vars="step", value_vars=["instagram_homophily", "facebook_homophily", "whatsapp_homophily"], var_name="platform", value_name="homophily")
    st.plotly_chart(px.line(plot, x="step", y="homophily", color="platform"), use_container_width=True)
with c4:
    st.subheader("Engagement & satisfaction")
    st.plotly_chart(px.line(df, x="step", y=["avg_engagement", "avg_satisfaction"]), use_container_width=True)

st.subheader("Platform mechanics")
platform_table = pd.DataFrame({
    "Platform": ["Instagram", "Facebook", "WhatsApp"],
    "Primary role": ["Discovery / algorithmic reach", "Community / groups", "Private high-trust transmission"],
    "Network prior": ["Scale-free", "Clustered small-world", "Strong-tie small-world"],
    "Default influence weight": [0.35, 0.25, 0.40],
    "Reach": [ig, fb, wa],
    "Trust": [cfg.instagram_trust, cfg.facebook_trust, cfg.whatsapp_trust],
})
st.dataframe(platform_table, use_container_width=True, hide_index=True)

st.subheader("Experiment: homophily × STEPPS")
if st.button("Run 3×3 experiment"):
    grid = {"homophily": [0.2, 0.6, 0.9], "content_emotion": [0.2, 0.6, 0.9]}
    exp = sweep(cfg, grid, repetitions=3)
    st.session_state["exp"] = exp
if "exp" in st.session_state:
    exp = st.session_state["exp"]
    pivot = exp.groupby(["homophily", "content_emotion"], as_index=False)["adoption_rate"].mean()
    st.plotly_chart(px.density_heatmap(pivot, x="homophily", y="content_emotion", z="adoption_rate", text_auto=".1%"), use_container_width=True)
    st.dataframe(exp[["homophily", "content_emotion", "adoption_rate", "peak_adoption", "churn_rate", "viral_R_proxy"]], use_container_width=True, hide_index=True)

st.download_button("Download simulation CSV", df.to_csv(index=False), "dsdvc_mp_simulation.csv", "text/csv")
