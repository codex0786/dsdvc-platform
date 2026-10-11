import streamlit as st
import pandas as pd
import plotly.express as px
from model import Params, DSDVCModel, scenario_grid

st.set_page_config(page_title='DSDVC Diffusion Model', page_icon='◈', layout='wide')
st.title('Dynamic Social Diffusion, Virality & Churn')
st.caption('Bass diffusion × Schelling-style community dynamics × Berger STEPPS × customer lifecycle')

with st.sidebar:
    st.header('Model Controls')
    n_agents = st.slider('Agents', 100, 2000, 500, 100)
    periods = st.slider('Periods', 20, 300, 120, 10)
    seed = st.number_input('Random seed', 0, 99999, 42)
    st.subheader('Bass')
    p = st.slider('Innovation p', 0.0, 0.20, 0.02, 0.005)
    q = st.slider('Imitation q', 0.0, 1.0, 0.40, 0.05)
    initial = st.slider('Initial adoption', 0.0, 0.20, 0.02, 0.005)
    st.subheader('Market')
    market = st.slider('Market condition', 0.0, 1.0, 0.65, 0.05)
    competition = st.slider('Competition', 0.0, 1.0, 0.30, 0.05)
    push = st.slider('Push intensity', 0.0, 1.0, 0.50, 0.05)
    churn = st.slider('Base churn', 0.0, 0.30, 0.08, 0.01)
    retention = st.slider('Retention investment', 0.0, 1.0, 0.25, 0.05)
    st.subheader('Schelling')
    homophily = st.slider('Homophily', 0.0, 1.0, 0.55, 0.05)
    threshold = st.slider('Staying threshold', 0.0, 1.0, 0.50, 0.05)
    rewiring = st.slider('Rewiring rate', 0.0, 0.30, 0.06, 0.01)
    st.subheader('STEPPS')
    vals = {}
    for label, key, default in [('Social currency','social_currency',.60),('Triggers','triggers',.55),('Emotion','emotion',.70),('Public','public',.60),('Practical value','practical_value',.55),('Stories','stories',.65)]:
        vals[key] = st.slider(label, 0.0, 1.0, default, 0.05)
    frequency = st.slider('Content frequency', 0.0, 1.0, 0.50, 0.05)
    quality = st.slider('Content quality', 0.0, 1.0, 0.60, 0.05)
    decay = st.slider('Content decay', 0.0, 0.30, 0.08, 0.01)
    run = st.button('Run simulation', type='primary', use_container_width=True)

params = Params(n_agents=n_agents, periods=periods, p=p, q=q, initial_adoption=initial,
                market_condition=market, competition=competition, push=push, churn_rate=churn,
                retention_investment=retention, homophily=homophily, staying_threshold=threshold,
                rewiring_rate=rewiring, content_frequency=frequency, content_quality=quality,
                content_decay=decay, seed=int(seed), **vals)

if 'df' not in st.session_state or run:
    model = DSDVCModel(params)
    with st.spinner('Running agent-based simulation...'):
        st.session_state.df = model.run()
    st.session_state.params = params

df = st.session_state.df
last = df.iloc[-1]

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric('Active adopters', f"{int(last.active_adopters):,}")
c2.metric('Adoption rate', f"{last.adoption_rate:.1%}")
c3.metric('Churned', f"{int(last.churned):,}")
c4.metric('STEPPS score', f"{last.stepps_score:.2f}")
c5.metric('Viral R', f"{last.viral_R:.2f}")

st.subheader('Diffusion & lifecycle')
fig = px.line(df, x='t', y=['active_adopters','cumulative_adopters','churned'], labels={'value':'Agents','t':'Period','variable':'Series'})
st.plotly_chart(fig, use_container_width=True)

col1,col2 = st.columns(2)
with col1:
    fig2 = px.line(df, x='t', y=['community_fit','homophily','cross_community_exposure'], labels={'value':'Index','t':'Period'})
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    fig3 = px.line(df, x='t', y=['shares_this_period','viral_R','content_attention'], labels={'value':'Value','t':'Period'})
    st.plotly_chart(fig3, use_container_width=True)

st.subheader('Scenario experiment')
st.write('Compare homophily × STEPPS strength. This is the first strategic experiment built into the model.')
if st.button('Run 3×3 scenario grid'):
    base = st.session_state.params
    with st.spinner('Running 9 simulations...'):
        grid = scenario_grid(base, ['homophily','content_quality'], [[.20,.55,.85],[.30,.60,.90]], periods=min(80, periods))
    st.dataframe(grid, use_container_width=True)
    fig4 = px.scatter(grid, x='homophily', y='final_active', size='peak_active', color='content_quality', hover_data=['viral_R','cross_community_exposure'])
    st.plotly_chart(fig4, use_container_width=True)

st.subheader('Latest simulation data')
st.dataframe(df.tail(20), use_container_width=True)
st.download_button('Download CSV', df.to_csv(index=False).encode(), 'dsdvc_simulation.csv', 'text/csv')

with st.expander('Model specification'):
    st.markdown('''
**Adoption hazard** = external innovation + local imitation + STEPPS content exposure + community fit, adjusted by market conditions, competition and push.

**Sharing** uses a logistic probability driven by the six STEPPS dimensions, engagement, influence, trust and community fit, with content decay.

**Community dynamics** use a Schelling-style staying threshold. Agents below the threshold probabilistically rewire toward more compatible contacts.

**Churn** rises with low satisfaction, low engagement, competition and poor community fit; retention investment reduces churn.
''')
