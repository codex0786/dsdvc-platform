from dsdvc_mp.model import ModelConfig, DSDVCMP

def test_model_runs():
    m = DSDVCMP(ModelConfig(n_agents=100, steps=10, seed=1))
    df = m.run()
    assert len(df) == 10
    assert 0 <= df.adoption_rate.iloc[-1] <= 1
    assert (df.instagram_homophily >= 0).all()
    assert (df.whatsapp_homophily >= 0).all()

def test_platforms_exist():
    m = DSDVCMP(ModelConfig(n_agents=100, steps=2))
    assert set(m.networks) == {"instagram", "facebook", "whatsapp"}
