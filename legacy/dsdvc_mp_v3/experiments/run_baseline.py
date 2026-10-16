from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsdvc_mp.model import ModelConfig, DSDVCMP, sweep

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

cfg = ModelConfig(n_agents=1000, steps=120, seed=42)
model = DSDVCMP(cfg)
df = model.run()
df.to_csv(OUT / "baseline_timeseries.csv", index=False)
print(model.summary())

exp = sweep(cfg, {
    "homophily": [0.2, 0.5, 0.8],
    "content_emotion": [0.2, 0.5, 0.8],
    "whatsapp_reach": [0.15, 0.30, 0.60],
}, repetitions=3)
exp.to_csv(OUT / "platform_network_experiment.csv", index=False)
print(exp.groupby(["homophily", "content_emotion", "whatsapp_reach"])["adoption_rate"].mean())
