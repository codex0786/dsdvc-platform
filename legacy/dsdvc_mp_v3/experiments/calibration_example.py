from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsdvc_mp.data import load_observations, estimate_platform_rates

obs = load_observations(Path(__file__).resolve().parents[1] / 'data' / 'sample_platform_observations.csv')
print(estimate_platform_rates(obs).to_string(index=False))
print('\nThese empirical rates can be used as calibration targets for scipy.optimize or Bayesian calibration.')
