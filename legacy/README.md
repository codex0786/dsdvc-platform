# Legacy models (preserved, executable)

| Folder | Origin | What it is |
|---|---|---|
| `dsdvc_mp_v2/` | original `DSDVC_Python_Deployment` archive (`dsdvc_model/`) | Single-network ABM: Bass + Schelling rewiring + STEPPS + churn + firm response |
| `dsdvc_mp_v3/` | original `DSDVC_MP_v2_Complete` archive (`dsdvc_mp/`) | Three-layer multiplex ABM (Instagram / Facebook / WhatsApp) |

Naming note: the source archives label the multiplex prototype "MP v2". Here the single-network
prototype is `v2` and the multiplex prototype is `v3` (the baseline named in the brief). The source
files are copied **byte-for-byte**; `CHECKSUMS.sha256` records their hashes and
`tests/test_legacy_regression.py` verifies both the hashes and the original mechanics.

Run:
```bash
make legacy-v3     # streamlit app of the multiplex prototype
make legacy-v2     # streamlit app of the single-network prototype
```
Programmatic:
```python
from legacy.dsdvc_mp_v3.model import ModelConfig, DSDVCMP
from legacy.dsdvc_mp_v2 import Params, DSDVCModel
```
