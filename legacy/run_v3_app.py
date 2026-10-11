"""Run the preserved v3 Streamlit app unmodified: its code imports `dsdvc_mp`, so alias the package."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import legacy.dsdvc_mp_v3 as v3  # noqa: E402
import legacy.dsdvc_mp_v3.model  # noqa: E402,F401

sys.modules["dsdvc_mp"] = v3
sys.modules["dsdvc_mp.model"] = v3.model
runpy.run_path(str(ROOT / "dsdvc_mp_v3" / "app.py"), run_name="__main__")
