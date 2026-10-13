"""Entry point so `streamlit run dashboard/app.py` works from the repository root (spec §64)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dsdvc.core.runtime import configure_threads  # noqa: E402

configure_threads()  # cap OpenMP/BLAS threads at the container's real CPU quota before numpy/lightgbm load
from dsdvc.dashboard.app import main  # noqa: E402

main()
