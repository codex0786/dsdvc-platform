"""Entry point so `streamlit run dashboard/app.py` works from the repository root (spec §64)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dsdvc.dashboard.app import main  # noqa: E402

main()
