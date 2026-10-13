"""Entry point so `streamlit run dashboard/app.py` works from the repository root (spec §64)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dsdvc.core.runtime import configure_threads  # noqa: E402

configure_threads()  # cap OpenMP/BLAS threads at the container's real CPU quota before numpy/lightgbm load
import streamlit as st  # noqa: E402

from dsdvc.dashboard.admin import admin_page  # noqa: E402
from dsdvc.dashboard.app import main as research_lab  # noqa: E402
from dsdvc.dashboard.auth import gate  # noqa: E402
from dsdvc.dashboard.workflow import workflow_page  # noqa: E402

st.set_page_config(page_title="DSDVC Marketing Intelligence", layout="wide")
ctx = gate()  # sign-in (SSO / dev / public demo) and client-workspace selection
pages = [st.Page(workflow_page, title="Client workflow", icon=":material/work:", default=True),
         st.Page(research_lab, title="Research lab", icon=":material/science:", url_path="lab")]
if not ctx.public_demo and (ctx.is_org_admin or ctx.can("admin")):
    pages.append(st.Page(admin_page, title="Administration", icon=":material/admin_panel_settings:", url_path="admin"))
st.navigation(pages).run()
