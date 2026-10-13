import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
_tmp = tempfile.mkdtemp(prefix="dsdvc-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["DSDVC_BOOTSTRAP_ADMIN_KEY"] = "dsdvc_testadmin_" + "s3cret-test-key"
for k in ("META_ACCESS_TOKEN", "INSTAGRAM_BUSINESS_ACCOUNT_ID", "FACEBOOK_PAGE_ID", "WHATSAPP_BUSINESS_TOKEN",
          "WHATSAPP_BUSINESS_ACCOUNT_ID", "BRAVE_SEARCH_API_KEY", "REDIS_URL", "MLFLOW_TRACKING_URI"):
    os.environ.pop(k, None)
os.environ.setdefault("DSDVC_DATA_DIR", tempfile.mkdtemp(prefix="dsdvc-data-"))
