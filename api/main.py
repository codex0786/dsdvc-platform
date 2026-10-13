"""Entry point so `uvicorn api.main:app` works from the repository root (spec §64)."""
from dsdvc.core.runtime import configure_threads

configure_threads()
from dsdvc.api.main import app  # noqa: E402,F401
