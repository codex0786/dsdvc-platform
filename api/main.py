"""Entry point so `uvicorn api.main:app` works from the repository root (spec §64)."""
from dsdvc.api.main import app  # noqa: F401
