"""Container-aware CPU limits.

Hosted containers (Streamlit Cloud, Kubernetes, Docker with --cpus) often report the host's CPU
count while enforcing a much smaller cgroup quota. OpenMP libraries (LightGBM, scikit-learn,
numpy BLAS) then start one thread per reported CPU and thrash. ``configure_threads`` caps them
at the real quota; call it before heavy libraries are imported.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")


def available_cpus() -> int:
    n = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            n = min(n, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    quota = None
    try:  # cgroup v2: "max 100000" or "200000 100000"
        q, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if q != "max":
            quota = int(q) / int(period)
    except (OSError, ValueError):
        try:  # cgroup v1
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0:
                quota = q / period
        except (OSError, ValueError):
            pass
    if quota:
        n = min(n, max(1, math.floor(quota)))
    return max(1, n)


def configure_threads(max_threads: int | None = None) -> int:
    """Set thread-count env vars (unless the operator already set them). Returns the count."""
    n = available_cpus() if max_threads is None else max(1, min(max_threads, available_cpus()))
    for v in THREAD_VARS:
        if not os.environ.get(v, "").strip().isdigit():  # unset, empty or malformed -> our value
            os.environ[v] = str(n)
    n = int(os.environ["OMP_NUM_THREADS"])
    try:  # also cap libraries that were already loaded (e.g. numpy imported by Streamlit itself)
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except Exception:  # noqa: BLE001 - best effort
        pass
    return n
