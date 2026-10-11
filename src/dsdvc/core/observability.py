"""Observability (spec §66): request/model latency, ingestion failures, prediction errors, data
freshness and versions, exposed in Prometheus text format; optional MLflow and Redis hooks."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager

from dsdvc import MODEL_VERSION, SIMULATION_VERSION, __version__
from dsdvc.config.settings import get_settings

log = logging.getLogger("dsdvc")


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters: dict[tuple, float] = defaultdict(float)
        self.hist: dict[tuple, list[float]] = defaultdict(list)
        self.gauges: dict[tuple, float] = {}

    def inc(self, name: str, value: float = 1.0, **labels):
        with self._lock:
            self.counters[(name, tuple(sorted(labels.items())))] += value

    def observe(self, name: str, value: float, **labels):
        with self._lock:
            h = self.hist[(name, tuple(sorted(labels.items())))]
            h.append(value)
            if len(h) > 5000:
                del h[:1000]

    def gauge(self, name: str, value: float, **labels):
        with self._lock:
            self.gauges[(name, tuple(sorted(labels.items())))] = value

    @contextmanager
    def timer(self, name: str, **labels):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - t) * 1000, **labels)

    def prometheus(self) -> str:
        def lab(ls):
            return "{" + ",".join(f'{k}="{v}"' for k, v in ls) + "}" if ls else ""
        lines = [f'dsdvc_build_info{{version="{__version__}",model_version="{MODEL_VERSION}",'
                 f'simulation_version="{SIMULATION_VERSION}"}} 1']
        with self._lock:
            for (n, ls), v in self.counters.items():
                lines.append(f"{n}_total{lab(ls)} {v}")
            for (n, ls), v in self.gauges.items():
                lines.append(f"{n}{lab(ls)} {v}")
            for (n, ls), vals in self.hist.items():
                if vals:
                    s = sorted(vals)
                    for q in (0.5, 0.9, 0.99):
                        ql = ls + (("quantile", str(q)),)
                        lines.append(f"{n}_ms{lab(ql)} {s[min(len(s) - 1, int(q * len(s)))]:.3f}")
                    lines.append(f"{n}_ms_count{lab(ls)} {len(s)}")
        return "\n".join(lines) + "\n"


METRICS = Metrics()


class Cache:
    """Redis if REDIS_URL is set and reachable, else in-process dict (simulation results are
    deterministic given config+seed, so caching by config hash is safe)."""

    def __init__(self):
        self._mem: dict[str, str] = {}
        self._r = None
        url = get_settings().redis_url
        if url:
            try:
                import redis
                self._r = redis.Redis.from_url(url, socket_timeout=1)
                self._r.ping()
            except Exception:  # noqa: BLE001
                log.warning("Redis unavailable; using in-process cache")
                self._r = None

    @staticmethod
    def key(obj) -> str:
        return "dsdvc:" + hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:24]

    def get(self, k: str):
        v = self._r.get(k) if self._r else self._mem.get(k)
        return json.loads(v) if v else None

    def set(self, k: str, v, ttl: int = 86400):
        s = json.dumps(v, default=str)
        if self._r:
            self._r.setex(k, ttl, s)
        else:
            if len(self._mem) > 512:
                self._mem.pop(next(iter(self._mem)))
            self._mem[k] = s


def log_to_mlflow(run_name: str, params: dict, metrics: dict, tags: dict | None = None) -> bool:
    uri = get_settings().mlflow_tracking_uri
    if not uri:
        return False
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: str(v)[:250] for k, v in list(params.items())[:100]})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v == v})
            mlflow.set_tags({"model_version": MODEL_VERSION, **(tags or {})})
        return True
    except Exception:  # noqa: BLE001
        METRICS.inc("dsdvc_mlflow_failures")
        return False
