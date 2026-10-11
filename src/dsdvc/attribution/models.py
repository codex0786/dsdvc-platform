"""Marketing attribution (spec §30).

Rule-based (last click, first touch, linear, time decay, position based) and data-driven
(Markov removal effect, Shapley value) attribution distribute *attributed* conversions.
They are NOT causal: incremental conversions require experiments (``incrementality``).
The output always reports both quantities side by side when an experiment is supplied.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from math import factorial

import numpy as np
import pandas as pd
from scipy import stats

from dsdvc.core.literature import cites


def _norm(d: dict) -> dict:
    tot = sum(d.values())
    return {k: v / tot for k, v in d.items()} if tot else d


def rule_based(paths: pd.DataFrame, method: str, half_life: float = 2.0, ends_weight: float = 0.4) -> dict[str, float]:
    """paths: columns path (list of channels, chronological) and converted (bool)."""
    credit: dict[str, float] = defaultdict(float)
    for path in paths.loc[paths.converted, "path"]:
        if not path:
            continue
        k = len(path)
        if method == "last_click":
            w = [0] * (k - 1) + [1]
        elif method == "first_touch":
            w = [1] + [0] * (k - 1)
        elif method == "linear":
            w = [1 / k] * k
        elif method == "time_decay":
            raw = [0.5 ** ((k - 1 - i) / half_life) for i in range(k)]
            w = [x / sum(raw) for x in raw]
        elif method == "position_based":
            if k == 1:
                w = [1.0]
            elif k == 2:
                w = [0.5, 0.5]
            else:
                mid = (1 - 2 * ends_weight) / (k - 2)
                w = [ends_weight] + [mid] * (k - 2) + [ends_weight]
        else:
            raise ValueError(method)
        for c, x in zip(path, w):
            credit[c] += x
    return dict(credit)


@cites("anderl2016")
def markov_attribution(paths: pd.DataFrame, order: int = 1) -> dict[str, float]:
    """First-order Markov chain with removal effects, scaled to total conversions."""
    trans = defaultdict(lambda: defaultdict(float))
    for path, conv in zip(paths.path, paths.converted):
        seq = ["start"] + list(path) + ["conv" if conv else "null"]
        for a, b in zip(seq[:-1], seq[1:]):
            trans[a][b] += 1
    states = sorted(set(trans) | {b for d in trans.values() for b in d})
    idx = {s: i for i, s in enumerate(states)}

    def p_conv(removed: str | None) -> float:
        n = len(states)
        P = np.zeros((n, n))
        for a, d in trans.items():
            tot = sum(d.values())
            for b, c in d.items():
                bb = "null" if b == removed else b
                P[idx[a], idx[bb]] += c / tot
        for s in ("conv", "null"):
            if s in idx:
                P[idx[s]] = 0
                P[idx[s], idx[s]] = 1
        if removed in idx:
            P[idx[removed]] = 0
            P[idx[removed], idx["null"]] = 1
        v = np.zeros(n)
        v[idx["start"]] = 1
        for _ in range(200):
            v = v @ P
        return float(v[idx["conv"]]) if "conv" in idx else 0.0

    base = p_conv(None)
    channels = [s for s in states if s not in ("start", "conv", "null")]
    removal = {c: max(0.0, 1 - p_conv(c) / base) if base > 0 else 0.0 for c in channels}
    total = paths.converted.sum()
    return {c: v * total for c, v in _norm(removal).items()}


@cites("shao_li2011")
def shapley_attribution(paths: pd.DataFrame, max_channels: int = 12) -> dict[str, float]:
    """Shapley values with v(S) = conversions among users whose channel set is a subset of S
    (a common data-driven MTA formulation). Exact for <= max_channels channels."""
    sets = paths.path.map(lambda p: frozenset(p))
    conv = paths.groupby(sets)["converted"].sum().to_dict()
    channels = sorted({c for p in paths.path for c in p})
    if len(channels) > max_channels:
        raise ValueError("too many channels for exact Shapley")
    v_cache: dict[frozenset, float] = {}

    def v(S: frozenset) -> float:
        if S not in v_cache:
            v_cache[S] = float(sum(c for k, c in conv.items() if k and k <= S))
        return v_cache[S]

    n = len(channels)
    phi = {}
    for c in channels:
        others = [x for x in channels if x != c]
        tot = 0.0
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for S in itertools.combinations(others, r):
                S = frozenset(S)
                tot += w * (v(S | {c}) - v(S))
        phi[c] = tot
    return phi


def all_attributions(paths: pd.DataFrame) -> pd.DataFrame:
    out = {m: rule_based(paths, m) for m in ("last_click", "first_touch", "linear", "time_decay", "position_based")}
    out["markov"] = markov_attribution(paths)
    out["shapley"] = shapley_attribution(paths)
    return pd.DataFrame(out).fillna(0.0).sort_index()


@cites("lewis_rao2015", "gordon2019")
def incrementality_test(treat_conv: int, treat_n: int, ctrl_conv: int, ctrl_n: int, level: float = 0.9) -> dict:
    """Holdout / geo / conversion-lift test: incremental conversions with a Newcombe-style CI."""
    p1, p0 = treat_conv / treat_n, ctrl_conv / ctrl_n
    diff = p1 - p0
    se = np.sqrt(p1 * (1 - p1) / treat_n + p0 * (1 - p0) / ctrl_n)
    z = stats.norm.ppf(0.5 + level / 2)
    incr = diff * treat_n
    return {"treatment_rate": p1, "control_rate": p0, "lift_abs": diff, "lift_rel": diff / p0 if p0 > 0 else float("nan"),
            "incremental_conversions": incr, "ci_low": (diff - z * se) * treat_n, "ci_high": (diff + z * se) * treat_n,
            "p_value": float(2 * (1 - stats.norm.cdf(abs(diff) / se))) if se > 0 else float("nan"),
            "attributed_vs_incremental_note": "Incremental = treatment minus counterfactual control; "
                                              "attributed conversions from MTA can exceed this."}


def required_sample_size(base_rate: float, rel_lift: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-arm sample size for a two-proportion test (Lewis & Rao 2015 show this is often huge)."""
    p0, p1 = base_rate, base_rate * (1 + rel_lift)
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p0 + p1) / 2
    n = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2 / (p1 - p0) ** 2
    return int(np.ceil(n))
