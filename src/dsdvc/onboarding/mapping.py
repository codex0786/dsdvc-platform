"""Column mapping and value parsing for client exports.

``suggest_mapping`` proposes canonical names for each source column with a confidence score;
an analyst confirms or edits it; ``apply_mapping`` produces the canonical frame. Parsing handles
currency symbols, Indian digit grouping (1,20,000), percentages, Yes/No flags and day-first dates.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dsdvc.onboarding.templates import CHANNEL_ALIASES, SPEND_TOKENS, DatasetSpec

IGNORE = "(ignore)"


def normalise(name: str) -> str:
    s = re.sub(r"[₹$€£%()\[\]?/+&.-]", " ", str(name).lower())
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _tokens(name: str) -> set[str]:
    return set(normalise(name).split("_")) | {normalise(name)}


@dataclass
class ColumnMatch:
    source: str
    target: str
    confidence: float
    reason: str


@dataclass
class Mapping:
    dataset: str
    matches: list[ColumnMatch] = field(default_factory=list)

    def as_dict(self) -> dict[str, str]:
        return {m.source: m.target for m in self.matches if m.target != IGNORE}

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([m.__dict__ for m in self.matches])

    @classmethod
    def from_table(cls, dataset: str, table: pd.DataFrame) -> "Mapping":
        return cls(dataset, [ColumnMatch(r.source, r.target, float(getattr(r, "confidence", 1.0)), getattr(r, "reason", "edited"))
                             for r in table.itertuples()])


def _channel_for(tokens: set[str]) -> str | None:
    best = None
    for canon, aliases in CHANNEL_ALIASES.items():
        hits = [a for a in aliases if a in tokens or any(t.startswith(a) and len(a) >= 4 for t in tokens)]
        if hits:
            score = max(len(h) for h in hits)
            if best is None or score > best[1]:
                best = (canon, score)
    return best[0] if best else None


def suggest_mapping(df: pd.DataFrame, spec: DatasetSpec, sample_rows: int = 50) -> Mapping:
    out = Mapping(spec.key)
    used: set[str] = set()
    for col in df.columns:
        toks = _tokens(col)
        norm = normalise(col)
        values = df[col].dropna().astype(str).head(sample_rows)
        numeric_share = values.map(lambda v: _parse_number(v) is not None).mean() if len(values) else 0
        # 1) explicit canonical / synonym match
        cand = None
        for c in spec.columns:
            syns = {normalise(x) for x in c.synonyms} | {c.name}
            if norm in syns:
                cand = (c.name, 0.98, "exact synonym")
                break
            overlap = toks & syns
            if overlap and (cand is None or cand[1] < 0.85):
                cand = (c.name, 0.85, f"keyword '{sorted(overlap)[0]}'")
        # 2) spend columns -> spend_<channel>
        if spec.channel_pattern and numeric_share > 0.8:
            ch = _channel_for(toks)
            is_spend = bool(toks & set(SPEND_TOKENS)) or ch is not None
            if ch and is_spend and (cand is None or cand[0] in ("revenue", "conversions") and not (toks & {"sales", "revenue"})):
                cand = (spec.channel_pattern.format(channel=ch), 0.9 if toks & set(SPEND_TOKENS) else 0.75,
                        f"channel '{ch}'" + (" + spend keyword" if toks & set(SPEND_TOKENS) else ""))
        # 3) fuzzy fallback
        if cand is None:
            names = [c.name for c in spec.columns]
            m = difflib.get_close_matches(norm, names, n=1, cutoff=0.75)
            if m:
                cand = (m[0], 0.6, "fuzzy name match")
        if cand is None or cand[0] in used:
            out.matches.append(ColumnMatch(col, IGNORE, 0.0, "no confident match" if cand is None else f"duplicate of {cand[0]}"))
            continue
        used.add(cand[0])
        out.matches.append(ColumnMatch(col, *cand))
    return out


_NUM_RE = re.compile(r"^\(?-?[\d,]*\.?\d+\)?$")


def _parse_number(v) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip().lower()
    if s in ("", "-", "na", "n/a", "nan", "null", "none"):
        return None
    if s in ("yes", "y", "true"):
        return 1.0
    if s in ("no", "n", "false"):
        return 0.0
    mult = 1.0
    for suf, m in (("cr", 1e7), ("crore", 1e7), ("lakh", 1e5), ("lac", 1e5), ("l", 1e5), ("k", 1e3), ("m", 1e6), ("mn", 1e6)):
        if s.endswith(suf) and re.search(r"\d", s[: -len(suf)] or ""):
            s, mult = s[: -len(suf)].strip(), m
            break
    pct = s.endswith("%")
    s = re.sub(r"(₹|rs\.?|inr|usd|\$|€|£|%|\s)", "", s)
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    if not _NUM_RE.match(s):
        return None
    try:
        x = float(s.replace(",", "")) * mult
    except ValueError:
        return None
    if pct:
        x /= 100
    return -x if neg else x


def parse_dates(series: pd.Series) -> tuple[pd.Series, str]:
    """Parse dates, choosing day-first vs month-first by which yields fewer failures / a regular series."""
    s = series.astype(str).str.strip()
    best = None
    for dayfirst in (True, False):
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed")
        bad = parsed.isna().mean()
        mono = parsed.dropna().is_monotonic_increasing
        score = (bad, not mono)
        if best is None or score < best[0]:
            best = (score, parsed, "day-first" if dayfirst else "month-first")
    return best[1], best[2]


def apply_mapping(df: pd.DataFrame, mapping: Mapping, spec: DatasetSpec) -> tuple[pd.DataFrame, list[str]]:
    """Rename + parse values. Returns the canonical frame and a list of parse notes."""
    notes = []
    m = mapping.as_dict()
    out = pd.DataFrame(index=df.index)
    for src, tgt in m.items():
        col = df[src]
        cs = spec.column(tgt)
        dtype = cs.dtype if cs else ("number" if tgt.startswith("spend_") else "string")
        if dtype == "date":
            parsed, how = parse_dates(col)
            out[tgt] = parsed
            notes.append(f"{src} → {tgt}: parsed as {how} dates" + (f"; {parsed.isna().sum()} unparseable" if parsed.isna().any() else ""))
        elif dtype in ("number", "integer", "boolean"):
            vals = col.map(_parse_number)
            bad = int((vals.isna() & col.notna() & (col.astype(str).str.strip() != "")).sum())
            out[tgt] = pd.to_numeric(vals, errors="coerce")
            if bad:
                notes.append(f"{src} → {tgt}: {bad} value(s) could not be read as numbers")
        else:
            out[tgt] = col.astype(str)
    long = _maybe_long_format(df, m)
    if long is not None:
        out, n2 = long
        notes += n2
    if "date" in out:
        out = out.sort_values("date").reset_index(drop=True)
    return out, notes


def _maybe_long_format(df: pd.DataFrame, m: dict) -> tuple[pd.DataFrame, list[str]] | None:
    """Detect (date, channel, spend) long tables and pivot them to wide spend_ columns."""
    cols = {normalise(c): c for c in df.columns}
    ch_col = next((cols[k] for k in ("channel", "media", "platform", "source") if k in cols), None)
    sp_col = next((cols[k] for k in ("spend", "cost", "amount_spent", "investment") if k in cols), None)
    date_src = next((s for s, t in m.items() if t == "date"), None)
    if not (ch_col and sp_col and date_src):
        return None
    dates, _ = parse_dates(df[date_src])
    tmp = pd.DataFrame({"date": dates, "channel": df[ch_col].astype(str), "spend": df[sp_col].map(_parse_number)})
    tmp["channel"] = tmp.channel.map(lambda c: _channel_for(_tokens(c)) or normalise(c))
    wide = tmp.pivot_table(index="date", columns="channel", values="spend", aggfunc="sum").add_prefix("spend_").reset_index()
    other = [t for t in m.values() if t not in ("date",) and not t.startswith("spend_")]
    if other:
        extra = pd.DataFrame({"date": dates})
        for s, t in m.items():
            if t in other:
                extra[t] = pd.to_numeric(df[s].map(_parse_number), errors="coerce")
        wide = wide.merge(extra.groupby("date").sum(numeric_only=True).reset_index(), on="date", how="left")
    return wide, [f"Detected long format ({ch_col}/{sp_col}); pivoted to {wide.shape[1] - 1} columns"]
