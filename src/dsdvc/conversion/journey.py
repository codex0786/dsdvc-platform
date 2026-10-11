"""Consumer journey state machine (spec §24).

    UNAWARE -> AWARE -> INTERESTED -> CONSIDERING -> TRIAL -> ADOPTED -> ENGAGED -> ADVOCATE
    ADOPTED -> DISSATISFIED -> CHURNED -> POSSIBLE_RECOVERY -> (ADOPTED | CHURNED)

Each allowed transition has its own probability model (logistic in transition-specific
covariates). ``JourneyModel.fit`` estimates them from a stage-transition panel; the ABM uses
the same structure with simulation parameters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsdvc.core.literature import cites
from dsdvc.core.stats import fit_logit
from dsdvc.data.schemas import JourneyStage as S

STAGES = [s.value for s in S]
STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}

TRANSITIONS: dict[str, list[str]] = {
    "UNAWARE": ["AWARE"],
    "AWARE": ["INTERESTED"],
    "INTERESTED": ["CONSIDERING"],
    "CONSIDERING": ["TRIAL"],
    "TRIAL": ["ADOPTED", "CHURNED"],
    "ADOPTED": ["ENGAGED", "DISSATISFIED"],
    "ENGAGED": ["ADVOCATE", "DISSATISFIED"],
    "ADVOCATE": ["DISSATISFIED"],
    "DISSATISFIED": ["CHURNED", "ENGAGED"],
    "CHURNED": ["POSSIBLE_RECOVERY"],
    "POSSIBLE_RECOVERY": ["ADOPTED", "CHURNED"],
}

ACTIVE_CUSTOMER = {"ADOPTED", "ENGAGED", "ADVOCATE", "DISSATISFIED"}
FUNNEL = ["UNAWARE", "AWARE", "INTERESTED", "CONSIDERING", "TRIAL", "ADOPTED", "ENGAGED", "ADVOCATE"]


def is_allowed(a: str, b: str) -> bool:
    return b == a or b in TRANSITIONS.get(a, [])


@cites("rogers2003")
class JourneyModel:
    """Per-transition logistic models fitted on panel rows
    (consumer_id, period, stage_from, stage_to, <covariates>)."""

    def __init__(self, covariates: list[str]):
        self.covariates = covariates
        self.models: dict[tuple[str, str], object] = {}

    def fit(self, panel: pd.DataFrame, min_rows: int = 30) -> pd.DataFrame:
        bad = panel[[not is_allowed(a, b) for a, b in zip(panel.stage_from, panel.stage_to)]]
        if len(bad):
            raise ValueError(f"{len(bad)} rows contain transitions not allowed by the journey state machine")
        rows = []
        for a, nexts in TRANSITIONS.items():
            sub = panel[panel.stage_from == a]
            for b in nexts:
                if len(sub) < min_rows:
                    rows.append({"from": a, "to": b, "n": len(sub), "status": "DATA_UNAVAILABLE"})
                    continue
                y = (sub.stage_to == b).to_numpy()
                res = fit_logit(sub[self.covariates], y, l2=1.0)
                self.models[(a, b)] = res
                rows.append({"from": a, "to": b, "n": len(sub), "rate": float(y.mean()), "status": "estimated",
                             **{f"b_{k}": v for k, v in zip(res.names, np.round(res.coef, 3))}})
        return pd.DataFrame(rows)

    def transition_probs(self, stage: str, x: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {}
        for b in TRANSITIONS.get(stage, []):
            m = self.models.get((stage, b))
            if m is not None:
                out[b] = m.predict(x[self.covariates].to_numpy())
        return out


def funnel_table(stage_counts: dict[str, int]) -> pd.DataFrame:
    """Cumulative 'reached at least stage k' funnel with step conversion rates."""
    order = FUNNEL
    reached = []
    for i, s in enumerate(order):
        later = set(order[i:]) | ({"DISSATISFIED", "CHURNED", "POSSIBLE_RECOVERY"} if i <= order.index("ADOPTED") else set())
        reached.append(sum(stage_counts.get(k, 0) for k in later))
    df = pd.DataFrame({"stage": order, "reached": reached})
    df["step_rate"] = df["reached"] / df["reached"].shift(1)
    return df
