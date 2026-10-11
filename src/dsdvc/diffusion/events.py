"""Event types and a lightweight event bus for the simulator (spec §52).

The simulator advances in discrete ticks (1 tick = 1 day by default); within a tick, state
changes are emitted as typed events and dispatched to subscribers. New mechanisms are added
by subscribing handlers rather than editing the core loop. High-volume events
(CONTENT_EXPOSED, CONTENT_ENGAGED, RECOMMENDATION_UPDATED) are aggregated per tick to keep
memory bounded; all others are stored individually.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import pandas as pd


class EventType(str, Enum):
    CONTENT_CREATED = "CONTENT_CREATED"
    CONTENT_EXPOSED = "CONTENT_EXPOSED"
    CONTENT_ENGAGED = "CONTENT_ENGAGED"
    CONTENT_SHARED = "CONTENT_SHARED"
    CROSS_PLATFORM_TRANSFER = "CROSS_PLATFORM_TRANSFER"
    AWARENESS_CREATED = "AWARENESS_CREATED"
    TRIAL_STARTED = "TRIAL_STARTED"
    ADOPTION_OCCURRED = "ADOPTION_OCCURRED"
    ADVOCACY_OCCURRED = "ADVOCACY_OCCURRED"
    CHURN_OCCURRED = "CHURN_OCCURRED"
    RECOMMENDATION_UPDATED = "RECOMMENDATION_UPDATED"
    COMMUNITY_REWIRED = "COMMUNITY_REWIRED"
    CAMPAIGN_LAUNCHED = "CAMPAIGN_LAUNCHED"
    COMPETITOR_ACTION = "COMPETITOR_ACTION"


AGGREGATED = {EventType.CONTENT_EXPOSED, EventType.CONTENT_ENGAGED, EventType.RECOMMENDATION_UPDATED,
              EventType.AWARENESS_CREATED}


@dataclass
class Event:
    t: int
    type: EventType
    agent: int = -1
    platform: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


class EventBus:
    def __init__(self):
        self._subs: dict[EventType, list[Callable[[Event], None]]] = defaultdict(list)
        self.log: list[Event] = []
        self.counts: dict[tuple[int, str], int] = defaultdict(int)

    def subscribe(self, etype: EventType, handler: Callable[[Event], None]) -> None:
        self._subs[etype].append(handler)

    def emit(self, ev: Event) -> None:
        self.counts[(ev.t, ev.type.value)] += 1
        if ev.type not in AGGREGATED:
            self.log.append(ev)
        for h in self._subs.get(ev.type, []):
            h(ev)

    def emit_count(self, t: int, etype: EventType, n: int, platform: str | None = None) -> None:
        """Aggregated emission for high-volume events (handlers receive one summary event)."""
        if n <= 0:
            return
        self.counts[(t, etype.value)] += int(n)
        for h in self._subs.get(etype, []):
            h(Event(t, etype, platform=platform, data={"count": int(n)}))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"t": e.t, "type": e.type.value, "agent": e.agent, "platform": e.platform, **e.data}
                             for e in self.log])

    def count_frame(self) -> pd.DataFrame:
        rows = [{"t": t, "type": k, "count": v} for (t, k), v in self.counts.items()]
        return pd.DataFrame(rows).pivot_table(index="t", columns="type", values="count", fill_value=0) if rows else pd.DataFrame()
