"""Optional LLM narrator (Claude) for the executive summary.

The narrator only rewrites computed evidence into prose. A numeric guard rejects any
narrative containing a number that does not appear in the evidence payload, so the LLM
cannot introduce figures. Without an API key/credentials the agent runs fully
deterministically and this module is never imported.
"""
from __future__ import annotations

import json
import re

SYSTEM = ("You are a marketing-science analyst. Write a concise executive summary (<= 180 words) of the evidence "
          "provided. Use ONLY numbers that appear in the evidence. Preserve epistemic labels (observed, estimated, "
          "simulated, predicted, recommended, DATA_UNAVAILABLE). Never say content 'will go viral'; use conditional "
          "probabilities. Mention the main data limitations.")


def _numbers(s: str) -> set[str]:
    return {n.rstrip(".").lstrip("0") or "0" for n in re.findall(r"\d+(?:\.\d+)?", s.replace(",", ""))}


class ClaudeNarrator:
    def __init__(self, model: str = "claude-opus-5-5", client=None):
        import anthropic
        self.client = client or anthropic.Anthropic()
        self.model = model

    def narrate(self, mem, sections: dict) -> str:
        evidence = {
            "request": mem.request,
            "recommendations": [c.model_dump(mode="json") for c in mem.recommendations],
            "conclusions": [c.model_dump(mode="json") for c in mem.conclusions],
            "unavailable": mem.unavailable,
            "expected_outcomes": {k: v for k, v in (sections.get("expected_outcomes") or {}).items()
                                  if isinstance(v, (int, float, str))},
        }
        payload = json.dumps(evidence, default=str)
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=2000,
            system=SYSTEM,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": f"Evidence (JSON):\n{payload}"}],
        )
        if resp.stop_reason == "refusal":
            return ""
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        extra = _numbers(text) - _numbers(payload)
        if extra:
            return ("[LLM narrative withheld: it contained numbers not present in the computed evidence: "
                    + ", ".join(sorted(extra)[:10]) + "]")
        return text
