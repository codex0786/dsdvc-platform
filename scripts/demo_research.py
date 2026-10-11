"""Run the research agent end to end and write a report (synthetic mode is explicit and labelled)."""
import argparse
from pathlib import Path

from dsdvc.research_agent.agent import MarketingResearchAgent, ResearchInputs
from dsdvc.research_agent.report import render_markdown

ap = argparse.ArgumentParser()
ap.add_argument("--request", default="I want to launch a premium skincare brand in India targeting women 18-30.")
ap.add_argument("--synthetic", action="store_true", help="explicitly allow labelled SYNTHETIC data to fill gaps")
ap.add_argument("--budget", type=float, default=600_000)
ap.add_argument("--out", default="outputs/research_report.md")
a = ap.parse_args()
mem, S = MarketingResearchAgent(allow_synthetic=a.synthetic).run(a.request, ResearchInputs(budget=a.budget))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(render_markdown(mem, S))
mem.save("research_memory")
print(f"wrote {a.out}")
