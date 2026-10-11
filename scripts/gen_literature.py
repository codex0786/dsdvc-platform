"""Regenerate docs/LITERATURE.md from the code's literature registry and @cites decorators."""
import importlib
import inspect
import pkgutil
from pathlib import Path

import dsdvc
from dsdvc.core.literature import LITERATURE

uses: dict[str, set[str]] = {}
for m in pkgutil.walk_packages(dsdvc.__path__, "dsdvc."):
    if m.name.startswith(("dsdvc.dashboard", "dsdvc.api")):
        continue  # entry points with side effects
    mod = importlib.import_module(m.name)
    for name, obj in inspect.getmembers(mod):
        if getattr(obj, "__module__", None) == mod.__name__:
            for k in getattr(obj, "__literature__", []):
                uses.setdefault(k, set()).add(f"`{mod.__name__}.{name}`")

lines = ["# Literature traceability", "",
         "Generated from `dsdvc.core.literature.LITERATURE` and the `@cites` decorators in code "
         "(regenerate with `python scripts/gen_literature.py`). *Supports* states narrowly what the work established; "
         "*Does not establish* guards against over-claiming.", "", "Topic summaries: [docs/literature/](literature/).", ""]
for k, v in LITERATURE.items():
    lines += [f"### `{k}`", v["citation"], "", f"- **Supports:** {v['supports']}",
              f"- **Does not establish:** {v['does_not_establish']}",
              f"- **Used by:** {', '.join(sorted(uses.get(k, []))) or '_documentation only_'}", ""]
out = Path(__file__).resolve().parents[1] / "docs" / "LITERATURE.md"
out.write_text("\n".join(lines))
print(f"wrote {out} ({len(uses)} of {len(LITERATURE)} references cited in code)")
