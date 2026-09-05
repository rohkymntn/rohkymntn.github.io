#!/usr/bin/env python3
"""Assemble projects.json from projects/<slug>/meta.json + report.html.

    python3 scripts/build_projects.py

Each project folder holds:
  meta.json   - tag, title, desc, tech[], date, hero, heroCaption
  report.html - the article body (HTML fragment) shown by the site's reader
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
out = []
for folder in sorted(p for p in (ROOT / "projects").iterdir() if p.is_dir()):
    meta_path, body_path = folder / "meta.json", folder / "report.html"
    if not meta_path.exists():
        continue
    meta = json.loads(meta_path.read_text())
    meta.setdefault("slug", folder.name)
    if body_path.exists():
        meta["body"] = body_path.read_text()
    out.append(meta)
out.sort(key=lambda m: m.get("order", 0))
(ROOT / "projects.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"wrote projects.json with {len(out)} project(s)")
