"""Append one line of usage numbers to a JSONL file: GitHub stars, forks and watchers, GitHub
traffic (if TRAFFIC_TOKEN is set), and PyPI downloads. Standard library only.

GitHub keeps traffic for 14 days, so a daily snapshot is the only way to keep the history.
Traffic counts are GitHub's 14-day totals; compare snapshots 14 days apart, don't sum them.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def get(url: str, token: str | None = None) -> dict | list | None:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "steadybatch-usage-snapshot"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except Exception as exc:  # one source failing shouldn't lose the others
        print(f"{url}: {type(exc).__name__}: {exc}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--package", default="steadybatch")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    api = f"https://api.github.com/repos/{args.repo}"
    row: dict = {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
    repo = get(api, os.environ.get("GH_TOKEN"))
    if isinstance(repo, dict):
        row.update(stars=repo.get("stargazers_count"), forks=repo.get("forks_count"),
                   watchers=repo.get("subscribers_count"), open_issues=repo.get("open_issues_count"))
    token = os.environ.get("TRAFFIC_TOKEN")
    if token:
        views, clones = get(f"{api}/traffic/views", token), get(f"{api}/traffic/clones", token)
        refs, paths = get(f"{api}/traffic/popular/referrers", token), get(f"{api}/traffic/popular/paths", token)
        if isinstance(views, dict):
            row.update(views_14d=views.get("count"), unique_visitors_14d=views.get("uniques"),
                       views_by_day={v["timestamp"][:10]: [v["count"], v["uniques"]] for v in views.get("views", [])})
        if isinstance(clones, dict):
            row.update(clones_14d=clones.get("count"), unique_cloners_14d=clones.get("uniques"))
        if isinstance(refs, list):
            row["referrers_14d"] = {r["referrer"]: [r["count"], r["uniques"]] for r in refs}
        if isinstance(paths, list):
            row["top_paths_14d"] = {p["path"]: [p["count"], p["uniques"]] for p in paths[:10]}
    pypi = get(f"https://pypistats.org/api/packages/{args.package}/recent")
    if isinstance(pypi, dict):
        row["pypi_downloads"] = pypi.get("data")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
