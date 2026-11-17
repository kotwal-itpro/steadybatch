#!/usr/bin/env python3
"""Start the tracker workflow on time when GitHub's own schedule runs late.

GitHub starts the */30 schedule only every few hours, so some submit slots are missed. Run this
every 15 minutes from any always-on machine with `gh` signed in (a launchd job, cron, ...). It
starts the workflow only when it is needed:

- a submit slot has started and no batch has been sent for it yet (dispatch with submit=true;
  submit itself still sends at most once per slot), or
- a batch is open and no run has started in the last 25 minutes (dispatch to collect).

It never starts a run while another tracker run is queued or running. Reads only public files.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

REPO = "kotwal-itpro/steadybatch"
RAW = f"https://raw.githubusercontent.com/{REPO}/main/"
SUBMIT_WINDOW = timedelta(hours=5)   # after this, leave the slot to the next one
COLLECT_GAP = timedelta(minutes=25)


def fetch(path: str) -> str:
    req = urllib.request.Request(RAW + path, headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def slot_start(now: datetime, hours: list[int]) -> datetime:
    day = now.replace(minute=0, second=0, microsecond=0)
    for back in range(0, 2):
        d = day - timedelta(days=back)
        for h in sorted(hours, reverse=True):
            s = d.replace(hour=h)
            if s <= now:
                return s
    raise ValueError("no slot hours")


def parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def main() -> int:
    now = datetime.now(timezone.utc)
    runs = json.loads(gh("run", "list", "-R", REPO, "--workflow", "tracker.yml", "-L", "5",
                         "--json", "status,createdAt"))
    if any(r["status"] in ("queued", "in_progress", "waiting", "pending") for r in runs):
        print(f"{now:%F %T} a run is already queued or running")
        return 0
    last_start = max((parse(r["createdAt"]) for r in runs), default=now - timedelta(days=1))

    cfg = json.loads(fetch("tracker/config.json"))
    open_batches = json.loads(fetch("tracker/open.json"))
    start = slot_start(now, cfg.get("slot_hours", [1, 7, 13, 19]))

    sent = [datetime.fromtimestamp(b["submitted_ts"], timezone.utc)
            for b in open_batches if b.get("submitted_ts")]
    for line in fetch("tracker/data/runs.jsonl").splitlines()[-40:]:
        if line.strip():
            row = json.loads(line)
            if row.get("submitted_at"):
                sent.append(parse(row["submitted_at"]))
    slot_done = any(s >= start for s in sent)

    if not slot_done and now - start < SUBMIT_WINDOW:
        why, submit = f"slot {start:%H}:00 UTC not sent yet", "true"
    elif open_batches and now - last_start > COLLECT_GAP:
        why, submit = f"{len(open_batches)} open batch(es) to collect", "false"
    else:
        print(f"{now:%F %T} nothing to do (slot {start:%H}:00 sent={slot_done}, open={len(open_batches)})")
        return 0
    if "--dry-run" not in sys.argv:
        gh("workflow", "run", "tracker.yml", "-R", REPO, "-f", f"submit={submit}")
    print(f"{now:%F %T} started the tracker: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
