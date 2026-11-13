"""Publish the tracker data as a new version of a Zenodo dataset (one per month).

The first run creates the dataset; later runs add a version to it, so there is one concept DOI
that always resolves to the newest data, plus one DOI per monthly snapshot. The record ID is kept
in a small state file that the workflow commits. Standard library only.

    ZENODO_TOKEN=... python scripts/zenodo_snapshot.py --state tracker/zenodo.json [--dry-run]

Does nothing without ZENODO_TOKEN, if this month is already published, or if there is no data.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION = """<p>Measurements of the OpenAI, Gemini and Anthropic batch APIs, taken four times a day by the
Batch API Tracker (<a href="https://kotwal-itpro.github.io/steadybatch/">live page</a>): turnaround from
the providers' own timestamps, lost and broken lines, answer drift between runs (as hashes, never the
answers), accuracy against known labels, model versions and cost. Includes a weekly 5,000-request batch
per provider. The inputs are synthetic support tickets, the same every run.</p>
<p>Files: runs.jsonl (one row per batch), runs.csv, the tracker config and the inputs. Code:
<a href="https://github.com/kotwal-itpro/steadybatch">steadybatch</a>.</p>"""


class Zenodo:
    def __init__(self, token: str, sandbox: bool):
        self.base = "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"
        self.token = token

    def call(self, method: str, url: str, body: dict | None = None, data: bytes | None = None) -> dict:
        if not url.startswith("http"):
            url = self.base + url
        headers = {"Authorization": f"Bearer {self.token}", "User-Agent": "steadybatch-data-release"}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        elif data is not None:
            headers["Content-Type"] = "application/octet-stream"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
        except urllib.error.HTTPError as exc:
            sys.exit(f"Zenodo {method} {url}: HTTP {exc.code}: {exc.read()[:500]!r}")
        return json.loads(raw) if raw else {}


def bundle() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in ["tracker/data/runs.jsonl", "tracker/data/selfhost.jsonl", "tracker/config.json",
                    "tracker/prices.json", "tracker/inputs.jsonl", "tracker/inputs-5k.jsonl",
                    "examples/support_ticket.schema.json", "examples/support_ticket.instructions.txt",
                    "docs/TRACKER-DESIGN.md"]:
            if (ROOT / rel).exists():
                z.write(ROOT / rel, rel)
    return buf.getvalue()


def metadata(month: str, n_runs: int) -> dict:
    return {"upload_type": "dataset",
            "title": "Batch API Tracker: measured turnaround, completeness and drift of LLM batch APIs",
            "creators": [{"name": "Kotwal, Ankur", "orcid": "0009-0009-4790-6866"}],
            "description": DESCRIPTION, "version": month, "license": "cc-by-4.0",
            "publication_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "notes": f"Snapshot of {n_runs} batch runs, through the end of {month}.",
            "keywords": ["LLM", "batch inference", "batch API", "benchmark", "reliability", "OpenAI",
                         "Gemini", "Anthropic"],
            "related_identifiers": [
                {"identifier": "https://github.com/kotwal-itpro/steadybatch", "relation": "isSupplementTo",
                 "resource_type": "software"},
                {"identifier": "https://kotwal-itpro.github.io/steadybatch/", "relation": "isDocumentedBy"}]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="tracker/zenodo.json")
    ap.add_argument("--dry-run", action="store_true", help="build the bundle and metadata, don't upload")
    args = ap.parse_args()
    state_path = ROOT / args.state
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    runs = ROOT / "tracker/data/runs.jsonl"
    n_runs = sum(1 for line in runs.read_text().splitlines() if line.strip()) if runs.exists() else 0
    # The snapshot on the 1st covers the month that just ended.
    now = datetime.now(timezone.utc)
    month = f"{now.year - (now.month == 1)}-{(now.month - 2) % 12 + 1:02d}"
    if n_runs == 0:
        print("no data yet, nothing to publish"); return
    if state.get("month") == month:
        print(f"{month} is already published ({state.get('doi')})"); return
    zipped, meta = bundle(), metadata(month, n_runs)
    if args.dry_run:
        print(json.dumps(meta, indent=1)); print(f"bundle: {len(zipped):,} bytes"); return
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ZENODO_TOKEN is not set, skipping"); return
    sandbox = os.environ.get("ZENODO_SANDBOX", "").lower() == "true"
    if state.get("sandbox", False) != sandbox and state.get("record_id"):
        sys.exit("the state file belongs to the other Zenodo (sandbox vs. real); move it aside first")
    z = Zenodo(token, sandbox)
    if state.get("record_id"):
        parent = z.call("POST", f"/deposit/depositions/{state['record_id']}/actions/newversion")
        draft = z.call("GET", parent["links"]["latest_draft"])
        for f in draft.get("files", []):  # a new version starts with the old files; replace them
            z.call("DELETE", f"/deposit/depositions/{draft['id']}/files/{f['id']}")
    else:
        draft = z.call("POST", "/deposit/depositions", body={})
    z.call("PUT", f"{draft['links']['bucket']}/batch-api-tracker-{month}.zip", data=zipped)
    z.call("PUT", f"/deposit/depositions/{draft['id']}", body={"metadata": meta})
    done = z.call("POST", f"/deposit/depositions/{draft['id']}/actions/publish")
    state = {"record_id": done["id"], "doi": done.get("doi"), "concept_doi": done.get("conceptdoi"),
             "month": month, "sandbox": sandbox, "published_at": now.isoformat(timespec="seconds")}
    state_path.write_text(json.dumps(state, indent=1) + "\n")
    print(f"published {month}: {state['doi']} (all versions: {state['concept_doi']})")


if __name__ == "__main__":
    main()
