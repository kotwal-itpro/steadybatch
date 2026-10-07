"""A small SQLite checkpoint so a job can stop and pick up where it left off.

If the process dies after submitting a batch, the batch ID is already
saved, so a restart polls that batch instead of paying for it twice.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from .models import Outcome, PreparedRequest, Request, Result

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    custom_id   TEXT PRIMARY KEY,
    ordinal     INTEGER NOT NULL,
    key         TEXT NOT NULL,
    model       TEXT NOT NULL,
    request     TEXT NOT NULL,
    schema      TEXT,
    state       TEXT NOT NULL DEFAULT 'pending',   -- pending | submitted | ok | failed
    attempts    INTEGER NOT NULL DEFAULT 0,
    batch_id    TEXT,
    outcome     TEXT,
    data        TEXT,
    text        TEXT,
    error       TEXT,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    history     TEXT NOT NULL DEFAULT '[]'
);
CREATE UNIQUE INDEX IF NOT EXISTS requests_key ON requests(key);
CREATE TABLE IF NOT EXISTS batches (
    batch_id     TEXT PRIMARY KEY,
    provider     TEXT NOT NULL,
    custom_ids   TEXT NOT NULL,
    submitted_at REAL NOT NULL,
    finished_at  REAL,
    state        TEXT NOT NULL DEFAULT 'open'      -- open | collected
);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)

    def close(self) -> None:
        self.db.close()

    # --- requests -----------------------------------------------------

    def add(self, prepared: list[PreparedRequest]) -> int:
        """Register work. Adding the same request again is a no-op."""
        start = self.db.execute("SELECT COALESCE(MAX(ordinal), -1) + 1 FROM requests").fetchone()[0]
        added = 0
        with self.db:
            for i, p in enumerate(prepared):
                cur = self.db.execute(
                    "INSERT OR IGNORE INTO requests (custom_id, ordinal, key, model, request, schema) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        p.custom_id,
                        start + i,
                        p.request.key,
                        p.model,
                        json.dumps(asdict(p.request)),
                        json.dumps(p.response_schema) if p.response_schema else None,
                    ),
                )
                if cur.rowcount == 0:
                    existing = self.db.execute(
                        "SELECT custom_id FROM requests WHERE key = ?", (p.request.key,)
                    ).fetchone()
                    if existing and existing["custom_id"] != p.custom_id:
                        raise ValueError(
                            f"key {p.request.key!r} is already in this job with different settings; "
                            "use a new checkpoint file or a new key"
                        )
                added += cur.rowcount
        return added

    def _prepared(self, row: sqlite3.Row) -> PreparedRequest:
        return PreparedRequest(
            custom_id=row["custom_id"],
            request=Request(**json.loads(row["request"])),
            model=row["model"],
            response_schema=json.loads(row["schema"]) if row["schema"] else None,
        )

    def ready_to_submit(self, max_attempts: int) -> list[PreparedRequest]:
        rows = self.db.execute(
            "SELECT * FROM requests WHERE state = 'pending' AND attempts < ? ORDER BY ordinal",
            (max_attempts,),
        ).fetchall()
        return [self._prepared(r) for r in rows]

    def mark_submitted(self, batch_id: str, provider: str, custom_ids: list[str]) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO batches (batch_id, provider, custom_ids, submitted_at) VALUES (?, ?, ?, ?)",
                (batch_id, provider, json.dumps(custom_ids), time.time()),
            )
            self.db.executemany(
                "UPDATE requests SET state = 'submitted', attempts = attempts + 1, batch_id = ? "
                "WHERE custom_id = ?",
                [(batch_id, cid) for cid in custom_ids],
            )

    def record(self, custom_id: str, outcome: Outcome, *, retryable: bool, data=None, text=None,
               error=None, input_tokens: int = 0, output_tokens: int = 0) -> None:
        if outcome is Outcome.OK:
            state = "ok"
        else:
            state = "pending" if retryable else "failed"
        row = self.db.execute("SELECT history FROM requests WHERE custom_id = ?", (custom_id,)).fetchone()
        if row is None:
            return  # a line we never asked for; ignore it
        history = json.loads(row["history"]) + [outcome.value + (f": {error}" if error else "")]
        with self.db:
            self.db.execute(
                "UPDATE requests SET state = ?, outcome = ?, data = ?, text = ?, error = ?, "
                "input_tokens = input_tokens + ?, output_tokens = output_tokens + ?, history = ? "
                "WHERE custom_id = ?",
                (state, outcome.value, json.dumps(data) if data is not None else None, text, error,
                 input_tokens, output_tokens, json.dumps(history), custom_id),
            )

    def give_up_on_exhausted(self, max_attempts: int) -> None:
        with self.db:
            self.db.execute(
                "UPDATE requests SET state = 'failed' WHERE state = 'pending' AND attempts >= ?",
                (max_attempts,),
            )

    # --- batches ------------------------------------------------------

    def open_batches(self) -> list[tuple[str, list[str]]]:
        rows = self.db.execute(
            "SELECT batch_id, custom_ids FROM batches WHERE state = 'open' ORDER BY submitted_at"
        ).fetchall()
        return [(r["batch_id"], json.loads(r["custom_ids"])) for r in rows]

    def release_batch(self, batch_id: str) -> None:
        """The provider refused this batch before doing any work: put its requests
        back as pending and give back the attempt, then close the batch."""
        with self.db:
            self.db.execute(
                "UPDATE requests SET state = 'pending', attempts = attempts - 1, batch_id = NULL "
                "WHERE batch_id = ? AND state = 'submitted'",
                (batch_id,),
            )
        self.close_batch(batch_id)

    def close_batch(self, batch_id: str) -> None:
        with self.db:
            self.db.execute(
                "UPDATE batches SET state = 'collected', finished_at = ? WHERE batch_id = ?",
                (time.time(), batch_id),
            )

    def batch_timings(self) -> list[tuple[str, float, float | None]]:
        rows = self.db.execute("SELECT batch_id, submitted_at, finished_at FROM batches").fetchall()
        return [(r["batch_id"], r["submitted_at"], r["finished_at"]) for r in rows]

    # --- results ------------------------------------------------------

    def results(self) -> Iterator[Result]:
        for r in self.db.execute("SELECT * FROM requests ORDER BY ordinal"):
            yield Result(
                key=r["key"],
                custom_id=r["custom_id"],
                outcome=Outcome(r["outcome"]) if r["outcome"] else Outcome.MISSING,
                data=json.loads(r["data"]) if r["data"] else None,
                text=r["text"],
                error=r["error"],
                attempts=r["attempts"],
                input_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"],
                history=json.loads(r["history"]),
            )

    def unfinished(self) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM requests WHERE state IN ('pending', 'submitted')"
        ).fetchone()[0]
