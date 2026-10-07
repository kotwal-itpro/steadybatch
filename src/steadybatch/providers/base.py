"""The interface every batch provider implements.

A provider only has to do four things: say what its limits are, submit a
list of requests, report a batch's status, and hand back whatever lines it
has. Everything else (retries, checks, ordering, checkpoints) lives in the
runner, so every provider gets the same treatment and the comparison is fair.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Iterator

from ..models import BatchStatus, PreparedRequest, RawResult


class BatchProvider(ABC):
    name: str = "base"
    max_requests_per_batch: int = 10_000
    max_bytes_per_batch: int = 100 * 1024 * 1024

    def size_of(self, req: PreparedRequest) -> int:
        """Rough size of one request line in the upload file."""
        return len(json.dumps(self.to_line(req)).encode("utf-8")) + 1

    def to_line(self, req: PreparedRequest) -> dict:
        """The provider-specific JSON for one request. Used for sizing and upload."""
        return {"custom_id": req.custom_id, "messages": req.request.messages}

    def is_over_capacity(self, exc: Exception) -> bool:
        """True if `submit` failed because the account's queue or quota is full.

        The runner then sends smaller batches instead of stopping. Providers that
        can tell this apart from other errors override it.
        """
        return False

    @abstractmethod
    def submit(self, batch: list[PreparedRequest]) -> str:
        """Send a batch. Return the provider's batch ID."""

    @abstractmethod
    def status(self, batch_id: str) -> BatchStatus:
        """Report where a batch is."""

    @abstractmethod
    def results(self, batch_id: str) -> Iterator[RawResult]:
        """Yield every line the provider returned, successes and errors alike."""
