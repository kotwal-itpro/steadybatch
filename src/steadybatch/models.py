"""Plain data types shared by the runner and the providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class Request:
    """One unit of work: a record you want the model to process.

    ``key`` is your own identifier for the record (a row id, a file name).
    It must be unique within a job; steadybatch uses it to put results
    back in order and to make retries safe.
    """

    key: str
    messages: list[dict[str, str]]
    system: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0


@dataclass(frozen=True)
class PreparedRequest:
    """A Request plus everything a provider needs to send it."""

    custom_id: str
    request: Request
    model: str
    response_schema: dict[str, Any] | None = None


class BatchState(str, Enum):
    PENDING = "pending"      # accepted, not started
    RUNNING = "running"
    DONE = "done"            # provider says it finished (results may still be incomplete)
    FAILED = "failed"        # the whole batch failed or expired


@dataclass
class BatchStatus:
    state: BatchState
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    detail: str = ""
    # True when the provider refused the work because the account's queue is full,
    # not because anything is wrong with the requests.
    over_capacity: bool = False


@dataclass
class RawResult:
    """What a provider returned for one line, before any checking."""

    custom_id: str
    ok: bool
    text: str | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class Outcome(str, Enum):
    OK = "ok"
    ERROR = "error"          # the provider returned an error for this line
    MISSING = "missing"      # the provider never returned this line at all
    INVALID = "invalid"      # a response came back but failed the schema check


@dataclass
class Result:
    key: str
    custom_id: str
    outcome: Outcome
    data: Any = None
    text: str | None = None
    error: str | None = None
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    history: list[str] = field(default_factory=list)
