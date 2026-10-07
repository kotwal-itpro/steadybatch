"""Split work into batches that stay under a provider's limits.

Providers cap a batch by request count and by file size. Going over the
size cap does not always produce a clear error, so we stay under both,
with a safety margin on size.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")


def chunk(
    items: Iterable[T],
    max_count: int,
    max_bytes: int,
    size_of: Callable[[T], int],
    margin: float = 0.9,
) -> Iterator[list[T]]:
    if max_count < 1 or max_bytes < 1:
        raise ValueError("limits must be positive")
    byte_budget = int(max_bytes * margin)
    batch: list[T] = []
    used = 0
    for item in items:
        size = size_of(item)
        if size > byte_budget:
            raise ValueError(
                f"a single request is {size} bytes, over the per-batch budget of {byte_budget}"
            )
        if batch and (len(batch) >= max_count or used + size > byte_budget):
            yield batch
            batch, used = [], 0
        batch.append(item)
        used += size
    if batch:
        yield batch
