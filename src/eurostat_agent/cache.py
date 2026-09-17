"""Execution-scoped metadata caching for deterministic Eurostat access."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from eurostat_agent.data import Observation
from eurostat_agent.metadata import (
    Codelist,
    CodelistRef,
    DatasetMetadata,
    DataStructure,
)
from eurostat_agent.tracing import TraceRecorder


@dataclass(frozen=True)
class CacheEntry[T]:
    """One immutable cached value and the time it entered the cache."""

    value: T
    cached_at: datetime


@dataclass(frozen=True)
class CacheResult[T]:
    """One cache lookup result with explicit hit/miss information."""

    value: T
    hit: bool
    cached_at: datetime


class CacheableClient(Protocol):
    """Client surface needed by the execution-scoped caching wrapper."""

    def get_structure(self, dataset_code: str) -> DataStructure: ...

    def get_codelist(self, ref: CodelistRef) -> Codelist: ...

    def get_dataset_metadata(self, dataset_code: str) -> DatasetMetadata: ...

    def fetch_series(
        self,
        dataset_code: str,
        filters: dict[str, str],
    ) -> tuple[Observation, ...]: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


@contextmanager
def _trace_cache_access(
    recorder: TraceRecorder | None,
    *,
    resource: str,
    cache_key: str,
    hit: bool,
) -> Iterator[None]:
    if recorder is None:
        yield
        return

    with recorder.stage(
        "metadata_cache",
        metadata=(
            ("resource", resource),
            ("cache_key", cache_key),
            ("cache_hit", str(hit).lower()),
        ),
    ):
        yield


class MetadataCache:
    """In-memory structural metadata cache intended for one execution.

    DataStructure and versioned Codelist objects are cached. DatasetMetadata is
    deliberately excluded because it carries source-freshness information used
    by statistical provenance.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._clock = clock
        self._structures: dict[str, CacheEntry[DataStructure]] = {}
        self._codelists: dict[CodelistRef, CacheEntry[Codelist]] = {}

    def _cached_at(self) -> datetime:
        cached_at = self._clock()
        if cached_at.tzinfo is None or cached_at.utcoffset() is None:
            raise ValueError("Cache timestamp must be timezone-aware.")
        return cached_at

    def get_structure(
        self,
        dataset_code: str,
        loader: Callable[[str], DataStructure],
        *,
        trace_recorder: TraceRecorder | None = None,
    ) -> CacheResult[DataStructure]:
        """Return one dataset structure, loading it only on the first lookup."""

        key = dataset_code.casefold()
        entry = self._structures.get(key)
        hit = entry is not None

        with _trace_cache_access(
            trace_recorder,
            resource="data_structure",
            cache_key=key,
            hit=hit,
        ):
            if entry is None:
                entry = CacheEntry(
                    value=loader(dataset_code),
                    cached_at=self._cached_at(),
                )
                self._structures[key] = entry

        return CacheResult(
            value=entry.value,
            hit=hit,
            cached_at=entry.cached_at,
        )

    def get_codelist(
        self,
        ref: CodelistRef,
        loader: Callable[[CodelistRef], Codelist],
        *,
        trace_recorder: TraceRecorder | None = None,
    ) -> CacheResult[Codelist]:
        """Return one exact versioned codelist, loading it once per execution."""

        entry = self._codelists.get(ref)
        hit = entry is not None
        cache_key = f"{ref.agency}/{ref.id}/{ref.version}"

        with _trace_cache_access(
            trace_recorder,
            resource="codelist",
            cache_key=cache_key,
            hit=hit,
        ):
            if entry is None:
                entry = CacheEntry(
                    value=loader(ref),
                    cached_at=self._cached_at(),
                )
                self._codelists[ref] = entry

        return CacheResult(
            value=entry.value,
            hit=hit,
            cached_at=entry.cached_at,
        )

    def clear(self) -> None:
        """Explicitly invalidate all structural metadata cached in this instance."""

        self._structures.clear()
        self._codelists.clear()


class MetadataCachingClient:
    """Client wrapper that caches only structural metadata for one execution."""

    def __init__(
        self,
        source: CacheableClient,
        *,
        cache: MetadataCache | None = None,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        self._source = source
        self._cache = cache if cache is not None else MetadataCache()
        self._trace_recorder = trace_recorder

    @property
    def cache(self) -> MetadataCache:
        """Expose the execution-scoped cache for explicit inspection/invalidation."""

        return self._cache

    def get_structure(self, dataset_code: str) -> DataStructure:
        """Return a cached dataset structure when available."""

        return self._cache.get_structure(
            dataset_code,
            self._source.get_structure,
            trace_recorder=self._trace_recorder,
        ).value

    def get_codelist(self, ref: CodelistRef) -> Codelist:
        """Return a cached exact-version codelist when available."""

        return self._cache.get_codelist(
            ref,
            self._source.get_codelist,
            trace_recorder=self._trace_recorder,
        ).value

    def get_dataset_metadata(self, dataset_code: str) -> DatasetMetadata:
        """Retrieve provenance metadata live rather than caching freshness data."""

        return self._source.get_dataset_metadata(dataset_code)

    def fetch_series(
        self,
        dataset_code: str,
        filters: dict[str, str],
    ) -> tuple[Observation, ...]:
        """Retrieve observations live; statistical observations are not cached."""

        return self._source.fetch_series(dataset_code, filters)
