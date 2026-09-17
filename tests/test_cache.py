from datetime import UTC, datetime

from eurostat_agent.cache import MetadataCache, MetadataCachingClient
from eurostat_agent.data import Observation
from eurostat_agent.metadata import (
    Code,
    Codelist,
    CodelistRef,
    DatasetMetadata,
    DataStructure,
    Dimension,
)
from eurostat_agent.tracing import TraceRecorder


def _structure() -> DataStructure:
    return DataStructure(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="72.0",
        dimensions=(
            Dimension(
                id="sex",
                position=3,
                codelist=CodelistRef(
                    agency="ESTAT",
                    id="SEX",
                    version="2.0",
                ),
            ),
        ),
        measure_id="OBS_VALUE",
    )


def _codelist() -> Codelist:
    return Codelist(
        id="SEX",
        agency="ESTAT",
        version="2.0",
        codes=(
            Code(id="F", label="Females"),
            Code(id="M", label="Males"),
        ),
    )


def test_metadata_cache_returns_explicit_miss_then_hit() -> None:
    calls = 0
    cached_at = datetime(2026, 9, 17, 18, 0, tzinfo=UTC)
    cache = MetadataCache(clock=lambda: cached_at)

    def load_structure(dataset_code: str) -> DataStructure:
        nonlocal calls
        calls += 1
        assert dataset_code == "DEMO_PJAN"
        return _structure()

    first = cache.get_structure("DEMO_PJAN", load_structure)
    second = cache.get_structure("demo_pjan", load_structure)

    assert calls == 1
    assert first.hit is False
    assert second.hit is True
    assert first.value == second.value == _structure()
    assert first.cached_at == second.cached_at == cached_at


def test_metadata_cache_keys_codelists_by_exact_versioned_reference() -> None:
    calls = 0
    cache = MetadataCache(clock=lambda: datetime(2026, 9, 17, 18, 0, tzinfo=UTC))
    sex_2 = CodelistRef(agency="ESTAT", id="SEX", version="2.0")
    sex_3 = CodelistRef(agency="ESTAT", id="SEX", version="3.0")

    def load_codelist(ref: CodelistRef) -> Codelist:
        nonlocal calls
        calls += 1
        return Codelist(
            id=ref.id,
            agency=ref.agency,
            version=ref.version,
            codes=(),
        )

    assert cache.get_codelist(sex_2, load_codelist).hit is False
    assert cache.get_codelist(sex_2, load_codelist).hit is True
    assert cache.get_codelist(sex_3, load_codelist).hit is False
    assert calls == 2


def test_metadata_caching_client_does_not_cache_provenance_or_observations() -> None:
    class Source:
        def __init__(self) -> None:
            self.structure_calls = 0
            self.codelist_calls = 0
            self.metadata_calls = 0
            self.series_calls = 0

        def get_structure(self, dataset_code: str) -> DataStructure:
            self.structure_calls += 1
            return _structure()

        def get_codelist(self, ref: CodelistRef) -> Codelist:
            self.codelist_calls += 1
            return _codelist()

        def get_dataset_metadata(self, dataset_code: str) -> DatasetMetadata:
            self.metadata_calls += 1
            return DatasetMetadata(
                id="DEMO_PJAN",
                agency="ESTAT",
                version="1.0",
                data_updated_at="2026-08-14T23:00:00+0200",
            )

        def fetch_series(
            self,
            dataset_code: str,
            filters: dict[str, str],
        ) -> tuple[Observation, ...]:
            self.series_calls += 1
            return (
                Observation(
                    dataset_code="DEMO_PJAN",
                    dimensions={"unit": "NR"},
                    time_period="2024",
                    value=1.0,
                ),
            )

    source = Source()
    client = MetadataCachingClient(source)
    ref = CodelistRef(agency="ESTAT", id="SEX", version="2.0")

    client.get_structure("DEMO_PJAN")
    client.get_structure("DEMO_PJAN")
    client.get_codelist(ref)
    client.get_codelist(ref)

    client.get_dataset_metadata("DEMO_PJAN")
    client.get_dataset_metadata("DEMO_PJAN")
    client.fetch_series("DEMO_PJAN", {"TIME_PERIOD": "2024"})
    client.fetch_series("DEMO_PJAN", {"TIME_PERIOD": "2024"})

    assert source.structure_calls == 1
    assert source.codelist_calls == 1
    assert source.metadata_calls == 2
    assert source.series_calls == 2


def test_metadata_cache_records_observed_hits_and_misses_in_trace() -> None:
    class Source:
        def get_structure(self, dataset_code: str) -> DataStructure:
            return _structure()

        def get_codelist(self, ref: CodelistRef) -> Codelist:
            return _codelist()

        def get_dataset_metadata(self, dataset_code: str) -> DatasetMetadata:
            return DatasetMetadata(
                id="DEMO_PJAN",
                agency="ESTAT",
                version="1.0",
                data_updated_at="2026-08-14T23:00:00+0200",
            )

        def fetch_series(
            self,
            dataset_code: str,
            filters: dict[str, str],
        ) -> tuple[Observation, ...]:
            return ()

    recorder = TraceRecorder("execution-cache")
    client = MetadataCachingClient(Source(), trace_recorder=recorder)

    client.get_structure("DEMO_PJAN")
    client.get_structure("demo_pjan")

    assert tuple(event.stage for event in recorder.trace.events) == (
        "metadata_cache",
        "metadata_cache",
    )
    assert dict(recorder.trace.events[0].metadata) == {
        "resource": "data_structure",
        "cache_key": "demo_pjan",
        "cache_hit": "false",
    }
    assert dict(recorder.trace.events[1].metadata) == {
        "resource": "data_structure",
        "cache_key": "demo_pjan",
        "cache_hit": "true",
    }


def test_metadata_cache_clear_explicitly_invalidates_entries() -> None:
    calls = 0
    cache = MetadataCache()

    def load_structure(dataset_code: str) -> DataStructure:
        nonlocal calls
        calls += 1
        return _structure()

    cache.get_structure("DEMO_PJAN", load_structure)
    cache.clear()
    result = cache.get_structure("DEMO_PJAN", load_structure)

    assert result.hit is False
    assert calls == 2
