from eurostat_agent.catalogue import (
    DatasetIndex,
    DatasetRecord,
    search_datasets,
)


def _record(
    code: str,
    title: str,
    description: str | None = None,
    paths: tuple[tuple[str, ...], ...] = (),
) -> DatasetRecord:
    return DatasetRecord(
        code=code,
        title=title,
        product_type="dataset",
        description=description,
        last_update=None,
        last_modified=None,
        data_start=None,
        data_end=None,
        value_count=None,
        paths=paths,
    )


def test_search_ranks_exact_code_match_first() -> None:
    index = DatasetIndex(
        (
            _record(
                code="une_rt_a",
                title="Unemployment by sex and age - annual data",
            ),
            _record(
                code="other_unemployment",
                title="Unemployment statistics",
            ),
        )
    )

    results = index.search("une_rt_a")

    assert [result.record.code for result in results] == [
        "une_rt_a",
    ]


def test_search_matches_exact_title_case_insensitively() -> None:
    index = DatasetIndex(
        (
            _record(
                code="une_rt_a",
                title="Unemployment by sex and age - annual data",
            ),
            _record(
                code="other_dataset",
                title="Employment statistics",
            ),
        )
    )

    results = index.search("UNEMPLOYMENT BY SEX AND AGE - ANNUAL DATA")

    assert len(results) == 1
    assert results[0].record.code == "une_rt_a"
    assert results[0].score == 90
    assert results[0].match_type == "exact_title"


def test_search_matches_normalized_title() -> None:
    index = DatasetIndex(
        (
            _record(
                code="une_rt_a",
                title="Unemployment by sex and age - annual data",
            ),
            _record(
                code="lfsa_urgan",
                title="Unemployment rates by age and sex",
            ),
        )
    )

    results = index.search("unemployment by sex and age annual data")

    assert len(results) == 1
    assert results[0].record.code == "une_rt_a"
    assert results[0].score == 80
    assert results[0].match_type == "normalized_title"


def test_search_preserves_multiple_records_with_same_exact_title() -> None:
    index = DatasetIndex(
        (
            _record(
                code="dataset_a",
                title="Population by age and sex",
            ),
            _record(
                code="dataset_b",
                title="Population by age and sex",
            ),
        )
    )

    results = index.search("Population by age and sex")

    assert [result.record.code for result in results] == [
        "dataset_a",
        "dataset_b",
    ]
    assert all(result.score == 90 for result in results)
    assert all(result.match_type == "exact_title" for result in results)


def test_search_matches_phrase_inside_title() -> None:
    index = DatasetIndex(
        (
            _record(
                code="une_rt_a",
                title="Unemployment by sex and age - annual data",
            ),
            _record(
                code="lfsa_urgan",
                title="Unemployment rates by age and sex",
            ),
            _record(
                code="employment",
                title="Employment statistics",
            ),
        )
    )

    results = index.search("unemployment")

    assert [result.record.code for result in results] == [
        "une_rt_a",
        "lfsa_urgan",
    ]
    assert all(result.score == 70 for result in results)
    assert all(result.match_type == "title_phrase" for result in results)


def test_search_requires_majority_title_token_coverage() -> None:
    index = DatasetIndex(
        (
            _record(
                code="une_rt_a",
                title="Unemployment by sex and age - annual data",
            ),
            _record(
                code="une_total",
                title="Unemployment statistics",
            ),
            _record(
                code="employment_age",
                title="Employment by age",
            ),
        )
    )

    results = index.search("unemployment age")

    assert [result.record.code for result in results] == [
        "une_rt_a",
    ]

    assert results[0].score == 69
    assert results[0].match_type == "title_tokens"


def test_search_matches_phrase_inside_description() -> None:
    index = DatasetIndex(
        (
            _record(
                code="demo_population",
                title="Population statistics",
                description=("Population on 1 January by age and sex."),
            ),
            _record(
                code="employment",
                title="Employment statistics",
                description="Employment indicators by age.",
            ),
        )
    )

    results = index.search("population on 1 january")

    assert [result.record.code for result in results] == [
        "demo_population",
    ]
    assert results[0].score == 65
    assert results[0].match_type == "description_phrase"


def test_search_matches_phrase_inside_catalogue_path() -> None:
    index = DatasetIndex(
        (
            _record(
                code="isoc_ci_ac_i",
                title="Individuals - internet activities",
                paths=(
                    (
                        "Cross cutting topics",
                        "Skills-related statistics",
                        "Skills supply - self-reported measures",
                        "Digital skills - ICT usage in households and individuals",
                    ),
                ),
            ),
            _record(
                code="other_dataset",
                title="Internet access statistics",
                paths=(
                    (
                        "Science, technology, digital society",
                        "Connectivity",
                    ),
                ),
            ),
        )
    )

    results = index.search("digital skills")

    assert [result.record.code for result in results] == [
        "isoc_ci_ac_i",
    ]
    assert results[0].score == 60
    assert results[0].match_type == "path_phrase"


def test_search_ignores_stopwords_in_title_token_scoring() -> None:
    index = DatasetIndex(
        (
            _record(
                code="population_age_sex",
                title="Population on 1 January by age and sex",
            ),
            _record(
                code="age_only",
                title="Statistics by age and year",
            ),
            _record(
                code="generic",
                title="Indicators by sector and year",
            ),
        )
    )

    results = index.search("population by age and sex")

    assert [result.record.code for result in results] == [
        "population_age_sex",
    ]

    assert results[0].score == 69
    assert results[0].match_type == "title_tokens"


def test_search_keeps_majority_title_token_match() -> None:
    index = DatasetIndex(
        (
            _record(
                code="population_age",
                title="Population by age group",
            ),
            _record(
                code="population_only",
                title="Population statistics",
            ),
        )
    )

    results = index.search("population age sex")

    assert [result.record.code for result in results] == [
        "population_age",
    ]

    assert results[0].score == 62
    assert results[0].match_type == "title_tokens"


def test_search_prefers_more_specific_title_when_token_scores_tie() -> None:
    index = DatasetIndex(
        (
            _record(
                code="regional",
                title=("Population on 1 January by age, sex and NUTS 2 region"),
            ),
            _record(
                code="general",
                title="Population on 1 January by age and sex",
            ),
            _record(
                code="specialized",
                title=(
                    "Population by sex, age group, "
                    "current activity status and NUTS 3 region"
                ),
            ),
        )
    )

    results = index.search("population by age and sex")

    assert [result.record.code for result in results] == [
        "general",
        "regional",
        "specialized",
    ]

    assert all(result.score == 69 for result in results)
    assert all(result.match_type == "title_tokens" for result in results)


def test_search_prefers_fresher_dataset_when_lexical_scores_tie() -> None:
    old_record = DatasetRecord(
        code="old_population",
        title="Population by sex and age",
        product_type="dataset",
        description=None,
        last_update="25.04.2014",
        last_modified=None,
        data_start="2001",
        data_end="2001",
        value_count=None,
        paths=(),
    )

    current_record = DatasetRecord(
        code="current_population",
        title="Population on 1 January by age and sex",
        product_type="dataset",
        description=None,
        last_update="14.08.2026",
        last_modified=None,
        data_start="1960",
        data_end="2025",
        value_count=None,
        paths=(),
    )

    index = DatasetIndex(
        (
            old_record,
            current_record,
        )
    )

    results = index.search("population by age and sex")

    assert [result.record.code for result in results] == [
        "current_population",
        "old_population",
    ]

    assert all(result.score == 69 for result in results)


def test_search_does_not_treat_future_projection_horizon_as_fresher() -> None:
    projected_record = DatasetRecord(
        code="projected_population",
        title="Population by age and sex projection",
        product_type="dataset",
        description=None,
        last_update="29.06.2026",
        last_modified=None,
        data_start="2022",
        data_end="2100",
        value_count=None,
        paths=(),
    )

    observed_record = DatasetRecord(
        code="observed_population",
        title="Population by age and sex observed",
        product_type="dataset",
        description=None,
        last_update="14.08.2026",
        last_modified=None,
        data_start="1960",
        data_end="2025",
        value_count=None,
        paths=(),
    )

    index = DatasetIndex(
        (
            projected_record,
            observed_record,
        )
    )

    results = index.search("population age sex")

    assert [result.record.code for result in results] == [
        "observed_population",
        "projected_population",
    ]

    assert all(result.score == 69 for result in results)


def test_search_respects_result_limit() -> None:
    index = DatasetIndex(
        (
            _record(
                code="population_a",
                title="Population statistics A",
            ),
            _record(
                code="population_b",
                title="Population statistics B",
            ),
            _record(
                code="population_c",
                title="Population statistics C",
            ),
        )
    )

    results = index.search(
        "population",
        limit=2,
    )

    assert [result.record.code for result in results] == [
        "population_a",
        "population_b",
    ]


def test_search_rejects_non_positive_limit() -> None:
    index = DatasetIndex(
        (
            _record(
                code="population",
                title="Population statistics",
            ),
        )
    )

    for limit in (0, -1):
        try:
            index.search("population", limit=limit)
        except ValueError as exc:
            assert str(exc) == "Search limit must be positive."
        else:
            raise AssertionError("Expected ValueError")


def test_search_datasets_exposes_public_search_tool() -> None:
    index = DatasetIndex(
        (
            _record(
                code="population_a",
                title="Population statistics A",
            ),
            _record(
                code="population_b",
                title="Population statistics B",
            ),
        )
    )

    results = search_datasets(
        "population",
        index=index,
        limit=1,
    )

    assert len(results) == 1
    assert results[0].record.code == "population_a"
