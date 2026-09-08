from __future__ import annotations

import re
from dataclasses import dataclass
from xml.etree import ElementTree

_NAMESPACE = {"nt": "urn:eu.europa.ec.eurostat.navtree"}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
}


@dataclass(frozen=True)
class DatasetRecord:
    code: str
    title: str
    product_type: str
    description: str | None
    last_update: str | None
    last_modified: str | None
    data_start: str | None
    data_end: str | None
    value_count: int | None
    paths: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class DatasetSearchResult:
    record: DatasetRecord
    score: int
    match_type: str


@dataclass(frozen=True)
class CatalogueSnapshot:
    source_url: str
    creation_date: str
    records: tuple[DatasetRecord, ...]


class DatasetIndex:
    def __init__(
        self,
        records: tuple[DatasetRecord, ...],
    ) -> None:
        self._records = records

        self._by_code = {record.code.casefold(): record for record in records}

        by_title: dict[str, list[DatasetRecord]] = {}
        by_normalized_title: dict[str, list[DatasetRecord]] = {}

        for record in records:
            title_key = record.title.casefold()
            normalized_title_key = _normalize_text(record.title)

            by_title.setdefault(
                title_key,
                [],
            ).append(record)

            by_normalized_title.setdefault(
                normalized_title_key,
                [],
            ).append(record)

        self._by_title = {title: tuple(matches) for title, matches in by_title.items()}

        self._by_normalized_title = {
            title: tuple(matches) for title, matches in by_normalized_title.items()
        }

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> tuple[DatasetSearchResult, ...]:
        if limit <= 0:
            raise ValueError("Search limit must be positive.")

        normalized_query = query.strip().casefold()

        if not normalized_query:
            return ()

        exact_code_match = self._by_code.get(normalized_query)

        if exact_code_match is not None:
            return (
                DatasetSearchResult(
                    record=exact_code_match,
                    score=100,
                    match_type="exact_code",
                ),
            )

        exact_title_matches = self._by_title.get(normalized_query)

        if exact_title_matches is not None:
            results = tuple(
                DatasetSearchResult(
                    record=record,
                    score=90,
                    match_type="exact_title",
                )
                for record in exact_title_matches
            )

            return results[:limit]

        normalized_query_text = _normalize_text(query)

        normalized_title_matches = self._by_normalized_title.get(normalized_query_text)

        if normalized_title_matches is not None:
            results = tuple(
                DatasetSearchResult(
                    record=record,
                    score=80,
                    match_type="normalized_title",
                )
                for record in normalized_title_matches
            )

            return results[:limit]

        query_tokens = {
            token for token in normalized_query_text.split() if token not in _STOPWORDS
        }

        matches: list[DatasetSearchResult] = []

        for record in self._records:
            normalized_title = _normalize_text(record.title)

            best_score = 0
            best_match_type: str | None = None

            if _contains_phrase(
                normalized_title,
                normalized_query_text,
            ):
                best_score = 70
                best_match_type = "title_phrase"

            title_tokens = set(normalized_title.split())

            matched_title_tokens = query_tokens & title_tokens

            if matched_title_tokens and query_tokens:
                coverage = len(matched_title_tokens) / len(query_tokens)

                if coverage > 0.5:
                    title_token_score = 50 + int(19 * coverage)

                    if title_token_score > best_score:
                        best_score = title_token_score
                        best_match_type = "title_tokens"

            if record.description is not None:
                normalized_description = _normalize_text(record.description)

                if (
                    _contains_phrase(
                        normalized_description,
                        normalized_query_text,
                    )
                    and 65 > best_score
                ):
                    best_score = 65
                    best_match_type = "description_phrase"

            path_phrase_match = any(
                _contains_phrase(
                    _normalize_text(path_part),
                    normalized_query_text,
                )
                for path in record.paths
                for path_part in path
            )

            if path_phrase_match and 60 > best_score:
                best_score = 60
                best_match_type = "path_phrase"

            if best_match_type is not None:
                matches.append(
                    DatasetSearchResult(
                        record=record,
                        score=best_score,
                        match_type=best_match_type,
                    )
                )

        matches.sort(
            key=lambda result: (
                -result.score,
                _descending_observed_data_end_key(result.record),
                _specificity_penalty(
                    result,
                    query_tokens,
                ),
                _descending_update_key(result.record.last_update),
            )
        )

        return tuple(matches[:limit])


def search_datasets(
    query: str,
    *,
    index: DatasetIndex,
    limit: int = 10,
) -> tuple[DatasetSearchResult, ...]:
    return index.search(
        query,
        limit=limit,
    )


def parse_catalogue(
    xml: str,
) -> tuple[DatasetRecord, ...]:
    root = ElementTree.fromstring(xml)

    return _parse_catalogue_root(root)


def parse_catalogue_snapshot(
    xml: str,
    *,
    source_url: str,
) -> CatalogueSnapshot:
    root = ElementTree.fromstring(xml)

    creation_date = root.get("creationDate")

    if creation_date is None or not creation_date.strip():
        raise ValueError("Catalogue snapshot is missing a creation date.")

    if not source_url.strip():
        raise ValueError("Catalogue snapshot source URL must not be empty.")

    records = _parse_catalogue_root(root)

    return CatalogueSnapshot(
        source_url=source_url,
        creation_date=creation_date.strip(),
        records=records,
    )


def _parse_catalogue_root(
    root: ElementTree.Element,
) -> tuple[DatasetRecord, ...]:
    records: dict[str, DatasetRecord] = {}

    for branch in root.findall(
        "./nt:branch",
        _NAMESPACE,
    ):
        _parse_branch(
            branch,
            (),
            records,
        )

    return tuple(records.values())


def _parse_branch(
    branch: ElementTree.Element,
    parent_path: tuple[str, ...],
    records: dict[str, DatasetRecord],
) -> None:
    title = _english_title(branch)
    path = parent_path + (title,)

    children = branch.find(
        "./nt:children",
        _NAMESPACE,
    )

    if children is None:
        return

    _parse_children(
        children,
        path,
        records,
    )


def _parse_leaf(
    leaf: ElementTree.Element,
    parent_path: tuple[str, ...],
    records: dict[str, DatasetRecord],
) -> None:
    code_node = leaf.find(
        "./nt:code",
        _NAMESPACE,
    )

    if code_node is None or code_node.text is None:
        raise ValueError("Catalogue leaf is missing a code.")

    code = code_node.text.strip()
    title = _english_title(leaf)
    product_type = leaf.get("type")

    if product_type is None:
        raise ValueError(f"Catalogue leaf {code!r} is missing a type.")

    description = _optional_english_text(
        leaf,
        "shortDescription",
    )

    last_update = _optional_text(
        leaf,
        "lastUpdate",
    )

    last_modified = _optional_text(
        leaf,
        "lastModified",
    )

    data_start = _optional_text(
        leaf,
        "dataStart",
    )

    data_end = _optional_text(
        leaf,
        "dataEnd",
    )

    value_count = _optional_int(
        leaf,
        "values",
    )

    candidate = DatasetRecord(
        code=code,
        title=title,
        product_type=product_type,
        description=description,
        last_update=last_update,
        last_modified=last_modified,
        data_start=data_start,
        data_end=data_end,
        value_count=value_count,
        paths=(parent_path,),
    )

    existing = records.get(code)

    if existing is None:
        records[code] = candidate
    else:
        if _metadata_without_paths(existing) != _metadata_without_paths(candidate):
            raise ValueError(f"Catalogue code {code!r} has inconsistent metadata.")

        records[code] = DatasetRecord(
            code=existing.code,
            title=existing.title,
            product_type=existing.product_type,
            description=existing.description,
            last_update=existing.last_update,
            last_modified=existing.last_modified,
            data_start=existing.data_start,
            data_end=existing.data_end,
            value_count=existing.value_count,
            paths=(existing.paths + (parent_path,)),
        )

    children = leaf.find(
        "./nt:children",
        _NAMESPACE,
    )

    if children is None:
        return

    child_path = parent_path + (title,)

    _parse_children(
        children,
        child_path,
        records,
    )


def _parse_children(
    children: ElementTree.Element,
    parent_path: tuple[str, ...],
    records: dict[str, DatasetRecord],
) -> None:
    for child in children:
        if child.tag == _qualified_name("branch"):
            _parse_branch(
                child,
                parent_path,
                records,
            )

        elif child.tag == _qualified_name("leaf"):
            _parse_leaf(
                child,
                parent_path,
                records,
            )


def _english_title(
    element: ElementTree.Element,
) -> str:
    title = element.find(
        "./nt:title[@language='en']",
        _NAMESPACE,
    )

    if title is None or title.text is None:
        raise ValueError("Catalogue element is missing an English title.")

    return title.text.strip()


def _optional_english_text(
    element: ElementTree.Element,
    name: str,
) -> str | None:
    node = element.find(
        f"./nt:{name}[@language='en']",
        _NAMESPACE,
    )

    if node is None or node.text is None:
        return None

    value = node.text.strip()

    return value or None


def _optional_text(
    element: ElementTree.Element,
    name: str,
) -> str | None:
    node = element.find(
        f"./nt:{name}",
        _NAMESPACE,
    )

    if node is None or node.text is None:
        return None

    value = node.text.strip()

    return value or None


def _optional_int(
    element: ElementTree.Element,
    name: str,
) -> int | None:
    value = _optional_text(
        element,
        name,
    )

    if value is None:
        return None

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Catalogue element {name!r} must contain an integer."
        ) from exc


def _metadata_without_paths(
    record: DatasetRecord,
) -> tuple[
    str,
    str,
    str,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    int | None,
]:
    return (
        record.code,
        record.title,
        record.product_type,
        record.description,
        record.last_update,
        record.last_modified,
        record.data_start,
        record.data_end,
        record.value_count,
    )


def _qualified_name(
    local_name: str,
) -> str:
    return f"{{{_NAMESPACE['nt']}}}{local_name}"


def _normalize_text(
    text: str,
) -> str:
    normalized = text.casefold()

    normalized = re.sub(
        r"[^\w]+",
        " ",
        normalized,
    )

    return " ".join(normalized.split())


def _contains_phrase(
    text: str,
    phrase: str,
) -> bool:
    text_tokens = text.split()
    phrase_tokens = phrase.split()

    if not phrase_tokens:
        return False

    phrase_length = len(phrase_tokens)

    return any(
        text_tokens[index : index + phrase_length] == phrase_tokens
        for index in range(len(text_tokens) - phrase_length + 1)
    )


def _specificity_penalty(
    result: DatasetSearchResult,
    query_tokens: set[str],
) -> int:
    if result.match_type != "title_tokens":
        return 0

    title_tokens = {
        token
        for token in _normalize_text(result.record.title).split()
        if token not in _STOPWORDS
    }

    extra_tokens = title_tokens - query_tokens

    return len(extra_tokens)


def _descending_observed_data_end_key(
    record: DatasetRecord,
) -> tuple[int, int, int]:
    data_end_parts = _parse_data_end(record.data_end)

    if data_end_parts is None:
        return (0, 0, 0)

    update_parts = _parse_update_date(record.last_update)

    if update_parts is not None:
        update_year = update_parts[0]
        data_end_year = data_end_parts[0]

        if data_end_year > update_year:
            return (0, 0, 0)

    year, month, day = data_end_parts

    return (
        -year,
        -month,
        -day,
    )


def _descending_update_key(
    last_update: str | None,
) -> tuple[int, int, int]:
    update_parts = _parse_update_date(last_update)

    if update_parts is None:
        return (0, 0, 0)

    year, month, day = update_parts

    return (
        -year,
        -month,
        -day,
    )


def _parse_data_end(
    data_end: str | None,
) -> tuple[int, int, int] | None:
    if data_end is None:
        return None

    match = re.fullmatch(
        r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?",
        data_end,
    )

    if match is None:
        return None

    year = int(match.group(1))

    month = int(match.group(2)) if match.group(2) is not None else 0

    day = int(match.group(3)) if match.group(3) is not None else 0

    return (
        year,
        month,
        day,
    )


def _parse_update_date(
    last_update: str | None,
) -> tuple[int, int, int] | None:
    if last_update is None:
        return None

    match = re.fullmatch(
        r"(\d{2})\.(\d{2})\.(\d{4})",
        last_update,
    )

    if match is None:
        return None

    day = int(match.group(1))
    month = int(match.group(2))
    year = int(match.group(3))

    return (
        year,
        month,
        day,
    )
