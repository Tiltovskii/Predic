"""Offline discovery helpers for an explicitly authorized HLTV capture.

The HTTP layer deliberately accepts only a pre-built immutable manifest.  This
module bridges the two collection phases without adding another network client:

* generate bounded results-listing URLs locally;
* turn already captured results listings into exact match URLs;
* turn already captured match pages into the exact map-stats links they expose.

Every extraction re-verifies the saved raw object before using it.  Nothing in
this module makes an HTTP request.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qs, urljoin, urlsplit

from .hltv_capture import CaptureCorruptionError, capture_index, parsed_capture_records
from .hltv_offline import HtmlNode, _find_all, _parse_dom, _text


DISCOVERY_PARSER_VERSION = "hltv-discovery-v1"
_MATCH_PATH = re.compile(r"^/matches/(\d+)(?:/|$)")
_MAP_STATS_PATH = re.compile(r"/mapstatsid/(\d+)(?:/|$)")
_HLTV_HOST = "www.hltv.org"


class HltvDiscoveryError(ValueError):
    """Raised when a captured listing or match cannot safely produce a manifest."""


@dataclass(frozen=True)
class ResultLink:
    """One match link, tied to the date header that contained it."""

    match_id: str
    url: str
    card_position: int
    listed_date: date
    date_heading_raw: str


@dataclass(frozen=True)
class ResultsPage:
    """The useful links extracted from a single saved HLTV results page."""

    matches: tuple[ResultLink, ...]
    pagination_urls: tuple[str, ...]
    has_listing_marker: bool


_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_RESULT_DATE_HEADING = re.compile(
    r"^Results\s+for\s+([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(\d{4})$",
    re.IGNORECASE,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute_hltv_url(href: str, base_url: str) -> str | None:
    """Resolve an on-site link; discovery never exports cross-site URLs."""

    absolute = urljoin(base_url, href)
    parsed = urlsplit(absolute)
    if (
        parsed.scheme.lower() != "https"
        or parsed.hostname is None
        or parsed.hostname.casefold() != _HLTV_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        return None
    return absolute


def _match_id(url: str) -> str | None:
    match = _MATCH_PATH.match(urlsplit(url).path)
    return match.group(1) if match is not None else None


def _map_stats_id(url: str) -> str | None:
    match = _MAP_STATS_PATH.search(urlsplit(url).path)
    return match.group(1) if match is not None else None


def _results_sublist_for_card(card: HtmlNode) -> HtmlNode:
    """Return the exact date-group container for one results card.

    A global headline, a sidebar headline, or a preceding group must never be
    used as a substitute.  The date is a coverage assertion, so ambiguity is
    an error rather than a best-effort guess.
    """

    ancestor = card.parent
    while ancestor is not None:
        if "results-sublist" in ancestor.classes:
            return ancestor
        ancestor = ancestor.parent
    raise HltvDiscoveryError(
        "result card has no enclosing results-sublist date group"
    )


def _parse_results_heading(headline: HtmlNode) -> tuple[date, str]:
    raw = _text(headline)
    normalized = " ".join(raw.replace("\u00a0", " ").split())
    match = _RESULT_DATE_HEADING.fullmatch(normalized)
    if match is None:
        raise HltvDiscoveryError(
            "results date heading is not a recognized English full-date label: "
            + repr(raw)
        )
    month = _MONTHS.get(match.group(1).casefold())
    if month is None:
        raise HltvDiscoveryError(f"results date heading has an unknown month: {raw!r}")
    try:
        return date(int(match.group(3)), month, int(match.group(2))), raw
    except ValueError as error:
        raise HltvDiscoveryError(f"results date heading is invalid: {raw!r}") from error


def _listing_date_for_card(card: HtmlNode) -> tuple[date, str]:
    group = _results_sublist_for_card(card)
    # HLTV's date grouping places the label directly inside the sublist.  Do
    # not recursively search here: a nested component's or sidebar's label
    # could otherwise leak into the card's coverage provenance.
    headlines = [
        child
        for child in group.children
        if isinstance(child, HtmlNode) and "standard-headline" in child.classes
    ]
    if len(headlines) != 1:
        raise HltvDiscoveryError(
            "results-sublist must contain exactly one direct standard-headline "
            f"(found {len(headlines)})"
        )
    return _parse_results_heading(headlines[0])


def parse_results_html(html: str, *, source_url: str) -> ResultsPage:
    """Extract exact match and pagination links from a local results snapshot.

    A results page can legitimately contain no matches for a date window.  It
    must nevertheless contain a recognizable results-listing container; this
    avoids accepting a 200 challenge/soft-block page as an empty result set.
    """

    root = _parse_dom(html)
    result_cards = _find_all(root, class_name="result-con")
    has_listing_marker = bool(result_cards) or any(
        _find_all(root, class_name=class_name)
        for class_name in ("results-holder", "results-sublist", "results-all")
    )
    if not has_listing_marker:
        raise HltvDiscoveryError(
            "captured page has no recognizable HLTV results-listing container"
        )

    matches: list[ResultLink] = []
    for card_position, card in enumerate(result_cards, start=1):
        listed_date, date_heading_raw = _listing_date_for_card(card)
        seen_in_card: set[str] = set()
        for anchor in _find_all(card, tag="a"):
            url = _absolute_hltv_url(anchor.attrs.get("href", ""), source_url)
            if url is None:
                continue
            match_id = _match_id(url)
            if match_id is None or match_id in seen_in_card:
                continue
            seen_in_card.add(match_id)
            matches.append(
                ResultLink(
                    match_id=match_id,
                    url=url,
                    card_position=card_position,
                    listed_date=listed_date,
                    date_heading_raw=date_heading_raw,
                )
            )

    pagination_urls: list[str] = []
    seen_pagination: set[str] = set()
    for anchor in _find_all(root, tag="a"):
        classes = anchor.classes
        if not ({"pagination-prev", "pagination-next"} & classes):
            continue
        href = anchor.attrs.get("href", "").strip()
        if not href:
            # HLTV marks an inactive edge arrow with the pagination class but
            # no href.  Resolving an empty href would manufacture the current
            # page as a false graph edge.
            continue
        url = _absolute_hltv_url(href, source_url)
        if url is None or url in seen_pagination:
            continue
        seen_pagination.add(url)
        pagination_urls.append(url)

    return ResultsPage(
        matches=tuple(matches),
        pagination_urls=tuple(pagination_urls),
        has_listing_marker=has_listing_marker,
    )


def generate_results_manifest(
    start_date: date,
    end_date: date,
    *,
    window_days: int,
    url_template: str,
) -> list[dict[str, object]]:
    """Create immutable date-window listing entries without network access.

    The template is intentionally explicit rather than hard-coding undocumented
    query parameters.  For example, after a sentinel verifies the route, a
    caller can use ``https://www.hltv.org/results?startDate={start_date}&endDate={end_date}``.
    """

    if end_date < start_date:
        raise ValueError("end_date must not precede start_date")
    if window_days < 1 or window_days > 31:
        raise ValueError("window_days must be between 1 and 31")
    if "{start_date}" not in url_template or "{end_date}" not in url_template:
        raise ValueError("url_template must contain {start_date} and {end_date}")

    entries: list[dict[str, object]] = []
    window_start = start_date
    while window_start <= end_date:
        window_end = min(window_start + timedelta(days=window_days - 1), end_date)
        try:
            url = url_template.format(
                start_date=window_start.isoformat(), end_date=window_end.isoformat()
            )
        except (KeyError, ValueError) as error:
            raise ValueError(f"invalid url_template: {error}") from error
        parsed = urlsplit(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("url_template must render an absolute URL")
        offsets = parse_qs(parsed.query, keep_blank_values=True).get("offset")
        if offsets is not None and offsets != ["0"]:
            raise ValueError(
                "url_template must render a root results URL with no offset or offset=0"
            )
        entries.append(
            {
                "record_id": (
                    f"hltv-results:{window_start.isoformat()}:{window_end.isoformat()}"
                ),
                "page_type": "results",
                "url": url,
                "discovery": {
                    # Aggregate coverage is only meaningful when this page is
                    # an explicit first page for the declared window.  Child
                    # pages carry a separate, parent-provenance kind below.
                    "kind": "hltv-results-root",
                    "parser_version": DISCOVERY_PARSER_VERSION,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                },
            }
        )
        window_start = window_end + timedelta(days=1)
    return entries


def _verified_stream_captures(
    state_db: str | Path,
    *,
    streams: Sequence[str],
    page_type: str,
    allow_partial: bool,
) -> list[dict[str, object]]:
    names = tuple(streams)
    if not names or any(not isinstance(stream, str) or not stream for stream in names):
        raise HltvDiscoveryError("at least one non-empty capture stream is required")
    if len(set(names)) != len(names):
        raise HltvDiscoveryError("capture streams must be unique")

    verified: list[dict[str, object]] = []
    for stream in names:
        captures = capture_index(state_db, stream=stream, allow_partial=allow_partial)
        if not captures:
            raise HltvDiscoveryError(f"capture stream {stream!r} has no completed pages")
        wrong_types = sorted(
            {
                str(capture["page_type"])
                for capture in captures
                if capture["page_type"] != page_type
            }
        )
        if wrong_types:
            raise HltvDiscoveryError(
                f"capture stream {stream!r} mixes {page_type!r} with "
                + ", ".join(repr(item) for item in wrong_types)
            )
        for raw_capture in captures:
            capture = dict(raw_capture)
            capture["stream"] = stream
            path = Path(str(capture["object_path"]))
            if not path.is_file():
                raise CaptureCorruptionError(f"captured object is missing: {path}")
            actual_hash = _sha256_file(path)
            expected_hash = str(capture["content_sha256"])
            if actual_hash != expected_hash:
                raise CaptureCorruptionError(
                    f"captured object hash mismatch for {path}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
            verified.append(capture)
    return verified


def _expected_listing_window(capture: dict[str, object]) -> tuple[date, date] | None:
    metadata = capture.get("manifest_metadata", {})
    if not isinstance(metadata, dict):
        raise HltvDiscoveryError("capture manifest metadata must be an object")
    discovery = metadata.get("discovery")
    if discovery is None:
        return None
    if not isinstance(discovery, dict):
        raise HltvDiscoveryError("capture discovery metadata must be an object")
    start = discovery.get("window_start")
    end = discovery.get("window_end")
    if start is None and end is None:
        return None
    if not isinstance(start, str) or not isinstance(end, str):
        raise HltvDiscoveryError(
            "date-window discovery metadata must contain string window_start and window_end"
        )
    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError as error:
        raise HltvDiscoveryError(
            "date-window discovery metadata must use ISO calendar dates"
        ) from error
    if end_date < start_date:
        raise HltvDiscoveryError("date-window discovery end precedes its start")
    return start_date, end_date


def _assert_listing_url_keeps_window(
    url: str, *, window_start: date, window_end: date, label: str
) -> None:
    query = parse_qs(urlsplit(url).query, keep_blank_values=True)
    start = window_start.isoformat()
    end = window_end.isoformat()
    if query.get("startDate") != [start] or query.get("endDate") != [end]:
        raise HltvDiscoveryError(
            f"{label} does not retain the expected results window "
            f"{start} through {end}: {url}"
        )


def _read_validated_results_listing(
    capture: dict[str, object],
) -> tuple[ResultsPage, tuple[date, date] | None]:
    """Read one verified listing and prove its date-window semantics locally."""

    expected_window = _expected_listing_window(capture)
    if expected_window is not None:
        window_start, window_end = expected_window
        _assert_listing_url_keeps_window(
            str(capture["source_url"]),
            window_start=window_start,
            window_end=window_end,
            label="requested results URL",
        )
        _assert_listing_url_keeps_window(
            str(capture["final_url"]),
            window_start=window_start,
            window_end=window_end,
            label="final results URL",
        )
        if _results_url_key(str(capture["source_url"])) != _results_url_key(
            str(capture["final_url"])
        ):
            raise HltvDiscoveryError(
                "date-window results redirect changed the requested listing identity: "
                f"{capture['source_url']} -> {capture['final_url']}"
            )
    path = Path(str(capture["object_path"]))
    try:
        html = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise HltvDiscoveryError(f"captured results page is not UTF-8: {path}") from error
    listing = parse_results_html(html, source_url=str(capture["final_url"]))
    if expected_window is not None:
        window_start, window_end = expected_window
        for match in listing.matches:
            if not window_start <= match.listed_date <= window_end:
                raise HltvDiscoveryError(
                    "results content contains a match outside its declared window "
                    f"{window_start.isoformat()} through {window_end.isoformat()}: "
                    f"{match.match_id} is listed under {match.listed_date.isoformat()}"
                )
        for pagination_url in listing.pagination_urls:
            _assert_listing_url_keeps_window(
                pagination_url,
                window_start=window_start,
                window_end=window_end,
                label="results pagination URL",
            )
    return listing, expected_window


def _results_url_key(url: str) -> tuple[str, str, str, tuple[tuple[str, tuple[str, ...]], ...]]:
    """Compare results URLs independent of query ordering and ``offset=0``."""

    parsed = urlsplit(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    if query.get("offset") == ["0"]:
        del query["offset"]
    return (
        parsed.scheme.casefold(),
        (parsed.hostname or "").casefold(),
        parsed.path or "/",
        tuple(sorted((key, tuple(values)) for key, values in query.items())),
    )


def _capture_parent_proof(capture: dict[str, object]) -> dict[str, str]:
    return {
        "stream": str(capture["stream"]),
        "record_id": str(capture["record_id"]),
        "requested_url": str(capture["source_url"]),
        "final_url": str(capture["final_url"]),
        "content_sha256": str(capture["content_sha256"]),
        "observed_at": str(capture["observed_at"]),
    }


def _capture_discovery(capture: dict[str, object]) -> dict[str, object] | None:
    metadata = capture.get("manifest_metadata", {})
    if not isinstance(metadata, dict):
        raise HltvDiscoveryError("capture manifest metadata must be an object")
    discovery = metadata.get("discovery")
    if discovery is None:
        return None
    if not isinstance(discovery, dict):
        raise HltvDiscoveryError("capture discovery metadata must be an object")
    return discovery


def _pagination_depth(capture: dict[str, object]) -> int:
    discovery = _capture_discovery(capture)
    if discovery is None or discovery.get("kind") != "hltv-results-pagination":
        return 0
    depth = discovery.get("pagination_depth")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise HltvDiscoveryError(
            "pagination discovery metadata must contain a positive integer pagination_depth"
        )
    return depth


def _validate_pagination_lineage(captures: Sequence[dict[str, object]]) -> None:
    """Verify every derived listing's immutable parent evidence.

    A child stream is deliberately captured separately from its parent.  This
    makes resume safe, but means we must validate the parent proof before using
    it to claim coverage.  The caller therefore always supplies the complete
    root+child stream set being aggregated.
    """

    by_key = {
        (str(capture["stream"]), str(capture["record_id"])): capture
        for capture in captures
    }
    parsed_parent_pages: dict[tuple[str, str], ResultsPage] = {}
    for capture in captures:
        discovery = _capture_discovery(capture)
        if discovery is None or discovery.get("kind") != "hltv-results-pagination":
            continue
        parents = discovery.get("parents")
        if not isinstance(parents, list) or not parents:
            raise HltvDiscoveryError(
                "derived results pagination capture must retain non-empty parent proofs"
            )
        child_window = _expected_listing_window(capture)
        if child_window is None:
            raise HltvDiscoveryError(
                "derived results pagination capture must retain its date window"
            )
        parent_depths: list[int] = []
        for parent in parents:
            if not isinstance(parent, dict):
                raise HltvDiscoveryError("results pagination parent proof must be an object")
            stream = parent.get("stream")
            record_id = parent.get("record_id")
            if not isinstance(stream, str) or not isinstance(record_id, str):
                raise HltvDiscoveryError(
                    "results pagination parent proof needs stream and record_id"
                )
            actual = by_key.get((stream, record_id))
            if actual is None:
                raise HltvDiscoveryError(
                    "derived results pagination parent is absent from the supplied stream "
                    f"set: {stream}/{record_id}"
                )
            expected_proof = _capture_parent_proof(actual)
            for key, expected in expected_proof.items():
                if parent.get(key) != expected:
                    raise HltvDiscoveryError(
                        "results pagination parent proof disagrees with the saved "
                        f"capture for {stream}/{record_id}: {key}"
                    )
            if _expected_listing_window(actual) != child_window:
                raise HltvDiscoveryError(
                    "derived results pagination child changed its parent's date window"
                )
            parent_url = parent.get("pagination_url")
            if not isinstance(parent_url, str):
                raise HltvDiscoveryError(
                    "results pagination parent proof needs its exact pagination_url"
                )
            if _results_url_key(parent_url) != _results_url_key(
                str(capture["source_url"])
            ):
                raise HltvDiscoveryError(
                    "results pagination parent proof does not point to this child URL"
                )
            parent_key = (stream, record_id)
            listing = parsed_parent_pages.get(parent_key)
            if listing is None:
                listing, _ = _read_validated_results_listing(actual)
                parsed_parent_pages[parent_key] = listing
            if _results_url_key(parent_url) not in {
                _results_url_key(url) for url in listing.pagination_urls
            }:
                raise HltvDiscoveryError(
                    "results pagination parent proof cites a URL not discovered in its "
                    f"saved HTML: {parent_url}"
                )
            parent_depth = _pagination_depth(actual)
            if parent_depth != _pagination_depth(capture) - 1:
                raise HltvDiscoveryError(
                    "results pagination parent depth must be exactly one less than "
                    "its child depth"
                )
            parent_depths.append(parent_depth)
        if _pagination_depth(capture) != parent_depths[0] + 1:
            raise HltvDiscoveryError(
                "results pagination depth does not follow its recorded parent proof"
            )


def derive_results_pagination_manifest(
    state_db: str | Path,
    *,
    streams: Sequence[str],
    max_entries: int = 1_000,
    max_depth: int = 32,
    allow_partial: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Create the *next* immutable results-listing manifest offline.

    The function never changes a capture stream or makes HTTP requests.  Pass
    every root and already captured child stream; it emits only pagination URLs
    that are not already present in that union.  A caller captures the emitted
    file under a new stream name, then calls this function again with the
    enlarged stream set.  This is the resumable pagination cascade.
    """

    if max_entries < 1:
        raise ValueError("max_entries must be positive")
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    captures = _verified_stream_captures(
        state_db,
        streams=streams,
        page_type="results",
        allow_partial=allow_partial,
    )
    _validate_pagination_lineage(captures)
    captured_keys = {
        _results_url_key(str(capture["source_url"])) for capture in captures
    } | {_results_url_key(str(capture["final_url"])) for capture in captures}
    entries_by_url: dict[
        tuple[str, str, str, tuple[tuple[str, tuple[str, ...]], ...]], dict[str, object]
    ] = {}
    pagination_links_seen = 0
    already_captured = 0

    for capture in captures:
        listing, expected_window = _read_validated_results_listing(capture)
        if expected_window is None:
            raise HltvDiscoveryError(
                "cannot derive historical pagination from a listing without a date window"
            )
        window_start, window_end = expected_window
        depth = _pagination_depth(capture) + 1
        for url in listing.pagination_urls:
            pagination_links_seen += 1
            key = _results_url_key(url)
            if key in captured_keys:
                already_captured += 1
                continue
            if depth > max_depth:
                raise HltvDiscoveryError(
                    f"pagination depth {depth} exceeds the configured maximum {max_depth}"
                )
            parent = {
                **_capture_parent_proof(capture),
                "pagination_url": url,
            }
            existing = entries_by_url.get(key)
            if existing is None:
                entries_by_url[key] = {
                    "record_id": "hltv-results-page:"
                    + hashlib.sha256(url.encode("utf-8")).hexdigest()[:24],
                    "page_type": "results",
                    "url": url,
                    "discovery": {
                        "kind": "hltv-results-pagination",
                        "parser_version": DISCOVERY_PARSER_VERSION,
                        "pagination_depth": depth,
                        "window_start": window_start.isoformat(),
                        "window_end": window_end.isoformat(),
                        "parents": [parent],
                    },
                }
                continue
            discovery = existing["discovery"]
            assert isinstance(discovery, dict)
            if (
                discovery["window_start"] != window_start.isoformat()
                or discovery["window_end"] != window_end.isoformat()
            ):
                raise HltvDiscoveryError(
                    f"one pagination URL belongs to incompatible date windows: {url}"
                )
            if int(discovery["pagination_depth"]) != depth:
                raise HltvDiscoveryError(
                    "one pagination URL was discovered at incompatible graph depths: "
                    + url
                )
            parents = discovery["parents"]
            assert isinstance(parents, list)
            if parent not in parents:
                parents.append(parent)

    records = list(entries_by_url.values())
    if len(records) > max_entries:
        raise HltvDiscoveryError(
            f"pagination derivation would emit {len(records)} entries, over max_entries={max_entries}"
        )
    for record in records:
        discovery = record["discovery"]
        assert isinstance(discovery, dict)
        parents = discovery["parents"]
        assert isinstance(parents, list)
        parents.sort(key=lambda item: (str(item["stream"]), str(item["record_id"])))
    report = {
        "parser_version": DISCOVERY_PARSER_VERSION,
        "results_capture_streams": list(streams),
        "results_capture_pages": len(captures),
        "pagination_links_seen": pagination_links_seen,
        "already_captured_pagination_links": already_captured,
        "unfetched_pagination_urls": [record["url"] for record in records],
        "pagination_entries_emitted": len(records),
        "coverage_complete": not records,
    }
    return records, report


def extract_match_manifest(
    state_db: str | Path,
    *,
    stream: str | None = None,
    streams: Sequence[str] | None = None,
    allow_partial: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Turn verified captured results pages into a deterministic match manifest.

    ``stream`` is retained for the single-stream workflow.  A caller that has
    followed pagination through immutable child streams must pass ``streams``
    explicitly so coverage is computed over the whole stated graph.
    """

    if streams is not None and stream is not None:
        raise HltvDiscoveryError("pass either stream or streams, not both")
    selected_streams = tuple(streams) if streams is not None else ((stream,) if stream else ())

    captures = _verified_stream_captures(
        state_db,
        streams=selected_streams,
        page_type="results",
        allow_partial=allow_partial,
    )
    _validate_pagination_lineage(captures)
    entries_by_match: dict[str, dict[str, object]] = {}
    captured_page_keys = {
        _results_url_key(str(capture["source_url"])) for capture in captures
    } | {_results_url_key(str(capture["final_url"])) for capture in captures}
    pagination_urls: list[str] = []
    seen_pagination: set[str] = set()
    duplicate_match_links = 0
    match_listing_dates: dict[str, set[date]] = {}

    for capture in captures:
        listing, _expected_window = _read_validated_results_listing(capture)
        for match in listing.matches:
            match_id = match.match_id
            url = match.url
            provenance = {
                "kind": "hltv-results-listing",
                "parser_version": DISCOVERY_PARSER_VERSION,
                "listing_capture_stream": capture["stream"],
                "listing_capture_record_id": capture["record_id"],
                "listing_requested_url": capture["source_url"],
                "listing_final_url": capture["final_url"],
                "listing_content_sha256": capture["content_sha256"],
                "listing_observed_at": capture["observed_at"],
                "card_position": match.card_position,
                # This is the results-listing grouping date, used only to
                # prove the historic coverage window.  It is not event_at.
                "listed_date": match.listed_date.isoformat(),
                "date_heading_raw": match.date_heading_raw,
                "date_selector_trace": "nearest .results-sublist > direct .standard-headline",
            }
            existing = entries_by_match.get(match_id)
            if existing is None:
                match_listing_dates[match_id] = {match.listed_date}
                entries_by_match[match_id] = {
                    "record_id": f"hltv-match:{match_id}",
                    "page_type": "match",
                    "url": url,
                    "discovery": [provenance],
                }
                continue
            duplicate_match_links += 1
            existing_dates = match_listing_dates[match_id]
            if match.listed_date not in existing_dates:
                raise HltvDiscoveryError(
                    f"match {match_id} occurs under incompatible results dates: "
                    + ", ".join(
                        item.isoformat() for item in sorted((*existing_dates, match.listed_date))
                    )
                )
            existing_dates.add(match.listed_date)
            discoveries = existing["discovery"]
            assert isinstance(discoveries, list)
            discoveries.append(provenance)
            if url != existing["url"]:
                aliases = existing.setdefault("discovered_url_aliases", [])
                assert isinstance(aliases, list)
                if url not in aliases:
                    aliases.append(url)
        for url in listing.pagination_urls:
            if url not in seen_pagination:
                seen_pagination.add(url)
                pagination_urls.append(url)

    unfetched_pagination_urls = [
        url for url in pagination_urls if _results_url_key(url) not in captured_page_keys
    ]
    records = list(entries_by_match.values())
    report = {
        "parser_version": DISCOVERY_PARSER_VERSION,
        "results_capture_streams": list(selected_streams),
        "results_capture_pages": len(captures),
        "matches_discovered": len(records),
        "duplicate_match_links": duplicate_match_links,
        "pagination_links_seen": len(pagination_urls),
        "unfetched_pagination_urls": unfetched_pagination_urls,
        "coverage_complete": not unfetched_pagination_urls,
    }
    return records, report


def _declared_root_windows(
    captures: Sequence[dict[str, object]],
) -> list[tuple[date, date]]:
    """Return only explicitly declared first-page windows.

    A page that merely has date-window metadata is not enough to establish
    coverage: it may be a manually supplied ``offset=100`` page.  Legacy
    untyped listings can still be inspected/extracted, but cannot make the
    aggregate coverage claim until they are re-captured from a root manifest.
    """

    windows: set[tuple[date, date]] = set()
    for capture in captures:
        discovery = _capture_discovery(capture)
        if discovery is not None and discovery.get("kind") == "hltv-results-pagination":
            continue
        if discovery is None or discovery.get("kind") != "hltv-results-root":
            raise HltvDiscoveryError(
                "aggregate historical coverage requires each root listing to carry "
                "discovery.kind='hltv-results-root'"
            )
        window = _expected_listing_window(capture)
        if window is None:
            raise HltvDiscoveryError(
                "aggregate historical coverage requires every root listing to retain "
                "date-window discovery metadata"
            )
        for label, url in (
            ("requested root results URL", str(capture["source_url"])),
            ("final root results URL", str(capture["final_url"])),
        ):
            offsets = parse_qs(urlsplit(url).query, keep_blank_values=True).get("offset")
            if offsets is not None and offsets != ["0"]:
                raise HltvDiscoveryError(
                    f"{label} must omit offset or use exactly offset=0: {url}"
                )
        windows.add(window)
    if not windows:
        raise HltvDiscoveryError("aggregate historical coverage has no root listing windows")
    return sorted(windows)


def _window_coverage(
    windows: Sequence[tuple[date, date]], *, start: date, end: date
) -> tuple[list[tuple[date, date]], list[tuple[date, date]]]:
    """Return date gaps and overlapping declared root windows in a target range."""

    gaps: list[tuple[date, date]] = []
    overlaps: list[tuple[date, date]] = []
    covered_through = start - timedelta(days=1)
    for raw_start, raw_end in windows:
        window_start = max(raw_start, start)
        window_end = min(raw_end, end)
        if window_end < start or window_start > end:
            continue
        if window_start <= covered_through:
            overlaps.append((window_start, min(window_end, covered_through)))
        elif window_start > covered_through + timedelta(days=1):
            gaps.append((covered_through + timedelta(days=1), window_start - timedelta(days=1)))
        if window_end > covered_through:
            covered_through = window_end
    if covered_through < end:
        gaps.append((covered_through + timedelta(days=1), end))
    return gaps, overlaps


def aggregate_match_manifest(
    state_db: str | Path,
    *,
    streams: Sequence[str],
    expected_start: date,
    expected_end: date,
    allow_partial: bool = False,
    require_complete: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Aggregate root+child results streams into a checkable historic corpus.

    Unlike the single-stream compatibility helper, this function proves both
    pagination closure and that the declared root date windows cover the exact
    requested interval.  It is still entirely offline.  Complete coverage is
    required by default so stdout cannot quietly become a trainable-looking
    partial corpus; diagnostics must opt in explicitly.
    """

    if expected_end < expected_start:
        raise ValueError("expected_end must not precede expected_start")
    captures = _verified_stream_captures(
        state_db,
        streams=streams,
        page_type="results",
        allow_partial=allow_partial,
    )
    _validate_pagination_lineage(captures)
    root_windows = _declared_root_windows(captures)
    out_of_range_windows = [
        (window_start, window_end)
        for window_start, window_end in root_windows
        if window_start < expected_start or window_end > expected_end
    ]
    if out_of_range_windows:
        rendered = ", ".join(
            f"{window_start.isoformat()}..{window_end.isoformat()}"
            for window_start, window_end in out_of_range_windows
        )
        raise HltvDiscoveryError(
            "aggregate stream set contains root date windows outside the requested "
            f"range {expected_start.isoformat()}..{expected_end.isoformat()}: {rendered}"
        )
    records, report = extract_match_manifest(
        state_db,
        streams=streams,
        allow_partial=allow_partial,
    )
    gaps, overlaps = _window_coverage(
        root_windows, start=expected_start, end=expected_end
    )
    report.update(
        {
            "expected_window": {
                "start": expected_start.isoformat(),
                "end": expected_end.isoformat(),
            },
            "declared_root_windows": [
                {"start": window_start.isoformat(), "end": window_end.isoformat()}
                for window_start, window_end in root_windows
            ],
            "root_window_gaps": [
                {"start": gap_start.isoformat(), "end": gap_end.isoformat()}
                for gap_start, gap_end in gaps
            ],
            "root_window_overlaps": [
                {"start": overlap_start.isoformat(), "end": overlap_end.isoformat()}
                for overlap_start, overlap_end in overlaps
            ],
        }
    )
    report["coverage_complete"] = bool(report["coverage_complete"]) and not gaps
    if require_complete and not report["coverage_complete"]:
        raise HltvDiscoveryError(
            "historical results coverage is incomplete; inspect pagination and root-window "
            "diagnostics before emitting a match manifest"
        )
    return records, report


def extract_mapstats_manifest(
    state_db: str | Path,
    *,
    stream: str,
    allow_partial: bool = False,
    require_complete: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Turn verified match captures into exact map-stats URLs and parent links.

    A played map without its exact linked statistics page is a missing branch
    of the corpus, not an empty statistic.  Refuse to emit a trainable-looking
    partial manifest unless the caller explicitly asks for diagnostics.
    """

    entries_by_map_stats: dict[str, dict[str, object]] = {}
    total_maps = 0
    played_maps = 0
    missing_played_map_stats: list[dict[str, object]] = []
    parent_identity_by_map_stats: dict[
        str, tuple[object, object, object, tuple[object, ...], object, object]
    ] = {}

    for record in parsed_capture_records(
        state_db, stream=stream, allow_partial=allow_partial
    ):
        if record.get("kind") != "map":
            continue
        total_maps += 1
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise HltvDiscoveryError("map parser record has no object payload")
        if payload.get("status") == "played":
            played_maps += 1
        map_stats_id = payload.get("map_stats_id")
        map_stats_url = payload.get("map_stats_url")
        if map_stats_id is None or map_stats_url is None:
            if payload.get("status") == "played":
                missing_played_map_stats.append(
                    {
                        "match_id": payload.get("match_id"),
                        "map_id": payload.get("map_id"),
                        "reason": "played_map_has_no_linked_map_stats_url",
                    }
                )
            continue
        if not isinstance(map_stats_id, str) or not isinstance(map_stats_url, str):
            raise HltvDiscoveryError("map stats ID and URL must be strings when present")
        url_id = _map_stats_id(map_stats_url)
        if url_id != map_stats_id:
            raise HltvDiscoveryError(
                f"map stats URL {map_stats_url!r} disagrees with map_stats_id {map_stats_id!r}"
            )
        team_ids = payload.get("team_ids")
        if not isinstance(team_ids, list) or len(team_ids) != 2:
            raise HltvDiscoveryError(
                "linked map stats must have exactly two parent match-team IDs"
            )
        parent_identity = (
            payload.get("match_id"),
            payload.get("map_id"),
            payload.get("map_order"),
            tuple(team_ids),
            payload.get("score_a"),
            payload.get("score_b"),
        )
        provenance = {
            "kind": "hltv-match-page",
            "parser_version": DISCOVERY_PARSER_VERSION,
            "parent_match_record_id": record["record_id"],
            "parent_match_id": payload.get("match_id"),
            "parent_map_id": payload.get("map_id"),
            "parent_map_order": payload.get("map_order"),
            "parent_team_ids": payload.get("team_ids"),
            "parent_score_a": payload.get("score_a"),
            "parent_score_b": payload.get("score_b"),
            "parent_capture": record.get("capture_provenance"),
        }
        existing = entries_by_map_stats.get(map_stats_id)
        if existing is None:
            parent_identity_by_map_stats[map_stats_id] = parent_identity
            entries_by_map_stats[map_stats_id] = {
                "record_id": f"hltv-map-stats:{map_stats_id}",
                "page_type": "map-stats",
                "url": map_stats_url,
                "discovery": [provenance],
            }
            continue
        if parent_identity_by_map_stats[map_stats_id] != parent_identity:
            raise HltvDiscoveryError(
                f"map_stats_id {map_stats_id} is linked by incompatible parent maps"
            )
        if existing["url"] != map_stats_url:
            raise HltvDiscoveryError(
                f"map_stats_id {map_stats_id} has conflicting URLs: "
                f"{existing['url']!r} and {map_stats_url!r}"
            )
        discoveries = existing["discovery"]
        assert isinstance(discoveries, list)
        discoveries.append(provenance)

    records = list(entries_by_map_stats.values())
    report = {
        "parser_version": DISCOVERY_PARSER_VERSION,
        "maps_seen": total_maps,
        "played_maps_seen": played_maps,
        "map_stats_discovered": len(records),
        "missing_played_map_stats": missing_played_map_stats,
        "complete_for_linked_maps": not missing_played_map_stats,
    }
    if require_complete and missing_played_map_stats:
        raise HltvDiscoveryError(
            "played maps are missing exact linked map-stats pages; request "
            "diagnostics explicitly before emitting a partial manifest"
        )
    return records, report


def records_to_jsonl(records: list[dict[str, object]]) -> str:
    """Canonical JSONL helper used by tests and optional external orchestration."""

    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
        for record in records
    )
