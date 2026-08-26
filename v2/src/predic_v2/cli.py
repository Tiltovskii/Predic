from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from .audit import audit
from .bo3_capture import (
    audit_bo3_capture,
    bo3_capture_index,
    capture_bo3,
    plan_bo3_capture,
)
from .db import connect, initialize
from .hltv_capture import (
    clear_host_circuit,
    capture_index,
    capture_manifest,
    parsed_capture_records,
    plan_capture,
)
from .hltv_offline import parse_file, records_to_jsonl
from .hltv_discovery import (
    aggregate_match_manifest,
    derive_results_pagination_manifest,
    extract_mapstats_manifest,
    extract_match_manifest,
    generate_results_manifest,
)
from .legacy import import_legacy_csv
from .materialize import materialize_raw_stream
from .raw_jsonl import import_jsonl
from .tedtay import import_tedtay_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="predic-data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize a data store")
    init_parser.add_argument("--db", required=True)

    legacy_parser = subparsers.add_parser(
        "import-legacy", help="Import and deduplicate the old Predic CS CSV"
    )
    legacy_parser.add_argument("--db", required=True)
    legacy_parser.add_argument("--csv", required=True)
    legacy_parser.add_argument("--from-date", type=date.fromisoformat)

    tedtay_parser = subparsers.add_parser(
        "import-tedtay-dataset",
        aliases=("import-tedtay-csgo-dataset", "import-tedtay-csgo-pro-matches"),
        help=(
            "Import downloaded TedTay historic_games_list.csv + game_data_rh.csv "
            "as a non-point-in-time research bootstrap"
        ),
    )
    tedtay_parser.add_argument("--db", required=True)
    tedtay_parser.add_argument("--historic-games-list", required=True)
    tedtay_parser.add_argument("--game-data-rh", required=True)
    tedtay_parser.add_argument("--from-date", type=date.fromisoformat)
    tedtay_parser.add_argument("--batch-size", type=int, default=500)

    audit_parser = subparsers.add_parser(
        "audit", help="Run structural and no-future checks"
    )
    audit_parser.add_argument("--db", required=True)

    jsonl_parser = subparsers.add_parser(
        "import-jsonl", help="Resume an authorized immutable JSONL export"
    )
    jsonl_parser.add_argument("--db", required=True)
    jsonl_parser.add_argument("--jsonl", required=True)
    jsonl_parser.add_argument("--source", required=True)
    jsonl_parser.add_argument("--stream", required=True)
    jsonl_parser.add_argument("--batch-size", type=int, default=1000)
    jsonl_parser.add_argument("--max-records", type=int)
    jsonl_parser.add_argument("--point-in-time-eligible", action="store_true")
    jsonl_parser.add_argument("--license-ref")

    hltv_parser = subparsers.add_parser(
        "parse-hltv-html",
        help="Parse an already captured local HLTV HTML file without network access",
    )
    hltv_parser.add_argument("--html", required=True)
    hltv_parser.add_argument(
        "--page-type", choices=("auto", "match", "map-stats"), default="auto"
    )
    hltv_parser.add_argument("--source-url")
    hltv_parser.add_argument("--observed-at")

    capture_parser = subparsers.add_parser(
        "capture-hltv-html",
        help="Capture explicitly authorized HLTV URLs with durable resume state",
    )
    capture_parser.add_argument("--manifest", required=True)
    capture_parser.add_argument("--policy", required=True)
    capture_parser.add_argument("--state-db", required=True)
    capture_parser.add_argument("--output-dir", required=True)
    capture_parser.add_argument("--stream", required=True)
    capture_parser.add_argument("--max-pages", type=int)
    capture_parser.add_argument("--max-http-requests", type=int)
    capture_parser.add_argument("--timeout-seconds", type=float, default=30.0)
    plan_capture_parser = subparsers.add_parser(
        "plan-hltv-capture",
        help="Validate an HLTV permission policy and URL manifest without network use",
    )
    plan_capture_parser.add_argument("--manifest", required=True)
    plan_capture_parser.add_argument("--policy", required=True)
    plan_capture_parser.add_argument("--max-pages", type=int)
    plan_capture_parser.add_argument("--max-http-requests", type=int)

    capture_index_parser = subparsers.add_parser(
        "export-hltv-capture-index",
        help="Export completed raw captures as deterministic JSONL",
    )
    capture_index_parser.add_argument("--state-db", required=True)
    capture_index_parser.add_argument("--stream", required=True)
    capture_index_parser.add_argument("--allow-partial", action="store_true")

    parse_captures_parser = subparsers.add_parser(
        "parse-hltv-captures",
        help="Verify and parse every completed raw capture into typed JSONL",
    )
    parse_captures_parser.add_argument("--state-db", required=True)
    parse_captures_parser.add_argument("--stream", required=True)
    parse_captures_parser.add_argument("--allow-partial", action="store_true")

    results_manifest_parser = subparsers.add_parser(
        "generate-hltv-results-manifest",
        help="Generate bounded, date-windowed HLTV results listing URLs without network access",
    )
    results_manifest_parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    results_manifest_parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    results_manifest_parser.add_argument("--window-days", type=int, default=7)
    results_manifest_parser.add_argument(
        "--url-template",
        required=True,
        help="must contain {start_date} and {end_date}",
    )

    extract_match_parser = subparsers.add_parser(
        "extract-hltv-match-manifest",
        help="Verify captured HLTV results listings and emit a match capture manifest",
    )
    extract_match_parser.add_argument("--state-db", required=True)
    extract_match_parser.add_argument(
        "--stream",
        action="append",
        required=True,
        help="repeat for every root and already captured pagination-child stream",
    )
    extract_match_parser.add_argument("--allow-partial", action="store_true")

    derive_pagination_parser = subparsers.add_parser(
        "derive-hltv-results-pagination-manifest",
        help="Offline: derive the next immutable HLTV results-pagination manifest",
    )
    derive_pagination_parser.add_argument("--state-db", required=True)
    derive_pagination_parser.add_argument(
        "--stream",
        action="append",
        required=True,
        help="repeat for every root and already captured pagination-child stream",
    )
    derive_pagination_parser.add_argument("--max-entries", type=int, default=1000)
    derive_pagination_parser.add_argument("--max-depth", type=int, default=32)
    derive_pagination_parser.add_argument("--allow-partial", action="store_true")

    aggregate_match_parser = subparsers.add_parser(
        "aggregate-hltv-match-manifest",
        help="Offline: prove historic listing coverage and emit an aggregated match manifest",
    )
    aggregate_match_parser.add_argument("--state-db", required=True)
    aggregate_match_parser.add_argument(
        "--stream",
        action="append",
        required=True,
        help="repeat for every root and already captured pagination-child stream",
    )
    aggregate_match_parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    aggregate_match_parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    aggregate_match_parser.add_argument("--allow-partial", action="store_true")
    aggregate_match_parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="diagnostics only; otherwise no manifest is emitted until coverage closes",
    )

    extract_mapstats_parser = subparsers.add_parser(
        "extract-hltv-mapstats-manifest",
        help="Verify captured HLTV match pages and emit exact linked map-stats URLs",
    )
    extract_mapstats_parser.add_argument("--state-db", required=True)
    extract_mapstats_parser.add_argument("--stream", required=True)
    extract_mapstats_parser.add_argument("--allow-partial", action="store_true")
    extract_mapstats_parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="diagnostics only; otherwise no manifest is emitted if a played map lacks stats",
    )

    materialize_parser = subparsers.add_parser(
        "materialize-hltv-stream",
        help="Materialize one bounded typed-HLTV raw-ingest window into normalized tables",
    )
    materialize_parser.add_argument("--db", required=True)
    materialize_parser.add_argument("--source", required=True)
    materialize_parser.add_argument("--stream", required=True)
    materialize_parser.add_argument("--max-records", type=int, default=1000)
    materialize_parser.add_argument("--after-raw-record-id")
    materialize_parser.add_argument(
        "--kind",
        action="append",
        choices=("series", "map", "ranking", "lineup", "player_map_stats"),
        help="repeatable; process one dependency-safe kind phase at a time",
    )
    materialize_parser.add_argument("--max-quarantine", type=int, default=200)

    review_host_parser = subparsers.add_parser(
        "review-hltv-host-circuit",
        help="Record human review and clear a persistent 401/403-style host stop",
    )
    review_host_parser.add_argument("--state-db", required=True)
    review_host_parser.add_argument("--host", required=True)
    review_host_parser.add_argument("--authorization-ref", required=True)
    review_host_parser.add_argument("--reason", required=True)

    plan_bo3_parser = subparsers.add_parser(
        "plan-bo3-capture",
        help="Validate an authorized BO3 API crawl plan without network use",
    )
    plan_bo3_parser.add_argument("--policy", required=True)
    plan_bo3_parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    plan_bo3_parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    plan_bo3_parser.add_argument(
        "--status", action="append", default=None,
        help="repeatable; defaults to finished and defwin",
    )
    plan_bo3_parser.add_argument("--window-days", type=int, default=7)
    plan_bo3_parser.add_argument("--page-limit", type=int, default=100)
    plan_bo3_parser.add_argument(
        "--profile", choices=("catalog", "training", "core", "rich", "exhaustive"),
        default="core",
    )

    capture_bo3_parser = subparsers.add_parser(
        "capture-bo3-json",
        help="Slow, resumable, explicitly authorized BO3 API raw capture",
    )
    capture_bo3_parser.add_argument("--state-db", required=True)
    capture_bo3_parser.add_argument("--output-dir", required=True)
    capture_bo3_parser.add_argument("--stream", required=True)
    capture_bo3_parser.add_argument("--policy", required=True)
    capture_bo3_parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    capture_bo3_parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    capture_bo3_parser.add_argument(
        "--status", action="append", default=None,
        help="repeatable; defaults to finished and defwin",
    )
    capture_bo3_parser.add_argument("--window-days", type=int, default=7)
    capture_bo3_parser.add_argument("--page-limit", type=int, default=100)
    capture_bo3_parser.add_argument(
        "--profile", choices=("catalog", "training", "core", "rich", "exhaustive"),
        default="core",
    )
    capture_bo3_parser.add_argument("--max-requests", type=int)
    capture_bo3_parser.add_argument(
        "--timeout-seconds", type=float, default=30.0,
    )

    audit_bo3_parser = subparsers.add_parser(
        "audit-bo3-capture",
        help="Report unfinished catalog windows and player-map gaps",
    )
    audit_bo3_parser.add_argument("--state-db", required=True)
    audit_bo3_parser.add_argument("--stream", required=True)
    audit_bo3_parser.add_argument("--max-samples", type=int, default=20)

    export_bo3_parser = subparsers.add_parser(
        "export-bo3-capture-index",
        help="Export the BO3 raw snapshot index as deterministic JSONL",
    )
    export_bo3_parser.add_argument("--state-db", required=True)
    export_bo3_parser.add_argument("--stream", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "parse-hltv-html":
        records = parse_file(
            args.html,
            page_type=args.page_type,
            source_url=args.source_url,
            observed_at=args.observed_at,
        )
        print(records_to_jsonl(records), end="")
        return
    if args.command == "capture-hltv-html":
        result = capture_manifest(
            args.state_db,
            args.manifest,
            args.output_dir,
            stream=args.stream,
            policy_path=args.policy,
            max_pages=args.max_pages,
            max_http_requests=args.max_http_requests,
            timeout_seconds=args.timeout_seconds,
        )
        failed = bool(result["stopped_reason"] or result["failure_count"])
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        if failed:
            raise SystemExit(2)
        return
    if args.command == "plan-hltv-capture":
        result = plan_capture(
            args.manifest,
            args.policy,
            max_pages=args.max_pages,
            max_http_requests=args.max_http_requests,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "export-hltv-capture-index":
        for record in capture_index(
            args.state_db,
            stream=args.stream,
            allow_partial=args.allow_partial,
        ):
            print(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        return
    if args.command == "parse-hltv-captures":
        for record in parsed_capture_records(
            args.state_db,
            stream=args.stream,
            allow_partial=args.allow_partial,
        ):
            print(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        return
    if args.command == "generate-hltv-results-manifest":
        for record in generate_results_manifest(
            args.start_date,
            args.end_date,
            window_days=args.window_days,
            url_template=args.url_template,
        ):
            print(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        return
    if args.command == "extract-hltv-match-manifest":
        records, report = extract_match_manifest(
            args.state_db,
            streams=args.stream,
            allow_partial=args.allow_partial,
        )
        for record in records:
            print(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return
    if args.command == "derive-hltv-results-pagination-manifest":
        records, report = derive_results_pagination_manifest(
            args.state_db,
            streams=args.stream,
            max_entries=args.max_entries,
            max_depth=args.max_depth,
            allow_partial=args.allow_partial,
        )
        for record in records:
            print(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return
    if args.command == "aggregate-hltv-match-manifest":
        records, report = aggregate_match_manifest(
            args.state_db,
            streams=args.stream,
            expected_start=args.start_date,
            expected_end=args.end_date,
            allow_partial=args.allow_partial,
            require_complete=not args.allow_incomplete,
        )
        for record in records:
            print(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return
    if args.command == "extract-hltv-mapstats-manifest":
        records, report = extract_mapstats_manifest(
            args.state_db,
            stream=args.stream,
            allow_partial=args.allow_partial,
            require_complete=not args.allow_incomplete,
        )
        for record in records:
            print(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return
    if args.command == "review-hltv-host-circuit":
        result = clear_host_circuit(
            args.state_db,
            authority=args.host,
            authorization_ref=args.authorization_ref,
            reason=args.reason,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "plan-bo3-capture":
        result = plan_bo3_capture(
            args.policy,
            start_date=args.start_date,
            end_date=args.end_date,
            statuses=args.status or ("finished", "defwin"),
            window_days=args.window_days,
            page_limit=args.page_limit,
            profile=args.profile,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "capture-bo3-json":
        result = capture_bo3(
            args.state_db,
            args.output_dir,
            stream=args.stream,
            policy_path=args.policy,
            start_date=args.start_date,
            end_date=args.end_date,
            statuses=args.status or ("finished", "defwin"),
            window_days=args.window_days,
            page_limit=args.page_limit,
            profile=args.profile,
            max_requests=args.max_requests,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        if result["stopped_reason"]:
            raise SystemExit(2)
        return
    if args.command == "audit-bo3-capture":
        result = audit_bo3_capture(
            args.state_db, stream=args.stream, max_samples=args.max_samples
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "export-bo3-capture-index":
        for record in bo3_capture_index(args.state_db, stream=args.stream):
            print(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        return

    connection = connect(args.db)
    initialize(connection)
    if args.command == "init":
        result = {"initialized": args.db}
    elif args.command == "import-legacy":
        result = import_legacy_csv(connection, args.csv, args.from_date)
    elif args.command in {
        "import-tedtay-dataset",
        "import-tedtay-csgo-dataset",
        "import-tedtay-csgo-pro-matches",
    }:
        result = import_tedtay_dataset(
            connection,
            args.historic_games_list,
            args.game_data_rh,
            args.from_date,
            batch_size=args.batch_size,
        )
    elif args.command == "audit":
        result = audit(connection)
    elif args.command == "import-jsonl":
        result = import_jsonl(
            connection,
            args.jsonl,
            source=args.source,
            stream=args.stream,
            batch_size=args.batch_size,
            max_records=args.max_records,
            point_in_time_eligible=args.point_in_time_eligible,
            license_ref=args.license_ref,
        )
    elif args.command == "materialize-hltv-stream":
        result = materialize_raw_stream(
            connection,
            source=args.source,
            stream=args.stream,
            max_records=args.max_records,
            after_raw_record_id=args.after_raw_record_id,
            record_kinds=args.kind,
            max_quarantine=args.max_quarantine,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
