from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from .audit import audit
from .baseline import (
    build_point_in_time_features,
    extract_bo3_match_table,
    train_catboost_baseline,
    walk_forward_catboost_backtest,
)
from .bo3_capture import (
    audit_bo3_capture,
    bo3_capture_index,
    capture_bo3,
    plan_bo3_capture,
    reprocess_bo3_game_snapshots,
    reprocess_bo3_player_snapshots,
)
from .db import connect, initialize
from .hltv_capture import (
    capture_index,
    capture_manifest,
    clear_host_circuit,
    parsed_capture_records,
    plan_capture,
)
from .hltv_discovery import (
    aggregate_match_manifest,
    derive_results_pagination_manifest,
    extract_mapstats_manifest,
    extract_match_manifest,
    generate_results_manifest,
)
from .hltv_offline import parse_file, records_to_jsonl
from .legacy import import_legacy_csv
from .map_baseline import (
    build_map_feature_table,
    walk_forward_map_catboost_backtest,
)
from .materialize import materialize_raw_stream
from .raw_jsonl import import_jsonl
from .tedtay import import_tedtay_dataset
from .valve_rankings import collect_valve_rankings


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
    results_manifest_parser.add_argument(
        "--start-date", type=date.fromisoformat, required=True
    )
    results_manifest_parser.add_argument(
        "--end-date", type=date.fromisoformat, required=True
    )
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
    aggregate_match_parser.add_argument(
        "--start-date", type=date.fromisoformat, required=True
    )
    aggregate_match_parser.add_argument(
        "--end-date", type=date.fromisoformat, required=True
    )
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
        "--status",
        action="append",
        default=None,
        help="repeatable; defaults to finished and defwin",
    )
    plan_bo3_parser.add_argument("--window-days", type=int, default=7)
    plan_bo3_parser.add_argument("--page-limit", type=int, default=100)
    plan_bo3_parser.add_argument(
        "--profile",
        choices=("catalog", "training", "core", "rich", "exhaustive"),
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
    capture_bo3_parser.add_argument(
        "--start-date", type=date.fromisoformat, required=True
    )
    capture_bo3_parser.add_argument(
        "--end-date", type=date.fromisoformat, required=True
    )
    capture_bo3_parser.add_argument(
        "--status",
        action="append",
        default=None,
        help="repeatable; defaults to finished and defwin",
    )
    capture_bo3_parser.add_argument("--window-days", type=int, default=7)
    capture_bo3_parser.add_argument("--page-limit", type=int, default=100)
    capture_bo3_parser.add_argument(
        "--profile",
        choices=("catalog", "training", "core", "rich", "exhaustive"),
        default="core",
    )
    capture_bo3_parser.add_argument("--max-requests", type=int)
    capture_bo3_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
    )
    capture_bo3_parser.add_argument(
        "--continue-on-quality-error",
        action="store_true",
        help="keep crawling while incomplete payloads remain queued for retry/audit",
    )
    capture_bo3_parser.add_argument(
        "--continue-on-network-error",
        action="store_true",
        help="keep crawling after transient network failures remain queued for retry",
    )
    capture_bo3_parser.add_argument(
        "--quarantine-incomplete-player-stats",
        action="store_true",
        help="store incomplete historical player payloads once without refetching",
    )
    capture_bo3_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel HTTP workers sharing the persisted global rate limit",
    )

    audit_bo3_parser = subparsers.add_parser(
        "audit-bo3-capture",
        help="Report unfinished catalog windows and player-map gaps",
    )
    audit_bo3_parser.add_argument("--state-db", required=True)
    audit_bo3_parser.add_argument("--stream", required=True)
    audit_bo3_parser.add_argument("--max-samples", type=int, default=20)

    reprocess_bo3_parser = subparsers.add_parser(
        "reprocess-bo3-player-snapshots",
        help="Offline rebuild of BO3 player indexes from durable raw JSON",
    )
    reprocess_bo3_parser.add_argument("--state-db", required=True)
    reprocess_bo3_parser.add_argument("--stream", required=True)
    reprocess_bo3_parser.add_argument("--after-game-id", type=int)
    reprocess_bo3_parser.add_argument("--max-games", type=int)

    reprocess_bo3_games_parser = subparsers.add_parser(
        "reprocess-bo3-game-snapshots",
        help="Offline rebuild of BO3 game-detail indexes from durable raw JSON",
    )
    reprocess_bo3_games_parser.add_argument("--state-db", required=True)
    reprocess_bo3_games_parser.add_argument("--stream", required=True)
    reprocess_bo3_games_parser.add_argument("--after-game-id", type=int)
    reprocess_bo3_games_parser.add_argument("--max-games", type=int)

    export_bo3_parser = subparsers.add_parser(
        "export-bo3-capture-index",
        help="Export the BO3 raw snapshot index as deterministic JSONL",
    )
    export_bo3_parser.add_argument("--state-db", required=True)
    export_bo3_parser.add_argument("--stream", required=True)

    valve_rankings_parser = subparsers.add_parser(
        "collect-valve-rankings",
        help="Export official historical Valve standings from Valve's git repository",
    )
    valve_rankings_parser.add_argument("--repo", required=True)
    valve_rankings_parser.add_argument("--output-csv", required=True)

    baseline_extract_parser = subparsers.add_parser(
        "extract-bo3-baseline-matches",
        help="Extract compact BO3 series outcomes without round/player metrics",
    )
    baseline_extract_parser.add_argument("--state-db", required=True)
    baseline_extract_parser.add_argument("--output-csv", required=True)
    baseline_extract_parser.add_argument("--stream", default="bo3-history-2020-2026-v2")

    baseline_features_parser = subparsers.add_parser(
        "build-baseline-features",
        help="Build chronological no-future team/player counter features",
    )
    baseline_features_parser.add_argument("--matches-csv", required=True)
    baseline_features_parser.add_argument("--output-csv", required=True)
    baseline_features_parser.add_argument("--rankings-csv")

    baseline_train_parser = subparsers.add_parser(
        "train-catboost-baseline",
        help="Train winner, BO3 score and round-share CatBoost baselines",
    )
    baseline_train_parser.add_argument("--features-csv", required=True)
    baseline_train_parser.add_argument("--output-dir", required=True)
    baseline_train_parser.add_argument("--train-before", default="2025-01-01")
    baseline_train_parser.add_argument("--test-from", default="2026-01-01")
    baseline_train_parser.add_argument("--iterations", type=int, default=900)
    baseline_train_parser.add_argument(
        "--feature-set",
        choices=("base", "core", "all", "core-veto"),
        default="core",
    )
    baseline_train_parser.add_argument("--veto-known-only", action="store_true")

    walk_forward_parser = subparsers.add_parser(
        "backtest-catboost-walk-forward",
        help="Simulate monthly retraining using only results known at each cutoff",
    )
    walk_forward_parser.add_argument("--features-csv", required=True)
    walk_forward_parser.add_argument("--output-dir", required=True)
    walk_forward_parser.add_argument("--test-from", default="2026-01-01")
    walk_forward_parser.add_argument("--validation-days", type=int, default=90)
    walk_forward_parser.add_argument("--iterations", type=int, default=900)
    walk_forward_parser.add_argument(
        "--feature-set",
        choices=("base", "core", "all", "core-veto"),
        default="core",
    )
    walk_forward_parser.add_argument("--veto-known-only", action="store_true")

    map_features_parser = subparsers.add_parser(
        "build-map-baseline-features",
        help="Expand causal series features into strict after-veto map rows",
    )
    map_features_parser.add_argument("--matches-csv", required=True)
    map_features_parser.add_argument("--series-features-csv", required=True)
    map_features_parser.add_argument("--output-csv", required=True)
    map_features_parser.add_argument(
        "--series-feature-set",
        choices=("base", "core", "all", "core-veto"),
        default="core-veto",
    )

    map_walk_forward_parser = subparsers.add_parser(
        "backtest-map-catboost-walk-forward",
        help="Backtest an individual-map winner model with monthly refits",
    )
    map_walk_forward_parser.add_argument("--features-csv", required=True)
    map_walk_forward_parser.add_argument("--output-dir", required=True)
    map_walk_forward_parser.add_argument("--test-from", default="2026-01-01")
    map_walk_forward_parser.add_argument("--validation-days", type=int, default=90)
    map_walk_forward_parser.add_argument("--iterations", type=int, default=900)
    map_walk_forward_parser.add_argument("--cohort-metadata-jsonl")

    argus_data_parser = subparsers.add_parser(
        "build-light-argus-dataset",
        help="Build causal per-player histories for target-aware map prediction",
    )
    argus_data_parser.add_argument("--state-db", required=True)
    argus_data_parser.add_argument("--matches-csv", required=True)
    argus_data_parser.add_argument("--map-features-csv", required=True)
    argus_data_parser.add_argument("--output-dir", required=True)
    argus_data_parser.add_argument("--raw-dir")
    argus_data_parser.add_argument("--stream", default="bo3-history-2020-2026-v2")
    argus_data_parser.add_argument("--max-history", type=int, default=32)

    argus_train_parser = subparsers.add_parser(
        "train-light-argus",
        help="Train a small target-aware player-history Transformer",
    )
    argus_train_parser.add_argument("--dataset-dir", required=True)
    argus_train_parser.add_argument("--output-dir", required=True)
    argus_train_parser.add_argument("--train-before", default="2025-01-01")
    argus_train_parser.add_argument("--test-from", default="2026-01-01")
    argus_train_parser.add_argument("--epochs", type=int, default=12)
    argus_train_parser.add_argument("--patience", type=int, default=3)
    argus_train_parser.add_argument("--batch-size", type=int, default=256)
    argus_train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    argus_train_parser.add_argument("--weight-decay", type=float, default=0.01)
    argus_train_parser.add_argument("--d-model", type=int, default=128)
    argus_train_parser.add_argument("--layers", type=int, default=3)
    argus_train_parser.add_argument("--heads", type=int, default=4)
    argus_train_parser.add_argument("--dropout", type=float, default=0.10)
    argus_train_parser.add_argument("--no-player-identity", action="store_true")
    argus_train_parser.add_argument("--no-team-identity", action="store_true")
    argus_train_parser.add_argument("--device", default="auto")
    argus_train_parser.add_argument("--seed", type=int, default=20260827)
    argus_train_parser.add_argument("--no-refit", action="store_true")
    argus_train_parser.add_argument("--monthly-refit", action="store_true")
    argus_train_parser.add_argument("--max-train-rows", type=int)
    argus_train_parser.add_argument("--max-validation-rows", type=int)
    argus_train_parser.add_argument("--max-test-rows", type=int)
    argus_train_parser.add_argument("--catboost-predictions-csv")
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
            continue_on_quality_error=args.continue_on_quality_error,
            continue_on_network_error=args.continue_on_network_error,
            quarantine_incomplete=args.quarantine_incomplete_player_stats,
            workers=args.workers,
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
    if args.command == "reprocess-bo3-player-snapshots":
        result = reprocess_bo3_player_snapshots(
            args.state_db,
            stream=args.stream,
            after_game_id=args.after_game_id,
            max_games=args.max_games,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "reprocess-bo3-game-snapshots":
        result = reprocess_bo3_game_snapshots(
            args.state_db,
            stream=args.stream,
            after_game_id=args.after_game_id,
            max_games=args.max_games,
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
    if args.command == "collect-valve-rankings":
        result = collect_valve_rankings(args.repo, args.output_csv)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "extract-bo3-baseline-matches":
        result = extract_bo3_match_table(
            args.state_db, args.output_csv, stream=args.stream
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "build-baseline-features":
        result = build_point_in_time_features(
            args.matches_csv,
            args.output_csv,
            rankings_csv=args.rankings_csv,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "train-catboost-baseline":
        result = train_catboost_baseline(
            args.features_csv,
            args.output_dir,
            train_before=args.train_before,
            test_from=args.test_from,
            iterations=args.iterations,
            feature_set=args.feature_set,
            veto_known_only=args.veto_known_only,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "backtest-catboost-walk-forward":
        result = walk_forward_catboost_backtest(
            args.features_csv,
            args.output_dir,
            test_from=args.test_from,
            validation_days=args.validation_days,
            iterations=args.iterations,
            feature_set=args.feature_set,
            veto_known_only=args.veto_known_only,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "build-map-baseline-features":
        result = build_map_feature_table(
            args.matches_csv,
            args.series_features_csv,
            args.output_csv,
            series_feature_set=args.series_feature_set,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "backtest-map-catboost-walk-forward":
        result = walk_forward_map_catboost_backtest(
            args.features_csv,
            args.output_dir,
            test_from=args.test_from,
            validation_days=args.validation_days,
            iterations=args.iterations,
            cohort_metadata_jsonl=args.cohort_metadata_jsonl,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "build-light-argus-dataset":
        from .light_argus_data import build_light_argus_dataset

        result = build_light_argus_dataset(
            args.state_db,
            args.matches_csv,
            args.map_features_csv,
            args.output_dir,
            raw_dir=args.raw_dir,
            stream=args.stream,
            max_history=args.max_history,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "train-light-argus":
        from .light_argus import train_light_argus

        result = train_light_argus(
            args.dataset_dir,
            args.output_dir,
            train_before=args.train_before,
            test_from=args.test_from,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
            dropout=args.dropout,
            use_player_identity=not args.no_player_identity,
            use_team_identity=not args.no_team_identity,
            device=args.device,
            seed=args.seed,
            refit=not args.no_refit,
            monthly_refit=args.monthly_refit,
            max_train_rows=args.max_train_rows,
            max_validation_rows=args.max_validation_rows,
            max_test_rows=args.max_test_rows,
            catboost_predictions_csv=args.catboost_predictions_csv,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
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
