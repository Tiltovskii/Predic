# Predic v2 data layer

This directory contains the point-in-time data foundation for a CS/CS2 match
prediction project. Network capture is disabled by default. The HLTV collector
may be enabled only for routes and uses covered by explicit written permission;
the permission reference, scope, time window, host/path/query allowlists, user
agent, contact, delay, and request limits are immutable inputs to every capture
stream. There is no anti-bot bypass, browser impersonation, proxy rotation, or
parallel request mode.

The initial implementation uses only Python's standard library and SQLite so it
can validate the contract before a large provider-specific backfill. Raw source
artifacts remain append-only; normalized rows retain source identity, revision,
and three distinct timestamps:

- `event_at`: when the game event happened;
- `known_at`: when the value became available to a predictor;
- `observed_at`: when our ingestion process retrieved it.

## Quick start

```bash
cd v2
python3 -m venv .venv
.venv/bin/pip install -e .
predic-data init --db data/predic.sqlite3
predic-data import-legacy \
  --db data/predic.sqlite3 \
  --csv ../CSGO/NN_csgo/mathes_for_a_5_years.csv \
  --from-date 2018-01-01
predic-data audit --db data/predic.sqlite3
```

The legacy importer deduplicates the mirrored team-order augmentation in the
old CSV and marks all reconstructed identities as low-confidence. It is useful
for bootstrapping and regression tests, not as a source of historically known
pre-match lineups or rankings. Its source snapshot has
`point_in_time_eligible = 0`; unknown legacy `known_at` values remain `NULL`.
Because the file has no stable match identifier, every legacy map is stored in
its own low-confidence series instead of guessing BO3/BO5 group boundaries.

## Downloaded TedTay CS:GO research bootstrap

`import-tedtay-dataset` imports the two **already downloaded** raw files from
[`tedtay/CS-GO-Pro-Matches-Comprehensive-Dataset`](https://github.com/tedtay/CS-GO-Pro-Matches-Comprehensive-Dataset):
`historic_games_list.csv` and `game_data_rh.csv`.

```bash
predic-data import-tedtay-dataset \
  --db data/predic.sqlite3 \
  --historic-games-list /path/to/historic_games_list.csv \
  --game-data-rh /path/to/game_data_rh.csv \
  --from-date 2018-01-01 \
  --batch-size 500
```

This adapter never executes the upstream scraper and makes no network request.
At audited upstream commit
`ce8a5f242a768c5698f8068828eeedc4fc134db1`, the repository declares an
[MIT license](https://github.com/tedtay/CS-GO-Pro-Matches-Comprehensive-Dataset/blob/main/LICENSE.txt)
whose `LICENSE.txt` SHA-256 is
`85ce7eda4c1d04cba58e5d9852703b1d978b31c196334bd5fa4dbf146136f285`.
That is a repository-declared license only: it does **not** establish that the
CSV data themselves are MIT-licensed or that their rights are verified. Its
README says the upstream data were scraped from HLTV using Selenium and
BeautifulSoup. The source is therefore recorded as a **research-bootstrap-only**
source, not as a replacement for a provider-specific authorization or license.
The bronze `source_snapshot` stores the exact repository URL, upstream commit,
license facts, explicit `dataset_rights_verified = false`, and SHA-256 values
of both downloaded CSV files; it always has `point_in_time_eligible = 0`, and
every imported `known_at` stays `NULL`.

The importer validates the exact raw header schemas, takes only the first row
for each exact `game_link` in each file, and parses `mapstatsid` from the link
as the durable source map ID. Every eligible historic row becomes one
low-confidence map/series plus two `series_participant` rows. A missing first
`game_data_rh.csv` row does not make the historic result disappear: it produces
zero lineup/stat rows and one immutable `tedtay_missing_game_data_rh` raw
record containing the exact historic CSV row. A map becomes one
low-confidence series; no BO3/BO5 grouping is inferred from team names and
dates. Teams are low-confidence name identities. Players are deliberately
low-confidence identities scoped to `(team, nickname)`, so two opponents with
the same nickname cannot silently merge.

The player-stat table order is reconciled to the historical map teams only
when its four captured half-score cells prove one final-score orientation.
Those fields do not encode overtime totals, and a tied final has no score-based
orientation. Such maps are still imported with their exact score (a draw has
`winner_team_id = NULL`) and both `series_participant` rows, but deliberately
get **zero** team-bound lineup/stat rows. Instead the full exact joined CSV
payload is written once to `raw_ingest_record` as
`tedtay_ambiguous_team_binding`, with a snapshot-scoped stream, `known_at =
NULL`, deterministic content hash, `mapstatsid`, and a reason. The JSON report
returns counts by kind/reason and at most 20 explicit samples, never an
unbounded map-ID list. This is intentionally not a source-order or
roster-history guess; a later evidence-backed resolver can consume those raw
records.

Malformed dates, links, score cells, and orientable lineups still fail closed.
The `--from-date` filter is applied to staged historic dates before full
score/lineup parsing, so known unfinished pre-2018 historical rows do not
prevent the intended 2018+ import. A source `date_unix_iso` is only an
un-zoned display timestamp; the importer uses its millisecond `date_unix` as
the normalized UTC event instant and accepts the corpus's observed UTC/UTC+1
display form. Each committed batch is one transaction. The importer hashes both
files before staging and again after staging, aborting if either source changed
in place. Re-running the same pair safely resumes after an interruption:
existing map/series/stat rows (including `observed_at` and every inserted
scalar), lineup rows, and quarantine records must be exactly identical or the
importer stops instead of overwriting data.

For each score-orientable map, it inserts exactly five starter
`lineup_member` rows per team and map-level player stats: kills/headshots,
assists/flash assists, deaths, KAST percentage, ADR, and rating. A legacy
plain-integer assists cell means `flash_assists = NULL`; `-` in KAST, ADR, or
rating remains `NULL`, with the original cells retained. The original K/D diff,
FK diff, and per-player raw cells remain in `metrics_json`.

## Authorized BO3 raw collection

BO3 is the rich 2020+ source. The collector uses the JSON service at
`api.bo3.gg`; it does not scrape rendered pages and does not spoof a browser.
It requires a separately confirmed BO3 permission policy because BO3 currently
publishes `ai-train=no` in its robots content signal. The committed
[`bo3_capture_policy.example.json`](examples/bo3_capture_policy.example.json)
is deliberately disabled. Copy it under ignored `v2/data/`, record the actual
permission reference/scope/date, keep an identifiable contact in the user
agent, and only then set `live_enabled` to `true`.

Validate a complete historical plan without making a request:

```bash
predic-data plan-bo3-capture \
  --policy data/bo3_capture_policy.local.json \
  --start-date 2020-06-15 \
  --end-date 2026-08-27 \
  --window-days 7 \
  --page-limit 100 \
  --profile core
```

The end date is exclusive. A live bounded run is:

```bash
predic-data capture-bo3-json \
  --state-db data/bo3-history-v2-state.sqlite3 \
  --output-dir data/bo3-history-raw \
  --stream bo3-history-2020-2026-v2 \
  --policy data/bo3_capture_policy.local.json \
  --start-date 2020-06-15 \
  --end-date 2026-08-27 \
  --status finished \
  --status defwin \
  --window-days 7 \
  --page-limit 100 \
  --profile core \
  --max-requests 100
```

Re-run the exact command to continue. One exclusive lock prevents two workers
from bypassing the shared rate limit. Request time is persisted across restarts;
`429` respects `Retry-After`, and `401/403/406/418/451` opens a durable host
circuit instead of trying different headers. Successful JSON is written and
fsynced under `objects/<sha-prefix>/<sha256>.json` before the corresponding task
and newly discovered child tasks are committed.

The catalog is split into closed half-open date windows. The first page records
`total.count`; every expected offset must be captured, and the number of unique
discovered matches must close exactly. This avoids silently resuming against a
single moving `2020-now` offset list. Each catalog response discovers match
details and played game IDs. The `core` profile then captures:

1. match details, including the available veto/pick/ban data;
2. each played game, including round/team economy data and `demo_url`;
3. `/games/{id}/players_stats`, including Steam profile/SteamID64, historical
   team binding, and player-map metrics.

For a time-boxed first pass, `--profile training` captures the complete catalog
and schedules every played map's player statistics immediately, while deferring
match detail, round/economy detail, demos, vetoes, and odds. In every profile
above `catalog`, player-stat tasks run before detail enrichment. Existing `core`
checkpoints adopt this player-first scheduling on resume, so the catalog is not
downloaded again and the full detail queue remains available for a later pass.

A finished map is not complete merely because its HTTP requests returned 200.
The quality gate requires ten distinct Steam profiles, two teams of five, and
non-null kills, deaths, assists, damage, ADR, and KAST. Partial responses remain
visible as retry/quarantine gaps and are never silently accepted. Inspect them
with:

```bash
predic-data audit-bo3-capture \
  --state-db data/bo3-history-v2-state.sqlite3 \
  --stream bo3-history-2020-2026-v2
```

`rich` additionally schedules kill/flash matrices plus grenade, hit-group, and
weapon endpoints. `exhaustive` also schedules player stats for every round and
can create several million requests, so it should only be enabled as a later
targeted enrichment. Use `training` when player timelines must land first and
resume as `core` to fill all details; demo URLs make
it possible to repair source gaps and derive event-level features later.

Upcoming/current matches belong in separate immutable snapshot streams, for
example one stream per capture time. Historical re-fetches have `known_at =
NULL`; a genuinely observed pre-match snapshot may use its capture time as
`known_at`. Never treat a historical BO3 `updated_at`, current team rank, AI
prediction, late map disclosure, or bookmaker value as information known before
the match.

### Swing

FACEIT and HLTV disclose the concept, not their fitted probability model or
coefficients. Both measure a change in win probability around an action,
roughly `P(win | state_after) - P(win | state_before)`, but their state,
credit allocation, and normalization are proprietary and differ. BO3 aggregate
round JSON is not sufficient to reproduce either metric exactly. A BO3 demo is
the correct input for a separately versioned `argus_swing_v1`: it exposes the
ordered kill/damage/flash/bomb state needed to train our own calibrated round
win-probability model. Do not label that proxy as FACEIT Swing or HLTV Round
Swing. See the official [FACEIT explanation](https://support.faceit.com/hc/en-us/articles/27123235446428-FACEIT-Season-8-Understanding-Round-Swing),
[HLTV Rating 3.0 launch](https://www.hltv.org/news/42485/introducing-rating-30),
and [HLTV's current adjustment](https://www.hltv.org/news/43047/rating-30-adjustments-go-live).

## Layers

1. `source_snapshot` is the bronze manifest for immutable files/API payloads.
2. Entity, match, map, lineup, stats, ranking, and odds tables are normalized
   silver data.
3. Training examples must use only point-in-time-eligible snapshots and be
   materialized with `known_at <= prediction_at`.

The schema keeps organization, competitive team/core, and lineup separate.
Provider connectors should be added only after their storage, ML, and betting
permissions are confirmed.

## Resumable authorized exports

`import-jsonl` ingests an immutable JSONL export without assuming a provider
schema. Each line is retained verbatim after canonical JSON serialization and
may contain `record_id`, `kind`, `event_at`, and `known_at` fields:

```json
{"record_id":"match-42","kind":"match","event_at":"2025-01-02T12:00:00Z","payload":{"winner":"team-a"}}
```

```bash
predic-data import-jsonl \
  --db data/predic.sqlite3 \
  --jsonl /path/to/authorized-export.jsonl \
  --source licensed-provider \
  --stream matches-2018-2026 \
  --batch-size 1000 \
  --license-ref /path/to/provider-license.txt
```

The byte cursor advances in the same transaction as each inserted batch. A
restart with the same command resumes exactly at the last committed offset;
stable record IDs make replay idempotent. If the file content changes, the
import stops and requires a new stream name instead of mixing revisions.

## Offline HLTV HTML adapter

The HLTV adapter has no HTTP client and accepts only an already captured local
HTML file. It emits deterministic typed JSONL records while leaving
`known_at = null`; a current rendering of an old match does not prove that a
field was historically available before the match.

```bash
predic-data parse-hltv-html \
  --html /path/to/authorized-match.html \
  --source-url https://www.hltv.org/matches/123/example \
  > authorized-match.jsonl
```

The output contains source numeric IDs, the document SHA-256, parser/schema
versions, raw metric cells, and warnings for partial or ambiguous fields.
Unplayed maps, missing ranks, absent Swing values, and unknown game/rating
versions remain explicit rather than being converted to sentinel numbers.
Map records keep named `score_a`/`score_b`, series map score, completed round
count, regulation/overtime counts, parsed half/OT segments, and the raw score
breakdown. Rulesets use `MR15`/`MR12` terminology; a first-to-16/first-to-13
winning score is not mislabeled as MR16/MR13.

## Authorized resumable HLTV capture

The repository contains a sequential raw HTML capture layer, but the checked-in
example policy has `live_enabled: false`. Copy the examples and replace every
placeholder only after checking the exact written permission:

```bash
cp examples/hltv_capture_policy.example.json data/hltv-policy.json
cp examples/hltv_capture_manifest.example.jsonl data/hltv-pilot.jsonl
```

The configured `contact` must occur in the `user_agent`. Allowed query keys are
explicit; redirects are followed manually and every hop must remain inside the
same configured scope. `robots_txt_mode: respect` currently fails closed.
`written_permission_override` is valid only when the written permission itself
explicitly covers the configured routes despite the public robots policy.

Validate the policy and every URL without creating state or using the network:

```bash
predic-data plan-hltv-capture \
  --policy data/hltv-policy.json \
  --manifest data/hltv-pilot.jsonl \
  --max-pages 20 \
  --max-http-requests 40
```

A successful plan means only that the files are structurally valid and every
URL is inside the declared scope. It does not prove that HLTV granted that
scope; the authorization fields must be copied from the actual written
permission before live capture is enabled.

After the policy matches the permission, a bounded pilot is:

```bash
predic-data capture-hltv-html \
  --policy data/hltv-policy.json \
  --manifest data/hltv-pilot.jsonl \
  --state-db data/hltv-capture.sqlite3 \
  --output-dir data/raw/hltv \
  --stream pilot-2026-08-25 \
  --max-pages 20 \
  --max-http-requests 40
```

Run the same command again to resume. Completed pages are never requested again;
if the process dies in the narrow interval after a response but before its
completion transaction, the current page may be requested once more. Raw bodies
are content-addressed and never overwritten. One shared state database must be
used for all streams so host-level cooldowns and blocks are preserved. A 401,
403, 406, 418, or 451 opens a manual-review circuit; 429 honors `Retry-After` and
the local exponential backoff. These stops and terminal page failures produce a
non-zero CLI exit status.

Do not retry a 401/403-style stop. After HLTV explicitly clarifies the route or
permission, record that review before starting a new stream; the command keeps
the original request timestamp and writes an audit row. It cannot clear a 429
cooldown.

```bash
predic-data review-hltv-host-circuit \
  --state-db data/hltv-capture.sqlite3 \
  --host www.hltv.org \
  --authorization-ref 'HLTV follow-up, 2026-08-25' \
  --reason 'HLTV confirmed the permitted historical listing route'
```

Once every pilot page succeeded, verify each raw SHA and parse the captures:

```bash
predic-data export-hltv-capture-index \
  --state-db data/hltv-capture.sqlite3 \
  --stream pilot-2026-08-25 \
  > data/hltv-pilot-captures.jsonl

predic-data parse-hltv-captures \
  --state-db data/hltv-capture.sqlite3 \
  --stream pilot-2026-08-25 \
  > data/hltv-pilot-parsed.jsonl

predic-data import-jsonl \
  --db data/predic.sqlite3 \
  --jsonl data/hltv-pilot-parsed.jsonl \
  --source authorized-hltv-capture \
  --stream pilot-2026-08-25-parsed \
  --license-ref /path/to/redacted-permission-reference.txt
```

`parse-hltv-captures` validates and stages the entire stream before writing its
first JSONL record, so a corrupt or unexpected later page cannot leave an
importable-looking prefix. Import only after the parse command exits with zero.

Do not pass `--point-in-time-eligible` for historical pages captured today.
Their records retain the real capture `observed_at`, but `known_at` remains null:
a current rendering of a 2018 match cannot prove what was known before that
match began. Partial export is rejected by default; `--allow-partial` exists for
explicit diagnostics only.

## Full-history discovery: results → matches → map stats

The old project paginated `/results` and immediately discarded URLs and source
IDs. v2 does not repeat that: it treats listing pages as their own authorized
capture type, preserves the raw listing SHA, then derives immutable manifests
for the next layer. No numeric-ID scan and no guessed map-stats URL is used.

### Current route status

The 2026-08-25 sentinel captured `/results` successfully and found ordinary
`offset` pagination. The separate historical request
`/results?startDate=2018-01-01&endDate=2018-01-07` returned HTTP 403. HLTV then
asked for slower sequential collection, so the local policy was tightened to a
five-minute interval and exactly one fresh historical probe was made in a new
stream. That probe also returned HTTP 403. The shared capture state therefore
has an open host circuit, and **no retry or bulk backfill may be started from
this repository yet**. This is not evidence that the written permission is
invalid; it is a concrete route-level response that needs a more specific HLTV
clarification.

Ask HLTV to confirm the exact approved historical listing URL(s), query
parameters, pagination semantics, rate limit, and whether this collector's
IP/user-agent must be allowlisted. In particular, ask whether the date filter
above is the intended route. Do not clear the host circuit until that answer is
recorded in the local policy/review command. A generic
all-time `/results` walk is not presented as a full-backfill route: the
successful page reported over 121k results, so one-page-at-a-time pagination
would be impractically deep without an explicitly approved bounded route.

### After HLTV confirms the exact historical route

Generate root date windows *locally*. The command below only writes a manifest;
do not submit its output until the template has been confirmed by HLTV and is
allowlisted in the policy.

```bash
predic-data generate-hltv-results-manifest \
  --start-date 2018-01-01 \
  --end-date 2018-01-31 \
  --window-days 7 \
  --url-template 'https://www.hltv.org/results?startDate={start_date}&endDate={end_date}' \
  > data/results-2018-01.jsonl
```

Capture a bounded root batch under a new immutable stream. The collector stores
the exact requested/final URL, raw body hash, and listing metadata. A listing
whose redirect changes identity, whose pagination drops the requested date
window, or whose displayed `Results for …` date is outside the claimed window
is rejected before it can produce a match manifest.

```bash
predic-data capture-hltv-html \
  --policy data/hltv-policy.json \
  --manifest data/results-2018-01.jsonl \
  --state-db data/hltv-capture.sqlite3 \
  --output-dir data/raw/hltv \
  --stream results-2018-01 \
  --max-pages 20 \
  --max-http-requests 40
```

If the approved date window has more than one results page, derive (offline)
the next child manifest from all captured pages. It emits only real pagination
links that are not already captured; every child retains a verified parent
stream, record ID, URLs, SHA, timestamp, and exact discovered link. Capture
that file under a **new** stream, add it to the repeated `--stream` list, and
repeat until the report says `coverage_complete: true`.

```bash
predic-data derive-hltv-results-pagination-manifest \
  --state-db data/hltv-capture.sqlite3 \
  --stream results-2018-01 \
  > data/results-2018-01-page-1.jsonl 2> data/results-2018-01-pagination-report.json

predic-data capture-hltv-html \
  --policy data/hltv-policy.json \
  --manifest data/results-2018-01-page-1.jsonl \
  --state-db data/hltv-capture.sqlite3 \
  --output-dir data/raw/hltv \
  --stream results-2018-01-page-1 \
  --max-pages 20 \
  --max-http-requests 40
```

Only after pagination closes, aggregate the full root+child graph. This command
fails closed by default: it writes no match JSONL if a pagination page or date
window is missing. `--allow-incomplete` is diagnostics only.

```bash
predic-data aggregate-hltv-match-manifest \
  --state-db data/hltv-capture.sqlite3 \
  --stream results-2018-01 \
  --stream results-2018-01-page-1 \
  --start-date 2018-01-01 \
  --end-date 2018-01-31 \
  > data/matches-2018-01.jsonl 2> data/results-2018-01-report.json
```

Capture that exact match manifest under its own stream. After those match
captures have passed parsing, derive the statistics manifest from actual
`mapstatsid` links embedded in the match HTML. This also fails closed if any
played map has no exact link; `--allow-incomplete` is diagnostics only:

```bash
predic-data extract-hltv-mapstats-manifest \
  --state-db data/hltv-capture.sqlite3 \
  --stream matches-2018-01 \
  > data/mapstats-2018-01.jsonl 2> data/matches-2018-01-report.json
```

Run results, matches, and map-stats as separate small immutable streams (for
example one short approved window at a time), always sharing the same capture
state DB. This makes long runs resumable, preserves the host cooldown/block
circuit across batches, and leaves a checkable parent-SHA chain from player
stats back to the listing page that discovered the match.

## Materializing model tables

`import-jsonl` is the bronze archive; it intentionally keeps every parsed
record verbatim. The next local step writes only verified relations into the
normalized tables used for modeling. Use dependency-safe phases for a large
stream: first `series`, then `map`, then `ranking`; after the corresponding
match stream is complete, materialize `player_map_stats` from the map-stats
stream. Each command returns `next_raw_record_id`; repeat the same phase with
that value as `--after-raw-record-id` while `has_more` is true.

```bash
predic-data materialize-hltv-stream \
  --db data/predic.sqlite3 \
  --source authorized-hltv-capture \
  --stream matches-2018-01-parsed \
  --kind series \
  --max-records 10000

predic-data materialize-hltv-stream \
  --db data/predic.sqlite3 \
  --source authorized-hltv-capture \
  --stream matches-2018-01-parsed \
  --kind map \
  --max-records 10000

predic-data materialize-hltv-stream \
  --db data/predic.sqlite3 \
  --source authorized-hltv-capture \
  --stream mapstats-2018-01-parsed \
  --kind player_map_stats \
  --max-records 10000
```

The materializer never turns the collection timestamp into `known_at`, invents
a map for orphan player stats, or maps a match-page lineup onto every map. Such
records remain retained in bronze and are returned in the bounded
`quarantined` report for follow-up. This is deliberate: it protects both the
training corpus and later point-in-time evaluation from invented joins.
