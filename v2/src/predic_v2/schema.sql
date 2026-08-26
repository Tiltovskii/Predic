PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS source_snapshot (
    snapshot_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_locator TEXT NOT NULL,
    source_revision TEXT,
    observed_at TEXT NOT NULL,
    published_at TEXT,
    content_sha256 TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    license_ref TEXT,
    point_in_time_eligible INTEGER NOT NULL DEFAULT 1 CHECK (point_in_time_eligible IN (0, 1)),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (source, source_locator, content_sha256)
);

CREATE TABLE IF NOT EXISTS organization (
    organization_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    source_snapshot_id TEXT NOT NULL REFERENCES source_snapshot(snapshot_id)
);

CREATE TABLE IF NOT EXISTS team_core (
    team_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    organization_id TEXT REFERENCES organization(organization_id),
    identity_confidence TEXT NOT NULL CHECK (identity_confidence IN ('low', 'medium', 'high')),
    valid_from TEXT,
    valid_to TEXT,
    source_snapshot_id TEXT NOT NULL REFERENCES source_snapshot(snapshot_id)
);

CREATE TABLE IF NOT EXISTS player (
    player_id TEXT PRIMARY KEY,
    canonical_nickname TEXT NOT NULL,
    identity_confidence TEXT NOT NULL CHECK (identity_confidence IN ('low', 'medium', 'high')),
    source_snapshot_id TEXT NOT NULL REFERENCES source_snapshot(snapshot_id)
);

CREATE TABLE IF NOT EXISTS entity_alias (
    source TEXT NOT NULL,
    entity_type TEXT NOT NULL CHECK (entity_type IN ('player', 'team', 'organization')),
    source_entity_id TEXT NOT NULL,
    canonical_entity_id TEXT NOT NULL,
    alias_at TEXT,
    source_snapshot_id TEXT NOT NULL REFERENCES source_snapshot(snapshot_id),
    PRIMARY KEY (source, entity_type, source_entity_id, canonical_entity_id)
);

CREATE TABLE IF NOT EXISTS series (
    series_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_series_id TEXT,
    scheduled_at TEXT,
    started_at TEXT,
    ended_at TEXT,
    known_at TEXT,
    observed_at TEXT NOT NULL,
    best_of INTEGER,
    lan_online TEXT,
    event_name TEXT,
    stage_name TEXT,
    status TEXT NOT NULL DEFAULT 'finished',
    winner_team_id TEXT REFERENCES team_core(team_id),
    identity_confidence TEXT NOT NULL CHECK (identity_confidence IN ('low', 'medium', 'high')),
    source_snapshot_id TEXT NOT NULL REFERENCES source_snapshot(snapshot_id)
);

CREATE TABLE IF NOT EXISTS series_participant (
    series_id TEXT NOT NULL REFERENCES series(series_id),
    team_id TEXT NOT NULL REFERENCES team_core(team_id),
    team_slot INTEGER NOT NULL CHECK (team_slot IN (1, 2)),
    source_snapshot_id TEXT NOT NULL REFERENCES source_snapshot(snapshot_id),
    PRIMARY KEY (series_id, team_id),
    UNIQUE (series_id, team_slot)
);

CREATE TABLE IF NOT EXISTS map_game (
    map_id TEXT PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES series(series_id),
    source_map_id TEXT,
    map_order INTEGER,
    map_name TEXT NOT NULL,
    game_version TEXT NOT NULL CHECK (game_version IN ('CSGO', 'CS2', 'UNKNOWN')),
    ruleset TEXT,
    started_at TEXT,
    ended_at TEXT,
    known_at TEXT,
    observed_at TEXT NOT NULL,
    team_a_id TEXT NOT NULL REFERENCES team_core(team_id),
    team_b_id TEXT NOT NULL REFERENCES team_core(team_id),
    score_a INTEGER,
    score_b INTEGER,
    winner_team_id TEXT REFERENCES team_core(team_id),
    picked_by_team_id TEXT REFERENCES team_core(team_id),
    legacy_target REAL,
    source_snapshot_id TEXT NOT NULL REFERENCES source_snapshot(snapshot_id),
    UNIQUE (source_snapshot_id, source_map_id)
);

CREATE TABLE IF NOT EXISTS lineup_member (
    map_id TEXT NOT NULL REFERENCES map_game(map_id),
    team_id TEXT NOT NULL REFERENCES team_core(team_id),
    player_id TEXT NOT NULL REFERENCES player(player_id),
    slot INTEGER NOT NULL,
    role TEXT,
    member_type TEXT NOT NULL DEFAULT 'starter',
    announced_at TEXT,
    known_at TEXT,
    actual_at TEXT,
    source_snapshot_id TEXT NOT NULL REFERENCES source_snapshot(snapshot_id),
    PRIMARY KEY (map_id, team_id, slot)
);

CREATE TABLE IF NOT EXISTS player_map_stats (
    map_id TEXT NOT NULL REFERENCES map_game(map_id),
    team_id TEXT NOT NULL REFERENCES team_core(team_id),
    player_id TEXT NOT NULL REFERENCES player(player_id),
    side TEXT NOT NULL DEFAULT 'BOTH',
    metric_version TEXT NOT NULL,
    known_at TEXT,
    observed_at TEXT NOT NULL,
    kills INTEGER,
    deaths INTEGER,
    assists INTEGER,
    flash_assists INTEGER,
    headshots INTEGER,
    traded_deaths INTEGER,
    opening_kills INTEGER,
    opening_deaths INTEGER,
    adr REAL,
    kast REAL,
    kpr REAL,
    dpr REAL,
    swing REAL,
    rating REAL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    source_snapshot_id TEXT NOT NULL REFERENCES source_snapshot(snapshot_id),
    PRIMARY KEY (map_id, player_id, side, metric_version)
);

CREATE TABLE IF NOT EXISTS ranking_snapshot (
    ranking_snapshot_id TEXT PRIMARY KEY,
    ranking_system TEXT NOT NULL,
    team_id TEXT NOT NULL REFERENCES team_core(team_id),
    rank INTEGER,
    points REAL,
    published_at TEXT,
    known_at TEXT,
    observed_at TEXT NOT NULL,
    metric_version TEXT NOT NULL,
    source_snapshot_id TEXT NOT NULL REFERENCES source_snapshot(snapshot_id)
);

CREATE TABLE IF NOT EXISTS odds_snapshot (
    odds_snapshot_id TEXT PRIMARY KEY,
    series_id TEXT REFERENCES series(series_id),
    map_id TEXT REFERENCES map_game(map_id),
    bookmaker TEXT NOT NULL,
    market_id TEXT NOT NULL,
    selection_id TEXT NOT NULL,
    market_type TEXT NOT NULL,
    decimal_odds REAL NOT NULL,
    available_size REAL,
    commission_rate REAL,
    in_play INTEGER NOT NULL DEFAULT 0,
    suspended INTEGER NOT NULL DEFAULT 0,
    captured_at TEXT NOT NULL,
    known_at TEXT NOT NULL,
    source_snapshot_id TEXT NOT NULL REFERENCES source_snapshot(snapshot_id)
);

CREATE TABLE IF NOT EXISTS ingestion_checkpoint (
    source TEXT NOT NULL,
    stream TEXT NOT NULL,
    cursor TEXT,
    high_watermark TEXT,
    updated_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (source, stream)
);

CREATE TABLE IF NOT EXISTS raw_ingest_record (
    raw_record_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    stream TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    record_kind TEXT NOT NULL,
    event_at TEXT,
    known_at TEXT,
    observed_at TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    source_snapshot_id TEXT NOT NULL REFERENCES source_snapshot(snapshot_id),
    UNIQUE (source, stream, source_record_id)
);

CREATE INDEX IF NOT EXISTS idx_map_game_started_at ON map_game(started_at);
CREATE INDEX IF NOT EXISTS idx_map_game_source_map_id ON map_game(source_map_id);
CREATE INDEX IF NOT EXISTS idx_series_participant_team ON series_participant(team_id);
CREATE INDEX IF NOT EXISTS idx_lineup_player ON lineup_member(player_id, actual_at);
CREATE INDEX IF NOT EXISTS idx_ranking_team_known ON ranking_snapshot(team_id, known_at);
CREATE INDEX IF NOT EXISTS idx_odds_known ON odds_snapshot(known_at);
CREATE INDEX IF NOT EXISTS idx_raw_record_event ON raw_ingest_record(source, stream, event_at);
