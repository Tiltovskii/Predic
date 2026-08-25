from __future__ import annotations

import sqlite3


def audit(connection: sqlite3.Connection) -> dict[str, object]:
    tables = (
        "source_snapshot",
        "team_core",
        "player",
        "series",
        "map_game",
        "lineup_member",
        "player_map_stats",
        "ranking_snapshot",
        "odds_snapshot",
        "raw_ingest_record",
    )
    counts = {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in tables
    }
    future_known = connection.execute(
        """
        SELECT COUNT(*)
        FROM map_game
        WHERE known_at IS NOT NULL
          AND started_at IS NOT NULL
          AND known_at > started_at
        """
    ).fetchone()[0]
    incomplete_lineups = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT map_id, team_id, COUNT(*) AS players
            FROM lineup_member
            GROUP BY map_id, team_id
            HAVING players <> 5
        )
        """
    ).fetchone()[0]
    duplicate_players = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT map_id, team_id, player_id, COUNT(*) AS appearances
            FROM lineup_member
            GROUP BY map_id, team_id, player_id
            HAVING appearances > 1
        )
        """
    ).fetchone()[0]
    invalid_scores = connection.execute(
        """
        SELECT COUNT(*)
        FROM map_game
        WHERE score_a IS NOT NULL AND score_b IS NOT NULL
          AND (score_a < 0 OR score_b < 0 OR score_a = score_b)
        """
    ).fetchone()[0]
    coverage_row = connection.execute(
        """
        SELECT MIN(started_at) AS first_date,
               MAX(started_at) AS last_date,
               COUNT(DISTINCT started_at) AS active_dates
        FROM map_game
        """
    ).fetchone()
    source_quality = [
        dict(row)
        for row in connection.execute(
            """
            SELECT s.source,
                   s.point_in_time_eligible,
                   COUNT(DISTINCT m.map_id) AS maps
            FROM source_snapshot AS s
            LEFT JOIN map_game AS m ON m.source_snapshot_id = s.snapshot_id
            GROUP BY s.source, s.point_in_time_eligible
            ORDER BY s.source, s.point_in_time_eligible
            """
        )
    ]
    yearly_maps = {
        row["year"]: row["maps"]
        for row in connection.execute(
            """
            SELECT substr(started_at, 1, 4) AS year, COUNT(*) AS maps
            FROM map_game
            GROUP BY substr(started_at, 1, 4)
            ORDER BY year
            """
        )
        if row["year"] is not None
    }
    return {
        "counts": counts,
        "coverage": {
            "first_date": coverage_row["first_date"],
            "last_date": coverage_row["last_date"],
            "active_dates": coverage_row["active_dates"],
        },
        "source_quality": source_quality,
        "yearly_maps": yearly_maps,
        "checks": {
            "map_known_after_start": future_known,
            "lineups_not_exactly_five": incomplete_lineups,
            "duplicate_players_within_lineup": duplicate_players,
            "invalid_or_tied_finished_scores": invalid_scores,
        },
        "ok": (
            future_known == 0
            and incomplete_lineups == 0
            and duplicate_players == 0
            and invalid_scores == 0
        ),
    }
