from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from urllib.parse import urljoin


PARSER_VERSION = "hltv-offline-html-v1"
SCHEMA_VERSION = "hltv-html-v1"

_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
_MATCH_ID = re.compile(r"/matches/(\d+)(?:/|$)")
_MAP_STATS_ID = re.compile(r"mapstatsid/(\d+)(?:/|$)")
_TEAM_ID = re.compile(r"/(?:team|stats/teams)/(\d+)(?:/|$)")
_PLAYER_ID = re.compile(r"/(?:player|stats/players)/(\d+)(?:/|$)")
_EVENT_ID = re.compile(r"/events/(\d+)(?:/|$)")


class HltvParseError(ValueError):
    """Raised when a local HTML snapshot cannot satisfy the parser contract."""


@dataclass
class HtmlNode:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[HtmlNode | str] = field(default_factory=list)
    parent: HtmlNode | None = None

    @property
    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())


class _DomParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = HtmlNode("document")
        self._stack = [self.root]

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        node = HtmlNode(
            tag=tag.casefold(),
            attrs={key.casefold(): value or "" for key, value in attrs},
            parent=self._stack[-1],
        )
        self._stack[-1].children.append(node)
        if node.tag not in _VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)
        if tag.casefold() not in _VOID_TAGS:
            self._stack.pop()

    def handle_endtag(self, tag: str) -> None:
        folded = tag.casefold()
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == folded:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self._stack[-1].children.append(data)


def _parse_dom(html: str) -> HtmlNode:
    parser = _DomParser()
    parser.feed(html)
    parser.close()
    return parser.root


def _walk(node: HtmlNode) -> Iterator[HtmlNode]:
    yield node
    for child in node.children:
        if isinstance(child, HtmlNode):
            yield from _walk(child)


def _find_all(
    node: HtmlNode,
    *,
    tag: str | None = None,
    class_name: str | None = None,
) -> list[HtmlNode]:
    return [
        item
        for item in _walk(node)
        if (tag is None or item.tag == tag)
        and (class_name is None or class_name in item.classes)
    ]


def _find_class_fragment(node: HtmlNode, fragment: str) -> list[HtmlNode]:
    return [
        item
        for item in _walk(node)
        if any(fragment in token for token in item.classes)
    ]


def _text(node: HtmlNode) -> str:
    chunks: list[str] = []

    def collect(item: HtmlNode) -> None:
        for child in item.children:
            if isinstance(child, str):
                chunks.append(child)
            elif child.tag == "br":
                chunks.append("\n")
            else:
                collect(child)

    collect(node)
    return re.sub(r"[ \t\r\f\v]+", " ", "".join(chunks)).strip()


def _first_text(node: HtmlNode, classes: Sequence[str]) -> str | None:
    for class_name in classes:
        matches = _find_all(node, class_name=class_name)
        if matches:
            value = _text(matches[0])
            if value:
                return value
    return None


def _link_id(href: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(href)
    return match.group(1) if match else None


def _entity_links(
    node: HtmlNode, pattern: re.Pattern[str]
) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    seen: set[str] = set()
    for anchor in _find_all(node, tag="a"):
        href = anchor.attrs.get("href", "")
        entity_id = _link_id(href, pattern)
        if entity_id is None or entity_id in seen:
            continue
        name = _text(anchor)
        if not name:
            name = anchor.attrs.get("title", "").strip()
        if not name:
            continue
        seen.add(entity_id)
        entities.append({"id": entity_id, "name": name, "href": href})
    return entities


def _canonical_url(root: HtmlNode) -> str | None:
    for link in _find_all(root, tag="link"):
        rel = set(link.attrs.get("rel", "").casefold().split())
        if "canonical" in rel and link.attrs.get("href"):
            return link.attrs["href"]
    return None


def _integer(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"-?\d+", value.replace("\u2212", "-"))
    return int(match.group()) if match else None


def _float(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", value.replace("\u2212", "-"))
    return float(match.group().replace(",", ".")) if match else None


def _number_pair(value: str | None) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    numbers = re.findall(r"\d+", value)
    if len(numbers) < 2:
        return _integer(value), None
    return int(numbers[0]), int(numbers[1])


def _ruleset(root: HtmlNode, game_version: str) -> tuple[str, str]:
    evidence_nodes: list[HtmlNode] = []
    for class_name in (
        "ruleset",
        "game-version",
        "cs-version",
        "match-info-note",
    ):
        evidence_nodes.extend(_find_all(root, class_name=class_name))
    evidence = " ".join(_text(node) for node in evidence_nodes)
    explicit = re.search(r"\bMR\s*(12|15)\b", evidence, re.IGNORECASE)
    if explicit:
        return f"MR{explicit.group(1)}", "explicit_page_label"
    if game_version == "CS2":
        return "MR12", "derived_from_explicit_game_version"
    if game_version == "CSGO":
        return "MR15", "derived_from_explicit_game_version"
    return "UNKNOWN", "insufficient_evidence"


def _round_counts(
    score_a: int | None,
    score_b: int | None,
    ruleset: str,
) -> dict[str, int | None]:
    if score_a is None or score_b is None:
        return {
            "completed_rounds": None,
            "regulation_rounds": None,
            "overtime_rounds": None,
        }
    completed = score_a + score_b
    regulation_limit = {"MR12": 24, "MR15": 30}.get(ruleset)
    if regulation_limit is None:
        return {
            "completed_rounds": completed,
            "regulation_rounds": None,
            "overtime_rounds": None,
        }
    return {
        "completed_rounds": completed,
        "regulation_rounds": min(completed, regulation_limit),
        "overtime_rounds": max(0, completed - regulation_limit),
    }


def _unix_to_iso(value: str | None) -> str | None:
    if value is None or not value.isdigit():
        return None
    timestamp = int(value)
    if timestamp > 10_000_000_000:
        timestamp /= 1000
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _document_time(root: HtmlNode) -> str | None:
    candidates: list[HtmlNode] = []
    for class_name in ("date", "match-time", "matchTime"):
        candidates.extend(_find_all(root, class_name=class_name))
    for node in candidates:
        parsed = _unix_to_iso(node.attrs.get("data-unix"))
        if parsed is not None:
            return parsed
    return None


def _content_hash(html: str) -> str:
    return hashlib.sha256(html.encode("utf-8")).hexdigest()


def _record(
    *,
    record_id: str,
    kind: str,
    entity_id: str,
    document_hash: str,
    event_at: str | None,
    payload: dict[str, object],
    warnings: Iterable[str] = (),
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "record_id": record_id,
        "kind": kind,
        "source_entity_id": entity_id,
        "source_document_sha256": document_hash,
        "event_at": event_at,
        "known_at": None,
        "payload": payload,
        "warnings": sorted(set(warnings)),
    }


def _revisioned_id(prefix: str, entity_id: str, document_hash: str) -> str:
    return f"hltv:{prefix}:{entity_id}:doc-{document_hash[:16]}"


def _match_teams(root: HtmlNode) -> list[dict[str, object]]:
    teams: list[dict[str, object]] = []
    for class_name in ("team1-gradient", "team2-gradient"):
        containers = _find_all(root, class_name=class_name)
        if containers:
            links = _entity_links(containers[0], _TEAM_ID)
            if links:
                teams.append(links[0])
    if len(teams) < 2:
        teams = _entity_links(root, _TEAM_ID)[:2]

    rankings = _find_all(root, class_name="teamRanking")
    for index, team in enumerate(teams):
        rank_text = _text(rankings[index]) if index < len(rankings) else None
        team["rank"] = (
            None
            if rank_text and "unranked" in rank_text.casefold()
            else _integer(rank_text)
        )
        team["rank_raw"] = rank_text
    return teams


def _lineup_players(node: HtmlNode) -> list[dict[str, object]]:
    players = [
        player
        for player in _entity_links(node, _PLAYER_ID)
        if "/player/" in player["href"]
    ]
    group: list[dict[str, object]] = []
    for player in players:
        anchor = next(
            (
                item
                for item in _find_all(node, tag="a")
                if _link_id(item.attrs.get("href", ""), _PLAYER_ID)
                == player["id"]
            ),
            None,
        )
        context = ""
        if anchor is not None:
            context_node = anchor.parent
            while context_node is not None and context_node.tag not in {
                "td",
                "tr",
                "table",
            }:
                context_node = context_node.parent
            if context_node is not None:
                context = _text(context_node).casefold()
        member_type = "starter"
        if "stand-in" in context or "standin" in context:
            member_type = "standin"
        elif "substitute" in context or "sub" in context.split():
            member_type = "substitute"
        group.append(
            {
                "player_id": player["id"],
                "nickname": player["name"],
                "member_type": member_type,
            }
        )
    return group


def _lineup_groups(
    root: HtmlNode,
) -> tuple[list[list[dict[str, object]]], list[str]]:
    warnings: list[str] = []

    # Historical layouts can place both teams in one table, one five-player
    # row per team. Prefer those explicit row boundaries over table position.
    for table in _find_all(root, tag="table"):
        row_groups = [
            players
            for row in _find_all(table, tag="tr")
            if len(players := _lineup_players(row)) == 5
        ]
        if len(row_groups) >= 2:
            return row_groups[:2], warnings

    groups: list[list[dict[str, object]]] = []
    # Newer layouts usually use one five-player table per team.
    for table in _find_all(root, tag="table"):
        players = _lineup_players(table)
        if len(players) == 5:
            groups.append(players)
        elif len(players) > 5:
            warnings.append(
                f"ambiguous_lineup_table_with_{len(players)}_unique_players"
            )
        if len(groups) == 2:
            break
    return groups, warnings


def _match_maps(
    root: HtmlNode,
    teams: list[dict[str, object]],
    match_id: str,
    game_version: str,
    ruleset: str,
    source_url: str | None,
) -> list[dict[str, object]]:
    maps: list[dict[str, object]] = []
    team_by_name = {
        str(team["name"]).casefold(): str(team["id"]) for team in teams
    }
    for order, holder in enumerate(
        _find_all(root, class_name="mapholder"), start=1
    ):
        map_name = _first_text(holder, ("mapname", "map-name")) or "UNKNOWN"
        score_nodes = _find_all(holder, class_name="results-team-score")
        scores = [_integer(_text(node)) for node in score_nodes[:2]]
        while len(scores) < 2:
            scores.append(None)
        card_team_nodes = _find_all(holder, class_name="results-teamname")
        card_team_names = [_text(node) for node in card_team_nodes[:2]]
        if len(card_team_names) < 2:
            card_team_names = [str(team["name"]) for team in teams[:2]]
        card_team_ids = [
            team_by_name.get(name.casefold()) for name in card_team_names
        ]
        while len(card_team_ids) < 2:
            card_team_ids.append(None)

        map_stats_id = None
        map_stats_url = None
        for anchor in _find_all(holder, tag="a"):
            href = anchor.attrs.get("href", "")
            map_stats_id = _link_id(href, _MAP_STATS_ID)
            if map_stats_id is not None:
                map_stats_url = urljoin(source_url, href) if source_url else href
                break
        entity_id = map_stats_id or f"{match_id}:position-{order}"

        picked_by = None
        for item in _walk(holder):
            if "pick" not in item.classes:
                continue
            if "results-left" in item.classes:
                picked_by = card_team_ids[0]
            elif "results-right" in item.classes:
                picked_by = card_team_ids[1]

        has_final_marker = any(
            {"won", "lost", "draw"} & item.classes for item in _walk(holder)
        )
        winner_team_id = None
        if has_final_marker and scores[0] is not None and scores[1] is not None:
            if scores[0] > scores[1]:
                winner_team_id = card_team_ids[0]
            elif scores[1] > scores[0]:
                winner_team_id = card_team_ids[1]
        if scores[0] is None or scores[1] is None:
            map_status = "unplayed"
        elif has_final_marker:
            map_status = "played"
        else:
            map_status = "live"
        breakdown_nodes = _find_class_fragment(holder, "half-score")
        breakdown_raw = [_text(node) for node in breakdown_nodes if _text(node)]
        half_scores: list[list[int]] = []
        for value in breakdown_raw:
            half_scores.extend(
                [int(left), int(right)]
                for left, right in re.findall(r"(\d+)\s*:\s*(\d+)", value)
            )
        round_counts = _round_counts(scores[0], scores[1], ruleset)
        maps.append(
            {
                "map_id": entity_id,
                "map_stats_id": map_stats_id,
                "map_stats_url": map_stats_url,
                "map_order": order,
                "map_name": map_name,
                "status": map_status,
                "team_ids": card_team_ids,
                "team_names": card_team_names,
                "scores": scores,
                "score_a": scores[0],
                "score_b": scores[1],
                "game_version": game_version,
                "ruleset": ruleset,
                **round_counts,
                "half_scores": half_scores,
                "overtime_segments": half_scores[2:],
                "round_breakdown_raw": breakdown_raw,
                "winner_team_id": winner_team_id,
                "picked_by_team_id": picked_by,
            }
        )
    return maps


def _veto(root: HtmlNode) -> dict[str, object] | None:
    candidates = _find_class_fragment(root, "veto")
    if not candidates:
        return None
    raw = _text(candidates[0])
    if not raw:
        return None
    actions: list[dict[str, object]] = []
    parts = re.split(r"(?=\b\d+\.\s*)", raw)
    for part in parts:
        match = re.match(
            r"\s*(\d+)\.\s*(.+?)\s+(removed|picked|left over)\s+(.+?)\s*(?:[.;]|$)",
            part,
            flags=re.IGNORECASE,
        )
        if match is None:
            decider = re.match(
                r"\s*(\d+)\.\s*(.+?)\s+was\s+left\s+over\s*(?:[.;]|$)",
                part,
                flags=re.IGNORECASE,
            )
            if decider is not None:
                actions.append(
                    {
                        "order": int(decider.group(1)),
                        "team_name": None,
                        "action": "decider",
                        "map_name": decider.group(2).strip(),
                        "raw_verb": "was left over",
                    }
                )
            continue
        verb = match.group(3).casefold()
        action = "ban" if verb == "removed" else "pick" if verb == "picked" else "decider"
        actions.append(
            {
                "order": int(match.group(1)),
                "team_name": match.group(2).strip(),
                "action": action,
                "map_name": match.group(4).strip(),
                "raw_verb": verb,
            }
        )
    return {"raw": raw, "actions": actions}


def _match_status(root: HtmlNode, maps: list[dict[str, object]]) -> str:
    status_nodes: list[HtmlNode] = []
    for class_name in ("match-info-note", "match-status", "countdown"):
        status_nodes.extend(_find_all(root, class_name=class_name))
    text = " ".join(_text(node) for node in status_nodes).casefold()
    if "walkover" in text or "walk-over" in text:
        return "walkover"
    if "forfeit" in text:
        return "forfeit"
    if "postponed" in text:
        return "postponed"
    if "cancelled" in text or "canceled" in text:
        return "cancelled"
    if re.search(r"\b(live|ongoing)\b", text) or any(
        item["status"] == "live" for item in maps
    ):
        return "live"
    if "finished" in text or "completed" in text:
        return "finished"
    if any(item["status"] == "played" for item in maps):
        return "finished"
    if "upcoming" in text or "countdown" in text:
        return "scheduled"
    return "unknown"


def _best_of(root: HtmlNode) -> int | None:
    candidates = []
    for class_name in ("preformatted-text", "match-info-note", "bestof"):
        candidates.extend(_text(node) for node in _find_all(root, class_name=class_name))
    candidates.append(_text(root))
    for value in candidates:
        match = re.search(r"(?:best\s+of|\bbo)\s*([1-7])\b", value, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _event(root: HtmlNode) -> dict[str, str] | None:
    links = _entity_links(root, _EVENT_ID)
    return links[0] if links else None


def _artifacts(root: HtmlNode) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    seen: set[str] = set()
    for anchor in _find_all(root, tag="a"):
        href = anchor.attrs.get("href", "")
        folded = href.casefold()
        kind = None
        if folded.endswith((".dem", ".dem.bz2", ".rar")) or "download/demo" in folded:
            kind = "demo"
        elif any(token in folded for token in ("youtube.com", "youtu.be", "twitch.tv")):
            kind = "vod"
        if kind is None or href in seen:
            continue
        seen.add(href)
        artifacts.append({"kind": kind, "url": href, "label": _text(anchor)})
    return artifacts


def _lan_online(root: HtmlNode) -> str | None:
    info = " ".join(
        filter(
            None,
            (
                _first_text(root, ("eventdesc", "match-info-note")),
                _first_text(root, ("location",)),
            ),
        )
    ).casefold()
    if re.search(r"\bonline\b", info):
        return "online"
    if re.search(r"\blan\b", info):
        return "lan"
    return None


def _parse_match(
    root: HtmlNode,
    *,
    document_hash: str,
    source_url: str | None,
) -> list[dict[str, object]]:
    warnings: list[str] = []
    locator = source_url or _canonical_url(root)
    match_id = _link_id(locator or "", _MATCH_ID)
    if match_id is None:
        match_id = f"unknown-{document_hash[:20]}"
        warnings.append("match_id_missing_fallback_to_document_hash")
    teams = _match_teams(root)
    if len(teams) != 2:
        warnings.append(f"expected_two_teams_found_{len(teams)}")
    event_at = _document_time(root)
    if event_at is None:
        warnings.append("event_at_missing")
    game_version = _game_version(root, locator)
    ruleset, ruleset_source = _ruleset(root, game_version)
    maps = _match_maps(
        root,
        teams,
        match_id,
        game_version,
        ruleset,
        locator,
    )
    status = _match_status(root, maps)
    lineup_groups, lineup_parse_warnings = _lineup_groups(root)
    warnings.extend(lineup_parse_warnings)
    if lineup_groups and len(lineup_groups) != 2:
        warnings.append(f"expected_two_lineups_found_{len(lineup_groups)}")
    if len(lineup_groups) == 2:
        first_players = {str(player["player_id"]) for player in lineup_groups[0]}
        second_players = {str(player["player_id"]) for player in lineup_groups[1]}
        overlap = sorted(first_players & second_players)
        if overlap:
            raise HltvParseError(
                "the same player IDs appear in both displayed lineups: "
                + ", ".join(overlap)
            )

    event = _event(root)
    team_ids = [str(team["id"]) for team in teams[:2]]
    series_scores = [0, 0]
    has_completed_map = False
    for map_payload in maps:
        if map_payload["status"] == "played":
            has_completed_map = True
        winner = map_payload["winner_team_id"]
        if winner in team_ids:
            series_scores[team_ids.index(str(winner))] += 1
    series_payload: dict[str, object] = {
        "match_id": match_id,
        "source_url": locator,
        "scheduled_at": event_at,
        "best_of": _best_of(root),
        "status": status,
        "lan_online": _lan_online(root),
        "game_version": game_version,
        "ruleset": ruleset,
        "ruleset_source": ruleset_source,
        "event": event,
        "teams": teams,
        "series_score_a": (
            series_scores[0] if len(team_ids) == 2 and has_completed_map else None
        ),
        "series_score_b": (
            series_scores[1] if len(team_ids) == 2 and has_completed_map else None
        ),
        "veto": _veto(root),
        "map_count": len(maps),
        "artifacts": _artifacts(root),
        "selector_trace": {
            "identity": "numeric_id_from_source_url_or_canonical_link",
            "teams": "team_container_then_numeric_team_links",
            "maps": "mapholder",
        },
    }
    records = [
        _record(
            record_id=_revisioned_id("series", match_id, document_hash),
            kind="series",
            entity_id=match_id,
            document_hash=document_hash,
            event_at=event_at,
            payload=series_payload,
            warnings=warnings,
        )
    ]

    for team in teams:
        rank = team.get("rank")
        if rank is None:
            continue
        team_id = str(team["id"])
        records.append(
            _record(
                record_id=_revisioned_id(
                    "ranking", f"{match_id}:{team_id}", document_hash
                ),
                kind="ranking",
                entity_id=team_id,
                document_hash=document_hash,
                event_at=event_at,
                payload={
                    "match_id": match_id,
                    "team_id": team_id,
                    "team_name": team["name"],
                    "rank": rank,
                    "rank_raw": team.get("rank_raw"),
                    "ranking_system": "hltv-match-page-rank",
                },
                warnings=("historical_known_at_unverified",),
            )
        )

    for index, lineup in enumerate(lineup_groups[:2]):
        if index >= len(teams):
            break
        team_id = str(teams[index]["id"])
        lineup_warnings = []
        if len(lineup) != 5:
            lineup_warnings.append(f"expected_five_players_found_{len(lineup)}")
        records.append(
            _record(
                record_id=_revisioned_id(
                    "lineup", f"{match_id}:{team_id}", document_hash
                ),
                kind="lineup",
                entity_id=f"{match_id}:{team_id}",
                document_hash=document_hash,
                event_at=event_at,
                payload={
                    "match_id": match_id,
                    "team_id": team_id,
                    "players": lineup,
                    "lineup_scope": "match_page_displayed",
                },
                warnings=lineup_warnings,
            )
        )

    for map_payload in maps:
        map_id = str(map_payload["map_id"])
        records.append(
            _record(
                record_id=_revisioned_id("map", map_id, document_hash),
                kind="map",
                entity_id=map_id,
                document_hash=document_hash,
                event_at=event_at,
                payload={"match_id": match_id, **map_payload},
                warnings=(
                    ("unplayed_map_not_training_eligible",)
                    if map_payload["status"] == "unplayed"
                    else ("live_map_not_training_eligible",)
                    if map_payload["status"] == "live"
                    else ()
                ),
            )
        )
    return records


def _cell_by_class(cells: list[HtmlNode], names: Sequence[str]) -> HtmlNode | None:
    for cell in cells:
        classes = cell.classes
        if any(name in classes for name in names):
            return cell
    return None


def _cell_text(cells: list[HtmlNode], names: Sequence[str]) -> str | None:
    cell = _cell_by_class(cells, names)
    return _text(cell) if cell is not None else None


def _header_key(value: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "", value.casefold())
    if normalized in {"player", "name"}:
        return "player"
    if normalized.startswith("op") and "kd" in normalized:
        return "opening"
    if normalized in {"kd", "killsdeaths"}:
        return "kills_deaths"
    if normalized.startswith("k") and "hs" in normalized:
        return "kills_headshots"
    if normalized.startswith("a") and (
        "flash" in normalized or normalized in {"af", "assistsf"}
    ):
        return "assists_flash"
    if normalized.startswith("d") and (
        "trade" in normalized or normalized in {"dt", "deathst"}
    ):
        return "deaths_traded"
    if normalized in {"k", "kills"}:
        return "kills"
    if normalized in {"a", "assists"}:
        return "assists"
    if normalized in {"d", "deaths"}:
        return "deaths"
    if normalized in {"hs", "headshots"}:
        return "headshots"
    if normalized == "adr":
        return "adr"
    if normalized == "kast" or normalized.startswith("kast"):
        return "kast"
    if normalized == "kpr":
        return "kpr"
    if normalized == "dpr":
        return "dpr"
    if normalized.startswith("swing"):
        return "swing"
    if normalized.startswith("rating"):
        return "rating"
    if normalized in {"mks", "multikills", "multikillrounds"}:
        return "multi_kill_rounds"
    if normalized.startswith("1vs") or normalized.startswith("clutch"):
        return "clutch_wins"
    return None


def _table_headers(table: HtmlNode) -> dict[str, int]:
    for row in _find_all(table, tag="tr"):
        headers = [
            child
            for child in row.children
            if isinstance(child, HtmlNode) and child.tag == "th"
        ]
        if not headers:
            continue
        mapped: dict[str, int] = {}
        for index, header in enumerate(headers):
            key = _header_key(_text(header))
            if key is not None:
                mapped[key] = index
        return mapped
    return {}


def _header_text(
    cells: list[HtmlNode], headers: dict[str, int], key: str
) -> str | None:
    index = headers.get(key)
    if index is None or index >= len(cells):
        return None
    return _text(cells[index])


def _metric_text(
    cells: list[HtmlNode],
    headers: dict[str, int],
    classes: Sequence[str],
    *header_keys: str,
) -> str | None:
    class_value = _cell_text(cells, classes)
    if class_value is not None:
        return class_value
    for key in header_keys:
        value = _header_text(cells, headers, key)
        if value is not None:
            return value
    return None


def _rating_version(table: HtmlNode, root: HtmlNode, has_swing: bool) -> str:
    for value in (_text(table), _text(root)):
        match = re.search(r"Rating\s*([123](?:\.\d)?)", value, re.IGNORECASE)
        if match:
            return f"hltv-rating-{match.group(1)}"
    return "hltv-rating-3.x" if has_swing else "hltv-rating-unknown"


def _game_version(root: HtmlNode, source_url: str | None) -> str:
    version_nodes: list[HtmlNode] = []
    for class_name in ("cs-version", "game-version", "match-info-box"):
        version_nodes.extend(_find_all(root, class_name=class_name))
    evidence = " ".join(
        [source_url or "", *(_text(node) for node in version_nodes)]
    ).casefold()
    if "csversion=cs2" in evidence or "counter-strike 2" in evidence:
        return "CS2"
    if "csversion=csgo" in evidence or "counter-strike: global offensive" in evidence:
        return "CSGO"
    return "UNKNOWN"


def _stats_tables(root: HtmlNode) -> list[HtmlNode]:
    tables: list[HtmlNode] = []
    for table in _find_all(root, tag="table"):
        if not ({"totalstats", "stats-table"} & table.classes):
            continue
        if _entity_links(table, _PLAYER_ID):
            tables.append(table)
    return tables


def _map_stats_teams(root: HtmlNode) -> tuple[list[dict[str, str]], str]:
    for class_name in ("match-info-box", "match-info"):
        for container in _find_all(root, class_name=class_name):
            teams = _entity_links(container, _TEAM_ID)
            if len(teams) >= 2:
                return teams[:2], f"scoped_{class_name}"
    return _entity_links(root, _TEAM_ID)[:2], "page_wide_fallback"


def _map_stats_score(root: HtmlNode) -> tuple[int | None, int | None, str]:
    containers: list[HtmlNode] = []
    for class_name in ("match-info-box", "match-info"):
        containers.extend(_find_all(root, class_name=class_name))
    for container in containers:
        score_nodes = _find_all(container, class_name="team-score")
        score_nodes.extend(_find_all(container, class_name="score"))
        values = [_integer(_text(node)) for node in score_nodes]
        numeric = [value for value in values if value is not None]
        if len(numeric) >= 2:
            return numeric[0], numeric[1], "scoped_score_nodes"
        score_containers = _find_all(container, class_name="match-info-score")
        score_texts = [_text(node) for node in score_containers] or [_text(container)]
        for score_text in score_texts:
            combined = re.search(r"\b(\d+)\s*:\s*(\d+)\b", score_text)
            if combined:
                return (
                    int(combined.group(1)),
                    int(combined.group(2)),
                    "scoped_score_text",
                )
    return None, None, "score_missing"


def _parse_player_row(
    row: HtmlNode,
    *,
    team_id: str | None,
    map_stats_id: str,
    metric_version: str,
    game_version: str,
    headers: dict[str, int],
) -> dict[str, object] | None:
    players = _entity_links(row, _PLAYER_ID)
    if not players:
        return None
    player = players[0]
    cells = [child for child in row.children if isinstance(child, HtmlNode) and child.tag == "td"]
    kills_raw = _metric_text(
        cells, headers, ("st-kills", "kills"), "kills", "kills_headshots"
    )
    assists_raw = _metric_text(
        cells,
        headers,
        ("st-assists", "assists"),
        "assists",
        "assists_flash",
    )
    deaths_raw = _metric_text(
        cells,
        headers,
        ("st-deaths", "deaths"),
        "deaths",
        "deaths_traded",
    )
    kills_deaths_raw = _header_text(cells, headers, "kills_deaths")
    if kills_deaths_raw is not None:
        paired_kills, paired_deaths = _number_pair(kills_deaths_raw)
        if kills_raw is None:
            kills_raw = str(paired_kills) if paired_kills is not None else None
        if deaths_raw is None:
            deaths_raw = str(paired_deaths) if paired_deaths is not None else None
    opening_raw = _metric_text(
        cells, headers, ("st-opening", "opening"), "opening"
    )
    opening_kills, opening_deaths = _number_pair(opening_raw)
    flash_assists = None
    if assists_raw:
        parentheses = re.search(r"\((\d+)\)", assists_raw)
        flash_assists = int(parentheses.group(1)) if parentheses else None
    headshots = _integer(
        _metric_text(
            cells,
            headers,
            ("st-headshots", "headshots", "st-hs"),
            "headshots",
        )
    )
    if headshots is None and kills_raw and headers.get("kills_headshots") is not None:
        parentheses = re.search(r"\((\d+)\)", kills_raw)
        headshots = int(parentheses.group(1)) if parentheses else None
    traded_deaths = _integer(
        _metric_text(
            cells,
            headers,
            ("st-traded-deaths", "traded-deaths"),
        )
    )
    if (
        traded_deaths is None
        and deaths_raw
        and headers.get("deaths_traded") is not None
    ):
        parentheses = re.search(r"\((\d+)\)", deaths_raw)
        traded_deaths = int(parentheses.group(1)) if parentheses else None

    raw_metrics = {
        " ".join(sorted(cell.classes)) or f"column-{index}": _text(cell)
        for index, cell in enumerate(cells)
    }
    return {
        "map_stats_id": map_stats_id,
        "team_id": team_id,
        "player_id": player["id"],
        "nickname": player["name"],
        "side": "BOTH",
        "game_version": game_version,
        "metric_version": metric_version,
        "kills": _integer(kills_raw),
        "deaths": _integer(deaths_raw),
        "assists": _integer(assists_raw),
        "flash_assists": flash_assists,
        "headshots": headshots,
        "traded_deaths": traded_deaths,
        "opening_kills": opening_kills
        if opening_raw is not None
        else _integer(_cell_text(cells, ("st-opening-kills", "opening-kills"))),
        "opening_deaths": opening_deaths
        if opening_raw is not None
        else _integer(_cell_text(cells, ("st-opening-deaths", "opening-deaths"))),
        "adr": _float(_metric_text(cells, headers, ("st-adr", "adr"), "adr")),
        "kast": _float(
            _metric_text(cells, headers, ("st-kast", "kast"), "kast")
        ),
        "kpr": _float(_metric_text(cells, headers, ("st-kpr", "kpr"), "kpr")),
        "dpr": _float(_metric_text(cells, headers, ("st-dpr", "dpr"), "dpr")),
        "swing": _float(
            _metric_text(cells, headers, ("st-swing", "swing"), "swing")
        ),
        "rating": _float(
            _metric_text(
                cells,
                headers,
                ("st-rating", "rating", "st-rating2"),
                "rating",
            )
        ),
        "multi_kill_rounds": _integer(
            _header_text(cells, headers, "multi_kill_rounds")
        ),
        "clutch_wins": _integer(_header_text(cells, headers, "clutch_wins")),
        "raw_metrics": raw_metrics,
        "selector_trace": "semantic_cell_class_then_normalized_header",
    }


def _parse_map_stats(
    root: HtmlNode,
    *,
    document_hash: str,
    source_url: str | None,
) -> list[dict[str, object]]:
    warnings: list[str] = []
    locator = source_url or _canonical_url(root)
    map_stats_id = _link_id(locator or "", _MAP_STATS_ID)
    if map_stats_id is None:
        map_stats_id = f"unknown-{document_hash[:20]}"
        warnings.append("map_stats_id_missing_fallback_to_document_hash")
    event_at = _document_time(root)
    teams, team_selector = _map_stats_teams(root)
    if len(teams) != 2:
        warnings.append(f"expected_two_teams_found_{len(teams)}")
    map_name = _first_text(root, ("mapname", "map-name"))
    game_version = _game_version(root, locator)
    if game_version == "UNKNOWN":
        warnings.append("game_version_unknown")
    ruleset, ruleset_source = _ruleset(root, game_version)
    score_a, score_b, score_source = _map_stats_score(root)
    round_counts = _round_counts(score_a, score_b, ruleset)

    records: list[dict[str, object]] = []
    tables = _stats_tables(root)
    if not tables:
        warnings.append("player_stats_tables_missing")
    for table_index, table in enumerate(tables[:2]):
        table_teams = _entity_links(table, _TEAM_ID)
        assignment_warning = None
        if len(table_teams) == 1:
            team_id = table_teams[0]["id"]
        else:
            team_id = teams[table_index]["id"] if table_index < len(teams) else None
            assignment_warning = "team_assignment_by_table_position"
        has_swing = bool(_find_all(table, class_name="st-swing")) or "Swing" in _text(table)
        metric_version = _rating_version(table, root, has_swing)
        headers = _table_headers(table)
        for row in _find_all(table, tag="tr"):
            player_stats = _parse_player_row(
                row,
                team_id=str(team_id) if team_id is not None else None,
                map_stats_id=map_stats_id,
                metric_version=metric_version,
                game_version=game_version,
                headers=headers,
            )
            if player_stats is None:
                continue
            player_id = str(player_stats["player_id"])
            entity_id = f"{map_stats_id}:{player_id}:BOTH:{metric_version}"
            player_warnings = list(warnings)
            if assignment_warning is not None:
                player_warnings.append(assignment_warning)
            if player_stats["swing"] is None and metric_version.startswith(
                "hltv-rating-3"
            ):
                player_warnings.append("rating_3_without_swing_value")
            records.append(
                _record(
                    record_id=_revisioned_id(
                        "player-map", entity_id, document_hash
                    ),
                    kind="player_map_stats",
                    entity_id=entity_id,
                    document_hash=document_hash,
                    event_at=event_at,
                    payload={
                        "source_url": locator,
                        "map_name": map_name,
                        "score_a": score_a,
                        "score_b": score_b,
                        "score_source": score_source,
                        "ruleset": ruleset,
                        "ruleset_source": ruleset_source,
                        **round_counts,
                        "team_selector_trace": team_selector,
                        **player_stats,
                    },
                    warnings=player_warnings,
                )
            )
    if records:
        return records
    return [
        _record(
            record_id=_revisioned_id("map-stats-document", map_stats_id, document_hash),
            kind="map_stats_document",
            entity_id=map_stats_id,
            document_hash=document_hash,
            event_at=event_at,
            payload={
                "source_url": locator,
                "map_name": map_name,
                "game_version": game_version,
                "score_a": score_a,
                "score_b": score_b,
                "score_source": score_source,
                "ruleset": ruleset,
                "ruleset_source": ruleset_source,
                **round_counts,
                "teams": teams,
            },
            warnings=warnings,
        )
    ]


def detect_page_type(root: HtmlNode, source_url: str | None = None) -> str:
    locator = source_url or _canonical_url(root) or ""
    if _MAP_STATS_ID.search(locator):
        return "map-stats"
    if _MATCH_ID.search(locator):
        return "match"
    if _stats_tables(root):
        return "map-stats"
    if _find_all(root, class_name="mapholder"):
        return "match"
    raise HltvParseError("cannot detect whether the local HTML is a match or map-stats page")


def _validate_source_identity(
    root: HtmlNode, source_url: str | None, page_type: str
) -> None:
    if source_url is None:
        return
    canonical = _canonical_url(root)
    if canonical is None:
        return
    pattern = _MATCH_ID if page_type == "match" else _MAP_STATS_ID
    source_id = _link_id(source_url, pattern)
    canonical_id = _link_id(canonical, pattern)
    if source_id is not None and canonical_id is not None and source_id != canonical_id:
        raise HltvParseError(
            f"source URL entity {source_id} disagrees with canonical entity {canonical_id}"
        )


def parse_html(
    html: str,
    *,
    page_type: str = "auto",
    source_url: str | None = None,
    observed_at: str | None = None,
) -> list[dict[str, object]]:
    """Parse an already captured HLTV HTML document without network access."""

    if page_type not in {"auto", "match", "map-stats"}:
        raise ValueError("page_type must be auto, match, or map-stats")
    root = _parse_dom(html)
    resolved_type = detect_page_type(root, source_url) if page_type == "auto" else page_type
    _validate_source_identity(root, source_url, resolved_type)
    document_hash = _content_hash(html)
    if resolved_type == "match":
        records = _parse_match(
            root, document_hash=document_hash, source_url=source_url
        )
    else:
        records = _parse_map_stats(
            root, document_hash=document_hash, source_url=source_url
        )
    for record in records:
        record["observed_at"] = observed_at
    return records


def parse_file(
    html_path: str | Path,
    *,
    page_type: str = "auto",
    source_url: str | None = None,
    observed_at: str | None = None,
) -> list[dict[str, object]]:
    path = Path(html_path)
    # Read bytes first so CRLF and other source-level details remain part of the
    # immutable document hash instead of being normalized by text I/O.
    html = path.read_bytes().decode("utf-8")
    return parse_html(
        html,
        page_type=page_type,
        source_url=source_url,
        observed_at=observed_at,
    )


def records_to_jsonl(records: Iterable[dict[str, object]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
        for record in records
    )
