from __future__ import annotations

import csv
import json
import re
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

_DATE_RE = re.compile(r"\bas of\s+(\d{4})[-_](\d{2})[-_](\d{2})\b", re.IGNORECASE)
_REGION_RE = re.compile(r"standings_(global|europe|americas|asia)(?:_|\.md)")


@dataclass(frozen=True)
class ValveRankingRow:
    ranking_system: str
    region: str
    published_at: str
    rank: int
    points: float
    team_name: str
    roster: tuple[str, ...]
    source_commit: str
    source_path: str

    @property
    def roster_signature(self) -> str:
        return ",".join(sorted(_normalise_token(player) for player in self.roster))

    def to_record(self) -> dict[str, object]:
        result = asdict(self)
        result["roster"] = json.dumps(self.roster, ensure_ascii=False)
        result["roster_signature"] = self.roster_signature
        return result


def _normalise_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def parse_standings_markdown(
    content: str, *, source_commit: str, source_path: str
) -> list[ValveRankingRow]:
    date_match = _DATE_RE.search(content)
    if date_match is None:
        raise ValueError(f"standings document has no as-of date: {source_path}")
    published = date(*map(int, date_match.groups())).isoformat()
    region_match = _REGION_RE.search(Path(source_path).name)
    if region_match is None:
        raise ValueError(f"cannot infer standings region from {source_path}")
    region = region_match.group(1)
    system = "valve_global" if region == "global" else "valve_regional"

    rows: list[ValveRankingRow] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 4 or not cells[0].isdigit():
            continue
        try:
            points = float(cells[1].replace(",", ""))
        except ValueError:
            continue
        team_name = cells[2].strip()
        roster = tuple(
            player.strip() for player in cells[3].split(",") if player.strip()
        )
        if not team_name or not roster:
            continue
        rows.append(
            ValveRankingRow(
                ranking_system=system,
                region=region,
                published_at=published,
                rank=int(cells[0]),
                points=points,
                team_name=team_name,
                roster=roster,
                source_commit=source_commit,
                source_path=source_path,
            )
        )
    if not rows:
        raise ValueError(f"standings document has no ranking rows: {source_path}")
    return rows


def _git(repo: Path, *args: str, allow_missing: bool = False) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode and not allow_missing:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _historic_root_documents(repo: Path) -> Iterable[tuple[str, str, str]]:
    for region in ("europe", "americas", "asia"):
        source_path = f"standings_{region}.md"
        history = _git(
            repo,
            "log",
            "--reverse",
            "--format=%H",
            "--",
            source_path,
        )
        for commit in filter(None, history.splitlines()):
            content = _git(repo, "show", f"{commit}:{source_path}", allow_missing=True)
            if content:
                yield commit, source_path, content


def _tree_documents(repo: Path) -> Iterable[tuple[str, str, str]]:
    commit = _git(repo, "rev-parse", "HEAD").strip()
    names = _git(repo, "ls-tree", "-r", "--name-only", "HEAD")
    for source_path in names.splitlines():
        # `live` contains every published snapshot. `invitation` is a monthly
        # subset of the same material and would only duplicate observations.
        if not source_path.startswith("live/"):
            continue
        if _REGION_RE.search(Path(source_path).name) is None:
            continue
        yield commit, source_path, _git(repo, "show", f"HEAD:{source_path}")


def collect_valve_rankings(
    repo_path: str | Path, output_csv: str | Path
) -> dict[str, object]:
    """Export the official Valve standings history from Valve's git repository.

    Old snapshots lived as mutable root files, so their git history is the
    source of truth. Newer snapshots are immutable files under ``live/``.
    """

    repo = Path(repo_path).resolve()
    if not (repo / ".git").exists():
        raise FileNotFoundError(f"not a git repository: {repo}")
    output = Path(output_csv).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Formatting-only commits can repeat the same snapshot. Prefer the latest
    # encountered representation of each exact ranked roster.
    deduplicated: dict[tuple[object, ...], ValveRankingRow] = {}
    documents = 0
    parse_errors: list[dict[str, str]] = []
    for commit, source_path, content in (
        *_historic_root_documents(repo),
        *_tree_documents(repo),
    ):
        try:
            parsed = parse_standings_markdown(
                content, source_commit=commit, source_path=source_path
            )
        except ValueError as error:
            parse_errors.append({"source_path": source_path, "error": str(error)})
            continue
        documents += 1
        for row in parsed:
            key = (
                row.ranking_system,
                row.region,
                row.published_at,
                row.rank,
                _normalise_token(row.team_name),
                row.roster_signature,
            )
            deduplicated[key] = row

    rows = sorted(
        deduplicated.values(),
        key=lambda row: (
            row.published_at,
            row.ranking_system,
            row.region,
            row.rank,
            row.team_name.casefold(),
        ),
    )
    fields = [
        "ranking_system",
        "region",
        "published_at",
        "rank",
        "points",
        "team_name",
        "roster",
        "roster_signature",
        "source_commit",
        "source_path",
    ]
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row.to_record() for row in rows)
    temporary.replace(output)

    dates_by_system: dict[str, set[str]] = {}
    for row in rows:
        dates_by_system.setdefault(row.ranking_system, set()).add(row.published_at)
    return {
        "source_repo": str(repo),
        "source_head": _git(repo, "rev-parse", "HEAD").strip(),
        "output_csv": str(output),
        "documents": documents,
        "rows": len(rows),
        "snapshots": {
            system: len(dates) for system, dates in sorted(dates_by_system.items())
        },
        "min_date": min((row.published_at for row in rows), default=None),
        "max_date": max((row.published_at for row in rows), default=None),
        "parse_errors": parse_errors,
    }
