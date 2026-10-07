"""Shared moving-player splits with position and historical holdout exclusions."""

from typing import Any

from .ingestion import partition


def split_rows(
    rows: list[dict[str, Any]],
    seed: int,
    excluded_players: set[str] | None = None,
    excluded_positions: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for part in ["discovery", "selection", "evaluation"]:
        candidates = [r for r in rows if partition(r["player"], seed) == part]
        groups[part] = [
            r
            for r in candidates
            if r["position"] not in seen
            and (
                part != "evaluation"
                or (
                    r["player"] not in (excluded_players or set())
                    and r["position"] not in (excluded_positions or set())
                )
            )
        ]
        if part == "evaluation":
            unique = []
            evaluation_positions: set[str] = set()
            for row in groups[part]:
                if row["position"] not in evaluation_positions:
                    unique.append(row)
                    evaluation_positions.add(row["position"])
            groups[part] = unique
        seen.update(r["position"] for r in candidates)
    return groups
