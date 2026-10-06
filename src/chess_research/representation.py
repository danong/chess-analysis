"""Elementary relationship predicates; no ratings or engine outcomes are read."""

from collections import Counter
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

import chess
import pyarrow as pa

from .analysis import replay
from .records import atomic_json, read_table, write_table

KINDS = list(range(1, 7))
RELATIONS = ["attack", "protect", "enemy_attack", "enemy_protect", "capture_own", "capture_enemy"]
PHASES = ["before", "after", "created", "removed", "retained"]


def feature_names() -> list[str]:
    elementary = [
        f"{phase}_{relation}_{a}_{b}"
        for phase in PHASES
        for relation in RELATIONS
        for a in KINDS
        for b in KINDS
    ]
    shared = [
        f"{phase}_same_target_{a}_{b}_{kind}"
        for phase in PHASES
        for a, b in combinations_with_replacement(RELATIONS, 2)
        for kind in KINDS
    ]
    return elementary + shared


def relationships(
    board: chess.Board, color: chess.Color, identities: dict[int, int]
) -> set[tuple[str, int, int]]:
    result = set()
    for square, piece in board.piece_map().items():
        for other in board.attacks(square):
            target = board.piece_at(other)
            if target is not None:
                relation = "protect" if target.color == piece.color else "attack"
                if piece.color != color:
                    relation = "enemy_" + relation
                result.add((relation, identities[square], identities[other]))
    for move in board.legal_moves:
        if board.is_capture(move):
            victim = move.to_square
            if board.is_en_passant(move):
                victim += -8 if board.turn else 8
            result.add(
                (
                    "capture_own" if board.turn == color else "capture_enemy",
                    identities[move.from_square],
                    identities[victim],
                )
            )
    return result


def feature_row(row: dict[str, Any]) -> dict[str, Any]:
    before = replay(row)
    identities = {s: s for s in before.piece_map()}
    kinds = {s: p.piece_type for s, p in before.piece_map().items()}
    old = relationships(before, before.turn, identities)
    move = chess.Move.from_uci(row["move"])
    after = before.copy()
    after.push(move)
    new_ids = identities.copy()
    new_ids[move.to_square] = identities[move.from_square]
    if before.is_castling(move):
        rank = chess.square_rank(move.from_square)
        rook_from = chess.square(7 if move.to_square > move.from_square else 0, rank)
        rook_to = chess.square(5 if move.to_square > move.from_square else 3, rank)
        new_ids[rook_to] = identities[rook_from]
    new = relationships(after, before.turn, new_ids)
    counts: Counter[str] = Counter()
    for phase, relations in [
        ("before", old),
        ("after", new),
        ("created", new - old),
        ("removed", old - new),
        ("retained", old & new),
    ]:
        by_target: dict[int, list[str]] = {}
        for relation, source, target in relations:
            by_target.setdefault(target, []).append(relation)
            counts[f"{phase}_{relation}_{kinds[source]}_{kinds[target]}"] += 1
        for target, labels in by_target.items():
            for a, b in combinations_with_replacement(RELATIONS, 2):
                if (a != b and a in labels and b in labels) or (a == b and labels.count(a) >= 2):
                    counts[f"{phase}_same_target_{a}_{b}_{kinds[target]}"] += 1
    result: dict[str, Any] = {"decision_id": row["decision_id"]}
    for name in feature_names():
        result[name] = counts[name]
    return result


def features(run: Path) -> None:
    rows = [feature_row(r) for r in read_table(run / "decisions.parquet")]
    names = feature_names()
    schema = pa.schema([("decision_id", pa.string())] + [(n, pa.int32()) for n in names])
    write_table(run / "features.parquet", rows, schema)
    atomic_json(
        run / "features.complete.json",
        {
            "rows": len(rows),
            "columns": names,
            "piece_types": {str(k): chess.piece_name(k) for k in KINDS},
            "identity": "original square, preserved across moves including castling and promotion",
        },
    )


def describe_rule(rule: list) -> str:
    """Mechanical prose drafted only after rule selection; no lesson labels."""
    descriptions = {}
    phases = {
        "before": "Before move",
        "after": "After move",
        "created": "Created",
        "removed": "Removed",
        "retained": "Retained",
    }
    for phase in PHASES:
        for relation in RELATIONS:
            actor = (
                "opponent"
                if relation.startswith("enemy_") or relation == "capture_enemy"
                else "mover"
            )
            verb = (
                "protection"
                if "protect" in relation
                else "legal capture"
                if "capture" in relation
                else "attack"
            )
            for a in KINDS:
                for b in KINDS:
                    descriptions[f"{phase}_{relation}_{a}_{b}"] = (
                        f"{phases[phase]}: {actor} {chess.piece_name(a)} {verb} of {chess.piece_name(b)} count"
                    )
        for relation_a, relation_b in combinations_with_replacement(RELATIONS, 2):
            for kind in KINDS:
                descriptions[f"{phase}_same_target_{relation_a}_{relation_b}_{kind}"] = (
                    f"{phases[phase]}: {chess.piece_name(kind)} pieces sharing {relation_a} and {relation_b} relationships count"
                )
    return " AND ".join(f"{descriptions[n]} {op} {t:g}" for n, op, t in rule)
