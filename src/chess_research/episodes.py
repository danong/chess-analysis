"""Deterministic played-versus-preferred engine trajectory episodes."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import chess
import pyarrow as pa

from .analysis import replay
from .records import atomic_json, read_table, write_table
from .splits import split_rows

PIECES = range(chess.PAWN, chess.KING + 1)
OUTCOMES = ("mover_win", "draw", "mover_loss")


def feature_names(horizon: int = 6) -> list[str]:
    """Return the explicit, location-free trajectory encoding allowlist."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    names: list[str] = []
    for trajectory in ("played", "preferred"):
        for step in range(1, horizon + 1):
            prefix = f"{trajectory}_step_{step}"
            for actor in ("mover", "opponent"):
                for piece in PIECES:
                    names.append(f"{prefix}_material_{actor}_{chess.piece_name(piece)}_delta")
                names.append(f"{prefix}_checks_{actor}")
                for piece in PIECES:
                    names.append(f"{prefix}_captures_{actor}_{chess.piece_name(piece)}")
                for piece in PIECES:
                    names.append(f"{prefix}_promotions_{actor}_to_{chess.piece_name(piece)}")
            for outcome in OUTCOMES:
                names.append(f"{prefix}_terminal_{outcome}")
    return names + [
        f"difference_{name.removeprefix('played_')}" for name in names[: len(names) // 2]
    ]


def _piece_counts(board: chess.Board, color: chess.Color) -> Counter[int]:
    return Counter(piece.piece_type for piece in board.piece_map().values() if piece.color == color)


def _validate_pv(board: chess.Board, pv: list[str], label: str) -> list[chess.Move]:
    moves: list[chess.Move] = []
    position = board.copy()
    for token in pv:
        try:
            move = chess.Move.from_uci(token)
        except ValueError as exc:
            raise ValueError(f"Invalid {label} move: {token}") from exc
        if move not in position.legal_moves:
            raise ValueError(f"Illegal {label} move: {token}")
        moves.append(move)
        position.push(move)
    return moves


def _terminal_code(board: chess.Board, mover: chess.Color) -> str | None:
    if not board.is_game_over():
        return None
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return "draw"
    return "mover_win" if outcome.winner == mover else "mover_loss"


def _trajectory(
    start: chess.Board, moves: list[chess.Move], mover: chess.Color, horizon: int
) -> tuple[list[float], bool]:
    """Encode legal state changes; terminal positions are held through the horizon."""
    counts: list[int] = []
    current = start.copy()
    terminal_seen = False
    allowed = feature_names(horizon)
    for step in range(1, horizon + 1):
        prefix = f"step_{step}"
        before_counts = {color: _piece_counts(current, color) for color in chess.COLORS}
        actor = current.turn
        changes: Counter[str] = Counter()
        if step <= len(moves) and not current.is_game_over():
            move = moves[step - 1]
            if move not in current.legal_moves:
                raise ValueError("Illegal trajectory continuation")
            captured = current.piece_at(move.to_square)
            if current.is_en_passant(move):
                captured = chess.Piece(chess.PAWN, not actor)
            current.push(move)
            side_name = "mover" if actor == mover else "opponent"
            if captured is not None:
                changes[
                    f"{prefix}_captures_{side_name}_{chess.piece_name(captured.piece_type)}"
                ] += 1
            if current.is_check():
                changes[f"{prefix}_checks_{side_name}"] += 1
            if move.promotion is not None:
                changes[
                    f"{prefix}_promotions_{side_name}_to_{chess.piece_name(move.promotion)}"
                ] += 1
        after_counts = {color: _piece_counts(current, color) for color in chess.COLORS}
        for color in chess.COLORS:
            side_name = "mover" if color == mover else "opponent"
            for piece in PIECES:
                delta = after_counts[color][piece] - before_counts[color][piece]
                if delta:
                    changes[f"{prefix}_material_{side_name}_{chess.piece_name(piece)}_delta"] += (
                        delta
                    )
        terminal = _terminal_code(current, mover)
        if terminal is not None and not terminal_seen:
            changes[f"{prefix}_terminal_{terminal}"] += 1
            terminal_seen = True
        # Feature order is shared with feature_names after the trajectory prefix.
        step_names = [
            name.removeprefix("played_")
            for name in allowed
            if name.startswith("played_step_" + str(step) + "_")
        ]
        counts.extend(changes.get(name, 0) for name in step_names)
    complete = len(moves) >= horizon or terminal_seen
    return [float(value) for value in counts], complete


def encode_episode(row: dict[str, Any], horizon: int = 6) -> dict[str, Any]:
    """Encode a joined decision/analysis row; incomplete continuations have no vector."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    board = replay(row)
    mover = board.turn
    preferred_pv = row.get("before_pv")
    after_pv = row.get("after_pv")
    if preferred_pv is None:
        raise ValueError("Missing before_pv")
    if after_pv is None:
        raise ValueError("Missing after_pv")
    preferred_moves = _validate_pv(board, preferred_pv, "preferred PV")
    if not preferred_moves:
        raise ValueError("Empty before_pv")
    played = chess.Move.from_uci(row["move"])
    if played not in board.legal_moves:
        raise ValueError("Illegal played move")
    after = board.copy()
    after.push(played)
    played_moves = _validate_pv(after, after_pv, "after PV")
    if not after_pv and not after.is_game_over():
        raise ValueError("Empty after_pv before a legal terminal position")
    # The played path includes the observed human move as its first ply.
    played_moves = [played, *played_moves]
    preferred_vector, preferred_available = _trajectory(board, preferred_moves, mover, horizon)
    played_vector, played_available = _trajectory(board, played_moves, mover, horizon)
    available = preferred_available and played_available
    if not available:
        return {"vector": [], "available": False, "informative": False}
    vector = (
        played_vector + preferred_vector + [a - b for a, b in zip(played_vector, preferred_vector)]
    )
    if len(vector) != len(feature_names(horizon)):
        raise AssertionError("Episode vector does not match feature allowlist")
    return {"vector": vector, "available": True, "informative": any(value != 0 for value in vector)}


EPISODES = pa.schema(
    [
        ("episode_id", pa.string()),
        ("decision_id", pa.string()),
        ("game_id", pa.string()),
        ("player", pa.string()),
        ("rating", pa.int32()),
        ("ply", pa.int32()),
        ("side", pa.string()),
        ("position", pa.string()),
        ("initial_fen", pa.string()),
        ("history", pa.list_(pa.string())),
        ("before_fen", pa.string()),
        ("fen", pa.string()),
        ("move", pa.string()),
        ("preferred_move", pa.string()),
        ("before_cp", pa.int32()),
        ("before_mate", pa.int32()),
        ("after_cp", pa.int32()),
        ("after_mate", pa.int32()),
        ("before_expected", pa.float64()),
        ("after_expected", pa.float64()),
        ("loss", pa.float64()),
        ("before_pv", pa.list_(pa.string())),
        ("after_pv", pa.list_(pa.string())),
        ("analysis_config_sha256", pa.string()),
        ("analysis_config_json", pa.string()),
        ("partition", pa.string()),
        ("vector", pa.list_(pa.float64())),
        ("available", pa.bool_()),
        ("informative", pa.bool_()),
    ]
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_episodes(run: Path, loss_threshold: float = 0.10, horizon: int = 6) -> Path:
    """Build immutable thresholded episodes plus full-sample split audit artifacts."""
    if not math.isfinite(loss_threshold) or loss_threshold < 0:
        raise ValueError("loss_threshold must be finite and non-negative")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if (run / "taxonomy.json").exists() or (run / "active-clusters.json").exists():
        raise ValueError("Cannot rebuild episodes after taxonomy or clusters exist")
    decisions_path = run / "decisions.parquet"
    active = json.loads((run / "active-analysis.json").read_text())
    analysis_dir = run / active["path"]
    analyses = [
        r for path in sorted(analysis_dir.glob("batch-*.parquet")) for r in read_table(path)
    ]
    analysis_by_id = {row["decision_id"]: row for row in analyses}
    if len(analysis_by_id) != len(analyses):
        raise ValueError("Duplicate analysis decision IDs")
    decisions = read_table(decisions_path)
    decision_ids = {r["decision_id"] for r in decisions}
    if len(decision_ids) != len(decisions) or set(analysis_by_id) != decision_ids:
        raise ValueError("Complete one-to-one analysis coverage required")
    analysis_config_path = analysis_dir / "config.json"
    analysis_config = json.loads(analysis_config_path.read_text())
    exclusions_path = run / "historical-exclusions.json"
    exclusions = json.loads(exclusions_path.read_text()) if exclusions_path.exists() else {}
    run_config = json.loads((run / "config.json").read_text())
    partitions = split_rows(
        decisions,
        run_config["seed"],
        set(exclusions.get("players", [])),
        set(exclusions.get("positions", [])),
    )
    partition_by_id = {r["decision_id"]: part for part, rows in partitions.items() for r in rows}
    source_hashes = {
        "decisions": _sha256(decisions_path),
        "analysis_config": _sha256(analysis_config_path),
        "analysis_batches": {
            path.name: _sha256(path) for path in sorted(analysis_dir.glob("batch-*.parquet"))
        },
        "historical_exclusions": _sha256(exclusions_path) if exclusions_path.exists() else None,
        "run_config": _sha256(run / "config.json"),
    }
    config = {
        "loss_threshold": loss_threshold,
        "horizon": horizon,
        "seed": run_config["seed"],
        "feature_names": feature_names(horizon),
        "source_hashes": source_hashes,
        "analysis_config": analysis_config,
        "analysis_config_sha256": _sha256(analysis_config_path),
        "score_perspective": "moving player",
        "partition_method": "splits.split_rows; discovery, selection, evaluation by moving-player identity",
        "encoding": "per-ply material count deltas, captures by victim type, checks, promotions, terminal result; played, preferred, and played-minus-preferred",
    }
    config_path = run / "episodes.config.json"
    complete_path = run / "episodes.complete.json"
    if complete_path.exists():
        old_config = json.loads(config_path.read_text()) if config_path.exists() else None
        complete = json.loads(complete_path.read_text())
        artifact = run / "episodes.parquet"
        if (
            old_config == config
            and artifact.exists()
            and complete.get("episodes_sha256") == _sha256(artifact)
        ):
            return run / "episodes.parquet"
        raise ValueError("Episode artifacts already exist with different inputs or settings")

    records: list[dict[str, Any]] = []
    for decision in decisions:
        analysis = analysis_by_id[decision["decision_id"]]
        if not math.isfinite(analysis["loss"]):
            raise ValueError(f"Non-finite loss for {decision['decision_id']}")
        if analysis["loss"] < loss_threshold:
            continue
        joined = decision | analysis
        encoded = encode_episode(joined, horizon)
        board = replay(decision)
        if not analysis.get("before_pv"):
            raise ValueError(f"Missing before_pv for {decision['decision_id']}")
        records.append(
            {
                "episode_id": decision["decision_id"],
                "decision_id": decision["decision_id"],
                "game_id": decision["game_id"],
                "player": decision["player"],
                "rating": decision["rating"],
                "ply": decision["ply"],
                "side": "White" if board.turn else "Black",
                "position": decision["position"],
                "initial_fen": decision["initial_fen"],
                "history": decision["history"],
                "before_fen": board.fen(),
                "fen": board.fen(),
                "move": decision["move"],
                "preferred_move": analysis["before_pv"][0],
                "before_cp": analysis["before_cp"],
                "before_mate": analysis["before_mate"],
                "after_cp": analysis["after_cp"],
                "after_mate": analysis["after_mate"],
                "before_expected": analysis["before_expected"],
                "after_expected": analysis["after_expected"],
                "loss": analysis["loss"],
                "before_pv": analysis["before_pv"],
                "after_pv": analysis["after_pv"],
                "analysis_config_sha256": config["analysis_config_sha256"],
                "analysis_config_json": json.dumps(analysis_config, sort_keys=True),
                "partition": partition_by_id.get(decision["decision_id"], "excluded"),
            }
            | encoded
        )
    write_table(run / "episodes.parquet", records, EPISODES)
    atomic_json(config_path, config)
    atomic_json(
        run / "partitions.json",
        {part: [row["decision_id"] for row in rows] for part, rows in partitions.items()},
    )
    atomic_json(
        run / "split-audit.json",
        {
            "excluded_unassigned_decision_ids": sorted(decision_ids - set(partition_by_id)),
            "all_decision_ids": sorted(decision_ids),
            "episode_ids": [record["episode_id"] for record in records],
        },
    )
    atomic_json(
        complete_path,
        {
            "decisions": len(decisions),
            "episodes": len(records),
            "available": sum(record["available"] for record in records),
            "informative": sum(record["informative"] for record in records),
            "episodes_sha256": _sha256(run / "episodes.parquet"),
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest(),
        },
    )
    return run / "episodes.parquet"
