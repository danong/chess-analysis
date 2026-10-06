"""Seeded sampling from a bounded, recorded archive prefix."""

import hashlib
import io
import random
from collections import Counter
from pathlib import Path

import chess.pgn
import zstandard

from .records import DECISIONS, Decision, atomic_json, write_table


def sample(archive: Path, run: Path, count: int, seed: int, fixture: bool = False) -> None:
    if (run / "decisions.parquet").exists():
        raise ValueError("Use a fresh run directory for a new sample")
    rng = random.Random(seed)
    counts: Counter[str] = Counter()
    rows: list[Decision] = []
    scanned = 0
    with archive.open("rb") as raw:
        stream = (
            zstandard.ZstdDecompressor().stream_reader(raw) if archive.suffix == ".zst" else raw
        )
        with io.TextIOWrapper(stream, encoding="utf-8") as text:
            while len(rows) < count and (game := chess.pgn.read_game(text)) is not None:
                scanned += 1
                h = game.headers
                if game.errors or h.get("Variant", "Standard") != "Standard":
                    continue
                if not fixture and (
                    not h.get("Event", "").lower().startswith("rated ")
                    or "rapid" not in h.get("Event", "").lower()
                    or h.get("WhiteTitle") == "BOT"
                    or h.get("BlackTitle") == "BOT"
                ):
                    continue
                board = game.board()
                candidates: list[Decision] = []
                history: list[str] = []
                for ply, move in enumerate(game.mainline_moves()):
                    side = "White" if board.turn else "Black"
                    player = h.get(side, "?").casefold()
                    try:
                        rating = int(h.get(side + "Elo", "600" if fixture else "0"))
                    except ValueError:
                        rating = 0
                    if move not in board.legal_moves:
                        raise ValueError("Illegal source replay")
                    if 400 <= rating <= 799 and player != "?":
                        game_id = h.get("Site", "?").rsplit("/", 1)[-1]
                        if game_id == "?" and not fixture:
                            raise ValueError("Missing source game ID")
                        if game_id == "?":
                            game_id = f"fixture-{scanned}"
                        candidates.append(
                            {
                                "decision_id": f"{game_id}:{ply}",
                                "game_id": game_id,
                                "player": player,
                                "rating": rating,
                                "ply": ply,
                                "initial_fen": game.board().fen(),
                                "history": history.copy(),
                                "move": move.uci(),
                                "position": " ".join(board.fen().split()[:4]),
                            }
                        )
                    history.append(move.uci())
                    board.push(move)
                rng.shuffle(candidates)
                for row in candidates:
                    if counts[row["player"]] < 10 and len(rows) < count:
                        counts[row["player"]] += 1
                        rows.append(row)
    if len({r["decision_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source game IDs")
    write_table(run / "decisions.parquet", rows, DECISIONS)
    stat = archive.stat()
    atomic_json(
        run / "config.json",
        {
            "seed": seed,
            "requested": count,
            "sampled": len(rows),
            "archive": str(archive.resolve()),
            "archive_size": stat.st_size,
            "archive_mtime_ns": stat.st_mtime_ns,
            "scanned_games": scanned,
            "stopped_at_game_boundary": True,
            "fixture": fixture,
            "cohort": [400, 799],
            "player_cap": 10,
            "sampling": "seeded shuffled decisions per streamed game",
        },
    )
    atomic_json(run / "sample.complete.json", {"rows": len(rows)})


def partition(player: str, seed: int) -> str:
    bucket = int.from_bytes(hashlib.sha256(f"{seed}:{player}".encode()).digest()[:8]) % 100
    return "discovery" if bucket < 60 else "selection" if bucket < 80 else "evaluation"
