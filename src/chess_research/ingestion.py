"""Seeded sampling from a bounded, recorded archive prefix."""

import hashlib
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

import chess.pgn
import zstandard

from .records import DECISIONS, Decision, atomic_json, read_table, write_table

PARTITIONS = ("discovery", "selection", "evaluation")


def _stratum_quotas(
    count: int, band_starts: list[int], only_partition: str | None = None
) -> dict[int, dict[str, int]]:
    """Allocate exact per-band counts across splits or to one requested split."""
    if count < 1 or not band_starts:
        raise ValueError("count and rating bands must be positive")
    if only_partition is not None and only_partition not in PARTITIONS:
        raise ValueError(f"only_partition must be one of {PARTITIONS}")
    ordered_bands = sorted(band_starts)
    band_quotas = {
        band: count // len(ordered_bands) + (index < count % len(ordered_bands))
        for index, band in enumerate(ordered_bands)
    }
    result: dict[int, dict[str, int]] = {}
    weights = {"discovery": 60, "selection": 20, "evaluation": 20}
    for band in ordered_bands:
        quota = band_quotas[band]
        if only_partition is not None:
            result[band] = {part: quota if part == only_partition else 0 for part in PARTITIONS}
            continue
        allocated = {part: quota * weight // 100 for part, weight in weights.items()}
        remainder = quota - sum(allocated.values())
        order = sorted(
            PARTITIONS,
            key=lambda part: (-(quota * weights[part] % 100), PARTITIONS.index(part)),
        )
        for part in order[:remainder]:
            allocated[part] += 1
        result[band] = allocated
    return result


def sample(
    archive: Path,
    run: Path,
    count: int,
    seed: int,
    fixture: bool = False,
    rating_min: int = 400,
    rating_max: int = 799,
    stratified: bool = False,
    prior_run: Path | None = None,
    only_partition: str | None = None,
) -> None:
    if count < 1:
        raise ValueError("count must be positive")
    if rating_min < 0 or rating_max < 0:
        raise ValueError("rating bounds must be nonnegative")
    if rating_min > rating_max:
        raise ValueError("rating_min must not exceed rating_max")
    if only_partition is not None and only_partition not in PARTITIONS:
        raise ValueError(f"only_partition must be one of {PARTITIONS}")
    if (run / "decisions.parquet").exists():
        raise ValueError("Use a fresh run directory for a new sample")
    if prior_run is not None and not (prior_run / "decisions.parquet").is_file():
        raise ValueError("prior_run must contain decisions.parquet")
    rng = random.Random(seed)
    counts: Counter[str] = Counter()
    bands: Counter[int] = Counter()
    partition_bands: Counter[tuple[int, str]] = Counter()
    # Rating bands stay anchored to round 200 point boundaries even when a
    # requested cohort starts or ends partway through a band.
    band_starts = list(range(rating_min // 200 * 200, rating_max + 1, 200))
    stratum_quotas = _stratum_quotas(count, band_starts, only_partition)
    quotas = {band: sum(values.values()) for band, values in stratum_quotas.items()}
    prior_rows = read_table(prior_run / "decisions.parquet") if prior_run else []
    prior_manifest_path = prior_run / "historical-exclusions.json" if prior_run else None
    prior_manifest = (
        json.loads(prior_manifest_path.read_text())
        if prior_manifest_path is not None and prior_manifest_path.is_file()
        else {}
    )
    prior_players = {r["player"] for r in prior_rows} | set(prior_manifest.get("players", []))
    prior_positions = {r["position"] for r in prior_rows} | set(prior_manifest.get("positions", []))
    rows: list[Decision] = []
    scanned = 0
    with archive.open("rb") as raw:
        stream = (
            zstandard.ZstdDecompressor().stream_reader(raw) if archive.suffix == ".zst" else raw
        )
        with io.TextIOWrapper(stream, encoding="utf-8") as text:
            while len(rows) < count and (game := chess.pgn.read_game(text)) is not None:
                scanned += 1
                if scanned % 10_000 == 0:
                    print(
                        f"Scanned {scanned:,} games; sampled {len(rows):,}/{count:,} decisions",
                        file=sys.stderr,
                    )
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
                    position = " ".join(board.fen().split()[:4])
                    split = partition(player, seed)
                    if (
                        rating_min <= rating <= rating_max
                        and player != "?"
                        and (only_partition is None or split == only_partition)
                        and not (
                            split == "evaluation"
                            and (player in prior_players or position in prior_positions)
                        )
                    ):
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
                                "position": position,
                            }
                        )
                    history.append(move.uci())
                    board.push(move)
                rng.shuffle(candidates)
                for row in candidates:
                    band = row["rating"] // 200 * 200
                    split = partition(row["player"], seed)
                    if (
                        counts[row["player"]] < 10
                        and len(rows) < count
                        and (
                            not stratified
                            or partition_bands[(band, split)] < stratum_quotas[band][split]
                        )
                    ):
                        counts[row["player"]] += 1
                        bands[band] += 1
                        partition_bands[(band, split)] += 1
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
            "cohort": [rating_min, rating_max],
            "stratified": stratified,
            "only_partition": only_partition,
            "band_counts": {str(b): bands[b] for b in band_starts},
            "band_quotas": {str(b): quotas[b] for b in band_starts} if stratified else None,
            "partition_band_counts": {
                str(b): {part: partition_bands[(b, part)] for part in PARTITIONS}
                for b in band_starts
            },
            "partition_band_quotas": {str(b): stratum_quotas[b] for b in band_starts}
            if stratified
            else None,
            "prefix_bias": (
                "Seeded sample from the scanned archive prefix; archive order can affect inclusion."
            ),
            "prior_run": str(prior_run.resolve()) if prior_run else None,
            "prior_manifest_sha256": hashlib.sha256(prior_manifest_path.read_bytes()).hexdigest()
            if prior_manifest_path is not None and prior_manifest_path.is_file()
            else None,
            "player_cap": 10,
            "sampling": "seeded shuffled decisions per streamed game"
            if not stratified
            else "seeded shuffled decisions per streamed game with per-band player-partition quotas",
        },
    )
    atomic_json(
        run / "historical-exclusions.json",
        {
            "players": sorted(prior_players),
            "positions": sorted(prior_positions),
            "source_decisions_sha256": hashlib.sha256(
                (prior_run / "decisions.parquet").read_bytes()
            ).hexdigest()
            if prior_run
            else None,
            "source_manifest_sha256": hashlib.sha256(prior_manifest_path.read_bytes()).hexdigest()
            if prior_manifest_path is not None and prior_manifest_path.is_file()
            else None,
        },
    )
    atomic_json(run / "sample.complete.json", {"rows": len(rows)})


def partition(player: str, seed: int) -> str:
    bucket = int.from_bytes(hashlib.sha256(f"{seed}:{player}".encode()).digest()[:8]) % 100
    return "discovery" if bucket < 60 else "selection" if bucket < 80 else "evaluation"
