"""Async Stockfish searches with independently committed restartable batches."""

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import chess
import chess.engine

from .doctor import ROOT
from .records import ANALYSES, Analysis, atomic_json, read_table, write_table

ANALYSIS_SCHEMA_VERSION = 2


def replay(row: dict[str, Any]) -> chess.Board:
    board = chess.Board(row["initial_fen"])
    for uci in row["history"]:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError("Illegal decision history")
        board.push(move)
    return board


def score_fields(score: chess.engine.Score, ply: int) -> tuple[int | None, int | None, float]:
    wdl = score.wdl(model="sf", ply=ply)
    return score.score(), score.mate(), wdl.expectation()


async def analyse(
    run: Path, nodes: int, workers: int = 4, batch_size: int = 32, ids: set[str] | None = None
) -> Path:
    engine_path = Path(os.environ.get("STOCKFISH_PATH", ROOT / ".tools/stockfish/stockfish"))
    config = {
        "nodes": nodes,
        "decisions_sha256": hashlib.sha256((run / "decisions.parquet").read_bytes()).hexdigest(),
        "workers": workers,
        "threads": 1,
        "hash_mib": 128,
        "engine_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
        "perspective": "moving player",
        "expected_score_model": "python-chess sf WDL",
        "chess_version": chess.__version__,
        "batch_size": batch_size,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
    }
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    target = run / f"analysis-{key}"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "config.json").exists():
        previous = json.loads((target / "config.json").read_text())
        if "engine_name" in previous:
            config["engine_name"] = previous["engine_name"]
    atomic_json(target / "config.json", config)
    rows = read_table(run / "decisions.parquet")
    if ids is not None:
        rows = [r for r in rows if r["decision_id"] in ids]
    done = {r["decision_id"] for p in target.glob("batch-*.parquet") for r in read_table(p)}
    pending = [r for r in rows if r["decision_id"] not in done]
    queue: asyncio.Queue[list[dict[str, Any]]] = asyncio.Queue()
    for i in range(0, len(pending), batch_size):
        queue.put_nowait(pending[i : i + batch_size])
    started = time.monotonic()

    async def worker() -> None:
        transport, engine = await chess.engine.popen_uci(str(engine_path))
        try:
            await engine.configure({"Threads": 1, "Hash": 128})
            config["engine_name"] = engine.id.get("name")
            while not queue.empty():
                batch = queue.get_nowait()
                results: list[Analysis] = []
                for row in batch:
                    board = replay(row)
                    color = board.turn
                    before = await engine.analyse(
                        board, chess.engine.Limit(nodes=nodes), game=object()
                    )
                    move = chess.Move.from_uci(row["move"])
                    if move not in board.legal_moves:
                        raise ValueError("Illegal played move")
                    board.push(move)
                    after = await engine.analyse(
                        board, chess.engine.Limit(nodes=nodes), game=object()
                    )
                    bcp, bm, be = score_fields(before["score"].pov(color), row["ply"])
                    acp, am, ae = score_fields(after["score"].pov(color), row["ply"] + 1)

                    def bound(info: Mapping[str, Any]) -> str:
                        if info.get("lowerbound"):
                            return "lower"
                        if info.get("upperbound"):
                            return "upper"
                        return "exact"

                    results.append(
                        {
                            "decision_id": row["decision_id"],
                            "before_cp": bcp,
                            "before_mate": bm,
                            "after_cp": acp,
                            "after_mate": am,
                            "before_expected": be,
                            "after_expected": ae,
                            "loss": be - ae,
                            "before_pv": [m.uci() for m in before.get("pv", [])],
                            "after_pv": [m.uci() for m in after.get("pv", [])],
                            "before_depth": before.get("depth"),
                            "after_depth": after.get("depth"),
                            "before_bound": bound(before),
                            "after_bound": bound(after),
                        }
                    )
                name = hashlib.sha256(
                    "\n".join(r["decision_id"] for r in batch).encode()
                ).hexdigest()[:16]
                write_table(target / f"batch-{name}.parquet", results, ANALYSES)
                print(f"Committed {len(results)} analyses", flush=True)
        finally:
            await engine.quit()
            transport.close()

    if pending:
        tasks = [asyncio.create_task(worker()) for _ in range(workers)]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    atomic_json(target / "config.json", config)
    atomic_json(
        target / "complete.json",
        {"rows": len(rows), "newly_analysed": len(pending), "seconds": time.monotonic() - started},
    )
    if ids is None:
        atomic_json(run / "active-analysis.json", {"path": target.name})
    return target


def load_analysis(run: Path) -> list[dict[str, Any]]:
    path = run / json.loads((run / "active-analysis.json").read_text())["path"]
    return [r for p in sorted(path.glob("batch-*.parquet")) for r in read_table(p)]
