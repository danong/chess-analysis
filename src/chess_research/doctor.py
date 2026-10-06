"""Small integration check for the local research environment."""

from __future__ import annotations

import asyncio
import importlib
import io
import os
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import chess
import chess.engine
import chess.pgn
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import typer
import zstandard as zstd
from rich.console import Console

app = typer.Typer(no_args_is_help=True)
console = Console()
ROOT = Path(__file__).resolve().parents[2]


async def check_engine(engine_path: Path) -> str:
    """Keep UCI I/O on one event loop; avoid background-thread wakeups."""
    transport, engine = await chess.engine.popen_uci(str(engine_path))
    try:
        await engine.configure({"Threads": 1, "Hash": 128})
        result = await engine.analyse(chess.Board(), chess.engine.Limit(nodes=10_000))
        if "score" not in result or not result.get("pv"):
            raise ValueError("Missing engine score or principal variation")
        mate_board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
        mate = await engine.analyse(mate_board, chess.engine.Limit(nodes=10_000))
        score = mate["score"]
        if score.white().mate() != 1 or score.black().mate() != -1:
            raise ValueError(f"Unexpected mate score/perspective: {score}")
        name = engine.id.get("name", "Unknown engine")
        await engine.quit()
        return name
    finally:
        transport.close()


@app.callback()
def main() -> None:
    """Tools for the chess discovery project."""


@app.command()
def doctor() -> None:
    """Check dependencies, streaming, storage, and the local engine."""
    try:
        console.print(f"Python {platform.python_version()}")
        matplotlib_cache = ROOT / ".cache" / "matplotlib"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
        for module in (
            "numpy",
            "scipy",
            "sklearn",
            "matplotlib",
            "jinja2",
            "pytest",
            "ruff",
            "mypy",
        ):
            importlib.import_module(module)
        console.print("[green]PASS[/green] dependency imports")

        fixture = ROOT / "tests" / "fixtures" / "tiny.pgn"
        compressed = zstd.ZstdCompressor().compress(fixture.read_bytes())
        games: list[chess.pgn.Game] = []
        with (
            zstd.ZstdDecompressor().stream_reader(io.BytesIO(compressed)) as stream,
            io.TextIOWrapper(stream, encoding="utf-8") as text,
        ):
            while (game := chess.pgn.read_game(text)) is not None:
                if game.errors:
                    raise ValueError(f"PGN errors: {game.errors}")
                board = game.board()
                for move in game.mainline_moves():
                    if move not in board.legal_moves:
                        raise ValueError(f"Illegal fixture move: {move}")
                    board.push(move)
                games.append(game)
        if len(games) != 2:
            raise ValueError("Expected two streamed games")
        exported = games[0].accept(chess.pgn.StringExporter())
        round_trip = chess.pgn.read_game(io.StringIO(exported))
        if round_trip is None or round_trip.end().board() != games[0].end().board():
            raise ValueError("PGN round trip changed the position")
        console.print("[green]PASS[/green] compressed stream, legal replay, PGN export")

        schema = pa.schema([("game_id", pa.string()), ("rating", pa.int32())])
        table = pa.Table.from_pylist(
            [{"game_id": "fixture-1", "rating": 600}, {"game_id": "fixture-2", "rating": 1000}],
            schema=schema,
        )
        with TemporaryDirectory(prefix="chess-doctor-") as temp:
            target = Path(temp) / "sample.parquet"
            pq.write_table(table, target)
            restored = pq.read_table(target)
            if not restored.equals(table):
                raise ValueError("Parquet round trip changed data or schema")
            with duckdb.connect() as connection:
                count = connection.execute(
                    "SELECT count(*) FROM read_parquet(?)", [str(target)]
                ).fetchone()
                if count != (2,):
                    raise ValueError("DuckDB could not query the Parquet sample")
        console.print("[green]PASS[/green] Parquet schema and DuckDB query")

        engine_path = Path(os.environ.get("STOCKFISH_PATH", ROOT / ".tools/stockfish/stockfish"))

        async def bounded_check() -> str:
            return await asyncio.wait_for(check_engine(engine_path), timeout=30)

        name = asyncio.run(bounded_check())
        console.print(f"[green]PASS[/green] {name} analysis and mate scores")
        console.print("[green]Environment ready for CPU pipeline work.[/green]")
    except Exception as exc:
        console.print(f"[red]FAIL[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(1) from exc
