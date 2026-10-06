"""Thin stage commands; persisted artifacts are the stage interfaces."""

import asyncio
from pathlib import Path

import typer

from . import analysis, discovery, ingestion, reporting, representation
from .doctor import doctor

app = typer.Typer(no_args_is_help=True)
app.command()(doctor)


@app.command()
def sample(
    archive: Path, run: Path, count: int = 100, seed: int = 20261006, fixture: bool = False
) -> None:
    if count < 1:
        raise typer.BadParameter("count must be positive")
    ingestion.sample(archive, run, count, seed, fixture)


@app.command()
def analyse(run: Path, nodes: int = 100_000, workers: int = 4, batch_size: int = 32) -> None:
    if min(nodes, workers, batch_size) < 1:
        raise typer.BadParameter("nodes, workers, and batch size must be positive")
    asyncio.run(analysis.analyse(run, nodes, workers, batch_size))


@app.command()
def features(run: Path) -> None:
    representation.features(run)


@app.command()
def discover(run: Path) -> None:
    discovery.discover(run)


@app.command()
def report(run: Path) -> None:
    reporting.report(run)
