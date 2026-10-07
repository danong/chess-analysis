"""Thin stage commands; persisted artifacts are the stage interfaces."""

import asyncio
from importlib import import_module
from pathlib import Path

import typer

from . import analysis, discovery, ingestion, reporting, representation
from .doctor import doctor

app = typer.Typer(no_args_is_help=True)
app.command()(doctor)


@app.command()
def sample(
    archive: Path,
    run: Path,
    count: int = 100,
    seed: int = 20261006,
    fixture: bool = False,
    rating_min: int = 400,
    rating_max: int = 799,
    stratified: bool = False,
    prior_run: Path | None = None,
) -> None:
    if count < 1:
        raise typer.BadParameter("count must be positive")
    if rating_min < 0 or rating_max < 0 or rating_min > rating_max:
        raise typer.BadParameter("rating bounds must be nonnegative and rating_min <= rating_max")
    ingestion.sample(
        archive,
        run,
        count,
        seed,
        fixture,
        rating_min,
        rating_max,
        stratified,
        prior_run,
    )


@app.command()
def analyse(run: Path, nodes: int = 100_000, workers: int = 4, batch_size: int = 32) -> None:
    if min(nodes, workers, batch_size) < 1:
        raise typer.BadParameter("nodes, workers, and batch size must be positive")
    asyncio.run(analysis.analyse(run, nodes, workers, batch_size))


@app.command(help="Legacy symbolic feature experiment; retained for historical comparison.")
def features(run: Path) -> None:
    representation.features(run)


@app.command(help="Legacy symbolic discovery experiment; retained for historical comparison.")
def discover(run: Path) -> None:
    discovery.discover(run)


@app.command(help="Legacy symbolic report; retained for historical comparison.")
def report(run: Path) -> None:
    reporting.report(run)


@app.command()
def episodes(run: Path, loss_threshold: float = 0.10, horizon: int = 6) -> None:
    if not 0 <= loss_threshold <= 1:
        raise typer.BadParameter("loss-threshold must be between 0 and 1")
    if horizon < 1:
        raise typer.BadParameter("horizon must be positive")
    episode_stage = import_module(".episodes", __package__)
    typer.echo(episode_stage.build_episodes(run, loss_threshold, horizon))


@app.command()
def cluster(run: Path, max_episodes: int = 200, distance: float = 0.3) -> None:
    if max_episodes < 1:
        raise typer.BadParameter("max-episodes must be positive")
    if not 0 < distance <= 2:
        raise typer.BadParameter("distance must be in (0, 2]")
    trajectories = import_module(".trajectories", __package__)
    typer.echo(trajectories.cluster(run, max_episodes, distance))


@app.command("review-packets")
def review_packets(run: Path, partition: str = "discovery") -> None:
    if partition not in {"discovery", "selection"}:
        raise typer.BadParameter("partition must be discovery or selection")
    trajectories = import_module(".trajectories", __package__)
    typer.echo(trajectories.review_packets(run, partition))


@app.command()
def freeze(run: Path, review: Path, selection_review: Path) -> None:
    trajectories = import_module(".trajectories", __package__)
    typer.echo(trajectories.freeze(run, review, selection_review))


@app.command()
def measure(run: Path, bootstrap: int = 1000, taxonomy_run: Path | None = None) -> None:
    if bootstrap < 1:
        raise typer.BadParameter("bootstrap must be positive")
    trajectories = import_module(".trajectories", __package__)
    typer.echo(trajectories.measure(run, bootstrap, taxonomy_run))


@app.command("engine-audit")
def engine_audit(run: Path, count: int = 20, nodes: int = 1_000_000) -> None:
    if count < 1:
        raise typer.BadParameter("count must be positive")
    if nodes < 1:
        raise typer.BadParameter("nodes must be positive")
    trajectories = import_module(".trajectories", __package__)
    typer.echo(trajectories.engine_audit(run, count, nodes))
