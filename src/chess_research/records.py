"""Explicit stage schemas and atomic artifact writes."""

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import pyarrow as pa
import pyarrow.parquet as pq


class Decision(TypedDict):
    decision_id: str
    game_id: str
    player: str
    rating: int
    ply: int
    initial_fen: str
    history: list[str]
    move: str
    position: str


class Analysis(TypedDict):
    decision_id: str
    before_cp: int | None
    before_mate: int | None
    after_cp: int | None
    after_mate: int | None
    before_expected: float
    after_expected: float
    loss: float
    before_pv: list[str]
    after_pv: list[str]
    before_depth: NotRequired[int | None]
    after_depth: NotRequired[int | None]
    before_bound: NotRequired[str | None]
    after_bound: NotRequired[str | None]


DECISIONS = pa.schema(
    [
        ("decision_id", pa.string()),
        ("game_id", pa.string()),
        ("player", pa.string()),
        ("rating", pa.int32()),
        ("ply", pa.int32()),
        ("initial_fen", pa.string()),
        ("history", pa.list_(pa.string())),
        ("move", pa.string()),
        ("position", pa.string()),
    ]
)
ANALYSES = pa.schema(
    [
        ("decision_id", pa.string()),
        ("before_cp", pa.int32()),
        ("before_mate", pa.int32()),
        ("after_cp", pa.int32()),
        ("after_mate", pa.int32()),
        ("before_expected", pa.float64()),
        ("after_expected", pa.float64()),
        ("loss", pa.float64()),
        ("before_pv", pa.list_(pa.string())),
        ("after_pv", pa.list_(pa.string())),
        ("before_depth", pa.int32()),
        ("after_depth", pa.int32()),
        ("before_bound", pa.string()),
        ("after_bound", pa.string()),
    ]
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def write_table(path: Path, rows: Iterable[Mapping[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist([dict(r) for r in rows], schema=schema), temp)
    os.replace(temp, path)


def read_table(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()
