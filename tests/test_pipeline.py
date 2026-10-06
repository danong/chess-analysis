"""Behavioral checks for leakage, replay, validation and crash recovery."""

import asyncio
import json
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor

from chess_research import analysis
from chess_research.analysis import replay, score_fields
from chess_research.discovery import extract_rules, matches, split_rows
from chess_research.ingestion import partition, sample
from chess_research.records import read_table
from chess_research.representation import feature_row

FIXTURE = Path(__file__).parent / "fixtures/tiny.pgn"


def test_sample_replay_caps_and_seed(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    sample(FIXTURE, a, 100, 12, True)
    sample(FIXTURE, b, 100, 12, True)
    rows = read_table(a / "decisions.parquet")
    assert rows == read_table(b / "decisions.parquet")
    for player in {r["player"] for r in rows}:
        assert sum(r["player"] == player for r in rows) <= 10
    for row in rows:
        board = replay(row)
        assert chess.Move.from_uci(row["move"]) in board.legal_moves
        assert row["position"] == " ".join(board.fen().split()[:4])


def test_score_perspective_and_mates():
    score = chess.engine.PovScore(chess.engine.Cp(200), chess.WHITE)
    assert score_fields(score.pov(chess.WHITE), 10)[0] == 200
    assert score_fields(score.pov(chess.BLACK), 10)[0] == -200
    mate = chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)
    assert score_fields(mate.pov(chess.BLACK), 10) == (None, -3, 0.0)
    assert score_fields(mate.pov(chess.WHITE), 10) == (None, 3, 1.0)


def test_features_ignore_ratings_and_outcomes(tmp_path):
    sample(FIXTURE, tmp_path, 100, 12, True)
    row = read_table(tmp_path / "decisions.parquet")[0]
    expected = feature_row(row)
    assert feature_row(row | {"rating": 9999, "loss": 123, "before_cp": 999}) == expected
    assert all(
        n == "decision_id"
        or n.split("_")[0] in {"before", "after", "created", "removed", "retained"}
        for n in expected
    )


def test_relationship_identity_and_special_moves():
    row = {
        "decision_id": "x",
        "initial_fen": "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
        "history": [],
        "move": "e1g1",
    }
    result = feature_row(row)
    assert result["after_protect_4_6"] == 1
    row = {
        "decision_id": "p",
        "initial_fen": "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "history": [],
        "move": "a7a8q",
    }
    assert feature_row(row)["created_attack_1_6"] == 1


def test_player_split_and_position_exclusion():
    players = {
        part: next(str(i) for i in range(10000) if partition(str(i), 7) == part)
        for part in ["discovery", "selection", "evaluation"]
    }
    rows = [{"player": p, "position": "shared"} for p in players.values()]
    rows += [{"player": p, "position": part} for part, p in players.items()]
    groups = split_rows(rows, 7)
    assert len(groups["discovery"]) == 2
    assert len(groups["selection"]) == len(groups["evaluation"]) == 1
    assert not (
        {r["player"] for r in groups["discovery"]} & {r["player"] for r in groups["evaluation"]}
    )


def test_rule_extraction_leaf_fidelity():
    x = np.arange(600).reshape(200, 3) % 13
    model = ExtraTreesRegressor(
        n_estimators=1, max_depth=3, min_samples_leaf=10, random_state=4
    ).fit(x, x[:, 0] * x[:, 1])
    rules = extract_rules(model, ["a", "b", "c"])
    groups = {}
    for row, leaf in zip(x, model.estimators_[0].apply(x), strict=True):
        matching = [
            i
            for i, rule in enumerate(rules)
            if matches(rule, dict(zip(["a", "b", "c"], row, strict=True)))
        ]
        assert len(matching) == 1
        groups.setdefault(leaf, set()).add(matching[0])
    assert all(len(ids) == 1 for ids in groups.values())
    assert len({next(iter(ids)) for ids in groups.values()}) == len(groups)


def test_interrupted_batch_recovery(tmp_path, monkeypatch):
    sample(FIXTURE, tmp_path, 100, 12, True)
    original = analysis.write_table
    calls = 0

    def interrupt(path, rows, schema):
        nonlocal calls
        calls += 1
        if calls == 2:
            path.with_suffix(".tmp").write_text("incomplete parquet")
            raise RuntimeError("simulated interruption")
        original(path, rows, schema)

    monkeypatch.setattr(analysis, "write_table", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        asyncio.run(analysis.analyse(tmp_path, 1000, workers=1, batch_size=3))
    batches = list(tmp_path.glob("analysis-*/batch-*.parquet"))
    assert len(batches) == 1
    committed = batches[0].read_bytes()
    monkeypatch.setattr(analysis, "write_table", original)
    target = asyncio.run(analysis.analyse(tmp_path, 1000, workers=1, batch_size=3))
    assert batches[0].read_bytes() == committed
    rows = [r for p in target.glob("batch-*.parquet") for r in read_table(p)]
    assert len(rows) == len({r["decision_id"] for r in rows}) == 13
    asyncio.run(analysis.analyse(tmp_path, 1000, workers=1, batch_size=3))
    assert json.loads((target / "complete.json").read_text())["newly_analysed"] == 0


def test_rated_human_filter(tmp_path):
    source = tmp_path / "source.pgn"
    blocks = []
    for i, (event, title, variant) in enumerate(
        [
            ("Rated rapid game", "", "Standard"),
            ("Unrated rapid game", "", "Standard"),
            ("Rated blitz game", "", "Standard"),
            ("Rated rapid game", "BOT", "Standard"),
            ("Rated rapid game", "", "Chess960"),
        ]
    ):
        blocks.append(f'''[Event "{event}"]
[Site "https://lichess.org/game{i}"]
[White "white{i}"]
[Black "black{i}"]
[WhiteElo "600"]
[BlackElo "1200"]
[WhiteTitle "{title}"]
[Variant "{variant}"]
[Result "*"]

1. e4 e5 *
''')
    source.write_text("\n".join(blocks))
    sample(source, tmp_path / "run", 100, 7)
    rows = read_table(tmp_path / "run/decisions.parquet")
    assert len(rows) == 1 and rows[0]["game_id"] == "game0"


def test_discovery_selection_evaluation_and_freeze(tmp_path):
    """Artificial numerical signal exercises learning, never research evidence."""
    import pyarrow as pa

    from chess_research.discovery import discover
    from chess_research.records import ANALYSES, DECISIONS, atomic_json, write_table
    from chess_research.representation import feature_names

    names = feature_names()
    decisions, features, outcomes = [], [], []
    for i in range(1200):
        identifier = str(i)
        decisions.append(
            {
                "decision_id": identifier,
                "game_id": identifier,
                "player": f"p{i}",
                "rating": 600,
                "ply": 0,
                "initial_fen": chess.STARTING_FEN,
                "history": [],
                "move": "e2e4",
                "position": f"artificial-{i}",
            }
        )
        features.append({"decision_id": identifier} | dict.fromkeys(names, 0) | {names[0]: i % 2})
        outcomes.append(
            {
                "decision_id": identifier,
                "before_expected": 0.5,
                "after_expected": 0.5 - 0.3 * (i % 2),
                "loss": 0.3 * (i % 2),
            }
        )
    write_table(tmp_path / "decisions.parquet", decisions, DECISIONS)
    write_table(
        tmp_path / "features.parquet",
        features,
        pa.schema([("decision_id", pa.string())] + [(n, pa.int32()) for n in names]),
    )
    write_table(tmp_path / "analysis-test/batch-test.parquet", outcomes, ANALYSES)
    atomic_json(tmp_path / "active-analysis.json", {"path": "analysis-test"})
    atomic_json(tmp_path / "config.json", {"seed": 7})
    discover(tmp_path)
    findings = json.loads((tmp_path / "findings.json").read_text())
    assert findings["model_fitted"]
    assert len(findings["candidates"]) == 1
    candidate = findings["candidates"][0]
    assert candidate["supported"]
    assert candidate["evaluation"]["player_bootstrap_ci"][0] > 0.29
    assert len(candidate["examples"]) == 10
    assert (tmp_path / "selected-rules.json").exists()
    with pytest.raises(ValueError, match="already frozen"):
        discover(tmp_path)


def test_annotated_report_exports_legal_history(tmp_path, monkeypatch):
    import io

    import chess.pgn

    from chess_research import reporting
    from chess_research.records import atomic_json

    sample(FIXTURE, tmp_path, 100, 12, True)
    target = asyncio.run(analysis.analyse(tmp_path, 1000, workers=1))
    rows = read_table(tmp_path / "decisions.parquet")
    identifier = rows[0]["decision_id"]
    outcomes = {r["decision_id"]: r for r in analysis.load_analysis(tmp_path)}
    atomic_json(
        tmp_path / "findings.json",
        {
            "candidates": [
                {
                    "id": 1,
                    "name": "Synthetic export test",
                    "rule": [],
                    "discovery": {},
                    "selection": {},
                    "evaluation": {},
                    "supported": False,
                    "examples": [identifier],
                    "counterexamples": [],
                }
            ],
            "partitions": {},
            "target": "Synthetic test",
            "model_fitted": False,
            "conclusion": "Not research evidence",
        },
    )

    async def fake_deeper(*args, **kwargs):
        assert kwargs["ids"] == {identifier}
        assert args[1] == 1_000_000
        return target

    monkeypatch.setattr(reporting, "analyse", fake_deeper)
    reporting.report(tmp_path)
    text = (tmp_path / "examples.pgn").read_text()
    game = chess.pgn.read_game(io.StringIO(text))
    assert game is not None and not game.errors
    assert "Engine expected-score loss" in text
    board = replay(rows[0])
    board.push_uci(rows[0]["move"])
    assert game.end().board() == board
    sensitivity = json.loads((tmp_path / "sensitivity.json").read_text())
    assert sensitivity[0]["original_loss"] == outcomes[identifier]["loss"]
