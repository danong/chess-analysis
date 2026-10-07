"""Synthetic tests for balanced rating-by-player-partition sampling."""

import json
from collections import Counter
from pathlib import Path

import chess
import chess.pgn

from chess_research.analysis import replay
from chess_research.ingestion import _stratum_quotas, partition, sample
from chess_research.records import DECISIONS, read_table, write_table


def _players(seed: int, part: str, count: int) -> list[str]:
    found = []
    index = 0
    while len(found) < count:
        name = f"{part}-player-{index}"
        if partition(name, seed) == part:
            found.append(name)
        index += 1
    return found


def test_partition_quota_rounding_sums_to_requested_count():
    expected = {
        1: {"discovery": 1, "selection": 0, "evaluation": 0},
        2: {"discovery": 1, "selection": 1, "evaluation": 0},
        7: {"discovery": 4, "selection": 2, "evaluation": 1},
        20: {"discovery": 12, "selection": 4, "evaluation": 4},
    }
    for count, partition_counts in expected.items():
        quotas = _stratum_quotas(count, [600])
        assert quotas == {600: partition_counts}
        assert sum(quotas[600].values()) == count
    evaluation_only = _stratum_quotas(2000, [400, 600, 800, 1000], "evaluation")
    assert evaluation_only == {
        band: {"discovery": 0, "selection": 0, "evaluation": 500} for band in (400, 600, 800, 1000)
    }


def _game(game_id: str, black: str, white_move: chess.Move) -> tuple[str, dict]:
    game = chess.pgn.Game()
    game.headers.update(
        {
            "Event": "Rated rapid game",
            "Site": f"https://lichess.org/{game_id}",
            "White": f"opponent-{game_id}",
            "Black": black,
            "WhiteElo": "1200",
            "BlackElo": "600",
            "Result": "*",
        }
    )
    board = game.board()
    node: chess.pgn.GameNode = game
    node = node.add_variation(white_move)
    board.push(white_move)
    position = " ".join(board.fen().split()[:4])
    black_move = next(iter(board.legal_moves))
    node.add_variation(black_move)
    row = {
        "decision_id": f"{game_id}:1",
        "game_id": game_id,
        "player": black,
        "rating": 600,
        "ply": 1,
        "initial_fen": chess.STARTING_FEN,
        "history": [white_move.uci()],
        "move": black_move.uci(),
        "position": position,
    }
    return str(game), row


def test_stratified_sampler_reserves_late_fresh_evaluation_quota(tmp_path):
    seed = 41
    opening_moves = list(chess.Board().legal_moves)
    discovery = _players(seed, "discovery", 12)
    selection = _players(seed, "selection", 4)
    old_evaluation = _players(seed, "evaluation", 4)
    fresh_evaluation = _players(seed, "evaluation", 4)
    while set(old_evaluation) & set(fresh_evaluation):
        old_evaluation = _players(seed, "evaluation", 8)[:4]
        fresh_evaluation = _players(seed, "evaluation", 8)[4:]

    specifications = []
    specifications.extend(zip(discovery, opening_moves[:12], strict=True))
    specifications.extend(zip(selection, opening_moves[12:16], strict=True))
    # Prior evaluation positions occur before fresh evaluation identities in the archive.
    specifications.extend(zip(old_evaluation, opening_moves[:4], strict=True))
    specifications.extend(zip(fresh_evaluation, opening_moves[16:20], strict=True))

    pgn_blocks, source_rows = [], []
    for index, (player, opening) in enumerate(specifications):
        block, row = _game(f"source-{index}", player, opening)
        pgn_blocks.append(block)
        source_rows.append(row)

    prior = tmp_path / "prior"
    prior.mkdir()
    old_rows = [row for row in source_rows if row["player"] in set(old_evaluation)]
    write_table(prior / "decisions.parquet", old_rows, DECISIONS)
    archive = tmp_path / "games.pgn"
    archive.write_text("\n\n".join(pgn_blocks) + "\n")

    run = tmp_path / "run"
    sample(
        archive,
        run,
        count=20,
        seed=seed,
        rating_min=600,
        rating_max=600,
        stratified=True,
        prior_run=prior,
    )

    rows = read_table(run / "decisions.parquet")
    config = json.loads((run / "config.json").read_text())
    assert len(rows) == 20
    assert Counter(partition(row["player"], seed) for row in rows) == {
        "discovery": 12,
        "selection": 4,
        "evaluation": 4,
    }
    assert {row["player"] for row in rows if partition(row["player"], seed) == "evaluation"} == set(
        fresh_evaluation
    )
    assert {row["player"] for row in rows}.isdisjoint(old_evaluation)
    assert {
        row["position"] for row in rows if partition(row["player"], seed) == "evaluation"
    }.isdisjoint({row["position"] for row in old_rows})
    assert config["partition_band_counts"]["600"] == {
        "discovery": 12,
        "selection": 4,
        "evaluation": 4,
    }
    assert config["partition_band_quotas"]["600"] == {
        "discovery": 12,
        "selection": 4,
        "evaluation": 4,
    }
    assert len({row["position"] for row in rows}) == len(rows)
    assert all(count <= 10 for count in Counter(row["player"] for row in rows).values())
    assert all(chess.Move.from_uci(row["move"]) in replay(row).legal_moves for row in rows)


def test_evaluation_only_sampler_fills_four_rating_bands(tmp_path):
    seed = 1024
    targets = _players(seed, "evaluation", 2000)
    opponents = _players(seed, "discovery", 2000)
    blocks = []
    bands = (400, 600, 800, 1000)
    for index, (target, opponent) in enumerate(zip(targets, opponents, strict=True)):
        rating = bands[index // 500]
        game = chess.pgn.Game()
        game.headers.update(
            {
                "Event": "Rated rapid game",
                "Site": f"https://lichess.org/eval-only-{index}",
                "White": opponent,
                "Black": target,
                "WhiteElo": str(rating),
                "BlackElo": str(rating),
                "Result": "*",
            }
        )
        node = game.add_variation(chess.Move.from_uci("e2e4"))
        node.add_variation(chess.Move.from_uci("e7e5"))
        blocks.append(str(game))
    archive = tmp_path / "evaluation-games.pgn"
    archive.write_text("\n\n".join(blocks) + "\n")

    run = tmp_path / "evaluation-run"
    sample(
        archive,
        run,
        count=2000,
        seed=seed,
        rating_min=400,
        rating_max=1199,
        stratified=True,
        only_partition="evaluation",
    )

    rows = read_table(run / "decisions.parquet")
    config = json.loads((run / "config.json").read_text())
    assert len(rows) == 2000
    assert {partition(row["player"], seed) for row in rows} == {"evaluation"}
    assert Counter(row["rating"] // 200 * 200 for row in rows) == {
        400: 500,
        600: 500,
        800: 500,
        1000: 500,
    }
    assert config["only_partition"] == "evaluation"
    assert config["partition_band_quotas"] == {
        str(band): {"discovery": 0, "selection": 0, "evaluation": 500} for band in bands
    }


def test_sampler_rejects_unknown_only_partition(tmp_path):
    import pytest

    archive = tmp_path / "unused.pgn"
    archive.write_text("")
    with pytest.raises(ValueError, match="only_partition"):
        sample(archive, tmp_path / "run", count=1, seed=1, only_partition="all")


def test_legacy_non_stratified_sampling_has_no_partition_quota(tmp_path):
    fixture = Path(__file__).parent / "fixtures/tiny.pgn"
    run = tmp_path / "run"
    sample(fixture, run, count=20, seed=8, fixture=True)
    config = json.loads((run / "config.json").read_text())
    assert config["stratified"] is False
    assert config["partition_band_quotas"] is None
