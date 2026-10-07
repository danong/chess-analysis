"""Edge coverage for description imports with no usable discovery vocabulary."""

import hashlib
import json

import chess
import pytest


def _write_semantics_run(run):
    from chess_research.episodes import EPISODES
    from chess_research.records import atomic_json, write_table

    row = {
        "episode_id": "disc",
        "decision_id": "disc",
        "game_id": "private-game-id",
        "player": "private-player-id",
        "rating": 1200,
        "ply": 0,
        "side": "White",
        "position": "position-disc",
        "initial_fen": chess.STARTING_FEN,
        "history": [],
        "before_fen": chess.STARTING_FEN,
        "fen": chess.STARTING_FEN,
        "move": "e2e4",
        "preferred_move": "d2d4",
        "before_cp": 40,
        "before_mate": None,
        "after_cp": -120,
        "after_mate": None,
        "before_expected": 0.8,
        "after_expected": 0.25,
        "loss": 0.55,
        "before_pv": ["d2d4", "d7d5"],
        "after_pv": ["e7e5", "g1f3"],
        "analysis_config_sha256": "synthetic",
        "analysis_config_json": "{}",
        "partition": "discovery",
        "vector": [],
        "available": True,
        "informative": True,
    }
    write_table(run / "episodes.parquet", [row], EPISODES)
    config = {"encoding": "stable synthetic", "stable_gate": {"version": 1}}
    atomic_json(run / "episodes.config.json", config)
    atomic_json(run / "stability.json", {"screened": 1, "decisions": []})
    atomic_json(
        run / "episodes.complete.json",
        {
            "episodes_sha256": hashlib.sha256((run / "episodes.parquet").read_bytes()).hexdigest(),
            "stability_sha256": hashlib.sha256((run / "stability.json").read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest(),
        },
    )


def test_all_ungrounded_descriptions_import_with_empty_frozen_model(tmp_path):
    from chess_research.records import read_table
    from chess_research.semantics import (
        PROMPT_VERSION,
        description_packets,
        import_descriptions,
    )

    _write_semantics_run(tmp_path)
    packet = json.loads(description_packets(tmp_path).read_text())
    example = packet["episodes"][0]
    external = tmp_path / "external-descriptions.json"
    external.write_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": packet["prompt_sha256"],
                "provider": "human-review",
                "records": [
                    {
                        "decision_id": "disc",
                        "episode_sha256": example["episode_sha256"],
                        "raw_explanation": "The supplied lines do not establish a concrete error.",
                        "normalized": "the and",
                        "grounded": False,
                        "reason": "insufficient evidence",
                        "author": "reviewer-7",
                    }
                ],
            }
        )
    )

    imported_path = import_descriptions(tmp_path, external)
    model = json.loads((tmp_path / "embedding-model.json").read_text())
    assert model["dimension"] == 0
    assert model["vocabulary"] == {}
    assert model["idf_vector"] == []
    assert model["training_ids"] == []
    record = json.loads(imported_path.read_text())["records"][0]
    assert record["author"] == "reviewer-7"
    episode = read_table(tmp_path / "episodes.parquet")[0]
    assert episode["vector"] == []
    assert episode["available"] is False
    assert episode["informative"] is False
    assert (tmp_path / "descriptions.complete.json").exists()


def test_description_source_cannot_alias_import_output(tmp_path):
    from chess_research.semantics import import_descriptions

    source = tmp_path / "descriptions.json"
    source.write_text("{}")
    with pytest.raises(ValueError, match="separate from descriptions.json"):
        import_descriptions(tmp_path, source)


def test_normalized_text_masks_chess_notation():
    from chess_research.semantics import _clean_normalized

    text = _clean_normalized("Queen Nxe4+ Qh5 R1e2 fxg8=Q# O-O-O e4 e8=Q# e2e4 wins material")
    assert text.split() == ["queen", "wins", "material"]
