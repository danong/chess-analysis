"""Synthetic checks for played-versus-preferred episode encoding and holdouts."""

import chess
import pytest


def _decision(fen: str, move: str, *, player: str = "p", position: str = "pos"):
    return {
        "decision_id": "game:0",
        "game_id": "game",
        "player": player,
        "rating": 650,
        "ply": 0,
        "initial_fen": fen,
        "history": [],
        "move": move,
        "position": position,
    }


def _mirror_uci(uci: str) -> str:
    move = chess.Move.from_uci(uci)
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    ).uci()


def _legal_pv(board: chess.Board, length: int, prefix: tuple[str, ...] = ()) -> list[str]:
    """Build a small legal synthetic line from the requested opening moves."""
    result = list(prefix)
    position = board.copy()
    for uci in result:
        position.push_uci(uci)
    while len(result) < length and not position.is_game_over():
        move = next(iter(position.legal_moves))
        result.append(move.uci())
        position.push(move)
    return result


def test_episode_encoder_uses_legal_capture_effects_and_ignores_metadata():
    from chess_research.episodes import encode_episode, feature_names

    # The rook can take the queen on e4; the alternative is a quiet king move.
    fen = "4k3/8/8/8/4q3/8/8/K3R3 w - - 0 1"
    board = chess.Board(fen)
    after_capture = board.copy()
    after_capture.push_uci("e1e4")
    after_quiet = board.copy()
    after_quiet.push_uci("a1a2")
    capture = _decision(fen, "e1e4") | {
        "before_pv": _legal_pv(board, 6, ("a1a2",)),
        "after_pv": _legal_pv(after_capture, 5),
    }
    quiet = _decision(fen, "a1a2") | {
        "before_pv": _legal_pv(board, 6, ("e1e4",)),
        "after_pv": _legal_pv(after_quiet, 5),
    }

    captured = encode_episode(capture, horizon=6)
    quiet_result = encode_episode(quiet, horizon=6)
    assert captured["available"] and captured["informative"]
    assert quiet_result["available"] and quiet_result["informative"]
    assert captured["vector"] != quiet_result["vector"]
    names = feature_names(6)
    assert captured["vector"][names.index("played_step_1_captures_mover_queen")] == 1
    assert captured["vector"][names.index("played_step_1_material_opponent_queen_delta")] == -1

    changed_metadata = capture | {
        "player": "another-player",
        "rating": 2400,
        "decision_id": "other-id",
        "game_id": "other-game",
        "loss": 0.99,
        "before_cp": -1200,
        "after_cp": 500,
    }
    assert encode_episode(changed_metadata, horizon=6)["vector"] == captured["vector"]


def test_episode_encoding_is_moving_player_perspective_invariant():
    from chess_research.episodes import encode_episode

    fen = "4k3/8/8/8/8/8/4p3/K3R3 w - - 0 1"
    board = chess.Board(fen)
    after = board.copy()
    after.push_uci("e1e2")
    white = _decision(fen, "e1e2") | {
        "before_pv": _legal_pv(board, 6),
        "after_pv": _legal_pv(after, 5),
    }
    mirrored_board = chess.Board(fen).mirror()
    black = _decision(mirrored_board.fen(), _mirror_uci("e1e2"), player="black-player") | {
        "before_pv": [_mirror_uci(move) for move in white["before_pv"]],
        "after_pv": [_mirror_uci(move) for move in white["after_pv"]],
    }

    assert encode_episode(white, horizon=6)["vector"] == encode_episode(black, horizon=6)["vector"]


def test_episode_availability_distinguishes_truncation_from_terminal_line():
    from chess_research.episodes import encode_episode

    ordinary = _decision(chess.STARTING_FEN, "e2e4") | {
        "before_pv": ["d2d4"],
        "after_pv": ["e7e5"],
    }
    truncated = encode_episode(ordinary, horizon=6)
    assert truncated["available"] is False

    # Qg7 is checkmate: a legal terminal line may end before the requested horizon.
    mate_fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
    board = chess.Board(mate_fen)
    mate_move = chess.Move.from_uci("f7g7")
    assert mate_move in board.legal_moves
    board.push(mate_move)
    assert board.is_checkmate()
    terminal = _decision(mate_fen, "f7g7") | {
        "before_pv": ["f7g7"],
        "after_pv": [],
    }
    result = encode_episode(terminal, horizon=6)
    assert result["available"] is True


def test_episode_encoder_rejects_illegal_principal_variation():
    from chess_research.episodes import encode_episode

    row = _decision(chess.STARTING_FEN, "e2e4") | {
        "before_pv": ["e2e5"],
        "after_pv": ["e7e5"],
    }
    with pytest.raises(ValueError, match="legal|illegal|PV|variation"):
        encode_episode(row, horizon=6)


def test_episode_encoding_records_promotion_and_does_not_end_at_claimable_draw():
    from chess_research.episodes import encode_episode, feature_names

    promotion_fen = "7k/P7/8/8/8/8/8/K7 w - - 0 1"
    promotion_board = chess.Board(promotion_fen)
    after_promotion = promotion_board.copy()
    after_promotion.push_uci("a7a8q")
    promotion = _decision(promotion_fen, "a7a8q") | {
        "before_pv": _legal_pv(promotion_board, 2, ("a7a8q",)),
        "after_pv": _legal_pv(after_promotion, 1),
    }
    promoted = encode_episode(promotion, horizon=2)
    names = feature_names(2)
    assert promoted["vector"][names.index("played_step_1_promotions_mover_to_queen")] == 1

    draw_fen = "7k/8/8/8/8/8/8/K6R w - - 99 1"
    draw_board = chess.Board(draw_fen)
    draw_after = draw_board.copy()
    draw_after.push_uci("h1h2")
    assert draw_after.can_claim_fifty_moves()
    claimable = _decision(draw_fen, "h1h2") | {
        "before_pv": _legal_pv(draw_board, 2),
        "after_pv": _legal_pv(draw_after, 1),
    }
    result = encode_episode(claimable, horizon=2)
    assert result["available"] is True
    assert result["vector"][names.index("played_step_1_terminal_draw")] == 0


def test_evaluation_exclusions_keep_all_decisions_as_denominators():
    # Find deterministic identities assigned to each partition.
    from chess_research.ingestion import partition
    from chess_research.splits import split_rows

    seed = 19
    players = {
        part: next(f"player-{i}" for i in range(10000) if partition(f"player-{i}", seed) == part)
        for part in ("discovery", "selection", "evaluation")
    }
    second_eval_player = next(
        f"player-{i}" for i in range(10000, 20000) if partition(f"player-{i}", seed) == "evaluation"
    )
    heldout_player = next(
        f"player-{i}" for i in range(20000, 30000) if partition(f"player-{i}", seed) == "evaluation"
    )
    rows = [
        {"decision_id": "eval-kept", "player": players["evaluation"], "position": "fresh"},
        {
            "decision_id": "eval-player-heldout",
            "player": heldout_player,
            "position": "old-player-pos",
        },
        {
            "decision_id": "eval-position-heldout",
            "player": second_eval_player,
            "position": "old-position",
        },
        {"decision_id": "discovery-nonmistake", "player": players["discovery"], "position": "disc"},
        {"decision_id": "selection-nonmistake", "player": players["selection"], "position": "sel"},
    ]
    groups = split_rows(
        rows,
        seed,
        excluded_players={heldout_player},
        excluded_positions={"old-position"},
    )
    assert [row["decision_id"] for row in groups["discovery"]] == ["discovery-nonmistake"]
    assert [row["decision_id"] for row in groups["selection"]] == ["selection-nonmistake"]
    assert [row["decision_id"] for row in groups["evaluation"]] == ["eval-kept"]


def test_episode_build_keeps_full_denominators_and_excludes_historical_eval(tmp_path):
    import json

    from chess_research.episodes import build_episodes
    from chess_research.ingestion import partition
    from chess_research.records import ANALYSES, DECISIONS, atomic_json, read_table, write_table

    seed = 23
    players = {
        part: next(f"current-{i}" for i in range(10000) if partition(f"current-{i}", seed) == part)
        for part in ("discovery", "selection", "evaluation")
    }
    old_player = next(
        f"old-player-{i}"
        for i in range(10000)
        if partition(f"old-player-{i}", seed) == "evaluation"
    )
    old_position_player = next(
        f"old-position-{i}"
        for i in range(10000)
        if partition(f"old-position-{i}", seed) == "evaluation"
    )
    specifications = [
        ("disc-mistake", players["discovery"], "disc-pos", 0.25),
        ("selection-nonmistake", players["selection"], "selection-pos", 0.01),
        ("eval-kept", players["evaluation"], "eval-pos", 0.25),
        ("eval-old-player", old_player, "old-player-pos", 0.25),
        ("eval-old-position", old_position_player, "old-pos", 0.25),
    ]
    decisions = [
        {
            "decision_id": decision_id,
            "game_id": decision_id,
            "player": player,
            "rating": 650,
            "ply": 0,
            "initial_fen": chess.STARTING_FEN,
            "history": [],
            "move": "e2e4",
            "position": position,
        }
        for decision_id, player, position, _loss in specifications
    ]
    analyses = [
        {
            "decision_id": decision_id,
            "before_cp": 0,
            "before_mate": None,
            "after_cp": 0,
            "after_mate": None,
            "before_expected": 0.5,
            "after_expected": 0.5 - loss,
            "loss": loss,
            "before_pv": ["d2d4"],
            "after_pv": ["e7e5"],
        }
        for decision_id, _player, _position, loss in specifications
    ]
    write_table(tmp_path / "decisions.parquet", decisions, DECISIONS)
    write_table(tmp_path / "analysis-one/batch-one.parquet", analyses, ANALYSES)
    atomic_json(tmp_path / "analysis-one/config.json", {"nodes": 1000, "threads": 1})
    atomic_json(tmp_path / "active-analysis.json", {"path": "analysis-one"})
    atomic_json(tmp_path / "config.json", {"seed": seed})
    atomic_json(
        tmp_path / "historical-exclusions.json",
        {
            "players": [old_player],
            "positions": ["old-pos"],
        },
    )

    episodes_path = build_episodes(tmp_path, loss_threshold=0.10, horizon=1)
    episode_rows = read_table(episodes_path)
    assert {row["decision_id"] for row in episode_rows} == {
        "disc-mistake",
        "eval-kept",
        "eval-old-player",
        "eval-old-position",
    }
    partitions = json.loads((tmp_path / "partitions.json").read_text())
    audit = json.loads((tmp_path / "split-audit.json").read_text())
    assert set(audit["all_decision_ids"]) == {row["decision_id"] for row in decisions}
    assert {row["decision_id"] for row in episode_rows} == set(audit["all_decision_ids"]) - {
        "selection-nonmistake"
    }
    assert "eval-old-player" not in partitions["evaluation"]
    assert "eval-old-position" not in partitions["evaluation"]
    by_id = {row["decision_id"]: row for row in episode_rows}
    assert by_id["eval-old-player"]["partition"] == "excluded"
    assert by_id["eval-old-position"]["partition"] == "excluded"
    assert (tmp_path / "episodes.complete.json").exists()


def _write_freeze_fixture(run, *, enough_support=False):
    from pathlib import Path

    from chess_research.episodes import EPISODES
    from chess_research.records import DECISIONS, atomic_json, write_table

    run.mkdir(parents=True, exist_ok=True)
    counts = (
        {"discovery": 30, "selection": 5, "evaluation": 1}
        if enough_support
        else {
            "discovery": 1,
            "selection": 1,
            "evaluation": 1,
        }
    )
    identifiers = {
        partition: [
            f"{partition[:4]}-id" if counts[partition] == 1 else f"{partition[:4]}-{i:02}"
            for i in range(counts[partition])
        ]
        for partition in counts
    }
    specs = [
        (identifier, partition) for partition, names in identifiers.items() for identifier in names
    ]
    decisions = [
        {
            "decision_id": identifier,
            "game_id": identifier,
            "player": f"{partition}-player-{identifier}",
            "rating": 650,
            "ply": 0,
            "initial_fen": chess.STARTING_FEN,
            "history": [],
            "move": "e2e4",
            "position": f"{partition}-position-{identifier}",
        }
        for identifier, partition in specs
    ]
    write_table(run / "decisions.parquet", decisions, DECISIONS)
    episode_rows = [
        {
            "episode_id": identifier,
            "decision_id": identifier,
            "game_id": identifier,
            "player": f"{partition}-player-{identifier}",
            "rating": 650,
            "ply": 0,
            "position": f"{partition}-position-{identifier}",
            "initial_fen": chess.STARTING_FEN,
            "history": [],
            "before_fen": chess.STARTING_FEN,
            "move": "e2e4",
            "preferred_move": "d2d4",
            "before_cp": 0,
            "before_mate": None,
            "after_cp": 0,
            "after_mate": None,
            "before_expected": 0.5,
            "after_expected": 0.4,
            "loss": 0.1,
            "before_pv": ["d2d4"],
            "after_pv": ["e7e5"],
            "analysis_config_sha256": "synthetic",
            "analysis_config_json": "{}",
            "partition": partition,
            "vector": [1.0],
            "available": True,
            "informative": True,
        }
        for identifier, partition in specs
    ]
    write_table(run / "episodes.parquet", episode_rows, EPISODES)
    atomic_json(
        run / "episodes.config.json",
        {
            "horizon": 1,
            "feature_names": ["synthetic"],
            "encoding": "synthetic test encoding",
            "loss_threshold": 0.1,
        },
    )
    atomic_json(
        run / "partitions.json",
        {part: identifiers[part] for part in identifiers},
    )
    atomic_json(
        run / "config.json",
        {
            "seed": 3,
            "cohort": [400, 799],
            "sampled": 3,
            "scanned_games": 3,
            "player_cap": 10,
            "archive": "synthetic",
            "prior_run": None,
        },
    )
    atomic_json(run / "analysis/config.json", {"nodes": 1000, "threads": 1})
    atomic_json(run / "active-analysis.json", {"path": "analysis"})
    from chess_research.trajectories import digest

    inputs = {
        name: digest(run / name)
        for name in ["episodes.parquet", "episodes.config.json", "partitions.json", "config.json"]
    }
    cluster_path = run / "clusters-synthetic.json"
    atomic_json(
        cluster_path,
        {
            "version": 1,
            "method": "synthetic test clustering",
            "distance": 0.3,
            "max_episodes": 2,
            "seed": 3,
            "inputs": inputs,
            "discovery_qualifying": counts["discovery"],
            "truncated": 0,
            "uninformative": 0,
            "sampled": counts["discovery"],
            "sample_band_counts": {"600": counts["discovery"]},
            "scaler_mean": [0.0],
            "scaler_scale": [1.0],
            "clusters": [
                {
                    "cluster_id": "c001",
                    "episode_ids": identifiers["discovery"],
                    "support": counts["discovery"],
                    "unique_players": counts["discovery"],
                    "unique_positions": counts["discovery"],
                    "unique_games": counts["discovery"],
                    "centroid": [1.0],
                    "radius": 1.0,
                    "mean_distance": 0.0,
                    "max_distance": 0.0,
                }
            ],
        },
    )
    atomic_json(run / "active-clusters.json", {"path": cluster_path.name})
    review_base = {"clusters_sha256": digest(cluster_path), "reviewer": "synthetic-review"}
    review = run / "review.json"
    selection = run / "selection-review.json"
    atomic_json(
        review,
        review_base
        | {
            "candidates": [
                {
                    "cluster_id": "c001",
                    "accepted": False,
                    "coherent": False,
                    "name": "",
                    "supporting_episode_ids": [],
                    "contradictions": [],
                    "reason": "Synthetic unsupported cluster",
                }
            ]
        },
    )
    atomic_json(selection, review_base | {"assignments": []})
    return Path(review), Path(selection), cluster_path


def test_semantic_cluster_uses_identity_scaling_and_freeze_binds_embedding_model(tmp_path):
    import hashlib
    import json

    from chess_research.records import atomic_json
    from chess_research.trajectories import cluster, digest, freeze, review_packets

    run = tmp_path / "semantic-run"
    _write_freeze_fixture(run, enough_support=True)
    model = {
        "version": 1,
        "method": "tfidf",
        "vocabulary": {"captures": 0},
        "idf_vector": [1.0],
        "prompt_version": "stable-error-description-v1",
        "dimension": 1,
        "training_ids": ["disc-00"],
    }
    atomic_json(run / "embedding-model.json", model)
    atomic_json(run / "descriptions.json", {"records": []})
    atomic_json(
        run / "stability.json",
        {"config": {"stable_gate": {"version": 1}}, "screened": 30, "decisions": []},
    )
    config_path = run / "episodes.config.json"
    config = json.loads(config_path.read_text()) | {
        "encoding": "TF-IDF external stable-error descriptions",
        "feature_names": ["captures"],
        "embedding_model_sha256": digest(run / "embedding-model.json"),
        "description_prompt_version": model["prompt_version"],
        "stable_gate": {"version": 1},
    }
    atomic_json(config_path, config)
    episodes_complete = {
        "episodes_sha256": digest(run / "episodes.parquet"),
        "stability_sha256": digest(run / "stability.json"),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
    }
    atomic_json(run / "episodes.complete.json", episodes_complete)
    atomic_json(
        run / "descriptions.complete.json",
        {
            "descriptions_sha256": digest(run / "descriptions.json"),
            "episodes_sha256": digest(run / "episodes.parquet"),
            "episodes_config_sha256": digest(config_path),
            "embedding_model_sha256": digest(run / "embedding-model.json"),
        },
    )

    cluster_path = cluster(run, max_episodes=100)
    clustered = json.loads(cluster_path.read_text())
    assert clustered["scaling"] == "identity for L2-normalized TF-IDF"
    assert clustered["scaler_mean"] == [0.0]
    assert clustered["scaler_scale"] == [1.0]
    assert clustered["inputs"]["embedding-model.json"] == digest(run / "embedding-model.json")
    assert "descriptions.complete.json" in clustered["inputs"]

    packets = review_packets(run, "discovery")
    review = json.loads((packets / "review.template.json").read_text())
    review["reviewer"] = "synthetic-review"
    for candidate in review["candidates"]:
        candidate.update(
            {
                "accepted": False,
                "coherent": False,
                "name": "",
                "supporting_episode_ids": [],
                "reason": "Reviewed as a synthetic rejected family.",
            }
        )
    review_path = run / "semantic-review.json"
    atomic_json(review_path, review)

    selection_packets = review_packets(run, "selection")
    selection_path = selection_packets / "selection-review.template.json"
    selection = json.loads(selection_path.read_text())
    selection["reviewer"] = "synthetic-review"
    for assignment in selection["assignments"]:
        assignment.update({"correct": True, "reason": "Synthetic identity-vector assignment."})
    atomic_json(selection_path, selection)

    taxonomy = json.loads(freeze(run, review_path, selection_path).read_text())
    assert taxonomy["inputs"]["embedding-model.json"] == digest(run / "embedding-model.json")

    # A separate evaluation run must bind its own semantic episode vectors to
    # the descriptions, model, and stable gate it was imported with.
    evaluation = tmp_path / "semantic-evaluation"
    _write_freeze_fixture(evaluation, enough_support=False)
    from chess_research.episodes import EPISODES
    from chess_research.records import DECISIONS, read_table, write_table

    id_map = {}
    evaluation_decisions = []
    for row in read_table(evaluation / "decisions.parquet"):
        new_id = f"evaluation-{row['decision_id']}"
        id_map[row["decision_id"]] = new_id
        evaluation_decisions.append(
            row
            | {
                "decision_id": new_id,
                "game_id": f"evaluation-{row['game_id']}",
                "player": f"evaluation-{row['player']}",
                "position": f"evaluation-{row['position']}",
            }
        )
    write_table(evaluation / "decisions.parquet", evaluation_decisions, DECISIONS)
    evaluation_episodes = []
    for row in read_table(evaluation / "episodes.parquet"):
        new_id = id_map[row["decision_id"]]
        evaluation_episodes.append(
            row
            | {
                "episode_id": new_id,
                "decision_id": new_id,
                "game_id": f"evaluation-{row['game_id']}",
                "player": f"evaluation-{row['player']}",
                "position": f"evaluation-{row['position']}",
            }
        )
    write_table(evaluation / "episodes.parquet", evaluation_episodes, EPISODES)
    partitions = json.loads((evaluation / "partitions.json").read_text())
    atomic_json(
        evaluation / "partitions.json",
        {part: [id_map[identifier] for identifier in ids] for part, ids in partitions.items()},
    )
    atomic_json(evaluation / "episodes.config.json", config)
    atomic_json(evaluation / "embedding-model.json", model)
    atomic_json(evaluation / "descriptions.json", {"records": []})
    atomic_json(
        evaluation / "stability.json",
        {"config": {"stable_gate": {"version": 1}}, "screened": 0, "decisions": []},
    )
    atomic_json(
        evaluation / "episodes.complete.json",
        {
            "episodes_sha256": digest(evaluation / "episodes.parquet"),
            "stability_sha256": digest(evaluation / "stability.json"),
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest(),
        },
    )
    evaluation_descriptions_complete = {
        "descriptions_sha256": digest(evaluation / "descriptions.json"),
        "episodes_sha256": digest(evaluation / "episodes.parquet"),
        "episodes_config_sha256": digest(evaluation / "episodes.config.json"),
        "embedding_model_sha256": digest(evaluation / "embedding-model.json"),
    }
    atomic_json(evaluation / "descriptions.complete.json", evaluation_descriptions_complete)
    evaluation_marker = evaluation_descriptions_complete | {"episodes_sha256": "tampered"}
    atomic_json(evaluation / "descriptions.complete.json", evaluation_marker)
    from chess_research.trajectories import measure

    with pytest.raises(ValueError, match="Stable semantic completion hash mismatch"):
        measure(evaluation, bootstrap=2, taxonomy_run=run)
    atomic_json(evaluation / "descriptions.complete.json", evaluation_descriptions_complete)
    assert (measure(evaluation, bootstrap=2, taxonomy_run=run)).exists()


def test_freeze_requires_source_evidence_and_explicit_selection_partition(tmp_path):
    import json

    from chess_research.records import atomic_json
    from chess_research.trajectories import freeze

    run = tmp_path / "run"
    review, selection, _cluster_path = _write_freeze_fixture(run)
    source = json.loads(review.read_text())
    source["candidates"][0]["supporting_episode_ids"] = ["eval-id"]
    atomic_json(review, source)
    with pytest.raises(ValueError, match="source discovery cluster"):
        freeze(run, review, selection)

    source["candidates"][0]["supporting_episode_ids"] = ["disc-id"]
    atomic_json(review, source)
    audit = json.loads(selection.read_text())
    audit["assignments"] = [
        {"decision_id": "eval-id", "cluster_id": "c001", "correct": True, "reason": "bad split"}
    ]
    atomic_json(selection, audit)
    with pytest.raises(ValueError, match="selection episode IDs"):
        freeze(run, review, selection)

    audit["assignments"] = []
    atomic_json(selection, audit)
    source["candidates"] = []
    atomic_json(review, source)
    with pytest.raises(ValueError, match="Every discovery cluster"):
        freeze(run, review, selection)
    assert not (run / "taxonomy.json").exists()


def test_clustering_is_repeatable_and_uses_only_discovery_episodes(tmp_path):
    import json

    from chess_research.trajectories import cluster

    run = tmp_path / "run"
    _write_freeze_fixture(run)
    first_path = cluster(run, max_episodes=10, distance=0.3)
    first_bytes = first_path.read_bytes()
    first = json.loads(first_bytes)
    second_path = cluster(run, max_episodes=10, distance=0.3)
    assert second_path == first_path
    assert second_path.read_bytes() == first_bytes
    assert first["clusters"]
    assert set(first["clusters"][0]["episode_ids"]) == {"disc-id"}


def test_nearest_rejected_family_does_not_fall_through_to_accepted_family():
    from chess_research.trajectories import assign

    rejected = {"cluster_id": "rejected", "centroid": [1.0, 0.0], "radius": 1.0}
    accepted = {"cluster_id": "accepted", "centroid": [0.0, 1.0], "radius": 1.0}
    model = {
        "scaler_mean": [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "routing_centroids": [rejected, accepted],
    }
    row_near_rejected = {"available": True, "informative": True, "vector": [1.0, 0.0]}
    row_near_accepted = {"available": True, "informative": True, "vector": [0.0, 1.0]}
    assert assign(row_near_rejected, model, [accepted])[0] is None
    assert assign(row_near_accepted, model, [accepted])[0] == "accepted"


def test_freeze_accepts_a_reviewed_supported_synthetic_family(tmp_path):
    import json

    from chess_research.records import atomic_json
    from chess_research.trajectories import freeze, measure, review_packets

    run = tmp_path / "run"
    review, selection, _cluster_path = _write_freeze_fixture(run, enough_support=True)
    source = json.loads(review.read_text())
    # The synthetic model's source cluster explicitly contains these discovery examples.
    from chess_research.trajectories import load_json

    model = load_json(run / "clusters-synthetic.json")
    source["candidates"] = [
        {
            "cluster_id": "c001",
            "accepted": True,
            "coherent": True,
            "name": "Observed synthetic line",
            "supporting_episode_ids": model["clusters"][0]["episode_ids"][:3],
            "contradictions": [],
            "reason": "Synthetic fixture with repeated evidence",
        }
    ]
    atomic_json(review, source)
    packet_dir = review_packets(run, partition="selection")
    selection = packet_dir / "selection-review.template.json"
    audit = json.loads(selection.read_text())
    audit["reviewer"] = "synthetic-review"
    for entry in audit["assignments"]:
        entry["correct"] = True
        entry["reason"] = "Synthetic repeated-vector assignment"
    atomic_json(selection, audit)
    reviewed_bytes = selection.read_bytes()

    taxonomy = json.loads(freeze(run, review, selection).read_text())
    assert selection.read_bytes() == reviewed_bytes
    from chess_research.trajectories import digest

    assert taxonomy["selection_review_sha256"] == digest(selection)
    assert [candidate["cluster_id"] for candidate in taxonomy["candidates"]] == ["c001"]
    assert taxonomy["candidates"][0]["selection_audit"]["correct"] == 5
    result = json.loads(measure(run, bootstrap=20).read_text())
    assert result["coverage"][1]["classified"] == 1
    assert result["families"][0]["bands"][1]["incidents"] == 1


def test_empty_frozen_taxonomy_measures_full_denominator_once(tmp_path):
    import json

    from chess_research.trajectories import freeze, measure

    run = tmp_path / "run"
    review, selection, _cluster_path = _write_freeze_fixture(run)
    freeze(run, review, selection)
    with pytest.raises(ValueError, match="already frozen"):
        freeze(run, review, selection)

    measured = measure(run, bootstrap=20)
    result = json.loads(measured.read_text())
    assert result["coverage"][1]["decisions"] == 1
    assert result["coverage"][1]["qualifying_mistakes"] == 1
    assert result["coverage"][1]["classified"] == 0
    with pytest.raises(ValueError, match="already measured"):
        measure(run, bootstrap=20)


def test_measure_refuses_changed_frozen_inputs(tmp_path):
    from chess_research.records import atomic_json
    from chess_research.trajectories import freeze, measure

    run = tmp_path / "run"
    review, selection, _cluster_path = _write_freeze_fixture(run)
    freeze(run, review, selection)
    atomic_json(run / "episodes.config.json", {"horizon": 99})
    with pytest.raises(ValueError, match="Frozen input changed"):
        measure(run, bootstrap=20)


def test_external_evaluation_keeps_taxonomy_frozen_and_requires_fresh_compatible_data(tmp_path):
    import json

    from chess_research.episodes import EPISODES
    from chess_research.records import DECISIONS, atomic_json, read_table, write_table
    from chess_research.trajectories import digest, freeze, measure

    source, fresh = tmp_path / "source", tmp_path / "fresh"
    review, selection, _ = _write_freeze_fixture(source)
    freeze(source, review, selection)
    original_taxonomy = digest(source / "taxonomy.json")
    _write_freeze_fixture(fresh)
    with pytest.raises(ValueError, match="repeats source players"):
        measure(fresh, bootstrap=20, taxonomy_run=source)
    decisions = read_table(fresh / "decisions.parquet")
    episodes = read_table(fresh / "episodes.parquet")
    for rows in [decisions, episodes]:
        for row in rows:
            if row["decision_id"] == "eval-id":
                row["player"], row["position"] = (
                    "fresh-evaluation-player",
                    "fresh-evaluation-position",
                )
    write_table(fresh / "decisions.parquet", decisions, DECISIONS)
    write_table(fresh / "episodes.parquet", episodes, EPISODES)
    config = json.loads((fresh / "episodes.config.json").read_text())
    atomic_json(fresh / "episodes.config.json", config | {"horizon": 2})
    with pytest.raises(ValueError, match="encoding differs: horizon"):
        measure(fresh, bootstrap=20, taxonomy_run=source)
    atomic_json(fresh / "episodes.config.json", config)
    result = json.loads(measure(fresh, bootstrap=20, taxonomy_run=source).read_text())
    assert result["coverage"][1]["decisions"] == 1
    assert result["taxonomy_sha256"] == original_taxonomy
    assert digest(source / "taxonomy.json") == original_taxonomy
