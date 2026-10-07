"""Contract checks for stable blunder gates and external text descriptions."""

import hashlib
import json

import chess
import pytest


def _scores(*, loss: float = 0.55, before: float = 0.8, after: float | None = None):
    return {
        "before_expected": before,
        "after_expected": before - loss if after is None else after,
        "loss": loss,
    }


def _deep(**updates):
    row = {
        "decision_id": "g:0",
        **_scores(),
        "before_cp": 40,
        "before_mate": None,
        "after_cp": -120,
        "after_mate": None,
        "before_pv": ["d2d4", "d7d5"],
        "after_pv": ["e7e5", "g1f3"],
        "before_bound": "exact",
        "after_bound": "exact",
        "played_move": "e2e4",
        "before_terminal": False,
        "after_terminal": False,
    }
    row.update(updates)
    if "after_expected" not in updates and {"loss", "before_expected"} & updates.keys():
        row["after_expected"] = row["before_expected"] - row["loss"]
    elif "after_expected" in updates and "loss" not in updates:
        row["loss"] = row["before_expected"] - row["after_expected"]
    return row


def test_stable_reason_requires_baseline_and_deep_loss_to_remain_catastrophic():
    from chess_research.stable import stable_reason

    baseline = _scores(loss=0.39)
    deep = _deep()
    assert stable_reason(baseline, deep) == "baseline_loss_below_minimum"
    assert stable_reason(_scores(loss=0.55), _deep(loss=0.399)) == "deep_loss_below_minimum"


def test_stable_reason_rejects_score_drift_pv_bounds_and_preferred_move_rescue():
    from chess_research.stable import stable_reason

    baseline = _scores()
    assert stable_reason(baseline, _deep(before_expected=0.91)) == "score_drift_exceeds_maximum"
    assert stable_reason(baseline, _deep(after_expected=0.36)) == "score_drift_exceeds_maximum"
    assert stable_reason(baseline, _deep(before_pv=[])) == "missing_deep_pv"
    assert stable_reason(baseline, _deep(after_pv=[])) == "missing_deep_pv"
    assert stable_reason(baseline, _deep(before_bound=None)) == "missing_or_unknown_before_bound"
    assert stable_reason(baseline, _deep(after_bound="lower")) == "after_score_bound_not_exact"
    assert (
        stable_reason(baseline, _deep(before_pv=["e2e4", "d7d5"]))
        == "deep_preferred_move_equals_played_move"
    )


def test_stable_reason_keeps_mate_scores_separate_and_allows_terminal_short_pv():
    from chess_research.stable import stable_reason

    baseline = _scores()
    mate = _deep(
        before_cp=None,
        before_mate=3,
        after_cp=None,
        after_mate=-2,
    )
    assert stable_reason(baseline, mate) is None
    terminal = _deep(after_pv=[], after_terminal=True, after_bound=None)
    assert stable_reason(baseline, terminal) is None


def _write_stable_run(run, monkeypatch, *, seed=9):
    from chess_research import stable
    from chess_research.ingestion import partition
    from chess_research.records import ANALYSES, DECISIONS, atomic_json, write_table

    players = {
        part: next(
            f"{part}-player-{n}"
            for n in range(10000)
            if partition(f"{part}-player-{n}", seed) == part
        )
        for part in ("discovery", "selection", "evaluation")
    }
    specs = [
        ("disc-stable", players["discovery"], 0.65, "stable"),
        ("disc-rescued", players["discovery"], 0.65, "rescued"),
        ("disc-illegal", players["discovery"], 0.65, "illegal"),
        ("selection-weak", players["selection"], 0.2, "weak"),
        ("eval-stable", players["evaluation"], 0.6, "eval"),
    ]
    decisions = [
        {
            "decision_id": identifier,
            "game_id": identifier,
            "player": player,
            "rating": 650,
            "ply": 0,
            "initial_fen": chess.STARTING_FEN,
            "history": [],
            "move": "e2e4",
            "position": f"position-{identifier}",
        }
        for identifier, player, _loss, _kind in specs
    ]
    baseline = [
        {
            "decision_id": identifier,
            "before_cp": 20,
            "before_mate": None,
            "after_cp": -100,
            "after_mate": None,
            "before_expected": 0.8,
            "after_expected": 0.8 - loss,
            "loss": loss,
            "before_pv": ["d2d4", "d7d5"],
            "after_pv": ["e7e5", "g1f3"],
            "before_depth": 12,
            "after_depth": 12,
            "before_bound": "exact",
            "after_bound": "exact",
        }
        for identifier, _player, loss, _kind in specs
    ]
    write_table(run / "decisions.parquet", decisions, DECISIONS)
    write_table(run / "analysis-base/batch-0.parquet", baseline, ANALYSES)
    analysis_settings = {
        "nodes": 100,
        "engine_sha256": "synthetic-engine",
        "chess_version": chess.__version__,
        "perspective": "moving player",
        "expected_score_model": "python-chess sf WDL",
    }
    atomic_json(run / "analysis-base/config.json", analysis_settings)
    atomic_json(run / "active-analysis.json", {"path": "analysis-base"})
    atomic_json(run / "config.json", {"seed": seed})

    deep_dir = run / "analysis-deep"
    deep_settings = analysis_settings | {"nodes": 1_000_000}
    atomic_json(deep_dir / "config.json", deep_settings)
    deep_rows = []
    for identifier, _player, loss, kind in specs:
        if kind == "weak":
            continue
        preferred = "e2e4" if kind == "rescued" else "d2d4"
        before_pv = ["d2d5"] if kind == "illegal" else [preferred, "d7d5"]
        deep_rows.append(
            {
                "decision_id": identifier,
                "before_cp": 20,
                "before_mate": None,
                "after_cp": -100,
                "after_mate": None,
                "before_expected": 0.8,
                "after_expected": 0.8 - (0.55 if kind != "stable" else loss - 0.05),
                "loss": 0.55 if kind != "stable" else loss - 0.05,
                "before_pv": before_pv,
                "after_pv": ["e7e5", "g1f3"],
                "before_depth": 30,
                "after_depth": 30,
                "before_bound": "exact",
                "after_bound": "exact",
            }
        )
    write_table(deep_dir / "batch-0.parquet", deep_rows, ANALYSES)
    calls = []

    async def fake_analyse(_run, _nodes, workers=4, batch_size=32, ids=None):
        calls.append(set(ids or set()))
        selected_dir = run / f"analysis-selected-{len(calls)}"
        from chess_research.records import read_table, write_table

        selected_rows = [
            row
            for row in read_table(deep_dir / "batch-0.parquet")
            if row["decision_id"] in (ids or set())
        ]
        write_table(selected_dir / "batch-0.parquet", selected_rows, ANALYSES)
        atomic_json(selected_dir / "config.json", deep_settings)
        return selected_dir

    monkeypatch.setattr(stable, "analyse", fake_analyse)
    return specs, calls


def test_stable_blunders_keeps_denominators_and_searches_only_requested_partition(
    tmp_path, monkeypatch
):
    from chess_research import stable
    from chess_research.records import read_table

    specs, calls = _write_stable_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="deep_nodes must exceed"):
        stable.stable_blunders(tmp_path, deep_nodes=100, partitions=("discovery",))
    result = stable.stable_blunders(tmp_path, deep_nodes=1_000_000, partitions=("discovery",))
    expected_screened = {"disc-stable", "disc-rescued", "disc-illegal"}
    assert calls == [expected_screened]
    ledger = json.loads((tmp_path / "stability.json").read_text())
    by_id = {entry["decision_id"]: entry for entry in ledger["decisions"]}
    assert set(by_id) == expected_screened
    assert by_id["disc-stable"]["stable"] is True
    assert by_id["disc-rescued"]["stable"] is False
    assert "deep_preferred_move_equals_played_move" in by_id["disc-rescued"]["rejection_reasons"]
    assert "illegal_before_pv" in by_id["disc-illegal"]["rejection_reasons"]
    episodes = read_table(result)
    assert [row["decision_id"] for row in episodes] == ["disc-stable"]

    partitions = json.loads((tmp_path / "partitions.json").read_text())
    audit = json.loads((tmp_path / "split-audit.json").read_text())
    assert {identifier for group in partitions.values() for identifier in group} == {
        spec[0] for spec in specs
    }
    assert set(audit["all_decision_ids"]) == {spec[0] for spec in specs}
    assert audit["partition_denominators"] == {part: len(partitions[part]) for part in partitions}
    assert audit["requested_partitions"] == ["discovery"]

    original_stability = (tmp_path / "stability.json").read_bytes()
    monkeypatch.setattr(stable, "STABLE_GATE_VERSION", stable.STABLE_GATE_VERSION + 1)
    with pytest.raises(ValueError, match="already exist|different inputs"):
        stable.stable_blunders(tmp_path, deep_nodes=1_000_000, partitions=("discovery",))
    assert (tmp_path / "stability.json").read_bytes() == original_stability


def _episode_row(identifier: str, partition: str) -> dict:
    return {
        "episode_id": identifier,
        "decision_id": identifier,
        "game_id": identifier,
        "player": f"player-{identifier}",
        "rating": 650,
        "ply": 0,
        "side": "White",
        "position": f"position-{identifier}",
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
        "partition": partition,
        "vector": [],
        "available": True,
        "informative": True,
    }


def _semantic_run(run, identifiers=("disc", "sel", "eval"), extra=()):
    from chess_research.episodes import EPISODES
    from chess_research.records import atomic_json, write_table

    partitions = dict(zip(("discovery", "selection", "evaluation"), identifiers, strict=True))
    rows = [_episode_row(identifier, part) for part, identifier in partitions.items()]
    rows.extend(_episode_row(identifier, part) for identifier, part in extra)
    write_table(run / "episodes.parquet", rows, EPISODES)
    config = {
        "encoding": "stable synthetic",
        "stable_gate": {"version": 1, "min_loss": 0.4, "max_score_drift": 0.1},
    }
    atomic_json(run / "episodes.config.json", config)
    atomic_json(
        run / "partitions.json", {part: [identifier] for part, identifier in partitions.items()}
    )
    atomic_json(run / "config.json", {"seed": 3})
    stability = {"screened": len(rows), "decisions": []}
    atomic_json(run / "stability.json", stability)
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
    return partitions


def _description_file(path, packet, records, *, provider="external-review"):
    from chess_research.semantics import PROMPT_VERSION

    path.write_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": packet["prompt_sha256"],
                "provider": provider,
                "records": records,
            }
        )
    )


def test_external_description_packets_validate_provenance_and_identity_free_examples(tmp_path):
    from chess_research.semantics import description_packets, import_descriptions

    _semantic_run(tmp_path)
    packet_path = description_packets(tmp_path, "discovery")
    packet = json.loads(packet_path.read_text())
    assert packet["partition"] == "discovery"
    assert len(packet["episodes"]) == 1
    example = packet["episodes"][0]
    assert {"decision_id", "episode_sha256", "fen", "played_pv", "preferred_pv"} <= set(example)
    assert not {"player", "rating", "game_id"} & set(example)

    bad_prompt = tmp_path / "bad-prompt.json"
    _description_file(bad_prompt, packet, [], provider="reviewer")
    value = json.loads(bad_prompt.read_text())
    value["prompt_sha256"] = "wrong"
    bad_prompt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="prompt provenance"):
        import_descriptions(tmp_path, bad_prompt)

    bad_id = tmp_path / "bad-id.json"
    _description_file(
        bad_id,
        packet,
        [
            {
                "decision_id": "not-in-run",
                "episode_sha256": "x",
                "raw_explanation": "x",
                "normalized": "x",
                "grounded": True,
            }
        ],
    )
    with pytest.raises(ValueError, match="missing or excluded"):
        import_descriptions(tmp_path, bad_id)

    wrong_fingerprint = tmp_path / "bad-fingerprint.json"
    _description_file(
        wrong_fingerprint,
        packet,
        [
            {
                "decision_id": "disc",
                "episode_sha256": "wrong",
                "raw_explanation": "x",
                "normalized": "x",
                "grounded": True,
            }
        ],
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        import_descriptions(tmp_path, wrong_fingerprint)

    duplicate = tmp_path / "duplicate.json"
    valid_record = {
        "decision_id": "disc",
        "episode_sha256": example["episode_sha256"],
        "raw_explanation": "A concrete explanation.",
        "normalized": "concrete description",
        "grounded": True,
    }
    _description_file(duplicate, packet, [valid_record, valid_record.copy()])
    with pytest.raises(ValueError, match="unique"):
        import_descriptions(tmp_path, duplicate)


def test_tfidf_is_fit_only_on_grounded_discovery_and_unavailable_text_stays_unclassified(tmp_path):
    from chess_research.records import read_table
    from chess_research.semantics import description_packets, import_descriptions

    partitions = _semantic_run(tmp_path, extra=(("missing", "selection"),))
    packet_path = description_packets(tmp_path, "discovery")
    packet = json.loads(packet_path.read_text())
    packets_by_id = {item["decision_id"]: item for item in packet["episodes"]}
    records = [
        {
            "decision_id": partitions["discovery"],
            "episode_sha256": packets_by_id[partitions["discovery"]]["episode_sha256"],
            "raw_explanation": "The queen moves to e4 while the rook promotes on a7a8q.",
            "normalized": "Queen captures material without a7a8q or e4 coordinates.",
            "grounded": True,
        },
        {
            "decision_id": partitions["selection"],
            "episode_sha256": "selection fingerprint is not in discovery packet",
            "raw_explanation": "Novel selection wording.",
            "normalized": "quantum galaxy vocabulary",
            "grounded": True,
        },
    ]
    # Build exact fingerprints for every supplied partition record.
    for part in ("selection", "evaluation"):
        part_packet = json.loads(description_packets(tmp_path, part).read_text())
        packets_by_id.update({item["decision_id"]: item for item in part_packet["episodes"]})
    records[1]["episode_sha256"] = packets_by_id[partitions["selection"]]["episode_sha256"]
    records.append(
        {
            "decision_id": partitions["evaluation"],
            "episode_sha256": packets_by_id[partitions["evaluation"]]["episode_sha256"],
            "raw_explanation": "Unsupported example.",
            "normalized": "another ungrounded term",
            "grounded": False,
            "reason": "The line does not explain this outcome.",
        }
    )
    source = tmp_path / "submitted-descriptions.json"
    _description_file(source, packet, records)
    import_descriptions(tmp_path, source)

    model = json.loads((tmp_path / "embedding-model.json").read_text())
    assert model["training_ids"] == [partitions["discovery"]]
    assert "quantum" not in model["vocabulary"]
    assert "galaxy" not in model["vocabulary"]
    assert not any(token in {"e4", "a7a8q"} for token in model["vocabulary"])
    rows = {row["decision_id"]: row for row in read_table(tmp_path / "episodes.parquet")}
    assert rows[partitions["discovery"]]["available"] is True
    assert rows[partitions["selection"]]["available"] is False
    assert rows[partitions["evaluation"]]["available"] is False
    assert rows["missing"]["available"] is False
    assert rows[partitions["selection"]]["vector"] == []


def test_external_frozen_tfidf_model_is_reused_without_refit_and_hash_checked(tmp_path):
    from chess_research.semantics import description_packets, import_descriptions

    source_run = tmp_path / "source"
    source_run.mkdir()
    ids = _semantic_run(source_run)
    packet = json.loads(description_packets(source_run, "discovery").read_text())
    episode = packet["episodes"][0]
    source_descriptions = source_run / "submitted.json"
    _description_file(
        source_descriptions,
        packet,
        [
            {
                "decision_id": ids["discovery"],
                "episode_sha256": episode["episode_sha256"],
                "raw_explanation": "The bishop captures a pawn.",
                "normalized": "bishop captures pawn",
                "grounded": True,
            }
        ],
    )
    import_descriptions(source_run, source_descriptions)
    model_path = source_run / "embedding-model.json"
    source_model = model_path.read_bytes()
    model_hash = hashlib.sha256(source_model).hexdigest()
    taxonomy = {"embedding_model_sha256": model_hash}
    taxonomy_bytes = json.dumps(taxonomy).encode()
    (source_run / "taxonomy.json").write_bytes(taxonomy_bytes)
    (source_run / "taxonomy.complete.json").write_text(
        json.dumps({"taxonomy_sha256": hashlib.sha256(taxonomy_bytes).hexdigest()})
    )

    target_run = tmp_path / "target"
    target_run.mkdir()
    target_ids = _semantic_run(target_run, ("new-disc", "new-sel", "new-eval"))
    target_packet = json.loads(description_packets(target_run, "discovery").read_text())
    target_episode = target_packet["episodes"][0]
    target_descriptions = target_run / "submitted.json"
    _description_file(
        target_descriptions,
        target_packet,
        [
            {
                "decision_id": target_ids["discovery"],
                "episode_sha256": target_episode["episode_sha256"],
                "raw_explanation": "The queen captures a rook.",
                "normalized": "quantum galaxy",
                "grounded": True,
            }
        ],
    )
    import_descriptions(target_run, target_descriptions, model_run=source_run)
    assert (target_run / "embedding-model.json").read_bytes() == source_model
    target_model = json.loads((target_run / "embedding-model.json").read_text())
    assert target_model["training_ids"] == [ids["discovery"]]

    (source_run / "embedding-model.json").write_text(json.dumps(target_model | {"dimension": 0}))
    fresh_run = tmp_path / "fresh"
    fresh_run.mkdir()
    fresh_ids = _semantic_run(fresh_run, ("fresh-disc", "fresh-sel", "fresh-eval"))
    fresh_packet = json.loads(description_packets(fresh_run, "discovery").read_text())
    fresh_episode = fresh_packet["episodes"][0]
    fresh_desc = fresh_run / "submitted.json"
    _description_file(
        fresh_desc,
        fresh_packet,
        [
            {
                "decision_id": fresh_ids["discovery"],
                "episode_sha256": fresh_episode["episode_sha256"],
                "raw_explanation": "x",
                "normalized": "x",
                "grounded": True,
            }
        ],
    )
    with pytest.raises(ValueError, match="taxonomy.*bind"):
        import_descriptions(fresh_run, fresh_desc, model_run=source_run)


def test_evaluation_split_keeps_one_occurrence_per_position():
    from chess_research.ingestion import partition
    from chess_research.splits import split_rows

    players = [f"p{i}" for i in range(100) if partition(f"p{i}", 42) == "evaluation"]
    rows = [
        {"decision_id": "first", "player": players[0], "position": "same"},
        {"decision_id": "repeat", "player": players[1], "position": "same"},
        {"decision_id": "other", "player": players[1], "position": "other"},
    ]
    assert [r["decision_id"] for r in split_rows(rows, 42)["evaluation"]] == ["first", "other"]
