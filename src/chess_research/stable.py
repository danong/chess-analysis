"""Deep-search stability gate for catastrophic blunder episodes."""

import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import chess

from .analysis import ANALYSIS_SCHEMA_VERSION, analyse, replay
from .episodes import EPISODES
from .records import atomic_json, read_table, write_table
from .splits import split_rows

STABLE_GATE_VERSION = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _analysis_rows(path: Path) -> list[dict[str, Any]]:
    return [row for batch in sorted(path.glob("batch-*.parquet")) for row in read_table(batch)]


def _finite_scores(analysis: dict[str, Any], decision_id: str) -> None:
    for field in ("before_expected", "after_expected", "loss"):
        value = analysis.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Non-finite {field} for {decision_id}")
    if not (0.0 <= analysis["before_expected"] <= 1.0):
        raise ValueError(f"Out-of-range before_expected for {decision_id}")
    if not (0.0 <= analysis["after_expected"] <= 1.0):
        raise ValueError(f"Out-of-range after_expected for {decision_id}")
    if not math.isclose(
        analysis["loss"], analysis["before_expected"] - analysis["after_expected"], abs_tol=1e-12
    ):
        raise ValueError(f"Inconsistent loss for {decision_id}")


def stable_reason(
    baseline: dict[str, Any],
    deep: dict[str, Any],
    min_loss: float = 0.4,
    max_score_drift: float = 0.1,
) -> str | None:
    """Return a stable-gate rejection reason, or None when the decision passes."""
    decision_id = str(deep.get("decision_id", baseline.get("decision_id", "unknown")))
    try:
        _finite_scores(baseline, decision_id)
        _finite_scores(deep, decision_id)
    except ValueError as exc:
        return str(exc)
    if baseline["loss"] < min_loss:
        return "baseline_loss_below_minimum"
    if deep["loss"] < min_loss:
        return "deep_loss_below_minimum"
    drift = max(
        abs(deep["before_expected"] - baseline["before_expected"]),
        abs(deep["after_expected"] - baseline["after_expected"]),
    )
    if drift > max_score_drift:
        return "score_drift_exceeds_maximum"
    before_pv = deep.get("before_pv")
    after_pv = deep.get("after_pv")
    if not before_pv:
        return "missing_deep_pv"
    if not after_pv and not deep.get("after_terminal"):
        return "missing_deep_pv"
    played_move = deep.get("played_move", baseline.get("played_move", baseline.get("move")))
    if played_move is not None and before_pv[0] == played_move:
        return "deep_preferred_move_equals_played_move"
    for label in ("before", "after"):
        bound = deep.get(f"{label}_bound")
        if deep.get(f"{label}_terminal") and bound is None:
            continue
        if bound not in {"exact", "lower", "upper"}:
            return f"missing_or_unknown_{label}_bound"
        if bound != "exact":
            return f"{label}_score_bound_not_exact"
    return None


def _pv_problem(decision: dict[str, Any], analysis: dict[str, Any]) -> str | None:
    board = replay(decision)
    before_pv = analysis.get("before_pv")
    after_pv = analysis.get("after_pv")
    if not before_pv:
        return "missing_deep_pv"
    move = chess.Move.from_uci(decision["move"])
    if move not in board.legal_moves:
        return "illegal_played_move"
    after_board = board.copy()
    after_board.push(move)
    if not after_pv and not after_board.is_game_over():
        return "missing_deep_pv"
    for label, pv, start in (
        ("before", before_pv, board),
        ("after", after_pv or [], after_board),
    ):
        position = start.copy()
        for token in pv:
            try:
                candidate = chess.Move.from_uci(token)
            except ValueError:
                return f"illegal_{label}_pv"
            if candidate not in position.legal_moves:
                return f"illegal_{label}_pv"
            position.push(candidate)
    return None


def _episode_row(
    decision: dict[str, Any],
    deep: dict[str, Any],
    partition: str,
    analysis_config_sha256: str,
    analysis_config_json: str,
) -> dict[str, Any]:
    board = replay(decision)
    return {
        "episode_id": decision["decision_id"],
        "decision_id": decision["decision_id"],
        "game_id": decision["game_id"],
        "player": decision["player"],
        "rating": decision["rating"],
        "ply": decision["ply"],
        "side": "White" if board.turn else "Black",
        "position": decision["position"],
        "initial_fen": decision["initial_fen"],
        "history": decision["history"],
        "before_fen": board.fen(),
        "fen": board.fen(),
        "move": decision["move"],
        "preferred_move": deep["before_pv"][0],
        "before_cp": deep["before_cp"],
        "before_mate": deep["before_mate"],
        "after_cp": deep["after_cp"],
        "after_mate": deep["after_mate"],
        "before_expected": deep["before_expected"],
        "after_expected": deep["after_expected"],
        "loss": deep["loss"],
        "before_pv": deep["before_pv"],
        "after_pv": deep["after_pv"],
        "analysis_config_sha256": analysis_config_sha256,
        "analysis_config_json": analysis_config_json,
        "partition": partition,
        "vector": [],
        "available": True,
        "informative": True,
    }


def stable_blunders(
    run: Path,
    min_loss: float = 0.4,
    max_score_drift: float = 0.1,
    deep_nodes: int = 1_000_000,
    partitions: tuple[str, ...] = ("discovery", "selection", "evaluation"),
) -> Path:
    """Reanalyse baseline catastrophic blunders and persist stable episode artifacts."""
    if not math.isfinite(min_loss) or not 0 <= min_loss <= 1:
        raise ValueError("min_loss must be finite and between zero and one")
    if not math.isfinite(max_score_drift) or not 0 <= max_score_drift <= 1:
        raise ValueError("max_score_drift must be finite and between zero and one")
    if deep_nodes < 1:
        raise ValueError("deep_nodes must be positive")
    valid_parts = {"discovery", "selection", "evaluation"}
    if (
        not partitions
        or len(set(partitions)) != len(partitions)
        or not set(partitions) <= valid_parts
    ):
        raise ValueError("partitions must be unique requested split names")
    if (run / "taxonomy.json").exists() or (run / "active-clusters.json").exists():
        raise ValueError("Cannot rebuild stable episodes after taxonomy or clusters exist")

    decisions_path = run / "decisions.parquet"
    decisions = read_table(decisions_path)
    decision_by_id = {row["decision_id"]: row for row in decisions}
    if len(decision_by_id) != len(decisions):
        raise ValueError("Duplicate decision IDs")
    active = json.loads((run / "active-analysis.json").read_text())
    baseline_dir = run / active["path"]
    baseline_rows = _analysis_rows(baseline_dir)
    baseline_by_id = {row["decision_id"]: row for row in baseline_rows}
    if len(baseline_by_id) != len(baseline_rows) or set(baseline_by_id) != set(decision_by_id):
        raise ValueError("Complete one-to-one baseline analysis coverage required")
    baseline_config_path = baseline_dir / "config.json"
    baseline_config = json.loads(baseline_config_path.read_text())
    if baseline_config.get("nodes") is not None and deep_nodes <= baseline_config["nodes"]:
        raise ValueError("deep_nodes must exceed the baseline analysis node limit")
    for decision_id, row in baseline_by_id.items():
        _finite_scores(row, decision_id)

    exclusions_path = run / "historical-exclusions.json"
    exclusions = json.loads(exclusions_path.read_text()) if exclusions_path.exists() else {}
    run_config_path = run / "config.json"
    run_config = json.loads(run_config_path.read_text())
    splits = split_rows(
        decisions,
        run_config["seed"],
        set(exclusions.get("players", [])),
        set(exclusions.get("positions", [])),
    )
    partition_by_id = {row["decision_id"]: part for part, rows in splits.items() for row in rows}
    requested = set(partitions)
    screened = [
        decision_by_id[decision_id]
        for decision_id, part in partition_by_id.items()
        if part in requested and baseline_by_id[decision_id]["loss"] >= min_loss
    ]
    screened_ids = {row["decision_id"] for row in screened}

    deep_dir = asyncio.run(analyse(run, deep_nodes, workers=4, ids=screened_ids))
    deep_rows = _analysis_rows(deep_dir)
    deep_by_id = {row["decision_id"]: row for row in deep_rows}
    if len(deep_by_id) != len(deep_rows) or set(deep_by_id) != screened_ids:
        raise ValueError("Complete one-to-one deep analysis coverage required")
    deep_config_path = deep_dir / "config.json"
    deep_config = json.loads(deep_config_path.read_text())
    for key in ("engine_sha256", "chess_version", "perspective", "expected_score_model"):
        if key in baseline_config and deep_config.get(key) != baseline_config[key]:
            raise ValueError(f"Deep analysis configuration mismatch for {key}")

    source_hashes = {
        "decisions": _sha256(decisions_path),
        "baseline_analysis_config": _sha256(baseline_config_path),
        "baseline_analysis_batches": {
            path.name: _sha256(path) for path in sorted(baseline_dir.glob("batch-*.parquet"))
        },
        "deep_analysis_config": _sha256(deep_config_path),
        "deep_analysis_batches": {
            path.name: _sha256(path) for path in sorted(deep_dir.glob("batch-*.parquet"))
        },
        "historical_exclusions": _sha256(exclusions_path) if exclusions_path.exists() else None,
        "run_config": _sha256(run_config_path),
    }
    config = {
        "loss_threshold": min_loss,
        "stable_gate_version": STABLE_GATE_VERSION,
        "stable_gate": {
            "version": STABLE_GATE_VERSION,
            "min_loss": min_loss,
            "max_score_drift": max_score_drift,
            "deep_nodes": deep_nodes,
            "require_exact_nonterminal_scores": True,
            "require_legal_nonempty_pvs": True,
            "reject_preferred_move_equal_to_played": True,
        },
        "min_loss": min_loss,
        "max_score_drift": max_score_drift,
        "deep_nodes": deep_nodes,
        "partitions": list(partitions),
        "seed": run_config["seed"],
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "baseline_analysis_config": baseline_config,
        "deep_analysis_config": deep_config,
        "analysis_config": deep_config,
        "source_hashes": source_hashes,
        "score_perspective": "moving player",
        "partition_method": "splits.split_rows; all decisions partitioned by moving-player identity",
        "screening": "baseline loss >= min_loss; deep loss >= min_loss; expected-score drift bounded at both positions; exact deep scores; legal PVs; deep preferred move differs from played move",
        "encoding": "stable episodes awaiting external semantic descriptions",
        "horizon": 0,
        "feature_names": [],
    }
    config_path = run / "episodes.config.json"
    complete_path = run / "episodes.complete.json"
    stability_path = run / "stability.json"
    artifact_path = run / "episodes.parquet"
    if complete_path.exists():
        old_config = json.loads(config_path.read_text()) if config_path.exists() else None
        complete = json.loads(complete_path.read_text())
        if (
            old_config == config
            and artifact_path.exists()
            and stability_path.exists()
            and complete.get("episodes_sha256") == _sha256(artifact_path)
            and complete.get("stability_sha256") == _sha256(stability_path)
        ):
            return artifact_path
        raise ValueError("Stable episode artifacts already exist with different inputs or settings")

    ledger: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    for decision in screened:
        decision_id = decision["decision_id"]
        baseline = baseline_by_id[decision_id]
        deep = deep_by_id[decision_id]
        _finite_scores(deep, decision_id)
        enriched_deep = deep | {"played_move": decision["move"]}
        start_board = replay(decision)
        played = chess.Move.from_uci(decision["move"])
        after_board = start_board.copy()
        if played in after_board.legal_moves:
            after_board.push(played)
        enriched_deep["before_terminal"] = start_board.is_game_over()
        enriched_deep["after_terminal"] = after_board.is_game_over()
        reasons: list[str] = []
        reason = stable_reason(baseline, enriched_deep, min_loss, max_score_drift)
        if reason:
            reasons.append(reason)
        pv_reason = _pv_problem(decision, deep)
        if pv_reason and pv_reason not in reasons:
            reasons.append(pv_reason)
        drift = max(
            abs(deep["before_expected"] - baseline["before_expected"]),
            abs(deep["after_expected"] - baseline["after_expected"]),
        )
        stable = not reasons
        ledger.append(
            {
                "decision_id": decision_id,
                "partition": partition_by_id[decision_id],
                "baseline": baseline,
                "deep": deep,
                "played_move": decision["move"],
                "max_expected_score_drift": drift,
                "stable": stable,
                "rejection_reasons": reasons,
            }
        )
        if stable:
            retained.append(
                _episode_row(
                    decision,
                    deep,
                    partition_by_id[decision_id],
                    _sha256(deep_config_path),
                    json.dumps(deep_config, sort_keys=True),
                )
            )

    write_table(artifact_path, retained, EPISODES)
    atomic_json(stability_path, {"config": config, "screened": len(ledger), "decisions": ledger})
    atomic_json(config_path, config)
    atomic_json(
        run / "partitions.json",
        {
            part: [row["decision_id"] for row in splits[part]]
            for part in ("discovery", "selection", "evaluation")
        },
    )
    atomic_json(
        run / "split-audit.json",
        {
            "excluded_unassigned_decision_ids": sorted(set(decision_by_id) - set(partition_by_id)),
            "all_decision_ids": sorted(decision_by_id),
            "partition_denominators": {
                part: len(splits[part]) for part in ("discovery", "selection", "evaluation")
            },
            "requested_partitions": list(partitions),
            "screened_decision_ids": [row["decision_id"] for row in ledger],
            "episode_ids": [row["episode_id"] for row in retained],
        },
    )
    atomic_json(
        complete_path,
        {
            "decisions": len(decisions),
            "screened": len(ledger),
            "episodes": len(retained),
            "available": len(retained),
            "informative": len(retained),
            "episodes_sha256": _sha256(artifact_path),
            "stability_sha256": _sha256(stability_path),
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest(),
        },
    )
    return artifact_path
