"""Discovery-only trajectory clustering, reviewed freezing, and held-out measurement.

Review is a persisted human/assistant activity, never a model call. Artifact hashes
bind the frozen classifier to its discovery inputs and reviewed evidence.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler

from .analysis import analyse, load_analysis, replay
from .doctor import ROOT
from .records import atomic_json, read_table

MIN_PLAYERS = 20
MIN_POSITIONS = 30
MIN_SUPPORT = 30


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def active_clusters(run: Path) -> tuple[Path, dict[str, Any]]:
    path = run / load_json(run / "active-clusters.json")["path"]
    result = load_json(path)
    for filename, expected in result["inputs"].items():
        if digest(run / filename) != expected:
            raise ValueError(f"Discovery input changed: {filename}")
    return path, result


def _validate_semantic_artifacts(run: Path) -> None:
    """Require completion manifests to bind stable episodes and frozen text vectors."""
    required = [
        "episodes.parquet",
        "episodes.config.json",
        "episodes.complete.json",
        "stability.json",
        "descriptions.json",
        "descriptions.complete.json",
        "embedding-model.json",
    ]
    if any(not (run / name).is_file() for name in required):
        raise ValueError("Complete stable semantic artifacts are required")
    episode = load_json(run / "episodes.complete.json")
    description = load_json(run / "descriptions.complete.json")
    config = load_json(run / "episodes.config.json")
    embedding = load_json(run / "embedding-model.json")
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    checks = [
        (episode.get("episodes_sha256"), digest(run / "episodes.parquet")),
        (episode.get("stability_sha256"), digest(run / "stability.json")),
        (episode.get("config_sha256"), config_hash),
        (description.get("descriptions_sha256"), digest(run / "descriptions.json")),
        (description.get("episodes_sha256"), digest(run / "episodes.parquet")),
        (description.get("episodes_config_sha256"), digest(run / "episodes.config.json")),
        (description.get("embedding_model_sha256"), digest(run / "embedding-model.json")),
        (config.get("embedding_model_sha256"), digest(run / "embedding-model.json")),
        (config.get("description_prompt_version"), embedding.get("prompt_version")),
    ]
    if any(actual != expected for actual, expected in checks):
        raise ValueError("Stable semantic completion hash mismatch")


def supported(group: dict[str, Any]) -> bool:
    return (
        group["support"] >= MIN_SUPPORT
        and group["unique_players"] >= MIN_PLAYERS
        and group["unique_positions"] >= MIN_POSITIONS
    )


def stratified_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Round-robin rating strata, one instance of each discovery position."""
    rng = np.random.default_rng(seed)
    buckets: dict[int, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for row in sorted(rows, key=lambda r: r["decision_id"]):
        if row["position"] not in seen:
            buckets.setdefault(row["rating"] // 200 * 200, []).append(row)
            seen.add(row["position"])
    for bucket in buckets.values():
        rng.shuffle(bucket)
    chosen: list[dict[str, Any]] = []
    while len(chosen) < count and any(buckets.values()):
        for band in sorted(buckets):
            if buckets[band] and len(chosen) < count:
                chosen.append(buckets[band].pop())
    return chosen


def cluster(run: Path, max_episodes: int = 200, distance: float = 0.3) -> Path:
    if (run / "taxonomy.json").exists():
        raise ValueError("Taxonomy is frozen; use a fresh run")
    if max_episodes < 1 or not 0 < distance <= 2:
        raise ValueError("max_episodes must be positive; cosine distance must be in (0, 2]")
    config = load_json(run / "config.json")
    partitions = load_json(run / "partitions.json")
    ids = set(partitions["discovery"])
    episodes = [r for r in read_table(run / "episodes.parquet") if r["decision_id"] in ids]
    eligible = [r for r in episodes if r["available"] and r["informative"]]
    rows = stratified_sample(eligible, max_episodes, config["seed"])
    inputs = {
        p: digest(run / p)
        for p in [
            "episodes.parquet",
            "episodes.config.json",
            "partitions.json",
            "config.json",
            "decisions.parquet",
        ]
    }
    semantic = (run / "embedding-model.json").exists()
    if semantic:
        _validate_semantic_artifacts(run)
        for filename in [
            "embedding-model.json",
            "descriptions.json",
            "stability.json",
            "episodes.complete.json",
            "descriptions.complete.json",
        ]:
            inputs[filename] = digest(run / filename)
    spec = {
        "version": 1,
        "method": "average-linkage agglomerative, cosine distance",
        "distance": distance,
        "max_episodes": max_episodes,
        "seed": config["seed"],
        "inputs": inputs,
    }
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    path = run / f"clusters-{key}.json"
    if path.exists():
        atomic_json(run / "active-clusters.json", {"path": path.name})
        return path
    description_authors = {}
    if semantic:
        description_authors = {
            r["decision_id"]: r.get("author") or "unspecified"
            for r in load_json(run / "descriptions.json")["records"]
        }
    groups = []
    scaler_mean: list[float] = []
    scaler_scale: list[float] = []
    zero_ids: list[str] = []
    if rows:
        x = np.array([r["vector"] for r in rows], dtype=float)
        # Scale units without subtracting the shared event signal: an identical
        # recurring trajectory must not become a zero vector after centering.
        if semantic:
            scaler_mean = np.zeros(x.shape[1]).tolist()
            scaler_scale = np.ones(x.shape[1]).tolist()
        else:
            scaler = StandardScaler(with_mean=False).fit(x)
            x = scaler.transform(x)
            scaler_mean, scaler_scale = np.zeros(x.shape[1]).tolist(), scaler.scale_.tolist()
        nonzero = np.linalg.norm(x, axis=1) > 1e-12
        zero_ids = [r["decision_id"] for r, ok in zip(rows, nonzero, strict=True) if not ok]
        rows = [r for r, ok in zip(rows, nonzero, strict=True) if ok]
        x = x[nonzero]
        labels = (
            AgglomerativeClustering(
                n_clusters=None, distance_threshold=distance, linkage="average", metric="cosine"
            ).fit_predict(x)
            if len(x) >= 2
            else np.zeros(len(x), dtype=int)
        )
        members = sorted(
            [np.flatnonzero(labels == label).tolist() for label in set(labels)],
            key=lambda indices: min(rows[i]["decision_id"] for i in indices),
        )
        for number, indices in enumerate(members, 1):
            group_rows = [rows[i] for i in indices]
            center = x[indices].mean(axis=0)
            distances = cosine_distances(x[indices], center.reshape(1, -1)).ravel()
            radius = min(distance, max(0.05, float(np.quantile(distances, 0.95)) + 0.05))
            groups.append(
                {
                    "cluster_id": f"c{number:03d}",
                    "episode_ids": [r["decision_id"] for r in group_rows],
                    "support": len(group_rows),
                    "unique_players": len({r["player"] for r in group_rows}),
                    "unique_positions": len({r["position"] for r in group_rows}),
                    "unique_games": len({r["game_id"] for r in group_rows}),
                    "description_author_counts": {
                        author: sum(
                            description_authors.get(r["decision_id"]) == author for r in group_rows
                        )
                        for author in sorted(
                            {
                                description_authors.get(r["decision_id"], "unspecified")
                                for r in group_rows
                            }
                        )
                    }
                    if semantic
                    else {},
                    "centroid": center.tolist(),
                    "radius": radius,
                    "mean_distance": float(distances.mean()),
                    "max_distance": float(distances.max()),
                }
            )
    result = spec | {
        "discovery_qualifying": len(episodes),
        "truncated": sum(not r["available"] for r in episodes),
        "uninformative": sum(r["available"] and not r["informative"] for r in episodes),
        "sampled": len(rows) + len(zero_ids),
        "sample_band_counts": {
            str(b): sum(r["rating"] // 200 * 200 == b for r in rows)
            for b in sorted({r["rating"] // 200 * 200 for r in rows})
        },
        "standardized_zero_ids": zero_ids,
        "scaling": "identity for L2-normalized TF-IDF"
        if semantic
        else "discovery standard deviations, without centering",
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "clusters": groups,
    }
    atomic_json(path, result)
    atomic_json(run / "active-clusters.json", {"path": path.name})
    return path


def assign(
    row: dict[str, Any], model: dict[str, Any], candidates: list[dict[str, Any]]
) -> tuple[str | None, float | None]:
    """Exactly one nearest family, or explicit rejection. Never reads rating/cost."""
    if not row["available"] or not row["informative"] or not candidates:
        return None, None
    vector = (np.array(row["vector"]) - model["scaler_mean"]) / model["scaler_scale"]
    if np.linalg.norm(vector) <= 1e-12:
        return None, None
    routing = model.get("routing_centroids", candidates)
    distances = cosine_distances(
        vector.reshape(1, -1), np.array([c["centroid"] for c in routing])
    ).ravel()
    index = int(np.argmin(distances))
    identifier = routing[index]["cluster_id"]
    candidate = next((c for c in candidates if c["cluster_id"] == identifier), None)
    d = float(distances[index])
    return (identifier if candidate is not None and d <= candidate["radius"] else None), d


def context(row: dict[str, Any]) -> tuple[str, str]:
    """Broad context for concentration flags, not discovery features."""
    board = replay(row)
    material = ",".join(
        str(len(board.pieces(kind, color)))
        for color in [board.turn, not board.turn]
        for kind in range(1, 6)
    )
    return " ".join(row["history"][:6]), material


def concentration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not rows:
        return result
    for label, values in [
        ("opening_prefix", [context(r)[0] for r in rows]),
        ("material_configuration", [context(r)[1] for r in rows]),
        ("player", [r["player"] for r in rows]),
        ("game", [r["game_id"] for r in rows]),
    ]:
        counts = {v: values.count(v) for v in set(values)}
        result[label] = {
            "distinct": len(counts),
            "largest_share": max(counts.values()) / len(rows),
            "concentrated": max(counts.values()) / len(rows) > 0.5,
        }
    return result


def example_game(row: dict[str, Any], label: str) -> chess.pgn.Game:
    game = chess.pgn.Game()
    game.setup(chess.Board(row["initial_fen"]))
    game.headers["Site"] = f"https://lichess.org/{row['game_id']}"
    game.headers["Annotator"] = "chess-research trajectory review"
    node: chess.pgn.GameNode = game
    for uci in row["history"]:
        node = node.add_variation(chess.Move.from_uci(uci))
    node.comment = (
        f"{label}; episode={row['decision_id']}; moving-player POV; "
        f"before cp={row['before_cp']} mate={row['before_mate']}; "
        f"after cp={row['after_cp']} mate={row['after_mate']}; loss={row['loss']:.4f}"
    )
    for name, line in [
        ("Played", [row["move"]] + row["after_pv"]),
        ("Preferred", row["before_pv"]),
    ]:
        variation = node
        board = replay(row)
        for number, uci in enumerate(line):
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError(f"Illegal review PV: {row['decision_id']}")
            variation = variation.add_variation(move)
            if number == 0:
                variation.comment = name
            board.push(move)
    return game


def review_packets(run: Path, partition: str = "discovery") -> Path:
    if partition not in {"discovery", "selection"}:
        raise ValueError("Review packets are limited to discovery and selection")
    path, model = active_clusters(run)
    target = run / "inspection" / path.stem
    target.mkdir(parents=True, exist_ok=True)
    episodes = {r["decision_id"]: r for r in read_table(run / "episodes.parquet")}
    ids = set(load_json(run / "partitions.json")[partition])
    candidates = [c for c in model["clusters"] if supported(c)]
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = {}
    rejected = 0
    if partition == "discovery":
        for group in model["clusters"]:
            rows = [episodes[i] for i in group["episode_ids"]]
            for row in rows:
                # Own-source membership, even when a centroid radius rejects a boundary point.
                v = (np.array(row["vector"]) - model["scaler_mean"]) / model["scaler_scale"]
                d = float(cosine_distances(v.reshape(1, -1), [group["centroid"]])[0, 0])
                grouped.setdefault(group["cluster_id"], []).append((row, d))
    else:
        for row in episodes.values():
            if row["decision_id"] in ids:
                assigned, assignment_distance = assign(row, model, candidates)
                if assigned:
                    assert assignment_distance is not None
                    grouped.setdefault(assigned, []).append((row, assignment_distance))
                else:
                    rejected += 1
    descriptions = {}
    if (run / "descriptions.json").exists():
        payload = load_json(run / "descriptions.json")
        descriptions = {r["decision_id"]: r for r in payload["records"]}
    packets, games = [], []
    for group in model["clusters"]:
        members = sorted(grouped.get(group["cluster_id"], []), key=lambda pair: pair[1])
        if not members:
            continue
        # Central and farthest examples, then fill across distinct players.
        ordering = members[:4] + list(reversed(members[-4:])) + members
        chosen: list[tuple[dict[str, Any], float]] = []
        seen_players: set[str] = set()
        for row, d in ordering:
            if row["player"] not in seen_players:
                seen_players.add(row["player"])
                chosen.append((row, d))
            if len(chosen) >= 12:
                break
        examples = []
        for row, d in chosen:
            board = replay(row)
            human = board.san(chess.Move.from_uci(row["move"]))
            alternatives = []
            for name, line in [
                ("played", [row["move"]] + row["after_pv"]),
                ("preferred", row["before_pv"]),
            ]:
                b = board.copy()
                sans = []
                for uci in line[:12]:
                    move = chess.Move.from_uci(uci)
                    sans.append(b.san(move))
                    b.push(move)
                alternatives.append((name, " ".join(sans)))
            examples.append(
                {
                    "decision_id": row["decision_id"],
                    "fen": board.fen(),
                    "human_san": human,
                    "description": descriptions.get(row["decision_id"]),
                    "distance": d,
                    "variations": dict(alternatives),
                    "before_cp": row["before_cp"],
                    "before_mate": row["before_mate"],
                    "after_cp": row["after_cp"],
                    "after_mate": row["after_mate"],
                    "loss": row["loss"],
                }
            )
            games.append(str(example_game(row, group["cluster_id"])))
        packets.append(
            {
                "cluster_id": group["cluster_id"],
                "support": len(members),
                "eligible_for_review_acceptance": supported(group),
                "source_support": group["support"],
                "source_players": group["unique_players"],
                "description_author_counts": group.get("description_author_counts", {}),
                "context": concentration([r for r, _ in members]),
                "examples": examples,
            }
        )
    atomic_json(
        target / f"{partition}.json",
        {
            "clusters_sha256": digest(path),
            "partition": partition,
            "unclassified": rejected,
            "packets": packets,
        },
    )
    if partition == "discovery":
        template = {
            "clusters_sha256": digest(path),
            "reviewer": "",
            "candidates": [
                {
                    "cluster_id": c["cluster_id"],
                    "accepted": False,
                    "coherent": False,
                    "name": "",
                    "supporting_episode_ids": [],
                    "contradictions": [],
                    "reason": "Pending review",
                }
                for c in model["clusters"]
            ],
        }
        template_name = "review.template.json"
    else:
        template = {
            "clusters_sha256": digest(path),
            "reviewer": "",
            "assignments": [
                {
                    "decision_id": e["decision_id"],
                    "cluster_id": p["cluster_id"],
                    "correct": None,
                    "reason": "Pending review",
                }
                for p in packets
                for e in p["examples"]
            ],
        }
        template_name = "selection-review.template.json"
    # Re-exporting packets must not erase a reviewer who filled in a template.
    if not (target / template_name).exists():
        atomic_json(target / template_name, template)
    (target / f"{partition}.pgn").write_text("\n\n".join(games) + "\n")
    lines = [
        f"# {partition.title()} trajectory review",
        "",
        "Ratings and player identities are omitted from review examples.",
        "",
    ]
    for p in packets:
        lines += [
            f"## {p['cluster_id']}",
            "",
            f"Support: {p['support']}; source players: {p['source_players']}; eligible: {p['eligible_for_review_acceptance']}.",
            "",
            f"Context: {p['context']}; description authors: {p['description_author_counts']}",
            "",
        ]
        for e in p["examples"]:
            lines += [
                f"- Description: {(e.get('description') or {}).get('raw_explanation', 'numeric trajectory')}. `{e['decision_id']}`: **{e['human_san']}**; distance {e['distance']:.3f}; loss {e['loss']:.3f}. Played: {e['variations']['played']}. Preferred: {e['variations']['preferred']}. FEN: `{e['fen']}`."
            ]
        lines.append("")
    (target / f"{partition}.md").write_text("\n".join(lines) + "\n")
    return target


def freeze(run: Path, review: Path, selection_review: Path) -> Path:
    target = run / "taxonomy.json"
    if target.exists():
        raise ValueError("Taxonomy already frozen")
    path, model = active_clusters(run)
    source = load_json(review)
    audit = load_json(selection_review)
    if any(r.get("clusters_sha256") != digest(path) for r in [source, audit]):
        raise ValueError("Review does not match active discovery artifact")
    if not source.get("reviewer") or not audit.get("reviewer"):
        raise ValueError("Both reviews require reviewer provenance")
    groups = {c["cluster_id"]: c for c in model["clusters"]}
    episodes = {r["decision_id"]: r for r in read_table(run / "episodes.parquet")}
    selection_ids = set(load_json(run / "partitions.json")["selection"])
    provisional = [c for c in groups.values() if supported(c)]
    audited: dict[str, list[tuple[bool, float]]] = {}
    seen: set[str] = set()
    for entry in audit["assignments"]:
        identifier = entry["decision_id"]
        if identifier in seen or identifier not in selection_ids or identifier not in episodes:
            raise ValueError("Selection audit needs unique selection episode IDs")
        seen.add(identifier)
        actual, d = assign(episodes[identifier], model, provisional)
        if (
            actual is None
            or d is None
            or actual != entry["cluster_id"]
            or type(entry["correct"]) is not bool
        ):
            raise ValueError(
                "Audit must review actual provisional assignments with boolean verdicts"
            )
        audited.setdefault(actual, []).append((entry["correct"], float(d)))
    candidates = []
    seen_groups: set[str] = set()
    for entry in source["candidates"]:
        identifier = entry["cluster_id"]
        if identifier not in groups or identifier in seen_groups:
            raise ValueError("Unknown or duplicate reviewed cluster")
        seen_groups.add(identifier)
        if type(entry["accepted"]) is not bool or type(entry["coherent"]) is not bool:
            raise ValueError("Review requires boolean acceptance/coherence verdicts")
        if not entry.get("reason", "").strip() or entry["reason"].strip() == "Pending review":
            raise ValueError("Every cluster requires a completed review rationale")
        group = groups[identifier]
        examples = entry["supporting_episode_ids"]
        if not set(examples) <= set(group["episode_ids"]):
            raise ValueError("Naming evidence must come from the source discovery cluster")
        if not entry["accepted"]:
            continue
        if (
            not supported(group)
            or not entry["coherent"]
            or not entry["name"]
            or len(set(examples)) < 3
        ):
            raise ValueError(
                "Accepted family needs support, coherence, name, and three source examples"
            )
        judgments = audited.get(identifier, [])
        positives = [d for correct, d in judgments if correct]
        negatives = [d for correct, d in judgments if not correct]
        if len(positives) < 5 or len(positives) / len(judgments) < 0.8:
            raise ValueError(
                "Accepted family needs five correct selection audits and >=80% audit accuracy"
            )
        radius = min(group["radius"], max(0.05, max(positives)))
        if negatives:
            radius = min(radius, max(0.0, min(negatives) - 1e-9))
        if sum(d <= radius for d in positives) < 5:
            raise ValueError(
                "Calibrated radius retains fewer than five correct reviewed assignments"
            )
        candidates.append(
            group
            | {
                "family_id": f"family-{identifier}",
                "name": entry["name"],
                "radius": radius,
                "review": entry,
                "selection_audit": {
                    "reviewed": len(judgments),
                    "correct": len(positives),
                    "radius": radius,
                },
            }
        )
    if seen_groups != set(groups):
        raise ValueError("Every discovery cluster needs an explicit accept/reject review")
    # Deterministically regenerate the central/boundary audit sample. A reviewer
    # may add judgments, but cannot omit difficult packet examples for acceptance.
    packet_dir = review_packets(run, "selection")
    packet = load_json(packet_dir / "selection.json")
    accepted_ids = {c["cluster_id"] for c in candidates}
    required = {
        e["decision_id"]
        for p in packet["packets"]
        if p["cluster_id"] in accepted_ids
        for e in p["examples"]
    }
    if not required <= seen:
        raise ValueError("Accepted families require review of every selection packet example")
    frozen = {
        "version": 1,
        "clusters_path": path.name,
        "clusters_sha256": digest(path),
        "inputs": model["inputs"],
        "scaler_mean": model["scaler_mean"],
        "scaler_scale": model["scaler_scale"],
        "candidates": candidates,
        "routing_centroids": provisional,
        "classification": "nearest supported discovery centroid; accept only reviewed families within calibrated cosine radius; otherwise unclassified",
        "reviews": {"discovery": source, "selection": audit},
        "review_sha256": digest(review),
        "selection_review_sha256": digest(selection_review),
    }
    frozen["taxonomy_id"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    atomic_json(target, frozen)
    atomic_json(run / "taxonomy.complete.json", {"taxonomy_sha256": digest(target)})
    return target


def interval(values: np.ndarray) -> list[float] | None:
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None


def engine_audit(run: Path, count: int = 20, nodes: int = 1_000_000) -> Path:
    """Discovery-only sensitivity audit; never replaces frozen cohort estimates."""
    if count < 1 or nodes < 1:
        raise ValueError("Audit count and nodes must be positive")
    if (run / "taxonomy.json").exists():
        raise ValueError("Audit engine sensitivity before freezing taxonomy")
    target = run / "engine-audit.json"
    if target.exists():
        raise ValueError("Engine audit already exists; preserve the recorded experiment")
    ids = set(load_json(run / "partitions.json")["discovery"])
    rows = stratified_sample(
        [r for r in read_table(run / "episodes.parquet") if r["decision_id"] in ids],
        count,
        load_json(run / "config.json")["seed"],
    )
    original = {r["decision_id"]: r for r in load_analysis(run)}
    selected = {r["decision_id"] for r in rows}
    findings = []
    if selected:
        path = asyncio.run(analyse(run, nodes, ids=selected))
        deeper = {r["decision_id"]: r for p in path.glob("batch-*.parquet") for r in read_table(p)}
        threshold = load_json(run / "episodes.config.json")["loss_threshold"]
        for identifier in sorted(selected):
            old, new = original[identifier], deeper[identifier]
            delta = new["loss"] - old["loss"]
            findings.append(
                {
                    "decision_id": identifier,
                    "original_loss": old["loss"],
                    "deep_loss": new["loss"],
                    "delta": delta,
                    "gate_changed": (old["loss"] >= threshold) != (new["loss"] >= threshold),
                    "unstable": abs(delta) > 0.05 or new["loss"] < threshold,
                    "before_pv": new["before_pv"],
                    "after_pv": new["after_pv"],
                }
            )
        deep_config = load_json(path / "config.json")
    else:
        deep_config = None
    atomic_json(
        target,
        {
            "nodes": nodes,
            "count": len(findings),
            "analysis_config": deep_config,
            "sample": "seeded rating-stratified discovery-only qualifying episodes",
            "unstable_count": sum(f["unstable"] for f in findings),
            "max_absolute_delta": max((abs(f["delta"]) for f in findings), default=None),
            "episodes": findings,
        },
    )
    return target


def measure(run: Path, bootstrap: int = 1000, taxonomy_run: Path | None = None) -> Path:
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    output = run / "measurement.json"
    if output.exists():
        raise ValueError("Evaluation already measured; do not revise taxonomy against this holdout")
    source = taxonomy_run or run
    taxonomy = load_json(source / "taxonomy.json")
    if (
        digest(source / "taxonomy.json")
        != load_json(source / "taxonomy.complete.json")["taxonomy_sha256"]
    ):
        raise ValueError("Frozen taxonomy changed")
    for filename, expected in taxonomy["inputs"].items():
        if digest(source / filename) != expected:
            raise ValueError(f"Frozen input changed: {filename}")
    if digest(source / taxonomy["clusters_path"]) != taxonomy["clusters_sha256"]:
        raise ValueError("Frozen discovery artifact changed")
    semantic = (source / "embedding-model.json").exists()
    if semantic:
        _validate_semantic_artifacts(source)
        if source.resolve() != run.resolve():
            _validate_semantic_artifacts(run)
    config = load_json(run / "config.json")
    ids = set(load_json(run / "partitions.json")["evaluation"])
    decisions = [r for r in read_table(run / "decisions.parquet") if r["decision_id"] in ids]
    episodes = {
        r["decision_id"]: r for r in read_table(run / "episodes.parquet") if r["decision_id"] in ids
    }
    if source.resolve() != run.resolve():
        source_config = load_json(source / "episodes.config.json")
        current_config = load_json(run / "episodes.config.json")
        for key in [
            "feature_names",
            "horizon",
            "loss_threshold",
            "encoding",
            "stable_gate",
            "embedding_model_sha256",
            "description_prompt_version",
        ]:
            if source_config.get(key) != current_config.get(key):
                raise ValueError(f"External evaluation encoding differs: {key}")
        original_engine = load_json(
            source / load_json(source / "active-analysis.json")["path"] / "config.json"
        )
        current_engine = load_json(
            run / load_json(run / "active-analysis.json")["path"] / "config.json"
        )
        for key in [
            "nodes",
            "engine_sha256",
            "threads",
            "hash_mib",
            "chess_version",
            "perspective",
            "expected_score_model",
        ]:
            if original_engine.get(key) != current_engine.get(key):
                raise ValueError(f"External evaluation engine differs: {key}")
        source_decisions = read_table(source / "decisions.parquet")
        if {r["player"] for r in source_decisions} & {r["player"] for r in decisions}:
            raise ValueError("External evaluation repeats source players")
        if {r["position"] for r in source_decisions} & {r["position"] for r in decisions}:
            raise ValueError("External evaluation repeats source positions")
    assigned = {i: assign(e, taxonomy, taxonomy["candidates"])[0] for i, e in episodes.items()}
    bands = list(range(config["cohort"][0] // 200 * 200, config["cohort"][1] + 1, 200))
    players = sorted({r["player"] for r in decisions})
    player_indices = {p: i for i, p in enumerate(players)}
    denominators = np.zeros((len(players), len(bands)))
    qualifying = np.zeros_like(denominators)
    qualifying_costs = np.zeros_like(denominators)
    for row in decisions:
        b = (row["rating"] - bands[0]) // 200
        p = player_indices[row["player"]]
        denominators[p, b] += 1
        qualifying[p, b] += row["decision_id"] in episodes
        if row["decision_id"] in episodes:
            qualifying_costs[p, b] += episodes[row["decision_id"]]["loss"]
    rng = np.random.default_rng(config["seed"])
    weights = (
        rng.multinomial(len(players), np.full(len(players), 1 / len(players)), size=bootstrap)
        if players
        else np.zeros((bootstrap, 0))
    )
    boot_den = weights @ denominators
    with np.errstate(divide="ignore", invalid="ignore"):
        boot_qualifying_cost = weights @ qualifying_costs / boot_den * 100
        boot_qualifying_rate = weights @ qualifying / boot_den * 100
    total_den = denominators.sum(axis=0)
    families = []
    for candidate in taxonomy["candidates"]:
        incidents, costs = np.zeros_like(denominators), np.zeros_like(denominators)
        hits = []
        for row in decisions:
            identifier = row["decision_id"]
            if assigned.get(identifier) == candidate["cluster_id"]:
                b = (row["rating"] - bands[0]) // 200
                p = player_indices[row["player"]]
                incidents[p, b] += 1
                costs[p, b] += episodes[identifier]["loss"]
                hits.append(episodes[identifier])
        with np.errstate(divide="ignore", invalid="ignore"):
            boot_rate = weights @ incidents / boot_den * 100
            boot_cost = weights @ costs / boot_den * 100
        band_results: list[dict[str, Any]] = []
        for j, band in enumerate(bands):
            band_hits = [r for r in hits if band <= r["rating"] < band + 200]
            losses = [r["loss"] for r in band_hits]
            q = qualifying[:, j].sum()
            n = total_den[j]
            cost = float(costs[:, j].sum() / n * 100) if n else None
            band_results.append(
                {
                    "band": f"{max(band, config['cohort'][0])}–{min(band + 199, config['cohort'][1])}",
                    "decisions": int(n),
                    "incidents": len(band_hits),
                    "incidents_per_100_decisions": len(band_hits) / n * 100 if n else None,
                    "cost_per_100_decisions": cost,
                    "rate_ci": interval(boot_rate[:, j]),
                    "cost_ci": interval(boot_cost[:, j]),
                    "qualifying_mistake_share": len(band_hits) / q if q else None,
                    "mean_loss": float(np.mean(losses)) if losses else None,
                    "median_loss": float(np.median(losses)) if losses else None,
                    "unique_players": len({r["player"] for r in band_hits}),
                    "unique_positions": len({r["position"] for r in band_hits}),
                    "sparse": len({r["player"] for r in band_hits}) < 20,
                    "context": concentration(band_hits),
                }
            )
        changes = []
        for j in range(len(bands) - 1):
            a = band_results[j]["cost_per_100_decisions"]
            b = band_results[j + 1]["cost_per_100_decisions"]
            changes.append(
                {
                    "from": band_results[j]["band"],
                    "to": band_results[j + 1]["band"],
                    "absolute_cost_reduction": a - b if a is not None and b is not None else None,
                    "relative_cost_reduction": (a - b) / a if a and b is not None else None,
                    "absolute_reduction_ci": interval(boot_cost[:, j] - boot_cost[:, j + 1]),
                    "absolute_incident_reduction": (
                        band_results[j]["incidents_per_100_decisions"]
                        - band_results[j + 1]["incidents_per_100_decisions"]
                        if total_den[j] and total_den[j + 1]
                        else None
                    ),
                    "absolute_incident_reduction_ci": interval(
                        boot_rate[:, j] - boot_rate[:, j + 1]
                    ),
                }
            )
        families.append(
            {
                "family_id": candidate["family_id"],
                "name": candidate["name"],
                "bands": band_results,
                "adjacent_changes": changes,
            }
        )
    coverage = []
    for j, band in enumerate(bands):
        eps = [e for e in episodes.values() if band <= e["rating"] < band + 200]
        n = int(total_den[j])
        classified_count = sum(assigned[e["decision_id"]] is not None for e in eps)
        coverage.append(
            {
                "band": f"{max(band, config['cohort'][0])}–{min(band + 199, config['cohort'][1])}",
                "decisions": n,
                "players": int(np.sum(denominators[:, j] > 0)),
                "sparse": int(np.sum(denominators[:, j] > 0)) < MIN_PLAYERS,
                "qualifying_mistakes": len(eps),
                "classified": classified_count,
                "assignment_coverage": classified_count / len(eps) if eps else None,
                "truncated": sum(not e["available"] for e in eps),
                "uninformative": sum(e["available"] and not e["informative"] for e in eps),
                "qualifying_incidents_per_100_decisions": len(eps) / n * 100 if n else None,
                "qualifying_cost_per_100_decisions": sum(e["loss"] for e in eps) / n * 100
                if n
                else None,
                "qualifying_rate_ci": interval(boot_qualifying_rate[:, j]),
                "qualifying_cost_ci": interval(boot_qualifying_cost[:, j]),
            }
        )
    stability_summary = None
    if (run / "stability.json").exists():
        ledger = load_json(run / "stability.json")
        screened = [r for r in ledger["decisions"] if r["decision_id"] in ids]
        reasons: dict[str, int] = {}
        for row in screened:
            for reason in row["rejection_reasons"]:
                reasons[reason] = reasons.get(reason, 0) + 1
        stability_summary = {
            "gate": ledger["config"]["stable_gate"],
            "baseline_candidates": len(screened),
            "confirmed": sum(r["stable"] for r in screened),
            "rejections": reasons,
            "ledger_sha256": digest(run / "stability.json"),
        }
    result = {
        "version": 1,
        "stability": stability_summary,
        "taxonomy_id": taxonomy["taxonomy_id"],
        "taxonomy_sha256": digest(source / "taxonomy.json"),
        "taxonomy_run": str(source.resolve()),
        "evaluation_inputs": {
            name: digest(run / name)
            for name in [
                "decisions.parquet",
                "episodes.parquet",
                "episodes.config.json",
                "partitions.json",
                "config.json",
            ]
        },
        "bootstrap": bootstrap,
        "seed": config["seed"],
        "coverage": coverage,
        "families": families,
        "denominator": "All sampled eligible evaluation decisions after historical and cross-partition position exclusions",
        "scope": "Conditional on the recorded archive prefix, rating strata, player cap, and qualifying-loss gate; not population prevalence or a causal developmental result",
    }
    write_report(run, taxonomy, result, source)
    atomic_json(output, result)
    return output


def write_report(
    run: Path, taxonomy: dict[str, Any], result: dict[str, Any], discovery_run: Path | None = None
) -> None:
    source = discovery_run or run
    path, model = active_clusters(source)
    prefix = os.path.relpath(source.resolve(), run.resolve())
    config = load_json(run / "config.json")
    partitions = load_json(run / "partitions.json")
    engine = load_json(run / load_json(run / "active-analysis.json")["path"] / "config.json")
    lines = [
        "# Stable-blunder semantic discovery proof"
        if (source / "embedding-model.json").exists()
        else "# Engine-trajectory discovery proof",
        "",
        f"Source: `{config['archive']}`; seed {config['seed']}; sampled {config['sampled']} decisions. {config['scanned_games']} prefix games scanned; player cap {config['player_cap']}.",
        "",
        f"Partitions: { {p: len(v) for p, v in partitions.items()} }. Historical exclusions: `{config.get('prior_run')}`. Moving-player splits, with earlier positions excluded from later partitions.",
        "",
        f"Engine: `{engine}`. No repository model calls, named tactical detectors, or rating features. External descriptions and discovery-only TF-IDF are used when recorded in the encoding.",
        "",
        f"Discovery: {model['sampled']} episodes; rating strata {model['sample_band_counts']}; {len(model['clusters'])} clusters; {len(taxonomy['candidates'])} accepted families. {model['truncated']} qualifying episodes unavailable for encoding and {model['uninformative']} uninformative before sampling.",
        "",
        f"Encoding: {load_json(run / 'episodes.config.json')['encoding']}; {len(load_json(run / 'episodes.config.json')['feature_names'])} dimensions; horizon {load_json(run / 'episodes.config.json')['horizon']}; expected-score loss gate {load_json(run / 'episodes.config.json')['loss_threshold']}. [Full schema and provenance](episodes.config.json).",
        "",
        f"Clustering: {model['method']}; threshold {model['distance']}; scaling {model.get('scaling', 'discovery standard deviations, without centering')}. Frozen taxonomy `{taxonomy['taxonomy_id']}`. Selection audits are purposive quality checks, not unbiased accuracy estimates.",
        "",
        f"Discovery source: `{source.resolve()}`. Inspection: [discovery]({prefix}/inspection/{path.stem}/discovery.md), [selection]({prefix}/inspection/{path.stem}/selection.md), and paired PGNs in the same directory.",
        "",
        "## Holdout coverage",
        "",
        "| Rating | Decisions | Players | Qualifying mistakes | Classified | Coverage | Unavailable descriptions/traces | Qualifying mistakes/100 decisions | 95% rate interval | Qualifying cost/100 decisions | 95% cost interval |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    if result.get("stability"):
        gate = result["stability"]
        lines[2:2] = [
            f"Stable gate: `{gate['gate']}`. Baseline candidates {gate['baseline_candidates']}; confirmed {gate['confirmed']}; rejection reasons `{gate['rejections']}`.",
            "",
            "TF-IDF measures lexical similarity, not neural semantic similarity. Descriptions are externally authored, position-grounded, and rating blind. Undescribed or rejected errors remain unclassified. Rating differences are observational associations, not evidence of learning or teachability.",
            "",
        ]
    if (source / "descriptions.json").exists():
        source_descriptions = load_json(source / "descriptions.json")
        evaluation_descriptions = load_json(run / "descriptions.json")
        lines[2:2] = [
            f"Description prompt: `{source_descriptions.get('prompt_version', 'unrecorded')}`; discovery/selection provider: {source_descriptions.get('provider', 'unrecorded')}. Evaluation provider: {evaluation_descriptions.get('provider', 'unrecorded')}. Generated descriptions and prompt hashes are retained in the linked JSON artifacts.",
            "",
        ]
    for b in result["coverage"]:
        value = b["assignment_coverage"]
        coverage = f"{100 * value:.1f}%" if value is not None else "NA"
        cost = b["qualifying_cost_per_100_decisions"]
        cost_text = f"{cost:.2f}" if cost is not None else "NA"
        rate = b["qualifying_incidents_per_100_decisions"]
        rate_text = f"{rate:.2f}" if rate is not None else "NA"
        rate_ci = (
            "–".join(f"{v:.2f}" for v in b["qualifying_rate_ci"])
            if b["qualifying_rate_ci"]
            else "NA"
        )
        cost_ci = (
            "–".join(f"{v:.2f}" for v in b["qualifying_cost_ci"])
            if b["qualifying_cost_ci"]
            else "NA"
        )
        lines.append(
            f"| {b['band']} | {b['decisions']} | {b['players']} | {b['qualifying_mistakes']} | {b['classified']} | {coverage} | {b['truncated']} | {rate_text} | {rate_ci} | {cost_text} | {cost_ci} |"
        )
    lines += [
        "",
        "Bands with fewer than 20 evaluation players are sparse; missing bands remain NA. Intervals are player-bootstrap estimates, not a validation of family coherence.",
        "",
    ]
    for family in result["families"]:
        lines += [
            "",
            f"## {family['name']}",
            "",
            "| Rating | Incidents/100 decisions | Cost/100 decisions | 95% cost interval | Players | Sparse |",
            "|---|---:|---:|---|---:|---|",
        ]
        for b in family["bands"]:
            lines.append(
                f"| {b['band']} | {b['incidents_per_100_decisions']} | {b['cost_per_100_decisions']} | {b['cost_ci']} | {b['unique_players']} | {b['sparse']} |"
            )
        lines += ["", f"Adjacent-band changes: `{family['adjacent_changes']}`", ""]
    lines += ["", "## Review outcome", ""]
    for entry in taxonomy["reviews"]["discovery"]["candidates"]:
        lines.append(
            f"- {entry['cluster_id']}: {'accepted' if entry['accepted'] else 'rejected'} — {entry['reason']}"
        )
    if not taxonomy["candidates"]:
        lines += [
            "",
            "**Inconclusive: no cluster passed coherence, support, and assignment review. Zero classified incidents reflect an empty taxonomy, not absence of chess errors.**",
        ]
    lines += [
        "",
        "## Limitations",
        "",
        result["scope"] + ".",
        "",
        "Expected-score gating uses Stockfish's WDL model, not measured low-rated player win probabilities, and saturates in already won/lost positions. Search budget and PV length affect eligibility; stability and explanation rejection can introduce nonrandom missingness. TF-IDF can fragment synonymous errors or group wording rather than mechanisms. Context checks use opening prefixes and material counts, not an authored skill taxonomy. Low assignment coverage limits family comparisons. No player-learning or teaching-effectiveness claim follows."
        if (source / "embedding-model.json").exists()
        else "Expected-score gating saturates in already won/lost positions; search budget and PV length affect eligibility. Material/event trajectories can group consequences while missing decision mechanisms or quiet positional errors. Context checks use opening prefixes and material counts, not an authored skill taxonomy. Low assignment coverage limits family comparisons. No player-learning or teaching-effectiveness claim follows.",
        "",
    ]
    sensitivity = source / "engine-audit.json"
    if sensitivity.exists():
        audit = load_json(sensitivity)
        lines += [
            f"[Engine sensitivity audit]({prefix}/engine-audit.json): {audit['count']} discovery episodes at {audit['nodes']} nodes; {audit['unstable_count']} unstable; maximum absolute loss change {audit['max_absolute_delta']}.",
            "",
        ]
    lines += ["![Rating cost curves](rating-costs.png)", ""]
    (run / "trajectory-report.md").write_text("\n".join(lines))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4))
    coordinates = np.arange(len(result["coverage"]))

    def plot_cost(rows: list[dict[str, Any]], field: str, ci_field: str, label: str) -> None:
        values = np.array([r[field] if r[field] is not None else np.nan for r in rows])
        lower = [
            max(0, v - r[ci_field][0]) if r[ci_field] else np.nan
            for v, r in zip(values, rows, strict=True)
        ]
        upper = [
            max(0, r[ci_field][1] - v) if r[ci_field] else np.nan
            for v, r in zip(values, rows, strict=True)
        ]
        axis.errorbar(coordinates, values, yerr=[lower, upper], marker="o", capsize=4, label=label)

    if result["families"]:
        for family in result["families"]:
            plot_cost(family["bands"], "cost_per_100_decisions", "cost_ci", family["name"])
    else:
        plot_cost(
            result["coverage"],
            "qualifying_cost_per_100_decisions",
            "qualifying_cost_ci",
            "All qualifying mistakes (no accepted families)",
        )
    axis.legend(fontsize="small")
    axis.set_xticks(coordinates, [b["band"] for b in result["coverage"]])
    axis.set_ylim(bottom=0)
    axis.set_ylabel("Expected-score loss\nper 100 sampled decisions")
    axis.set_xlabel("Moving-player Lichess rapid rating")
    figure.tight_layout()
    figure.savefig(run / "rating-costs.png", dpi=160)
    plt.close(figure)
