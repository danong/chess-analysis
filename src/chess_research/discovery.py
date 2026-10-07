"""Legacy symbolic experiment; active discovery lives in trajectories.py."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesRegressor

from .analysis import load_analysis
from .records import atomic_json, read_table
from .representation import describe_rule, feature_names
from .splits import split_rows


def extract_rules(
    model: ExtraTreesRegressor, names: list[str]
) -> list[list[tuple[str, str, float]]]:
    rules: dict[str, list[tuple[str, str, float]]] = {}
    for estimator in model.estimators_:
        tree = estimator.tree_

        def visit(node: int, bounds: dict[str, tuple[float, float]], tree: Any = tree) -> None:
            if tree.children_left[node] == tree.children_right[node]:
                rule = []
                for name, (low, high) in sorted(bounds.items()):
                    if np.isfinite(low):
                        rule.append((name, ">", low))
                    if np.isfinite(high):
                        rule.append((name, "<=", high))
                rules[json.dumps(rule)] = rule
                return
            name = names[tree.feature[node]]
            threshold = float(np.floor(tree.threshold[node]))
            low, high = bounds.get(name, (-np.inf, np.inf))
            visit(tree.children_left[node], bounds | {name: (low, min(high, threshold))})
            visit(tree.children_right[node], bounds | {name: (max(low, threshold), high)})

        visit(0, {})
    return list(rules.values())


def matches(rule: list, row: dict[str, Any]) -> bool:
    return all(row[n] > t if op == ">" else row[n] <= t for n, op, t in rule)


def evidence(rule: list, rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    hit = [r for r in rows if matches(rule, r)]
    miss = [r for r in rows if not matches(rule, r)]
    excess = (
        float(np.mean([r["loss"] for r in hit]) - np.mean([r["loss"] for r in miss]))
        if hit and miss
        else None
    )
    players = sorted({r["player"] for r in rows})
    grouped = {p: [r for r in rows if r["player"] == p] for p in players}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(1000 if excess is not None else 0):
        sampled = [r for p in rng.choice(players, len(players)) for r in grouped[p]]
        a = [r["loss"] for r in sampled if matches(rule, r)]
        b = [r["loss"] for r in sampled if not matches(rule, r)]
        if a and b:
            values.append(float(np.mean(a) - np.mean(b)))
    ci = np.quantile(values, [0.025, 0.975]).tolist() if values else None
    return {
        "support": len(hit),
        "distinct_players": len({r["player"] for r in hit}),
        "coverage": len(hit) / len(rows) if rows else 0,
        "excess": excess,
        "player_bootstrap_ci": ci,
        "mean_cost": float(np.mean([r["loss"] for r in hit])) if hit else None,
    }


def discover(run: Path) -> None:
    if (run / "findings.json").exists():
        raise ValueError("Findings already frozen; use a fresh run to avoid reusing evaluation")
    config = json.loads((run / "config.json").read_text())
    seed = config["seed"]
    decisions = read_table(run / "decisions.parquet")
    features = read_table(run / "features.parquet")
    outcomes = {r["decision_id"]: r for r in load_analysis(run)}
    by_id = {r["decision_id"]: r for r in features}
    if set(outcomes) != {r["decision_id"] for r in decisions}:
        raise ValueError("Complete analysis required")
    rows = [
        r | by_id[r["decision_id"]] | {"loss": outcomes[r["decision_id"]]["loss"]}
        for r in decisions
    ]
    names = [n for n in pq.read_schema(run / "features.parquet").names if n != "decision_id"]
    if names != feature_names():
        raise ValueError("Feature schema differs from elementary relationship allowlist")
    parts = split_rows(rows, seed)
    findings: dict[str, Any] = {
        "partitions": {k: len(v) for k, v in parts.items()},
        "excluded_repeated_positions": len(rows) - sum(map(len, parts.values())),
        "candidates": [],
        "target": "Engine-derived expected-score loss; signed before minus after, moving-player POV",
        "model": {"trees": 100, "max_depth": 3, "min_samples_leaf": 50, "seed": seed},
        "baseline": "nonmatching decisions in the same partition",
        "bootstrap_replicates": 1000,
    }
    atomic_json(
        run / "partitions.json", {k: [r["decision_id"] for r in v] for k, v in parts.items()}
    )
    train = parts["discovery"]
    findings["model_fitted"] = len(train) >= 100 and bool(names)
    if len(train) >= 100 and names:
        model = ExtraTreesRegressor(
            n_estimators=100, max_depth=3, min_samples_leaf=50, random_state=seed, n_jobs=1
        )
        model.fit([[r[n] for n in names] for r in train], [r["loss"] for r in train])
        ranked = []
        for rule in extract_rules(model, names):
            if not rule:
                continue
            d = evidence_fast(rule, train)
            s = evidence_fast(rule, parts["selection"])
            if (
                d["excess"] is not None
                and d["excess"] > 0
                and s["excess"] is not None
                and s["excess"] > 0
                and s["support"] >= 30
                and s["distinct_players"] >= 20
            ):
                ranked.append((s["coverage"] * s["excess"], rule))
        chosen = sorted(ranked, key=lambda item: item[0], reverse=True)[:3]
        atomic_json(run / "selected-rules.json", [rule for _, rule in chosen])
        for index, (_, rule) in enumerate(chosen, 1):
            ev = evidence(rule, parts["evaluation"], seed)
            hits = [r for r in parts["selection"] if matches(rule, r)]
            hits.sort(key=lambda r: r["loss"], reverse=True)
            diverse = []
            players: set[str] = set()
            ordered = [hits[int(i)] for i in np.linspace(0, len(hits) - 1, min(10, len(hits)))]
            for r in ordered + hits:
                if r["player"] not in players:
                    diverse.append(r["decision_id"])
                    players.add(r["player"])
                if len(diverse) == 10:
                    break
            findings["candidates"].append(
                {
                    "id": index,
                    "rule": rule,
                    "name": describe_rule(rule),
                    "discovery": evidence_fast(rule, train),
                    "selection": evidence(rule, parts["selection"], seed),
                    "evaluation": ev,
                    "supported": ev["support"] >= 30
                    and ev["distinct_players"] >= 20
                    and ev["excess"] is not None
                    and ev["excess"] > 0
                    and ev["player_bootstrap_ci"] is not None
                    and ev["player_bootstrap_ci"][0] > 0,
                    "examples": diverse,
                    "counterexamples": [
                        r["decision_id"] for r in sorted(hits, key=lambda r: r["loss"])[:3]
                    ],
                }
            )
    findings["conclusion"] = (
        "Candidates require example review; association does not establish teachability."
        if findings["candidates"]
        else "Inconclusive: no rules met selection requirements."
    )
    atomic_json(run / "findings.json", findings)
    atomic_json(run / "discover.complete.json", {"candidates": len(findings["candidates"])})


def evidence_fast(rule: list, rows: list[dict[str, Any]]) -> dict[str, Any]:
    hit = [r for r in rows if matches(rule, r)]
    miss = [r for r in rows if not matches(rule, r)]
    return {
        "support": len(hit),
        "distinct_players": len({r["player"] for r in hit}),
        "coverage": len(hit) / len(rows) if rows else 0,
        "excess": float(np.mean([r["loss"] for r in hit]) - np.mean([r["loss"] for r in miss]))
        if hit and miss
        else None,
    }
