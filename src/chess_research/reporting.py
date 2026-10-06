"""Saved findings, deeper example analysis, and annotated PGN export."""

import asyncio
import json
from pathlib import Path

import chess
import chess.pgn

from .analysis import analyse, load_analysis, replay
from .records import atomic_json, read_table


def report(run: Path) -> None:
    findings = json.loads((run / "findings.json").read_text())
    config = json.loads((run / "config.json").read_text())
    active = json.loads((run / "active-analysis.json").read_text())["path"]
    engine_config = json.loads((run / active / "config.json").read_text())
    decisions = {r["decision_id"]: r for r in read_table(run / "decisions.parquet")}
    outcomes = {r["decision_id"]: r for r in load_analysis(run)}
    ids = {i for c in findings["candidates"] for i in c["examples"] + c["counterexamples"]}
    deeper = {}
    if ids:
        path = asyncio.run(analyse(run, 1_000_000, ids=ids))
        deeper = {r["decision_id"]: r for p in path.glob("batch-*.parquet") for r in read_table(p)}
    lines = [
        "# Beginner cohort discovery proof",
        "",
        "Question: Can automatically learned rules identify recurring costly decisions that make sense as potential lessons?",
        "",
        f"Sample: {config['sampled']} decisions; {config['scanned_games']} scanned games; seed {config['seed']}; cohort 400–799; at most ten decisions per player.",
        "",
        f"Source: `{config['archive']}`. Claims apply only to this sampled archive prefix.",
        "",
        f"Synthetic fixture: {config['fixture']}. Partitions after position exclusion: {findings['partitions']}.",
        "",
        findings["target"] + ". Mate scores remain separate from centipawns.",
        "",
        f"Engine configuration: {engine_config}. Model fitted: {findings['model_fitted']}.",
        "",
        "100 ExtraTrees, depth 3, minimum leaf 50. Player split 60/20/20; earlier positions excluded from later partitions. Selection ranks coverage × excess cost. Final evaluation uses frozen rules and 1,000 player-bootstrap replicates.",
        "",
        findings["conclusion"],
        "",
    ]
    supported = sum(bool(c["supported"]) for c in findings["candidates"])
    lines += [
        f"Final evidence: {supported} of {len(findings['candidates'])} frozen candidates met the planned threshold.",
        "",
    ]
    if supported == 0:
        lines += ["Result: inconclusive. No candidate lesson is established by this run.", ""]
    games = []
    sensitivity = []
    for c in findings["candidates"]:
        lines += [
            f"## {c['name']}",
            "",
            f"Rule: `{c['rule']}`",
            "",
            f"Discovery: {c['discovery']}",
            f"Selection: {c['selection']}",
            f"Evaluation: {c['evaluation']}",
            f"Evidence threshold met: {c['supported']}",
            "",
            "Piece codes: 1 pawn, 2 knight, 3 bishop, 4 rook, 5 queen, 6 king. Predicates count attack, protection, or legal-capture relationships; retained predicates preserve the same pieces.",
            "",
        ]
        for identifier in dict.fromkeys(c["examples"] + c["counterexamples"]):
            row = decisions[identifier]
            original = outcomes[identifier]
            deep = deeper[identifier]
            sensitivity.append(
                {
                    "candidate": c["id"],
                    "decision_id": identifier,
                    "original_loss": original["loss"],
                    "deep_loss": deep["loss"],
                    "delta": deep["loss"] - original["loss"],
                }
            )
            game = chess.pgn.Game()
            game.setup(chess.Board(row["initial_fen"]))
            game.headers["Site"] = f"https://lichess.org/{row['game_id']}"
            game.headers["Annotator"] = "chess-research"
            node: chess.pgn.GameNode = game
            for uci in row["history"]:
                node = node.add_variation(chess.Move.from_uci(uci))
            node.comment = f"Rule {c['id']}; moving player {row['player']}; rating {row['rating']}. Before score cp={deep['before_cp']} mate={deep['before_mate']}; PV {' '.join(deep['before_pv'])}"
            played = node.add_variation(chess.Move.from_uci(row["move"]))
            played.comment = f"Engine expected-score loss {deep['loss']:.4f}; after cp={deep['after_cp']} mate={deep['after_mate']}; PV {' '.join(deep['after_pv'])}"
            board = replay(row)
            alternate = node
            for uci in deep["before_pv"]:
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    break
                alternate = alternate.add_variation(move)
                board.push(move)
            games.append(str(game))
            lines.append(
                f"- {identifier}: loss {original['loss']:.4f} → {deep['loss']:.4f} at one million nodes; PV {' '.join(deep['before_pv'])}."
            )
        lines += [
            "",
            "Teachability review remains required: inspect the annotated variations before assigning a lesson name.",
            "",
        ]
    lines += [
        "This proof examines immediate relationships and may miss deeper mechanisms. It does not demonstrate learning benefits or comprehensive mistake rankings."
    ]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    (run / "examples.pgn").write_text("\n\n".join(games))
    atomic_json(run / "sensitivity.json", sensitivity)
    atomic_json(run / "report.complete.json", {"examples": len(games)})
