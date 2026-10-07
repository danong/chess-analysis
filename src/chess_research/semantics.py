"""External error descriptions and discovery-only TF-IDF episode vectors.

Descriptions are authored outside the pipeline. This module only exports
evidence packets, validates returned records, and applies a frozen text model.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import chess
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

from .analysis import replay
from .episodes import EPISODES
from .records import atomic_json, read_table, write_table

PROMPT_VERSION = "stable-error-description-v1"
PROMPT = """Describe the concrete chess error shown by this decision.
Compare the played continuation with the engine-preferred continuation using only
the position, legal move sequences, and supplied engine scores. In a short phrase
or one sentence, state what the played choice does and what the better choice
preserves, creates, avoids, or exchanges. Be precise about compensation and
exchanges when the evidence supports them. Do not assign a chess category or
named tactical label, give advice, infer intent or psychology, or claim more than
the supplied lines show. If the position and lines do not support a concrete
description, set grounded=false and explain the evidence gap. Do not extend the
engine search. Return both a raw explanation that may refer to squares in this
example and a concise normalized formulation that describes the decision without
square names, move notation, ratings, opening labels, or player identity."""

_SQUARE = re.compile(r"(?<![a-z0-9])[a-h][1-8](?![a-z0-9])", re.IGNORECASE)
_UCI = re.compile(r"(?<![a-z0-9])[a-h][1-8][a-h][1-8][qrbn]?(?![a-z0-9])", re.IGNORECASE)
_SAN = re.compile(
    r"(?<![a-z0-9])(?:[KQRBN](?:[a-h1-8]{0,2})x?[a-h][1-8]|[a-h]x[a-h][1-8]|[a-h][1-8]|O-O(?:-O)?)(?:=[QRBN])?[+#]?(?![a-z0-9])"
)
_VECTORIZER_PARAMS = {
    "lowercase": True,
    "stop_words": "english",
    "ngram_range": [1, 2],
    "sublinear_tf": True,
    "max_features": 512,
    "min_df": 1,
    "strip_accents": "unicode",
    "norm": "l2",
    "use_idf": True,
    "smooth_idf": True,
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _san_line(board: chess.Board, pv: list[str], label: str) -> list[str]:
    result: list[str] = []
    current = board.copy()
    for uci in pv:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise ValueError(f"Invalid {label} move: {uci}") from exc
        if move not in current.legal_moves:
            raise ValueError(f"Illegal {label} move: {uci}")
        result.append(current.san(move))
        current.push(move)
    return result


def _episode_packet(row: dict[str, Any]) -> dict[str, Any]:
    board = replay(row)
    played_pv = [row["move"], *row["after_pv"]]
    played_san = _san_line(board, played_pv, "played PV")
    preferred_pv = list(row["before_pv"])
    preferred_san = _san_line(board, preferred_pv, "preferred PV")
    if not played_pv or not preferred_pv:
        raise ValueError(f"Missing played or preferred continuation: {row['decision_id']}")
    payload = {
        "decision_id": row["decision_id"],
        "fen": board.fen(),
        "side": "White" if board.turn else "Black",
        "human_uci": row["move"],
        "human_san": board.san(chess.Move.from_uci(row["move"])),
        "preferred_uci": row["preferred_move"],
        "played_san": played_san,
        "played_pv": played_pv,
        "preferred_san": preferred_san,
        "preferred_pv": preferred_pv,
        "before_cp": row["before_cp"],
        "before_mate": row["before_mate"],
        "after_cp": row["after_cp"],
        "after_mate": row["after_mate"],
        "loss": row["loss"],
    }
    return payload | {"episode_sha256": _canonical_hash(payload)}


def description_packets(run: Path, partition: str = "discovery") -> Path:
    """Export replay-verified, identity-free examples for external description."""
    if partition not in {"discovery", "selection", "evaluation"}:
        raise ValueError("Partition must be discovery, selection, or evaluation")
    _require_open_semantics_run(run)
    rows = read_table(run / "episodes.parquet")
    packets = [
        _episode_packet(row)
        for row in sorted(rows, key=lambda item: item["decision_id"])
        if row["partition"] == partition
    ]
    value = {
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "prompt": PROMPT,
        "partition": partition,
        "episodes": packets,
    }
    target = run / "description-packets" / f"{partition}.json"
    _write_immutable_json(target, value, "Description packet")
    return target


def _require_open_semantics_run(run: Path) -> None:
    if (run / "active-clusters.json").exists() or (run / "taxonomy.json").exists():
        raise ValueError("Cannot export or import descriptions after clusters or taxonomy exist")


def _write_immutable_json(path: Path, value: Any, label: str) -> None:
    if path.exists():
        old = json.loads(path.read_text())
        if old != value:
            raise ValueError(f"{label} already exists with different content")
        return
    atomic_json(path, value)


def _clean_normalized(text: str) -> str:
    # Mask chess coordinates and UCI even when an author accidentally includes
    # them in the position-independent formulation.
    return _SQUARE.sub(" ", _SAN.sub(" ", _UCI.sub(" ", text))).lower()


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=512,
        min_df=1,
        strip_accents="unicode",
        norm="l2",
        use_idf=True,
        smooth_idf=True,
    )


def _model_record(
    vectorizer: TfidfVectorizer, training_ids: list[str], hashes: list[str]
) -> dict[str, Any]:
    vocab = {
        token: int(index)
        for token, index in sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])
    }
    idf = [float(value) for value in vectorizer.idf_]
    return {
        "version": 1,
        "method": "tfidf",
        "vocabulary": vocab,
        "idf_vector": idf,
        "vectorizer_params": _VECTORIZER_PARAMS,
        "prompt_version": PROMPT_VERSION,
        "dimension": len(vocab),
        "training_ids": training_ids,
        "training_description_hashes": hashes,
    }


def _load_frozen_model(path: Path) -> tuple[dict[str, Any], TfidfVectorizer]:
    model = json.loads(path.read_text())
    if model.get("version") != 1 or model.get("method") != "tfidf":
        raise ValueError("Unsupported embedding model")
    if model.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Embedding model prompt version mismatch")
    vocabulary = model.get("vocabulary")
    idf = model.get("idf_vector")
    if not isinstance(vocabulary, dict) or not isinstance(idf, list):
        raise TypeError("Malformed embedding model")
    if len(vocabulary) != model.get("dimension") or len(idf) != len(vocabulary):
        raise ValueError("Embedding model dimension mismatch")
    if sorted(vocabulary.values()) != list(range(len(vocabulary))):
        raise ValueError("Embedding vocabulary indices must be contiguous")
    vectorizer = _vectorizer()
    vectorizer.vocabulary = vocabulary
    vectorizer.vocabulary_ = vocabulary
    vectorizer.fixed_vocabulary_ = True
    # Install the frozen IDF directly; there is no fit on selection/evaluation.

    transformer = TfidfTransformer(sublinear_tf=True, use_idf=True, smooth_idf=True, norm="l2")
    transformer.idf_ = np.asarray(idf, dtype=float)
    vectorizer._tfidf = transformer
    return model, vectorizer


def _external_model(model_run: Path) -> tuple[Path, str]:
    taxonomy_path = model_run / "taxonomy.json"
    complete_path = model_run / "taxonomy.complete.json"
    model_path = model_run / "embedding-model.json"
    if not taxonomy_path.exists() or not complete_path.exists() or not model_path.exists():
        raise ValueError("model_run needs a frozen taxonomy and embedding-model.json")
    if _digest(taxonomy_path) != json.loads(complete_path.read_text()).get("taxonomy_sha256"):
        raise ValueError("Source taxonomy changed")
    taxonomy = json.loads(taxonomy_path.read_text())
    expected = taxonomy.get("embedding_model_sha256")
    if expected is None:
        expected = taxonomy.get("inputs", {}).get("embedding-model.json")
    if expected != _digest(model_path):
        raise ValueError("Source taxonomy does not bind this embedding model hash")
    return model_path, _digest(model_path)


def import_descriptions(run: Path, descriptions: Path, model_run: Path | None = None) -> Path:
    """Validate external descriptions and persist frozen TF-IDF episode vectors."""
    run = Path(run)
    descriptions = Path(descriptions)
    _require_open_semantics_run(run)
    if descriptions.resolve() == (run / "descriptions.json").resolve():
        raise ValueError("External description source must be separate from descriptions.json")
    raw = json.loads(descriptions.read_text())
    expected_prompt_hash = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    if (
        raw.get("prompt_version") != PROMPT_VERSION
        or raw.get("prompt_sha256") != expected_prompt_hash
    ):
        raise ValueError("Description prompt provenance mismatch")
    if not raw.get("provider"):
        raise ValueError("External descriptions need provider provenance")
    records = raw.get("records")
    if not isinstance(records, list):
        raise TypeError("Description records must be a list")

    rows = read_table(run / "episodes.parquet")
    episode_complete_path = run / "episodes.complete.json"
    stability_path = run / "stability.json"
    if not episode_complete_path.exists() or not stability_path.exists():
        raise ValueError("Stable episode completion marker is required before description import")
    by_id = {row["decision_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate episode decision IDs")
    packet_by_id = {
        row["decision_id"]: _episode_packet(row)
        for row in rows
        if row["partition"] in {"discovery", "selection", "evaluation"}
    }
    imported: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Each description record must be an object")
        identifier = record.get("decision_id")
        if not isinstance(identifier, str) or identifier in imported:
            raise ValueError("Description decision IDs must be unique strings")
        episode = by_id.get(identifier)
        packet = packet_by_id.get(identifier)
        if (
            episode is None
            or packet is None
            or episode["partition"]
            not in {
                "discovery",
                "selection",
                "evaluation",
            }
        ):
            raise ValueError(f"Description ID is missing or excluded: {identifier}")
        if record.get("episode_sha256") != packet["episode_sha256"]:
            raise ValueError(f"Description episode fingerprint mismatch: {identifier}")
        if not isinstance(record.get("raw_explanation"), str) or not isinstance(
            record.get("normalized"), str
        ):
            raise TypeError(f"Description text fields must be strings: {identifier}")
        if type(record.get("grounded")) is not bool:
            raise ValueError(f"Description grounded flag must be boolean: {identifier}")
        imported[identifier] = {
            "decision_id": identifier,
            "episode_sha256": record["episode_sha256"],
            "raw_explanation": record["raw_explanation"],
            "normalized": record["normalized"],
            "grounded": record["grounded"],
            "reason": record.get("reason"),
            "author": record.get("author"),
            "partition": episode["partition"],
            "description_sha256": _canonical_hash(record),
        }

    output = run / "descriptions.json"
    source_hash = _digest(descriptions)
    provenance = {
        "prompt_version": raw["prompt_version"],
        "prompt_sha256": raw["prompt_sha256"],
        "provider": raw["provider"],
        "source_sha256": source_hash,
        "records": [imported[key] for key in sorted(imported)],
    }
    completion_path = run / "descriptions.complete.json"
    if output.exists():
        if json.loads(output.read_text()) != provenance:
            raise ValueError("Descriptions already imported with different content")
        if not completion_path.exists():
            raise ValueError(
                "Description import is incomplete; preserve artifacts and use a fresh run"
            )
        complete = json.loads(completion_path.read_text())
        if (
            complete.get("descriptions_sha256") != _digest(output)
            or complete.get("episodes_sha256") != _digest(run / "episodes.parquet")
            or complete.get("episodes_config_sha256") != _digest(run / "episodes.config.json")
            or complete.get("embedding_model_sha256") != _digest(run / "embedding-model.json")
            or complete.get("source_sha256") != source_hash
        ):
            raise ValueError("Imported description artifacts changed; use a fresh run")
        return output

    stable_complete = json.loads(episode_complete_path.read_text())
    if stable_complete.get("episodes_sha256") != _digest(
        run / "episodes.parquet"
    ) or stable_complete.get("stability_sha256") != _digest(stability_path):
        raise ValueError("Stable episode or confidence ledger changed before description import")
    current_config_path = run / "episodes.config.json"
    if not current_config_path.exists():
        raise ValueError("Stable episode config is required before description import")
    stable_config = json.loads(current_config_path.read_text())
    expected_config_sha = hashlib.sha256(
        json.dumps(stable_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if stable_complete.get("config_sha256") != expected_config_sha:
        raise ValueError("Stable episode config changed before description import")

    discovery = [
        item
        for item in provenance["records"]
        if item["partition"] == "discovery" and item["grounded"]
    ]
    discovery.sort(key=lambda item: item["decision_id"])
    texts = [_clean_normalized(item["normalized"]) for item in discovery]
    texts = [text for text in texts if text.strip()]
    discovery = [item for item in discovery if _clean_normalized(item["normalized"]).strip()]

    current_config = json.loads(current_config_path.read_text())
    if model_run is None:
        model_path = run / "embedding-model.json"
        if model_path.exists():
            raise ValueError(
                "Embedding model already exists; use model_run to reuse a frozen model"
            )
        if texts:
            vectorizer = _vectorizer()
            try:
                vectorizer.fit(texts)
            except ValueError as exc:
                if "empty vocabulary" not in str(exc).lower():
                    raise
                vectorizer = None
            if vectorizer is not None:
                model = _model_record(
                    vectorizer,
                    [item["decision_id"] for item in discovery],
                    [item["description_sha256"] for item in discovery],
                )
            else:
                model = {
                    "version": 1,
                    "method": "tfidf",
                    "vocabulary": {},
                    "idf_vector": [],
                    "vectorizer_params": _VECTORIZER_PARAMS,
                    "prompt_version": PROMPT_VERSION,
                    "dimension": 0,
                    "training_ids": [],
                    "training_description_hashes": [],
                }
        else:
            model = {
                "version": 1,
                "method": "tfidf",
                "vocabulary": {},
                "idf_vector": [],
                "vectorizer_params": _VECTORIZER_PARAMS,
                "prompt_version": PROMPT_VERSION,
                "dimension": 0,
                "training_ids": [],
                "training_description_hashes": [],
            }
            vectorizer = None
        atomic_json(model_path, model)
        model_source = {"kind": "discovery_fit", "sha256": _digest(model_path)}
    else:
        model_run = Path(model_run)
        source_model, source_hash = _external_model(model_run)
        source_config_path = model_run / "episodes.config.json"
        if not source_config_path.exists() or current_config.get("stable_gate") != json.loads(
            source_config_path.read_text()
        ).get("stable_gate"):
            raise ValueError("External embedding model stable_gate does not match this run")
        model_path = run / "embedding-model.json"
        if model_path.exists():
            if _digest(model_path) != source_hash:
                raise ValueError("Local embedding model exists with different content")
        else:
            shutil.copyfile(source_model, model_path)
        model, vectorizer = _load_frozen_model(model_path)
        model_source = {
            "kind": "frozen_model_run",
            "path": str(Path(model_run).resolve()),
            "sha256": source_hash,
        }

    model, frozen_vectorizer = _load_frozen_model(model_path)
    row_vectors: dict[str, list[float]] = {}
    row_available: dict[str, bool] = {}
    for identifier, item in imported.items():
        vector: list[float] = []
        if item["grounded"] and frozen_vectorizer is not None:
            cleaned = _clean_normalized(item["normalized"])
            if cleaned.strip() and model["dimension"]:
                vector = frozen_vectorizer.transform([cleaned]).toarray()[0].astype(float).tolist()
                if not any(abs(value) > 1e-12 for value in vector):
                    vector = []
        available = bool(vector)
        row_vectors[identifier] = vector
        row_available[identifier] = available

    updated_rows = []
    for row in rows:
        identifier = row["decision_id"]
        vector = row_vectors.get(identifier, [])
        available = row_available.get(identifier, False)
        updated_rows.append(
            row | {"vector": vector, "available": available, "informative": available}
        )
    write_table(run / "episodes.parquet", updated_rows, EPISODES)

    config_path = current_config_path
    config = current_config
    config.update(
        {
            "horizon": 0,
            "encoding": "TF-IDF external stable-error descriptions",
            "feature_names": [
                token for token, _ in sorted(model["vocabulary"].items(), key=lambda pair: pair[1])
            ],
            "embedding_method": "tfidf",
            "embedding_model_sha256": _digest(model_path),
            "embedding_model_source": model_source,
            "description_prompt_version": PROMPT_VERSION,
        }
    )
    atomic_json(config_path, config)
    atomic_json(output, provenance)
    episodes_complete_path = run / "episodes.complete.json"
    episodes_complete = json.loads(episodes_complete_path.read_text())
    episodes_complete.update(
        {
            "available": sum(bool(row["available"]) for row in updated_rows),
            "informative": sum(bool(row["informative"]) for row in updated_rows),
            "episodes_sha256": _digest(run / "episodes.parquet"),
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }
    )
    atomic_json(episodes_complete_path, episodes_complete)
    atomic_json(
        completion_path,
        {
            "descriptions_sha256": _digest(output),
            "episodes_sha256": _digest(run / "episodes.parquet"),
            "episodes_config_sha256": _digest(config_path),
            "embedding_model_sha256": _digest(model_path),
            "source_sha256": source_hash,
            "model_source": model_source,
        },
    )
    return output
