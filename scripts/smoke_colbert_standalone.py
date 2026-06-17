"""Run a standalone smoke retrieval against restored aichip_us ColBERTv2 artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="/content/HippoRAG")
    parser.add_argument("--query", default="Intel neural network accelerator memory buffer")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--out-json", default="/content/drive/MyDrive/hipporag_colbert_aichip_us/aichip_us_smoke_colbertv2_gpt-5.5_top50_damping0.5.json")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    if not work_dir.exists():
        raise FileNotFoundError(work_dir)
    os.chdir(work_dir)
    sys.path.insert(0, str(work_dir))

    os.environ.setdefault("OPENAI_API_KEY", "colab-dummy-key")
    os.environ.setdefault("OPENIE_MODEL", "gpt-5.5")

    cache_path = work_dir / "output" / "aichip_us_queries.named_entity_output.tsv"
    print("cache exists=", cache_path.exists())
    if cache_path.exists():
        cache = pd.read_csv(cache_path, sep="\t")
        print(cache.head().to_string(index=False))
        assert args.query in set(cache.get("query", [])), "smoke query missing from NER cache"

    from src.hipporag import HippoRAG

    rag = HippoRAG(
        "aichip_us",
        "openai",
        os.environ.get("OPENIE_MODEL", "gpt-5.5"),
        "colbertv2",
        doc_ensemble=True,
        graph_alg="ppr",
        damping=0.5,
    )
    ranks, scores, logs = rag.rank_docs(args.query, top_k=args.top_k)

    corpus = load_json(work_dir / "data" / "aichip_us_corpus.json")
    value_path = work_dir / "data" / "aichip_us_value_scores.json"
    values = load_json(value_path).get("records", {}) if value_path.exists() else {}
    docs = []
    for rank, score in zip(ranks, scores):
        item = corpus[int(rank)]
        value_record = values.get(str(item["idx"]), {})
        docs.append(
            {
                "rank": int(rank),
                "idx": int(item["idx"]),
                "title": item["title"],
                "score": float(score),
                "value_score": float(value_record.get("value_score", 0.0)),
                "value_components": value_record.get("value_components", {}),
            }
        )

    payload = {
        "dataset": "aichip_us",
        "retriever": "colbertv2",
        "llm": "openai",
        "model_name": os.environ.get("OPENIE_MODEL", "gpt-5.5"),
        "query": args.query,
        "top_k": args.top_k,
        "damping": 0.5,
        "sim_threshold": 0.8,
        "doc_ensemble": True,
        "ranks": [int(rank) for rank in ranks],
        "scores": [float(score) for score in scores],
        "docs": docs,
        "logs": json_safe(logs),
        "statistics": json_safe(getattr(rag, "statistics", {})),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("ranks=", ranks)
    print("scores=", scores[: args.top_k])
    print("logs keys=", logs.keys())
    print("named entities=", logs.get("named_entities"))
    print("linked nodes=", logs.get("linked_node_scores", [])[:5])
    print("wrote", out_json)


if __name__ == "__main__":
    main()
