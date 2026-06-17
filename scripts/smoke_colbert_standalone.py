"""Run a standalone smoke retrieval against restored aichip_us ColBERTv2 artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="/content/HippoRAG")
    parser.add_argument("--query", default="Intel neural network accelerator memory buffer")
    parser.add_argument("--top-k", type=int, default=10)
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
    print("ranks=", ranks)
    print("scores=", scores[: args.top_k])
    print("logs keys=", logs.keys())
    print("named entities=", logs.get("named_entities"))
    print("linked nodes=", logs.get("linked_node_scores", [])[:5])


if __name__ == "__main__":
    main()
