#!/usr/bin/env python3
"""Generate HippoRAG query NER cache TSV for ColBERTv2 Colab evaluation."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def clean(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            query = clean(row.get("query") or row.get("question"))
            triples = clean(row.get("triples"))
            if not query or not triples:
                continue
            try:
                rows[query] = json.loads(triples)
            except json.JSONDecodeError:
                rows[query] = ast.literal_eval(triples)
    return rows


def write_tsv(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "question", "triples"], delimiter="\t")
        writer.writeheader()
        for query in sorted(rows):
            writer.writerow(
                {
                    "query": query,
                    "question": query,
                    "triples": json.dumps(rows[query], ensure_ascii=False),
                }
            )


def call_openai_compatible_ner(query: str, model: str, reasoning_effort: str | None) -> dict[str, Any]:
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or "http://localhost:11435/v1"
    ).rstrip("/")
    prompt = (
        "Please extract all named entities that are important for solving the question below.\n"
        "Return only valid JSON with this schema: {\"named_entities\": [\"...\"]}\n\n"
        f"Question: {query}"
    )
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You're a very effective entity extraction system. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY') or 'codex-proxy'}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NER request failed: HTTP {exc.code}: {message}") from exc
    result = extract_json_object(payload["choices"][0]["message"]["content"])
    entities = result.get("named_entities", [])
    if not isinstance(entities, list):
        entities = []
    return {"named_entities": [clean(entity) for entity in entities if clean(entity)]}


def queries_from_qa(path: Path) -> list[str]:
    payload = load_json(path)
    return [clean(item["question"]) for item in payload["queries"] if clean(item.get("question"))]


def queries_from_thoughts(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = load_json(path)
    queries: list[str] = []
    methods = payload.get("methods", {})
    if isinstance(methods, dict):
        for rows in methods.values():
            if isinstance(rows, dict):
                iterator = rows.values()
            else:
                iterator = rows
            for row in iterator:
                if isinstance(row, dict):
                    thought = clean(row.get("thought"))
                else:
                    thought = clean(row)
                if thought:
                    queries.append(thought)
    return queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default="hipporag_v1/data/aichip_us_qa_dev.json")
    parser.add_argument("--existing", default="hipporag_v1/output/aichip_us_queries.named_entity_output.tsv")
    parser.add_argument("--extra-thoughts-json", default="")
    parser.add_argument("--out", default="colab_inputs_aichip_us/aichip_us_queries.named_entity_output.tsv")
    parser.add_argument("--model", default=os.environ.get("OPENIE_MODEL", "gpt-5.5"))
    parser.add_argument("--reasoning-effort", default="high")
    args = parser.parse_args()

    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:11435/v1")
    os.environ.setdefault("OPENAI_API_KEY", "codex-proxy")

    rows = load_existing(Path(args.existing))
    target_queries = queries_from_qa(Path(args.qa))
    if args.extra_thoughts_json:
        target_queries.extend(queries_from_thoughts(Path(args.extra_thoughts_json)))

    seen: set[str] = set()
    target_queries = [q for q in target_queries if not (q in seen or seen.add(q))]
    missing = [query for query in target_queries if query not in rows]
    print(f"target queries: {len(target_queries)}; cached: {len(target_queries) - len(missing)}; missing: {len(missing)}")
    for i, query in enumerate(missing, start=1):
        print(f"NER {i}/{len(missing)}: {query[:90]}", flush=True)
        rows[query] = call_openai_compatible_ner(query, args.model, args.reasoning_effort)
        write_tsv(Path(args.out), rows)

    write_tsv(Path(args.out), rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
