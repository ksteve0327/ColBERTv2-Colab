#!/usr/bin/env python3
"""Generate IRCoT-style thoughts for ColBERTv2 batch evaluation handoff."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def clean(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def clip(text: str, limit: int) -> str:
    text = clean(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def call_codex_proxy_json(
    prompt: str,
    model: str,
    reasoning_effort: str | None,
    max_tokens: int,
    temperature: float = 0.0,
) -> dict[str, Any]:
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or "http://localhost:11435/v1"
    ).rstrip("/")
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only valid JSON. Do not include markdown fences."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
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
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"codex-proxy request failed: HTTP {exc.code}: {message}") from exc
    return extract_json_object(payload["choices"][0]["message"]["content"])


def compact_item(row: dict[str, Any], context_limit: int) -> dict[str, Any]:
    return {
        "id": row["id"],
        "type": row.get("type", ""),
        "question": row["question"],
        "retrieved_patents": [
            {
                "idx": patent.get("idx"),
                "patent_id": patent.get("patent_id"),
                "title": patent.get("title"),
                "context": clip(patent.get("context", ""), context_limit),
            }
            for patent in row.get("retrieved_patents", [])
        ],
    }


def build_prompt(method: str, items: list[dict[str, Any]]) -> str:
    return (
        "You are reproducing the IRCoT-style multi-step retrieval step for HippoRAG on AI semiconductor patents.\n"
        "For each item, write exactly one concise retrieval thought that will be used as the next retrieval query.\n"
        "The thought should bridge from the original question to missing technical evidence in the retrieved patents.\n"
        "Do not answer the question. Do not mention gold labels. Use technical patent phrases.\n"
        "Also extract named entities from the thought for HippoRAG query trigger linking.\n\n"
        f"Retriever method: {method}\n\n"
        "Return only valid JSON with this schema:\n"
        '{"items":[{"id":"local_001","thought":"...","named_entities":["..."]}]}\n\n'
        f"Items:\n{json.dumps(items, ensure_ascii=False, indent=2)}"
    )


def normalize_items(result: dict[str, Any], expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows = result.get("items")
    if not isinstance(rows, list):
        raise ValueError("response.items must be a list")
    out: dict[str, dict[str, Any]] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        qid = clean(item.get("id"))
        thought = clean(item.get("thought"))
        entities = item.get("named_entities", [])
        if not qid or qid not in expected_ids or not thought:
            continue
        if not isinstance(entities, list):
            entities = []
        out[qid] = {
            "id": qid,
            "thought": thought,
            "named_entities": [clean(entity) for entity in entities if clean(entity)],
        }
    return out


def existing_by_method(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    if not path.exists():
        return {}
    payload = load_json(path)
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for method, rows in payload.get("methods", {}).items():
        method_rows = rows.values() if isinstance(rows, dict) else rows
        result[method] = {
            row["id"]: row
            for row in method_rows
            if isinstance(row, dict) and clean(row.get("id")) and clean(row.get("thought"))
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="hipporag_v1/output/aichip_us_colbert_step1_for_thoughts.json")
    parser.add_argument("--out", default="hipporag_v1/output/aichip_us_colbert_ircot_thoughts.json")
    parser.add_argument("--model", default=os.environ.get("OPENIE_MODEL", "gpt-5.5"))
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--context-limit", type=int, default=900)
    args = parser.parse_args()

    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:11435/v1")
    os.environ.setdefault("OPENAI_API_KEY", "codex-proxy")

    source = load_json(Path(args.input))
    cached = existing_by_method(Path(args.out))
    output = {
        "dataset": source.get("dataset", "aichip_us"),
        "source_path": args.input,
        "generator": {
            "provider": "codex-proxy",
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
        },
        "methods": {},
    }

    for method, rows in source["methods"].items():
        print(f"method: {method}", flush=True)
        method_cache = cached.get(method, {})
        result_rows: dict[str, dict[str, Any]] = dict(method_cache)
        missing_rows = [row for row in rows if row["id"] not in result_rows]
        print(f"  total={len(rows)} cached={len(result_rows)} missing={len(missing_rows)}", flush=True)
        for offset in range(0, len(missing_rows), args.batch_size):
            batch = missing_rows[offset : offset + args.batch_size]
            payload = [compact_item(row, args.context_limit) for row in batch]
            expected_ids = {row["id"] for row in batch}
            print(f"  batch {offset + 1}-{offset + len(batch)}/{len(missing_rows)}", flush=True)
            started = time.time()
            response = call_codex_proxy_json(
                build_prompt(method, payload),
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_tokens=args.max_tokens,
            )
            generated = normalize_items(response, expected_ids)
            missing_ids = sorted(expected_ids - set(generated))
            if missing_ids:
                raise RuntimeError(f"missing generated thoughts for {missing_ids}")
            for row in batch:
                generated_row = generated[row["id"]]
                result_rows[row["id"]] = {
                    **generated_row,
                    "type": row.get("type", ""),
                    "question": row["question"],
                }
            output["methods"][method] = [result_rows[row["id"]] for row in rows if row["id"] in result_rows]
            write_json(Path(args.out), output)
            print(f"  wrote checkpoint in {time.time() - started:.1f}s", flush=True)
        output["methods"][method] = [result_rows[row["id"]] for row in rows if row["id"] in result_rows]
        write_json(Path(args.out), output)

    write_json(Path(args.out), output)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
