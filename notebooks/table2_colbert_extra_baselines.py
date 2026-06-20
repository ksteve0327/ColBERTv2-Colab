# %% [markdown]
# # AIChip-US Table 2 Extra ColBERTv2 Baselines
#
# Runs the ColBERTv2 variants for the HippoRAG 1 Table 2-style extra baselines:
#
# - `Proposition (ColBERTv2)`
# - `RAPTOR (ColBERTv2)`
#
# This notebook/script is intended for Colab GPU. It does not run OpenIE or
# HippoRAG PPR. It only indexes retrieval units and maps unit hits back to
# patent document ids.
#
# Required Drive folder:
#
# `/content/drive/MyDrive/hipporag_colbert_aichip_us/`
#
# Required files:
#
# - `aichip_us_corpus.json`
# - `aichip_us_qa_dev.json`
#
# Optional files:
#
# - `aichip_us_propositions.json`; if missing, generated on Colab GPU with
#   `chentong00/propositionizer-wiki-flan-t5-large`.
# - `aichip_us_raptor_units.json`; generated locally by
#   `patent/table2_extra_baselines.py --methods raptor --build-missing`.

# %%
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any


def run(cmd: str, cwd: Path | str | None = None, check: bool = True):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, cwd=cwd, check=check)


DATASET = "aichip_us"
DRIVE_DIR = Path("/content/drive/MyDrive/hipporag_colbert_aichip_us")
WORK_DIR = Path("/content/HippoRAG")
CHECKPOINT = WORK_DIR / "exp" / "colbertv2.0"
TOP_K_DOCS = 5
TOP_K_UNITS = 80

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
run("nvidia-smi")

# %%
from google.colab import drive

drive.mount("/content/drive")
DRIVE_DIR.mkdir(parents=True, exist_ok=True)
print("Drive dir:", DRIVE_DIR)

# %%
# Install ColBERTv2 dependencies.
run("pip install -U pip")
run("pip install colbert-ai==0.2.19 transformers==4.37.2 tokenizers==0.15.2 pandas tqdm sentencepiece protobuf")
try:
    run("pip install faiss-gpu-cu12")
except Exception:
    run("pip install faiss-cpu")

# %%
# Prepare a lightweight HippoRAG checkout only for the ColBERT checkpoint path
# and familiar workspace layout. This notebook does not call HippoRAG PPR.
if not WORK_DIR.exists():
    run(f"git clone --branch legacy --single-branch https://github.com/OSU-NLP-Group/HippoRAG.git {WORK_DIR}")
else:
    print("Repo exists:", WORK_DIR)

if not CHECKPOINT.exists():
    run("mkdir -p exp", cwd=WORK_DIR)
    run(
        "wget -nc https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz "
        "-O exp/colbertv2.0.tar.gz",
        cwd=WORK_DIR,
    )
    run("tar -xzf exp/colbertv2.0.tar.gz -C exp", cwd=WORK_DIR)
print("Checkpoint:", CHECKPOINT, CHECKPOINT.exists())

# %%
def clean(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def recall_at_k(gold: list[int], retrieved: list[int], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(gold) & set(retrieved[:k])) / len(set(gold))


def score_method(queries: list[dict[str, Any]], rankings: list[list[int]]) -> dict[str, Any]:
    per_query = []
    for query, retrieved in zip(queries, rankings):
        gold = [int(doc["idx"]) for doc in query["gold_docs"]]
        row = {
            "id": query["id"],
            "type": query["type"],
            "gold_idx": gold,
            "retrieved_idx": retrieved,
            "R@2": recall_at_k(gold, retrieved, 2),
            "R@5": recall_at_k(gold, retrieved, 5),
        }
        per_query.append(row)
    summary = {}
    for group in ["local", "global", "average"]:
        rows = per_query if group == "average" else [row for row in per_query if row["type"] == group]
        summary[group] = {
            "R@2": round(sum(row["R@2"] for row in rows) / len(rows) * 100, 1) if rows else 0.0,
            "R@5": round(sum(row["R@5"] for row in rows) / len(rows) * 100, 1) if rows else 0.0,
        }
    return {"summary": summary, "per_query": per_query}


def markdown_table(report: dict[str, Any]) -> str:
    lines = [
        "# AIChip-US Extra ColBERTv2 Table 2 Baselines",
        "",
        "| Method | Local R@2 | Local R@5 | Global R@2 | Global R@5 | Average R@2 | Average R@5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, payload in report["methods"].items():
        s = payload["summary"]
        lines.append(
            f"| {method} | {s['local']['R@2']:.1f} | {s['local']['R@5']:.1f} | "
            f"{s['global']['R@2']:.1f} | {s['global']['R@5']:.1f} | "
            f"{s['average']['R@2']:.1f} | {s['average']['R@5']:.1f} |"
        )
    if report.get("skipped"):
        lines += ["", "Skipped methods:"]
        for method, reason in report["skipped"].items():
            lines.append(f"- `{method}`: {reason}")
    return "\n".join(lines) + "\n"


required_inputs = [DRIVE_DIR / f"{DATASET}_corpus.json", DRIVE_DIR / f"{DATASET}_qa_dev.json"]
if not all(path.exists() for path in required_inputs):
    archive = DRIVE_DIR / "colab_table2_extra_upload_aichip_us.tar.gz"
    if archive.exists():
        print("Extracting input archive:", archive)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(DRIVE_DIR)
    missing = [str(path) for path in required_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

corpus = load_json(DRIVE_DIR / f"{DATASET}_corpus.json")
qa = load_json(DRIVE_DIR / f"{DATASET}_qa_dev.json")
queries = qa["queries"]
print("corpus", len(corpus), "queries", len(queries))

# %%
# Proposition generation with the official Dense X / FactoidWiki propositionizer.
def fallback_sentence_props(text: str, limit: int = 12) -> list[str]:
    return [clean(part) for part in re.split(r"(?<=[.!?])\\s+|\\n+", text) if clean(part)][:limit]


def build_propositions_if_needed(path: Path) -> dict[str, Any]:
    if path.exists():
        print("Using existing propositions:", path)
        return load_json(path)

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "chentong00/propositionizer-wiki-flan-t5-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    items = []
    for i, doc in enumerate(corpus, start=1):
        print(f"proposition {i}/{len(corpus)}")
        input_text = f"Title: {doc.get('title', '')}. Section: . Content: {clean(doc.get('text', ''))}"
        encoded = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=768)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output_ids = model.generate(**encoded, max_new_tokens=512)
        output_text = tokenizer.decode(output_ids[0].detach().cpu(), skip_special_tokens=True)
        try:
            props = json.loads(output_text)
            if not isinstance(props, list):
                props = []
        except Exception:
            props = fallback_sentence_props(doc.get("text", ""))
        for prop in props:
            prop = clean(prop)
            if prop:
                items.append({"unit_id": f"prop_{len(items):06d}", "doc_idx": int(doc["idx"]), "text": prop})
        if i % 10 == 0:
            write_json(path, {"source": model_name, "items": items})
    payload = {"source": model_name, "items": items}
    write_json(path, payload)
    return payload


proposition_path = DRIVE_DIR / f"{DATASET}_propositions.json"
propositions = build_propositions_if_needed(proposition_path)
print("proposition units", len(propositions.get("items", [])))

# %%
from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig, Run, RunConfig


def build_colbert_index(units: list[dict[str, Any]], exp_name: str, index_name: str) -> tuple[Searcher, list[dict[str, Any]]]:
    units = [unit for unit in units if clean(unit.get("text"))]
    root = WORK_DIR / "data" / "lm_vectors" / "colbert_extra" / DATASET
    root.mkdir(parents=True, exist_ok=True)
    tsv_path = root / f"{exp_name}_{len(units)}.tsv"
    index_dir = root / exp_name / "indexes" / index_name
    if index_dir.exists() and not ((index_dir / "metadata.json").exists() or (index_dir / "plan.json").exists()):
        print("Removing incomplete ColBERT index:", index_dir)
        shutil.rmtree(index_dir)
    overwrite_mode = "reuse" if index_dir.exists() else "force_silent_overwrite"
    with tsv_path.open("w", encoding="utf-8") as f:
        for pid, unit in enumerate(units):
            text = clean(unit["text"]).replace("\t", " ").replace("\n", " ").replace('"', "'")
            f.write(f"{pid}\\t{text}\\n")
    with Run().context(RunConfig(nranks=1, experiment=exp_name, root=str(root))):
        config = ColBERTConfig(nbits=2, root=str(root), doc_maxlen=180, bsize=32)
        indexer = Indexer(checkpoint=str(CHECKPOINT), config=config)
        indexer.index(name=index_name, collection=str(tsv_path), overwrite=overwrite_mode)
    with Run().context(RunConfig(nranks=1, experiment=exp_name, root=str(root))):
        config = ColBERTConfig(root=str(root))
        searcher = Searcher(index=index_name, config=config)
    return searcher, units


def colbert_unit_rank(searcher: Searcher, units: list[dict[str, Any]], doc_key: str = "doc_idx") -> list[list[int]]:
    rankings = []
    for i, query in enumerate(queries, start=1):
        print(f"search {i}/{len(queries)}: {query['id']}")
        pids, _ranks, scores = searcher.search(query["question"], k=TOP_K_UNITS)
        doc_scores: dict[int, float] = {}
        for pid, score in zip(pids, scores):
            unit = units[int(pid)]
            if doc_key == "doc_idxs":
                doc_idxs = [int(idx) for idx in unit.get("doc_idxs", [])]
            else:
                doc_idxs = [int(unit["doc_idx"])]
            for doc_idx in doc_idxs:
                doc_scores[doc_idx] = max(doc_scores.get(doc_idx, float("-inf")), float(score))
        rankings.append(
            [
                doc_idx
                for doc_idx, _score in sorted(doc_scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)[
                    :TOP_K_DOCS
                ]
            ]
        )
    return rankings


report = {
    "qa_path": str(DRIVE_DIR / f"{DATASET}_qa_dev.json"),
    "n_queries": len(queries),
    "methods": {},
    "skipped": {},
    "sources": {
        "proposition": "https://github.com/chentong0/factoid-wiki",
        "raptor": "https://github.com/parthsarthi03/raptor",
        "colbertv2": "colbert-ai==0.2.19 + colbertv2.0 checkpoint",
    },
}

# %%
# Proposition (ColBERTv2)
prop_units = propositions.get("items", [])
if prop_units:
    prop_searcher, prop_units = build_colbert_index(prop_units, "proposition", "nbits_2")
    prop_rankings = colbert_unit_rank(prop_searcher, prop_units, "doc_idx")
    report["methods"]["Proposition (ColBERTv2)"] = score_method(queries, prop_rankings)
else:
    report["skipped"]["Proposition (ColBERTv2)"] = "empty proposition units"

# %%
# RAPTOR (ColBERTv2)
raptor_path = DRIVE_DIR / f"{DATASET}_raptor_units.json"
if raptor_path.exists():
    raptor_payload = load_json(raptor_path)
    raptor_units = raptor_payload.get("items", [])
    if raptor_units:
        raptor_searcher, raptor_units = build_colbert_index(raptor_units, "raptor", "nbits_2")
        raptor_rankings = colbert_unit_rank(raptor_searcher, raptor_units, "doc_idxs")
        report["methods"]["RAPTOR (ColBERTv2)"] = score_method(queries, raptor_rankings)
    else:
        report["skipped"]["RAPTOR (ColBERTv2)"] = "empty RAPTOR units"
else:
    report["skipped"]["RAPTOR (ColBERTv2)"] = f"missing {raptor_path.name}; upload local RAPTOR unit export first"

# %%
out_json = DRIVE_DIR / f"{DATASET}_retrieval_eval_extra_colbert.json"
out_md = DRIVE_DIR / f"{DATASET}_retrieval_eval_extra_colbert.md"
write_json(out_json, report)
out_md.write_text(markdown_table(report), encoding="utf-8")
print(out_json)
print(out_md)
print(markdown_table(report))
