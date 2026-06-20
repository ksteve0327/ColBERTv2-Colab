# %% [markdown]
# # AIChip-US RAPTOR (ColBERTv2) Evaluation
#
# This Colab notebook evaluates `RAPTOR (ColBERTv2)` for the AIChip-US patent
# QA dev set.
#
# Important: RAPTOR tree construction and summary-node generation are already
# done locally with codex-proxy. Colab only consumes the exported RAPTOR units
# and runs ColBERTv2 indexing/search on GPU.
#
# Required Drive folder:
#
# `/content/drive/MyDrive/hipporag_colbert_aichip_us/`
#
# Required files:
#
# - `aichip_us_corpus.json`
# - `aichip_us_qa_dev.json`
# - `aichip_us_raptor_units.json`
#
# Optional files:
#
# - `aichip_us_retrieval_eval_raptor.json`; merged into the final report as
#   the standard RAPTOR row.
# - `aichip_us_retrieval_eval_extra_colbert.json`; existing extra-baseline
#   report, e.g. Proposition (ColBERTv2), merged rather than overwritten.
# - `aichip_us_raptor_tree.pkl`; archived for reproducibility but not loaded.

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
TOP_K_UNITS = 120

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
run("nvidia-smi", check=False)

# %%
from google.colab import drive

drive.mount("/content/drive")
DRIVE_DIR.mkdir(parents=True, exist_ok=True)
print("Drive dir:", DRIVE_DIR)
run(f'find "{DRIVE_DIR}" -maxdepth 1 -type f | sort | sed -n "1,160p"', check=False)

# %%
# Install only the ColBERTv2-side dependencies. RAPTOR itself is not needed
# because Colab consumes the prebuilt `aichip_us_raptor_units.json`.
run("pip install -U pip")
run("pip install colbert-ai==0.2.19 transformers==4.37.2 tokenizers==0.15.2 pandas tqdm sentencepiece protobuf")
try:
    run("pip install faiss-gpu-cu12")
except Exception:
    run("pip install faiss-cpu")

# %%
# Prepare a lightweight HippoRAG checkout only for the ColBERTv2 checkpoint
# path and familiar workspace layout.
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


def ensure_required_inputs() -> None:
    required = [
        DRIVE_DIR / f"{DATASET}_corpus.json",
        DRIVE_DIR / f"{DATASET}_qa_dev.json",
        DRIVE_DIR / f"{DATASET}_raptor_units.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        archive = DRIVE_DIR / "colab_table2_extra_upload_aichip_us.tar.gz"
        if archive.exists():
            print("Extracting input archive:", archive)
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(DRIVE_DIR)
            missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files. Upload/copy these to Drive first: "
            + ", ".join(str(path) for path in missing)
        )


ensure_required_inputs()

corpus = load_json(DRIVE_DIR / f"{DATASET}_corpus.json")
qa = load_json(DRIVE_DIR / f"{DATASET}_qa_dev.json")
queries = qa["queries"]
raptor_payload = load_json(DRIVE_DIR / f"{DATASET}_raptor_units.json")
raptor_units = [unit for unit in raptor_payload.get("items", []) if clean(unit.get("text"))]

print("corpus", len(corpus))
print("queries", len(queries))
print("raptor units", len(raptor_units))
print("raptor doc coverage", len({int(doc_idx) for unit in raptor_units for doc_idx in unit.get("doc_idxs", [])}))

if not raptor_units:
    raise RuntimeError("RAPTOR units are empty.")

# %%
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
            "retrieved_idx": [int(idx) for idx in retrieved],
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


report_path = DRIVE_DIR / f"{DATASET}_retrieval_eval_extra_colbert.json"
if report_path.exists():
    report = load_json(report_path)
else:
    report = {
        "qa_path": str(DRIVE_DIR / f"{DATASET}_qa_dev.json"),
        "n_queries": len(queries),
        "methods": {},
        "skipped": {},
        "sources": {},
    }

report.setdefault("methods", {})
report.setdefault("skipped", {})
report.setdefault("sources", {})
report["sources"].update(
    {
        "raptor": "https://github.com/parthsarthi03/raptor",
        "colbertv2": "colbert-ai==0.2.19 + colbertv2.0 checkpoint",
        "raptor_units": str(DRIVE_DIR / f"{DATASET}_raptor_units.json"),
    }
)

# Merge the already-computed local RAPTOR row if available.
raptor_eval_path = DRIVE_DIR / f"{DATASET}_retrieval_eval_raptor.json"
if raptor_eval_path.exists():
    raptor_eval = load_json(raptor_eval_path)
    if "RAPTOR" in raptor_eval.get("methods", {}):
        report["methods"]["RAPTOR"] = raptor_eval["methods"]["RAPTOR"]
        report["skipped"].pop("RAPTOR", None)
        print("Merged local RAPTOR row:", report["methods"]["RAPTOR"]["summary"])
else:
    print("Optional local RAPTOR eval missing:", raptor_eval_path)

# %%
# RAPTOR (ColBERTv2): index RAPTOR units with ColBERTv2 and map unit hits back
# to their leaf patent document ids.
from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig, Run, RunConfig


def safe_colbert_text(text: str) -> str:
    return clean(text).replace("\t", " ").replace("\n", " ").replace('"', "'")


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
            f.write(f"{pid}\t{safe_colbert_text(unit['text'])}\n")
    with Run().context(RunConfig(nranks=1, experiment=exp_name, root=str(root))):
        config_colbert = ColBERTConfig(nbits=2, root=str(root), doc_maxlen=180, bsize=32)
        indexer = Indexer(checkpoint=str(CHECKPOINT), config=config_colbert)
        indexer.index(name=index_name, collection=str(tsv_path), overwrite=overwrite_mode)
    with Run().context(RunConfig(nranks=1, experiment=exp_name, root=str(root))):
        config_colbert = ColBERTConfig(root=str(root))
        searcher = Searcher(index=index_name, config=config_colbert)
    return searcher, units


def colbert_raptor_rank(searcher: Searcher, units: list[dict[str, Any]]) -> list[list[int]]:
    rankings: list[list[int]] = []
    for i, query in enumerate(queries, start=1):
        print(f"RAPTOR ColBERT search {i}/{len(queries)}: {query['id']}")
        pids, _ranks, scores = searcher.search(query["question"], k=min(TOP_K_UNITS, len(units)))
        doc_scores: dict[int, float] = {}
        for pid, score in zip(pids, scores):
            unit = units[int(pid)]
            for doc_idx in unit.get("doc_idxs", []):
                doc_scores[int(doc_idx)] = max(doc_scores.get(int(doc_idx), float("-inf")), float(score))
        rankings.append(
            [
                doc_idx
                for doc_idx, _score in sorted(doc_scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)[
                    :TOP_K_DOCS
                ]
            ]
        )
    return rankings


raptor_searcher, raptor_units = build_colbert_index(raptor_units, "raptor", "nbits_2")
raptor_colbert_rankings = colbert_raptor_rank(raptor_searcher, raptor_units)
write_json(
    DRIVE_DIR / f"{DATASET}_raptor_colbert_rankings.json",
    {q["id"]: r for q, r in zip(queries, raptor_colbert_rankings)},
)
report["methods"]["RAPTOR (ColBERTv2)"] = score_method(queries, raptor_colbert_rankings)
report["skipped"].pop("RAPTOR (ColBERTv2)", None)
print(report["methods"]["RAPTOR (ColBERTv2)"]["summary"])

# %%
# Save combined extra-baseline report.
out_json = DRIVE_DIR / f"{DATASET}_retrieval_eval_extra_colbert.json"
out_md = DRIVE_DIR / f"{DATASET}_retrieval_eval_extra_colbert.md"
write_json(out_json, report)
out_md.write_text(markdown_table(report), encoding="utf-8")

archive_path = DRIVE_DIR / f"{DATASET}_raptor_colbert_outputs.tar.gz"
archive_members = [
    f"{DATASET}_raptor_units.json",
    f"{DATASET}_raptor_colbert_rankings.json",
    f"{DATASET}_retrieval_eval_extra_colbert.json",
    f"{DATASET}_retrieval_eval_extra_colbert.md",
]
if (DRIVE_DIR / f"{DATASET}_raptor_tree.pkl").exists():
    archive_members.insert(1, f"{DATASET}_raptor_tree.pkl")
if (DRIVE_DIR / f"{DATASET}_retrieval_eval_raptor.json").exists():
    archive_members.append(f"{DATASET}_retrieval_eval_raptor.json")

run(
    f"tar -czf {archive_path} "
    f"-C {DRIVE_DIR} "
    + " ".join(archive_members),
    check=False,
)

print(out_json)
print(out_md)
print(markdown_table(report))
