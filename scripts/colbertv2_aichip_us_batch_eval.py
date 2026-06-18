# %% [markdown]
# # AIChip-US ColBERTv2 Batch Evaluation
#
# This Colab notebook fills the ColBERTv2-side retrieval artifacts for the
# HippoRAG 1 US patent reproduction report.
#
# It is designed for the current split workflow:
#
# - Colab GPU: ColBERTv2 corpus search and HippoRAG(ColBERTv2) PPR retrieval.
# - Local Mac codex-proxy: LLM-only steps such as IRCoT thought generation and
#   final QA answer generation, unless you explicitly expose an OpenAI-compatible
#   endpoint to Colab.
#
# Required Drive files under `/content/drive/MyDrive/hipporag_colbert_aichip_us/`:
#
# - `aichip_us_colbertv2_artifacts.tar.gz`
# - `aichip_us_corpus.json`
# - `aichip_us_id_map.json`
# - `aichip_us_qa_dev.json`
# - `aichip_us_queries.named_entity_output.tsv` with all QA questions cached
#
# Output files are copied back to the same Drive directory.

# %%
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None, check=True):
    print(f"$ {cmd}")
    return subprocess.run(cmd, cwd=cwd, shell=True, check=check)


DATASET = "aichip_us"
DRIVE_DIR = Path("/content/drive/MyDrive/hipporag_colbert_aichip_us")
WORK_DIR = Path("/content/HippoRAG")
ARCHIVE_NAME = f"{DATASET}_colbertv2_artifacts.tar.gz"
TOP_K = 5
OPENIE_MODEL = "gpt-5.5"

os.environ.setdefault("OPENIE_MODEL", OPENIE_MODEL)
os.environ.setdefault("OPENAI_API_KEY", "colab-dummy-key")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

run("nvidia-smi")

# %%
from google.colab import drive

drive.mount("/content/drive")
DRIVE_DIR.mkdir(parents=True, exist_ok=True)
print("Drive dir:", DRIVE_DIR)

# %%
if not WORK_DIR.exists():
    run(f"git clone --branch legacy --single-branch https://github.com/OSU-NLP-Group/HippoRAG.git {WORK_DIR}")
else:
    print("Repo already exists:", WORK_DIR)
run("git rev-parse --short HEAD", cwd=WORK_DIR)

# %%
# Install legacy HippoRAG + ColBERT dependencies.
# transformers is pinned because colbert-ai==0.2.19 imports AdamW from transformers.
run("pip install -U pip")
run(
    "pip install "
    "colbert-ai==0.2.19 "
    "langchain langchain-openai langchain-community langchain-together openai "
    "python-dotenv tiktoken tqdm ipdb thefuzz rank-bm25 pytrec_eval "
    "python-igraph pandas scipy transformers==4.37.2 tokenizers==0.15.2"
)
try:
    run("pip install faiss-gpu-cu12", check=True)
except Exception:
    run("pip install faiss-cpu", check=True)
run(
    "python - <<'PY'\n"
    "import transformers\n"
    "print('transformers', transformers.__version__, 'has AdamW', hasattr(transformers, 'AdamW'))\n"
    "PY"
)

# %%
# Download ColBERTv2 checkpoint if needed.
ckpt_dir = WORK_DIR / "exp" / "colbertv2.0"
if not ckpt_dir.exists():
    run("mkdir -p exp", cwd=WORK_DIR)
    run(
        "wget -nc https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz "
        "-O exp/colbertv2.0.tar.gz",
        cwd=WORK_DIR,
    )
    run("tar -xzf exp/colbertv2.0.tar.gz -C exp", cwd=WORK_DIR)
else:
    print("Checkpoint already exists:", ckpt_dir)

# %%
# Runtime patch for current Colab packages. This mirrors the local patch file
# but keeps this notebook self-contained.
patch_code = r'''
from pathlib import Path
import os

WORK_DIR = Path(os.environ.get("WORK_DIR", "/content/HippoRAG"))

def read(path):
    return path.read_text(encoding="utf-8")

def write(path, text):
    path.write_text(text, encoding="utf-8")

def patch_optional_chat_models(path):
    src = read(path)
    lines = src.splitlines()
    out = []
    i = 0
    target = "langchain_community.chat_models"
    removable = {"except ImportError:", "ChatOllama = type(None)", "ChatLlamaCpp = type(None)"}
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "try:":
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and target in lines[j] and "ChatOllama" in lines[j]:
                i = j + 1
                while i < len(lines) and (not lines[i].strip() or lines[i].strip() in removable):
                    i += 1
                continue
            if j >= len(lines) or (lines[j] and not lines[j].startswith((" ", "\t"))):
                i += 1
                continue
        if target in lines[i] and "ChatOllama" in lines[i]:
            while out and not out[-1].strip():
                out.pop()
            if out and out[-1].strip() == "try:":
                out.pop()
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].strip() in removable):
                i += 1
            continue
        if stripped in removable:
            i += 1
            continue
        out.append(lines[i])
        i += 1
    block = [
        "try:",
        "    from langchain_community.chat_models import ChatOllama, ChatLlamaCpp",
        "except ImportError:",
        "    ChatOllama = type(None)",
        "    ChatLlamaCpp = type(None)",
        "",
    ]
    insert_after = "from src.processing import extract_json_dict"
    for idx, line in enumerate(out):
        if line.strip() == insert_after:
            out[idx + 1:idx + 1] = block
            break
    else:
        out = block + out
    write(path, "\n".join(out) + "\n")

def patch_colbert_indexing(path):
    src = read(path)
    if "import os\n" not in src:
        src = src.replace("import json\n", "import json\nimport os\n")
    src = src.replace(
        "def colbertv2_index(corpus: list, dataset_name: str, exp_name: str, index_name='nbits_2', checkpoint_path='exp/colbertv2.0', overwrite=False):",
        "def colbertv2_index(corpus: list, dataset_name: str, exp_name: str, index_name='nbits_2', checkpoint_path='exp/colbertv2.0', overwrite='reuse'):",
    )
    src = src.replace(
        "corpus_processed = [x.replace('\\n', '\\t') for x in corpus]",
        "corpus_processed = [str(x).replace('\\n', '\\t') for x in corpus if str(x).strip()]",
    )
    src = src.replace(
        "corpus_tsv_file_path = f'data/lm_vectors/colbert/{dataset_name}_{exp_name}_{len(corpus_processed)}.tsv'\n    with open(corpus_tsv_file_path, 'w') as f:",
        "corpus_tsv_file_path = f'data/lm_vectors/colbert/{dataset_name}_{exp_name}_{len(corpus_processed)}.tsv'\n    os.makedirs(os.path.dirname(corpus_tsv_file_path), exist_ok=True)\n    with open(corpus_tsv_file_path, 'w') as f:",
    )
    write(path, src)

def patch_hipporag(path):
    src = read(path)
    src = src.replace(
        "colbertv2_index(self.phrases.tolist(), self.corpus_name, 'phrase', self.colbert_config['phrase_index_name'], overwrite=True)",
        "colbertv2_index(self.phrases.tolist(), self.corpus_name, 'phrase', self.colbert_config['phrase_index_name'], overwrite='reuse')",
    )
    src = src.replace(
        "colbertv2_index(self.dataset_df['paragraph'].tolist(), self.corpus_name, 'corpus', self.colbert_config['doc_index_name'], overwrite=True)",
        "colbertv2_index(self.dataset_df['paragraph'].tolist(), self.corpus_name, 'corpus', self.colbert_config['doc_index_name'], overwrite='reuse')",
    )
    src = src.replace("overwrite=False", "overwrite='reuse'")
    write(path, src)

patch_optional_chat_models(WORK_DIR / "src" / "named_entity_extraction_parallel.py")
patch_optional_chat_models(WORK_DIR / "src" / "openie_with_retrieval_option_parallel.py")
patch_colbert_indexing(WORK_DIR / "src" / "colbertv2_indexing.py")
patch_hipporag(WORK_DIR / "src" / "hipporag.py")
print("Patched legacy HippoRAG runtime files.")
'''
(WORK_DIR / "colbertv2_colab_runtime_patch.py").write_text(patch_code, encoding="utf-8")
run("python colbertv2_colab_runtime_patch.py", cwd=WORK_DIR)
run(
    "python -m py_compile "
    "src/named_entity_extraction_parallel.py src/openie_with_retrieval_option_parallel.py "
    "src/colbertv2_indexing.py src/hipporag.py",
    cwd=WORK_DIR,
)

# %%
# Restore ColBERTv2 artifacts and copy evaluation inputs.
# If this cell is run after older notebook cells, DRIVE_DIR/WORK_DIR may still
# be strings in the Colab kernel. Cast them here so Path "/" joins work.
DRIVE_DIR = Path(DRIVE_DIR)
WORK_DIR = Path(WORK_DIR)
archive = DRIVE_DIR / ARCHIVE_NAME
if not archive.exists():
    cleaned_archive = DRIVE_DIR / f"{DATASET}_colbertv2_artifacts.cleaned.tar.gz"
    if cleaned_archive.exists():
        archive = cleaned_archive
if not archive.exists():
    raise FileNotFoundError(f"Missing {archive}. Upload the previous ColBERTv2 artifact tar.gz first.")
run(f"tar -xzf {archive} -C {WORK_DIR}")

(WORK_DIR / "data").mkdir(exist_ok=True)
(WORK_DIR / "output").mkdir(exist_ok=True)

input_archive = DRIVE_DIR / f"colab_inputs_{DATASET}.tar.gz"
input_nested = DRIVE_DIR / f"colab_inputs_{DATASET}"
input_tmp = Path("/content/hipporag_colbert_eval_inputs")
if input_archive.exists():
    if input_tmp.exists():
        shutil.rmtree(input_tmp)
    input_tmp.mkdir(parents=True, exist_ok=True)
    run(f"tar -xzf {input_archive} -C {input_tmp}")
    INPUT_DIR = input_tmp / f"colab_inputs_{DATASET}"
elif input_nested.exists():
    INPUT_DIR = input_nested
else:
    INPUT_DIR = DRIVE_DIR
print("Using input dir:", INPUT_DIR)

data_files = [
    f"{DATASET}_corpus.json",
    f"{DATASET}_id_map.json",
    f"{DATASET}_qa_dev.json",
    f"{DATASET}_value_scores.json",
]
for name in data_files:
    src = INPUT_DIR / name
    if src.exists():
        shutil.copy2(src, WORK_DIR / "data" / name)
        print("copied", src.name)
    elif name.endswith("_value_scores.json"):
        print("optional missing", name)
    else:
        raise FileNotFoundError(src)

ner_src = INPUT_DIR / f"{DATASET}_queries.named_entity_output.tsv"
if not ner_src.exists():
    raise FileNotFoundError(
        f"Missing {ner_src}. Generate it locally with scripts/generate_query_ner_cache.py and upload it."
    )
shutil.copy2(ner_src, WORK_DIR / "output" / ner_src.name)
print("copied", ner_src.name)

print("Artifact directories:")
run(f"find {WORK_DIR}/data/lm_vectors/colbert/{DATASET} -maxdepth 4 -type d | sort | sed -n '1,80p'")

# %%
# Validate QA set and query NER cache coverage.
qa = json.load(open(WORK_DIR / "data" / f"{DATASET}_qa_dev.json", encoding="utf-8"))
corpus = json.load(open(WORK_DIR / "data" / f"{DATASET}_corpus.json", encoding="utf-8"))
id_map = json.load(open(WORK_DIR / "data" / f"{DATASET}_id_map.json", encoding="utf-8"))
queries = qa["queries"]
query_texts = [item["question"] for item in queries]

with open(WORK_DIR / "output" / f"{DATASET}_queries.named_entity_output.tsv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
cache_key = "query" if rows and "query" in rows[0] else "question"
cached_queries = {row[cache_key] for row in rows}
missing = [query for query in query_texts if query not in cached_queries]
print("QA queries:", len(query_texts), "NER cache rows:", len(rows), "missing:", len(missing))
if missing:
    print("\nMissing examples:")
    for item in missing[:5]:
        print("-", item)
    raise RuntimeError("Query NER cache does not cover all QA questions.")

# %%
# Evaluation helpers.
def recall_at_k(gold, retrieved, k):
    if not gold:
        return 0.0
    return len(set(gold) & set(retrieved[:k])) / len(set(gold))


def score_method(queries, rankings, k_values=(2, 5)):
    per_query = []
    for query, retrieved in zip(queries, rankings):
        gold = [int(doc["idx"]) for doc in query["gold_docs"]]
        row = {
            "id": query["id"],
            "type": query["type"],
            "gold_idx": gold,
            "retrieved_idx": [int(x) for x in retrieved],
        }
        for k in k_values:
            row[f"R@{k}"] = recall_at_k(gold, retrieved, k)
        per_query.append(row)

    summary = {}
    for group in ["local", "global", "average"]:
        rows = per_query if group == "average" else [row for row in per_query if row["type"] == group]
        summary[group] = {}
        for k in k_values:
            summary[group][f"R@{k}"] = round((sum(row[f"R@{k}"] for row in rows) / len(rows)) * 100, 1) if rows else 0.0
    return {"summary": summary, "per_query": per_query}


def markdown_retrieval_table(report, title):
    lines = [
        f"# {title}",
        "",
        "| Method | Local R@2 | Local R@5 | Global R@2 | Global R@5 | Average R@2 | Average R@5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, payload in report["methods"].items():
        s = payload["summary"]
        lines.append(
            "| {method} | {lr2:.1f} | {lr5:.1f} | {gr2:.1f} | {gr5:.1f} | {ar2:.1f} | {ar5:.1f} |".format(
                method=method,
                lr2=s["local"]["R@2"],
                lr5=s["local"]["R@5"],
                gr2=s["global"]["R@2"],
                gr5=s["global"]["R@5"],
                ar2=s["average"]["R@2"],
                ar5=s["average"]["R@5"],
            )
        )
    return "\n".join(lines) + "\n"


def context_for_doc(idx, limit=1200):
    item = corpus[int(idx)]
    meta = id_map.get(str(idx), {})
    text = re.sub(r"\s+", " ", item["text"]).strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "..."
    return {
        "idx": int(idx),
        "patent_id": meta.get("patent_id", ""),
        "title": meta.get("title") or item.get("title", ""),
        "context": text,
    }


def rrf_fuse(first, second, top_k=5, rrf_k=60):
    scores = {}
    for ranking in (first, second):
        for pos, doc_idx in enumerate(ranking, start=1):
            doc_idx = int(doc_idx)
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + pos)
    return [doc for doc, _ in sorted(scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)[:top_k]]

# %%
# ColBERTv2 corpus search.
sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)

from colbert import Searcher
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig


def colbert_rank(query_list, top_k=5):
    query_data = {i: query for i, query in enumerate(query_list)}
    root = f"data/lm_vectors/colbert/{DATASET}"
    with Run().context(RunConfig(nranks=1, experiment="corpus", root=root)):
        config = ColBERTConfig(root=root)
        searcher = Searcher(index="nbits_2", config=config)
        ranking = searcher.search_all(Queries(path=None, data=query_data), k=top_k)
    outputs = []
    for i in range(len(query_list)):
        outputs.append([int(docid) for docid, _rank, _score in ranking.data[i]][:top_k])
    return outputs


def hipporag_colbert_rank(query_list, top_k=5):
    from src.hipporag import HippoRAG

    colbert_config = {
        "root": f"data/lm_vectors/colbert/{DATASET}",
        "doc_index_name": "nbits_2",
        "phrase_index_name": "nbits_2",
    }
    rag = HippoRAG(
        DATASET,
        "openai",
        OPENIE_MODEL,
        "colbertv2",
        doc_ensemble=True,
        graph_alg="ppr",
        damping=0.5,
        sim_threshold=0.8,
        recognition_threshold=0.9,
        colbert_config=colbert_config,
    )
    outputs = []
    logs = []
    for i, query in enumerate(query_list, start=1):
        print(f"HippoRAG ColBERTv2 query {i}/{len(query_list)}")
        ranks, scores, query_logs = rag.rank_docs(query, top_k=top_k)
        outputs.append([int(x) for x in ranks])
        logs.append(query_logs)
    return outputs, logs

# %%
# Table 2 ColBERTv2 rows.
single_report = {
    "qa_path": f"data/{DATASET}_qa_dev.json",
    "n_queries": len(queries),
    "methods": {},
    "skipped": {},
}

print("Running ColBERTv2 single-step corpus retrieval...")
colbert_rankings = colbert_rank(query_texts, top_k=TOP_K)
single_report["methods"]["ColBERTv2"] = score_method(queries, colbert_rankings)

print("Running HippoRAG (ColBERTv2) single-step PPR retrieval...")
hippo_colbert_rankings, hippo_colbert_logs = hipporag_colbert_rank(query_texts, top_k=TOP_K)
hippo_scored = score_method(queries, hippo_colbert_rankings)
hippo_scored["logs"] = hippo_colbert_logs
single_report["methods"]["HippoRAG (ColBERTv2)"] = hippo_scored

out_json = WORK_DIR / "output" / f"{DATASET}_retrieval_eval_colbert.json"
out_md = WORK_DIR / "output" / f"{DATASET}_retrieval_eval_colbert.md"
out_json.write_text(json.dumps(single_report, ensure_ascii=False, indent=2), encoding="utf-8")
out_md.write_text(markdown_retrieval_table(single_report, "Patent QA ColBERTv2 Single-step Retrieval Performance"), encoding="utf-8")
shutil.copy2(out_json, DRIVE_DIR / out_json.name)
shutil.copy2(out_md, DRIVE_DIR / out_md.name)
print(out_md.read_text())
print("Copied Table 2 ColBERTv2 outputs to Drive.")

# %%
# Export Step-1 contexts for local codex-proxy IRCoT thought generation.
# Run local thought generation from this file, then upload
# `aichip_us_colbert_ircot_thoughts.json` to DRIVE_DIR and rerun the next cell.
step1_payload = {
    "dataset": DATASET,
    "qa_path": f"data/{DATASET}_qa_dev.json",
    "methods": {},
}
rankings_by_method = {
    "ColBERTv2": colbert_rankings,
    "HippoRAG (ColBERTv2)": hippo_colbert_rankings,
}
for method, rankings in rankings_by_method.items():
    rows = []
    for query, ranking in zip(queries, rankings):
        rows.append(
            {
                "id": query["id"],
                "type": query["type"],
                "question": query["question"],
                "retrieved_patents": [context_for_doc(idx) for idx in ranking[:TOP_K]],
            }
        )
    step1_payload["methods"][method] = rows

stage_a_json = WORK_DIR / "output" / f"{DATASET}_colbert_step1_for_thoughts.json"
stage_a_json.write_text(json.dumps(step1_payload, ensure_ascii=False, indent=2), encoding="utf-8")
shutil.copy2(stage_a_json, DRIVE_DIR / stage_a_json.name)
print("Wrote", DRIVE_DIR / stage_a_json.name)
print("If aichip_us_colbert_ircot_thoughts.json is not in Drive yet, stop here and generate it locally with codex-proxy.")

# %%
# Optional Table 3 ColBERTv2 rows.
# This cell runs only if DRIVE_DIR/aichip_us_colbert_ircot_thoughts.json exists.
# Expected JSON schema:
# {
#   "methods": {
#     "ColBERTv2": [{"id": "local_001", "thought": "..."}],
#     "HippoRAG (ColBERTv2)": [{"id": "local_001", "thought": "...", "named_entities": ["..."]}]
#   }
# }
thoughts_path = DRIVE_DIR / f"{DATASET}_colbert_ircot_thoughts.json"
if not thoughts_path.exists():
    print("No thoughts file found:", thoughts_path)
    print("Skipping Table 3 ColBERTv2 rows for now.")
else:
    thoughts_payload = json.load(open(thoughts_path, encoding="utf-8"))

    # Append thought NER rows if named_entities are provided.
    ner_path = WORK_DIR / "output" / f"{DATASET}_queries.named_entity_output.tsv"
    existing = {}
    with open(ner_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            existing[row["query"]] = row
    for rows_for_method in thoughts_payload.get("methods", {}).values():
        if isinstance(rows_for_method, dict):
            rows_iter = rows_for_method.values()
        else:
            rows_iter = rows_for_method
        for row in rows_iter:
            thought = row.get("thought", "")
            entities = row.get("named_entities")
            if thought and entities and thought not in existing:
                existing[thought] = {
                    "query": thought,
                    "question": thought,
                    "triples": json.dumps({"named_entities": entities}, ensure_ascii=False),
                }
    with open(ner_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "question", "triples"], delimiter="\t")
        writer.writeheader()
        for key in existing:
            writer.writerow(existing[key])

    method_rows = {}
    for method, rows_for_method in thoughts_payload["methods"].items():
        if isinstance(rows_for_method, dict):
            rows = list(rows_for_method.values())
        else:
            rows = rows_for_method
        method_rows[method] = {row["id"]: row for row in rows}

    multi_report = {
        "qa_path": f"data/{DATASET}_qa_dev.json",
        "single_step_path": str(out_json),
        "n_queries": len(queries),
        "methods": {},
        "skipped": {},
    }

    # IRCoT + ColBERTv2.
    colbert_thoughts = [method_rows["ColBERTv2"][q["id"]]["thought"] for q in queries]
    colbert_step2 = colbert_rank(colbert_thoughts, top_k=TOP_K)
    colbert_fused = [rrf_fuse(first, second, TOP_K) for first, second in zip(colbert_rankings, colbert_step2)]
    scored = score_method(queries, colbert_fused)
    for row, thought, first, second in zip(scored["per_query"], colbert_thoughts, colbert_rankings, colbert_step2):
        row["thought"] = thought
        row["first_step_idx"] = first
        row["second_step_idx"] = second
    multi_report["methods"]["IRCoT + ColBERTv2"] = scored

    # IRCoT + HippoRAG (ColBERTv2).
    hippo_thought_rows = method_rows.get("HippoRAG (ColBERTv2)", {})
    missing_ner = [
        row["thought"]
        for row in hippo_thought_rows.values()
        if row.get("thought") and not row.get("named_entities")
    ]
    if missing_ner:
        multi_report["skipped"]["IRCoT + HippoRAG (ColBERTv2)"] = (
            "thought named_entities missing; generate NER locally and include named_entities in thoughts JSON"
        )
    else:
        hippo_thoughts = [hippo_thought_rows[q["id"]]["thought"] for q in queries]
        hippo_step2, hippo_step2_logs = hipporag_colbert_rank(hippo_thoughts, top_k=TOP_K)
        hippo_fused = [rrf_fuse(first, second, TOP_K) for first, second in zip(hippo_colbert_rankings, hippo_step2)]
        scored = score_method(queries, hippo_fused)
        for row, thought, first, second in zip(scored["per_query"], hippo_thoughts, hippo_colbert_rankings, hippo_step2):
            row["thought"] = thought
            row["first_step_idx"] = first
            row["second_step_idx"] = second
        scored["logs"] = hippo_step2_logs
        multi_report["methods"]["IRCoT + HippoRAG (ColBERTv2)"] = scored

    multi_json = WORK_DIR / "output" / f"{DATASET}_ircot_retrieval_eval_colbert.json"
    multi_md = WORK_DIR / "output" / f"{DATASET}_ircot_retrieval_eval_colbert.md"
    multi_json.write_text(json.dumps(multi_report, ensure_ascii=False, indent=2), encoding="utf-8")
    multi_md.write_text(markdown_retrieval_table(multi_report, "Patent QA ColBERTv2 Multi-step Retrieval Performance"), encoding="utf-8")
    shutil.copy2(multi_json, DRIVE_DIR / multi_json.name)
    shutil.copy2(multi_md, DRIVE_DIR / multi_md.name)
    shutil.copy2(ner_path, DRIVE_DIR / ner_path.name)
    print(multi_md.read_text())
    print("Copied Table 3 ColBERTv2 outputs to Drive.")

# %%
# Export QA contexts for local codex-proxy Table 4 scoring.
# After this file is copied back, local code can generate ColBERTv2 QA answers
# with codex-proxy and merge the rows into the report.
qa_contexts = {
    "dataset": DATASET,
    "qa_path": f"data/{DATASET}_qa_dev.json",
    "methods": {},
}
qa_contexts["methods"]["ColBERTv2"] = [
    {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "retrieved_contexts": [context_for_doc(idx) for idx in ranking[:TOP_K]],
    }
    for q, ranking in zip(queries, colbert_rankings)
]
qa_contexts["methods"]["HippoRAG (ColBERTv2)"] = [
    {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "retrieved_contexts": [context_for_doc(idx) for idx in ranking[:TOP_K]],
    }
    for q, ranking in zip(queries, hippo_colbert_rankings)
]

multi_json = WORK_DIR / "output" / f"{DATASET}_ircot_retrieval_eval_colbert.json"
if multi_json.exists():
    multi = json.load(open(multi_json, encoding="utf-8"))
    for method, payload in multi.get("methods", {}).items():
        qa_contexts["methods"][method] = [
            {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "retrieved_contexts": [context_for_doc(idx) for idx in row["retrieved_idx"][:TOP_K]],
            }
            for q, row in zip(queries, payload["per_query"])
        ]

qa_contexts_json = WORK_DIR / "output" / f"{DATASET}_colbert_qa_contexts_for_local.json"
qa_contexts_json.write_text(json.dumps(qa_contexts, ensure_ascii=False, indent=2), encoding="utf-8")
shutil.copy2(qa_contexts_json, DRIVE_DIR / qa_contexts_json.name)
print("Wrote", DRIVE_DIR / qa_contexts_json.name)

# %%
print("Done. Files in Drive:")
run(f"find {DRIVE_DIR} -maxdepth 1 -type f | sort | sed -n '1,120p'", check=False)
