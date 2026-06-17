"""Runtime patch for the legacy HippoRAG ColBERTv2 Colab workflow.

Run inside Colab after cloning HippoRAG and before the smoke retrieval cell:

    %run /content/drive/MyDrive/hipporag_colbert_aichip_us/colbertv2_colab_runtime_patch.py

The patch is idempotent. It repairs legacy imports that break on current
LangChain packages, fixes ColBERT indexing paths, and makes smoke retrieval
reuse already-built ColBERT indexes instead of rebuilding them.
"""

from __future__ import annotations

import os
from pathlib import Path


WORK_DIR = Path(os.environ.get("WORK_DIR", "/content/HippoRAG"))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def patch_optional_chat_models(path: Path) -> None:
    """Make ChatOllama/ChatLlamaCpp imports optional and repair bad try blocks."""
    src = _read(path)
    lines = src.splitlines()
    out: list[str] = []
    i = 0
    target = "langchain_community.chat_models"
    removable = {
        "except ImportError:",
        "ChatOllama = type(None)",
        "ChatLlamaCpp = type(None)",
    }

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

            # A previous broken patch can leave a top-level try with no indented
            # body. Drop only that impossible import-area shape.
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

    _write(path, "\n".join(out) + "\n")


def patch_colbert_indexing(path: Path) -> None:
    src = _read(path)
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
    src = src.replace(
        "colbertv2_index(corpus_contents, args.dataset, 'corpus', checkpoint_path, overwrite=True)",
        "colbertv2_index(corpus_contents, args.dataset, 'corpus', checkpoint_path=checkpoint_path, overwrite=True)",
    )
    src = src.replace(
        "colbertv2_index(phrases, args.dataset, 'phrase', checkpoint_path, overwrite=True)",
        "colbertv2_index(phrases, args.dataset, 'phrase', checkpoint_path=checkpoint_path, overwrite=True)",
    )
    src = src.replace(
        "phrases = phrases.tolist()",
        "phrases = [str(phrase) for phrase in phrases.tolist() if str(phrase).strip()]",
    )
    src = src.replace(
        "colbertv2_index(corpus_contents, args.dataset, 'corpus', checkpoint_path=checkpoint_path, overwrite=False)",
        "colbertv2_index(corpus_contents, args.dataset, 'corpus', checkpoint_path=checkpoint_path, overwrite='reuse')",
    )
    src = src.replace(
        "colbertv2_index(phrases, args.dataset, 'phrase', checkpoint_path=checkpoint_path, overwrite=False)",
        "colbertv2_index(phrases, args.dataset, 'phrase', checkpoint_path=checkpoint_path, overwrite='reuse')",
    )
    _write(path, src)


def patch_hipporag_index_reuse(path: Path) -> None:
    src = _read(path)
    src = src.replace(
        "colbertv2_index(self.phrases.tolist(), self.corpus_name, 'phrase', self.colbert_config['phrase_index_name'], overwrite=True)",
        "colbertv2_index(self.phrases.tolist(), self.corpus_name, 'phrase', self.colbert_config['phrase_index_name'], overwrite='reuse')",
    )
    src = src.replace(
        "colbertv2_index(self.dataset_df['paragraph'].tolist(), self.corpus_name, 'corpus', self.colbert_config['doc_index_name'], overwrite=True)",
        "colbertv2_index(self.dataset_df['paragraph'].tolist(), self.corpus_name, 'corpus', self.colbert_config['doc_index_name'], overwrite='reuse')",
    )
    src = src.replace("overwrite=False", "overwrite='reuse'")
    _write(path, src)


def main() -> None:
    if not WORK_DIR.exists():
        raise FileNotFoundError(WORK_DIR)

    patch_optional_chat_models(WORK_DIR / "src" / "named_entity_extraction_parallel.py")
    patch_optional_chat_models(WORK_DIR / "src" / "openie_with_retrieval_option_parallel.py")
    patch_colbert_indexing(WORK_DIR / "src" / "colbertv2_indexing.py")
    patch_hipporag_index_reuse(WORK_DIR / "src" / "hipporag.py")

    print(f"Patched legacy HippoRAG files under {WORK_DIR}")


if __name__ == "__main__":
    main()
