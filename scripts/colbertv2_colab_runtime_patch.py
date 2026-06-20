"""Runtime patch for the legacy HippoRAG ColBERTv2 Colab workflow.

Run inside Colab after cloning HippoRAG and before the smoke retrieval cell:

    %run /content/drive/MyDrive/hipporag_colbert_aichip_us/colbertv2_colab_runtime_patch.py

The patch is idempotent. It repairs legacy imports that break on current
LangChain packages, fixes ColBERT indexing paths, and makes smoke retrieval
reuse already-built ColBERT indexes instead of rebuilding them.
"""

from __future__ import annotations

import os
import re
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


def patch_hipporag_colbert_linking(path: Path) -> None:
    """Improve ColBERTv2 trigger linking with exact/alias lookup and overlap reranking."""
    src = _read(path)

    if "import re\n" not in src:
        src = src.replace("import os\n", "import os\nimport re\n", 1)

    if "def _normalize_link_text(text):" not in src:
        helper_block = '''

def _normalize_link_text(text):
    return re.sub(r'\\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', str(text).lower())).strip()


def _link_tokens(text):
    return [token for token in _normalize_link_text(text).split() if token]


def _token_overlap(a, b):
    a_tokens = set(_link_tokens(a))
    b_tokens = set(_link_tokens(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))
'''
        needle = 'COLBERT_CKPT_DIR = "exp/colbertv2.0"\n'
        if needle not in src:
            raise ValueError("Could not find COLBERT_CKPT_DIR in hipporag.py")
        src = src.replace(needle, needle + helper_block, 1)

    if "    def _ensure_phrase_link_lookup(self):" not in src:
        class_helper_block = '''
    def _ensure_phrase_link_lookup(self):
        if hasattr(self, 'phrase_link_lookup'):
            return

        phrase_link_lookup = defaultdict(list)
        for phrase_id, phrase in enumerate(self.phrases):
            normalized_phrase = _normalize_link_text(phrase)
            if normalized_phrase:
                phrase_link_lookup[normalized_phrase].append(phrase_id)
        self.phrase_link_lookup = phrase_link_lookup

    def _get_phrase_aliases(self, query):
        normalized_query = _normalize_link_text(query)
        aliases = [normalized_query]

        if normalized_query.endswith('s') and len(normalized_query) > 3:
            aliases.append(normalized_query[:-1])

        acronym_aliases = {
            'gpus': 'gpu',
            'cpus': 'cpu',
            'cnns': 'cnn',
            'pims': 'pim',
            'processing in memory': 'pim',
            'processing in memory pim': 'pim',
            'pim': 'pim',
        }
        if normalized_query in acronym_aliases:
            aliases.append(acronym_aliases[normalized_query])
        if normalized_query == 'pim':
            aliases.extend(['processing in memory', 'processing in memory pim'])

        return list(dict.fromkeys(alias for alias in aliases if alias))

    def _exact_phrase_link(self, query):
        self._ensure_phrase_link_lookup()
        for alias in self._get_phrase_aliases(query):
            phrase_ids = self.phrase_link_lookup.get(alias)
            if phrase_ids:
                return min(phrase_ids, key=lambda phrase_id: self.phrase_to_num_doc[phrase_id])
        return None

'''
        needle = "    def link_node_by_colbertv2(self, query_ner_list):"
        if needle not in src:
            raise ValueError("Could not find link_node_by_colbertv2 in hipporag.py")
        src = src.replace(needle, class_helper_block + needle, 1)

    if "if not linking_score_map or not np.any(all_phrase_weights):" not in src:
        src = src.replace(
            "if len(query_ner_list) > 0:  # if no entities are found, assign uniform probability to documents\n                all_phrase_weights, linking_score_map = self.link_node_by_colbertv2(query_ner_list)",
            "if len(query_ner_list) > 0:  # if no entities are found, assign uniform probability to documents\n                all_phrase_weights, linking_score_map = self.link_node_by_colbertv2(query_ner_list)\n                if not linking_score_map or not np.any(all_phrase_weights):\n                    query_ner_list = []\n                    linking_score_map = {}",
            1,
        )

    method_start = src.find("    def link_node_by_colbertv2(self, query_ner_list):")
    method_end = src.find("    def link_node_by_dpr", method_start)
    if method_start == -1 or method_end == -1:
        raise ValueError("Could not replace link_node_by_colbertv2 in hipporag.py")

    improved_method = '''    def link_node_by_colbertv2(self, query_ner_list):
        phrase_ids = []
        max_scores = []
        linked_queries = []
        linking_debug = []
        colbert_link_top_k = int(os.environ.get('HIPPO_LINK_COLBERT_TOP_K', '8'))
        min_overlap_for_rerank = float(os.environ.get('HIPPO_LINK_MIN_OVERLAP', '0.25'))
        require_overlap = os.environ.get('HIPPO_LINK_REQUIRE_OVERLAP', '1') != '0'

        for query in query_ner_list:
            exact_phrase_id = self._exact_phrase_link(query)
            if exact_phrase_id is not None:
                phrase_ids.append(exact_phrase_id)
                max_scores.append(1.0)
                linked_queries.append(query)
                linking_debug.append({
                    'query_entity': query,
                    'linked_node': self.phrases[exact_phrase_id],
                    'score': 1.0,
                    'token_overlap': _token_overlap(query, self.phrases[exact_phrase_id]),
                    'method': 'exact',
                    'candidates': [],
                })
                continue

            queries = Queries(path=None, data={0: query})
            queries_ = [query]
            encoded_query = self.phrase_searcher.encode(queries_, full_length_search=False)
            max_score = self.get_colbert_max_score(query)
            ranking = self.phrase_searcher.search_all(queries, k=colbert_link_top_k)

            candidates = []
            for phrase_id, rank, score in ranking.data[0]:
                phrase = self.phrases[phrase_id]
                encoded_doc = self.phrase_searcher.checkpoint.docFromText([phrase]).float()
                real_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()
                normalized_score = float(real_score / max_score)
                overlap = _token_overlap(query, phrase)
                candidates.append((phrase_id, normalized_score, overlap, rank, phrase))

            if not candidates:
                continue

            overlapping_candidates = [candidate for candidate in candidates if candidate[2] >= min_overlap_for_rerank]
            if overlapping_candidates:
                phrase_id, normalized_score, overlap, rank, phrase = max(overlapping_candidates, key=lambda item: (item[2], item[1]))
                method = 'colbert_overlap_rerank'
            else:
                if require_overlap:
                    linking_debug.append({
                        'query_entity': query,
                        'linked_node': None,
                        'score': 0.0,
                        'token_overlap': 0.0,
                        'method': 'skipped_no_overlap',
                        'candidates': [
                            {
                                'linked_node': candidate_phrase,
                                'score': candidate_score,
                                'token_overlap': candidate_overlap,
                                'rank': candidate_rank,
                            }
                            for _, candidate_score, candidate_overlap, candidate_rank, candidate_phrase in candidates[:5]
                        ],
                    })
                    continue
                phrase_id, normalized_score, overlap, rank, phrase = max(candidates, key=lambda item: item[1])
                method = 'colbert_topk'

            phrase_ids.append(phrase_id)
            max_scores.append(normalized_score)
            linked_queries.append(query)
            linking_debug.append({
                'query_entity': query,
                'linked_node': phrase,
                'score': normalized_score,
                'token_overlap': overlap,
                'method': method,
                'candidates': [
                    {
                        'linked_node': candidate_phrase,
                        'score': candidate_score,
                        'token_overlap': candidate_overlap,
                        'rank': candidate_rank,
                    }
                    for _, candidate_score, candidate_overlap, candidate_rank, candidate_phrase in candidates[:5]
                ],
            })

        top_phrase_vec = np.zeros(len(self.phrases))

        for phrase_id in phrase_ids:
            if self.node_specificity:
                if self.phrase_to_num_doc[phrase_id] == 0:
                    weight = 1
                else:
                    weight = 1 / self.phrase_to_num_doc[phrase_id]
                top_phrase_vec[phrase_id] = weight
            else:
                top_phrase_vec[phrase_id] = 1.0

        self.linking_debug = linking_debug
        return top_phrase_vec, {(query, self.phrases[phrase_id]): max_score for phrase_id, max_score, query in zip(phrase_ids, max_scores, linked_queries)}

'''
    src = src[:method_start] + improved_method + src[method_end:]

    if "'linking_debug': getattr(self, 'linking_debug', [])," not in src:
        src, replacements = re.subn(
            r"(logs = \{'named_entities': query_ner_list,\s*"
            r"'linked_node_scores': \[list\(k\) \+ \[float\(v\)\] for k, v in linking_score_map\.items\(\)\],)",
            "\\1\n                    'linking_debug': getattr(self, 'linking_debug', []),",
            src,
            count=1,
        )
        if replacements == 0:
            raise ValueError("Could not patch HippoRAG retrieval logs with linking_debug")

    _write(path, src)


def main() -> None:
    if not WORK_DIR.exists():
        raise FileNotFoundError(WORK_DIR)

    patch_optional_chat_models(WORK_DIR / "src" / "named_entity_extraction_parallel.py")
    patch_optional_chat_models(WORK_DIR / "src" / "openie_with_retrieval_option_parallel.py")
    patch_colbert_indexing(WORK_DIR / "src" / "colbertv2_indexing.py")
    patch_hipporag_index_reuse(WORK_DIR / "src" / "hipporag.py")
    patch_hipporag_colbert_linking(WORK_DIR / "src" / "hipporag.py")

    print(f"Patched legacy HippoRAG files under {WORK_DIR}")


if __name__ == "__main__":
    main()
