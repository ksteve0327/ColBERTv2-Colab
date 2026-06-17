# Smoke retrieval: uses HippoRAG rank_docs directly, no full QA evaluation.
# Run this in the Colab notebook after WORK_DIR, DRIVE_DIR, and run() are defined.
import os
import shlex

os.environ.setdefault("OPENAI_API_KEY", "colab-dummy-key")
os.environ.setdefault("OPENIE_MODEL", "gpt-5.5")

patch_script = os.path.join(DRIVE_DIR, "colbertv2_colab_runtime_patch.py")
if not os.path.exists(patch_script):
    raise FileNotFoundError(f"Missing Colab runtime patch: {patch_script}")

run(f"cp {shlex.quote(patch_script)} {shlex.quote(os.path.join(WORK_DIR, 'colbertv2_colab_runtime_patch.py'))}")
run("python colbertv2_colab_runtime_patch.py", cwd=WORK_DIR)
run(
    "python -m py_compile "
    "src/named_entity_extraction_parallel.py "
    "src/openie_with_retrieval_option_parallel.py "
    "src/colbertv2_indexing.py "
    "src/hipporag.py",
    cwd=WORK_DIR,
)

code = r'''
import os, sys, pandas as pd
sys.path.insert(0, '/content/HippoRAG')
os.environ.setdefault('OPENAI_API_KEY', 'colab-dummy-key')
os.environ.setdefault('OPENIE_MODEL', 'gpt-5.5')
query = 'Intel neural network accelerator memory buffer'
cache_path = 'output/aichip_us_queries.named_entity_output.tsv'
print('cache exists=', os.path.exists(cache_path))
if os.path.exists(cache_path):
    cache = pd.read_csv(cache_path, sep='\t')
    print(cache.head().to_string(index=False))
    assert query in set(cache.get('query', [])), 'smoke query missing from NER cache'
from src.hipporag import HippoRAG
rag = HippoRAG('aichip_us', 'openai', os.environ.get('OPENIE_MODEL', 'gpt-5.5'), 'colbertv2', doc_ensemble=True, graph_alg='ppr', damping=0.5)
ranks, scores, logs = rag.rank_docs(query, top_k=10)
print('ranks=', ranks)
print('scores=', scores[:10])
print('logs keys=', logs.keys())
print('named entities=', logs.get('named_entities'))
print('linked nodes=', logs.get('linked_node_scores', [])[:5])
'''

open(os.path.join(WORK_DIR, "smoke_colbert.py"), "w").write(code)
run(
    'bash -o pipefail -c "PYTHONPATH=/content/HippoRAG python -u smoke_colbert.py 2>&1 | tee /content/smoke_colbert.log"',
    cwd=WORK_DIR,
)
run(f"cp /content/smoke_colbert.log {DRIVE_DIR}/checkpoints/ || true")
