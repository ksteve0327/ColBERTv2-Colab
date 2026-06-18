# ColBERTv2-Colab

Colab free-plan workflow and curated artifacts for the HippoRAG 1 `aichip_us`
ColBERTv2 backend run.
Because ColBERTv2 did not run on macOS in this HippoRAG 1 setup, the backend
run was executed in Google Colab instead.

## Contents

- `notebooks/colbertv2_aichip_us_colab.ipynb`: stripped-output Colab notebook.
- `inputs/colab_inputs_aichip_us.tar.gz`: prepared corpus, OpenIE, value score,
  query, and ID-map inputs required by the notebook.
- `scripts/colbertv2_colab_runtime_patch.py`: idempotent runtime patch for the
  legacy HippoRAG ColBERTv2 code on current Colab packages.
- `scripts/restore_colbertv2_artifacts.py`: restores the committed inputs and
  cleaned ColBERTv2 artifacts into `/content/HippoRAG`.
- `scripts/smoke_colbert_standalone.py`: standalone smoke retrieval check.
- `scripts/smoke_cell_patch.py`: smoke retrieval cell for the completed ColBERTv2
  graph/index.
- `artifacts/aichip_us_colbertv2_artifacts.cleaned.tar.gz`: filtered ColBERTv2
  artifacts containing only `aichip_us` graph/index outputs.
- `artifacts/aichip_us_colbertv2_artifacts.cleaned.manifest.txt`: exact file list
  included in the cleaned artifact archive.
- `security/`: public-release security scan script and result.

## Artifact Scope

The original Colab archive also contained legacy sample outputs for unrelated
datasets. The committed archive was rebuilt to include only:

- `output/aichip_us*`
- `output/kb_to_kb.tsv`
- `output/query_to_kb.tsv`
- `output/rel_kb_to_kb.tsv`
- `data/lm_vectors/colbert/aichip_us*`
- ColBERT nearest-neighbor files used by the `aichip_us` graph run

These committed artifacts are dataset-specific. They can be reused only for the
included `aichip_us` corpus. If you use a different dataset, do not reuse
`artifacts/aichip_us_colbertv2_artifacts.cleaned.tar.gz`; rebuild the graph,
nearest-neighbor files, and ColBERTv2 indexes for that dataset.

## Using Another Dataset

The workflow can be applied to another corpus, but the dataset must be prepared
in the same HippoRAG file format. Replace `aichip_us` with your dataset name and
provide these files:

```text
<dataset>_corpus.json
<dataset>.json
<dataset>_queries.named_entity_output.tsv
openie_<dataset>_results_ner_<model>_<n>.json
```

Optional but recommended:

```text
<dataset>_id_map.json
<dataset>_value_scores.json
```

For a new dataset, update the notebook variable:

```python
DATASET = "<dataset>"
```

Then rerun the ColBERTv2 build stages in the notebook:

1. Copy prepared input artifacts into `/content/HippoRAG/data` and
   `/content/HippoRAG/output`.
2. Run the first `create_graph.py` pass to generate `query_to_kb.tsv`,
   `kb_to_kb.tsv`, and `rel_kb_to_kb.tsv`.
3. Run `src/colbertv2_knn.py` for `kb_to_kb` and `query_to_kb`.
4. Run the second `create_graph.py --create_graph --cosine_sim_edges` pass.
5. Run `src/colbertv2_indexing.py` to build phrase and corpus indexes.
6. Run a smoke query and archive the new `output/` and
   `data/lm_vectors/colbert/` directories.

The helper scripts in `scripts/restore_colbertv2_artifacts.py`,
`scripts/smoke_colbert_standalone.py`, and `scripts/smoke_cell_patch.py` are
currently specialized for the committed `aichip_us` replay. For another dataset,
use the notebook flow or adapt those scripts to accept your dataset name, query,
and artifact paths.

## Colab Smoke

This repository is self-contained for replaying the completed `aichip_us`
ColBERTv2 run. In a Colab GPU runtime:

```bash
git clone https://github.com/ksteve0327/ColBERTv2-Colab /content/ColBERTv2-Colab
git clone https://github.com/OSU-NLP-Group/HippoRAG /content/HippoRAG
cd /content/HippoRAG && git checkout legacy
```

Then install the notebook dependencies and checkpoint as shown in
`notebooks/colbertv2_aichip_us_colab.ipynb`, or run the notebook setup cells
through the checkpoint download step.

To restore the committed inputs/artifacts and run a smoke retrieval:

```bash
python /content/ColBERTv2-Colab/scripts/restore_colbertv2_artifacts.py
PYTHONPATH=/content/HippoRAG python /content/ColBERTv2-Colab/scripts/smoke_colbert_standalone.py
```

Expected smoke output includes `ranks=...` and `scores=...` with no traceback.
The standalone smoke script also writes:

```text
/content/drive/MyDrive/hipporag_colbert_aichip_us/aichip_us_smoke_colbertv2_gpt-5.5_top50_damping0.5.json
```

Copy that JSON into the main HippoRAG workspace at `hipporag_v1/output/` to run
Contriever-vs-ColBERTv2 overlap and value-prior comparison.

Inside the notebook, after the setup cells have defined `DRIVE_DIR`, `WORK_DIR`,
and `run()`, the equivalent smoke cell is:

```python
DRIVE_DIR = "/content/drive/MyDrive/hipporag_colbert_aichip_us"
WORK_DIR = "/content/HippoRAG"

!cp -f "$DRIVE_DIR/colbertv2_colab_runtime_patch.py" "$WORK_DIR/colbertv2_colab_runtime_patch.py"
!cd "$WORK_DIR" && python colbertv2_colab_runtime_patch.py
!cd "$WORK_DIR" && python -m py_compile src/named_entity_extraction_parallel.py src/openie_with_retrieval_option_parallel.py src/colbertv2_indexing.py src/hipporag.py
%run -i /content/drive/MyDrive/hipporag_colbert_aichip_us/smoke_cell_patch.py
```

## Security Note

Notebook outputs were stripped before commit. The input archive, cleaned artifact,
and repository files were scanned for common credential patterns before
publication. See `security/SECURITY_CHECK.md`.
