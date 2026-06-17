# ColBERTv2-Colab

Colab free-plan workflow and curated artifacts for the HippoRAG 1 `aichip_us`
ColBERTv2 backend run.

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
