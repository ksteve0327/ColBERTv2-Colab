# ColBERTv2-Colab

Colab free-plan workflow and curated artifacts for the HippoRAG 1 `aichip_us`
ColBERTv2 backend run.

## Contents

- `notebooks/colbertv2_aichip_us_colab.ipynb`: stripped-output Colab notebook.
- `scripts/colbertv2_colab_runtime_patch.py`: idempotent runtime patch for the
  legacy HippoRAG ColBERTv2 code on current Colab packages.
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

After copying this repo's files into the expected Drive folder:

```python
DRIVE_DIR = "/content/drive/MyDrive/hipporag_colbert_aichip_us"
WORK_DIR = "/content/HippoRAG"

!cp -f "$DRIVE_DIR/colbertv2_colab_runtime_patch.py" "$WORK_DIR/colbertv2_colab_runtime_patch.py"
!cd "$WORK_DIR" && python colbertv2_colab_runtime_patch.py
!cd "$WORK_DIR" && python -m py_compile src/named_entity_extraction_parallel.py src/openie_with_retrieval_option_parallel.py src/colbertv2_indexing.py src/hipporag.py
%run -i /content/drive/MyDrive/hipporag_colbert_aichip_us/smoke_cell_patch.py
```

Expected smoke output includes `ranks=...` and `scores=...` with no traceback.

## Security Note

Notebook outputs were stripped before commit. The cleaned artifact and repository
files were scanned for common credential patterns before publication. See
`security/SECURITY_CHECK.md`.
