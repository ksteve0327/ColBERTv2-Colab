# Security Check

Public-release checks performed before pushing this repository:

- Notebook outputs stripped from `notebooks/colbertv2_aichip_us_colab.ipynb`
  `notebooks/colbertv2_aichip_us_batch_eval.ipynb`, and the Table 2
  extra-baseline notebooks.
- Prepared input archives added as `inputs/colab_inputs_aichip_us.tar.gz` and
  `inputs/colab_table2_extra_upload_aichip_us.tar.gz`.
- ColBERTv2 single-step, IRCoT, RAPTOR, Proposition, QA, and diagnostic result
  files added under `results/`.
- Colab build/import logs added under `artifacts/`.
- Original Colab archive filtered to remove unrelated benchmark outputs.
- Committed artifact rebuilt as `artifacts/aichip_us_colbertv2_artifacts.cleaned.tar.gz`.
- Repository files, input archive, and cleaned artifact contents scanned for common
  credential patterns.

Scan command:

```bash
python3 security/scan_public_release.py
```

Result:

```text
SECURITY SCAN PASSED
```
