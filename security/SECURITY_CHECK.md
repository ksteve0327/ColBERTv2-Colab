# Security Check

Public-release checks performed before pushing this repository:

- Notebook outputs stripped from `notebooks/colbertv2_aichip_us_colab.ipynb`.
- Prepared input archive added as `inputs/colab_inputs_aichip_us.tar.gz`.
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
