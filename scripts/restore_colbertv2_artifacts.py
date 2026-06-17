"""Restore the committed aichip_us ColBERTv2 run into a Colab HippoRAG clone."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}")


def extract_tar(src: Path, dst: Path) -> None:
    print(f"extracting {src} -> {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src, "r:gz") as tar:
        tar.extractall(dst)


def find_input_dir(tmpdir: Path) -> Path:
    nested = tmpdir / "colab_inputs_aichip_us"
    if nested.exists():
        return nested
    return tmpdir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default="/content/ColBERTv2-Colab")
    parser.add_argument("--work-dir", default="/content/HippoRAG")
    parser.add_argument("--drive-dir", default="/content/drive/MyDrive/hipporag_colbert_aichip_us")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    work_dir = Path(args.work_dir)
    drive_dir = Path(args.drive_dir)
    if not repo_dir.exists():
        raise FileNotFoundError(repo_dir)
    if not work_dir.exists():
        raise FileNotFoundError(f"{work_dir} does not exist. Run the notebook setup/clone cells first.")

    input_archive = repo_dir / "inputs" / "colab_inputs_aichip_us.tar.gz"
    artifact_archive = repo_dir / "artifacts" / "aichip_us_colbertv2_artifacts.cleaned.tar.gz"
    if not input_archive.exists():
        raise FileNotFoundError(input_archive)
    if not artifact_archive.exists():
        raise FileNotFoundError(artifact_archive)

    drive_dir.mkdir(parents=True, exist_ok=True)
    copy_if_exists(repo_dir / "scripts" / "colbertv2_colab_runtime_patch.py", drive_dir / "colbertv2_colab_runtime_patch.py")
    copy_if_exists(repo_dir / "scripts" / "smoke_cell_patch.py", drive_dir / "smoke_cell_patch.py")
    copy_if_exists(input_archive, drive_dir / "colab_inputs_aichip_us.tar.gz")
    copy_if_exists(artifact_archive, drive_dir / "aichip_us_colbertv2_artifacts.cleaned.tar.gz")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        extract_tar(input_archive, tmpdir)
        input_dir = find_input_dir(tmpdir)
        (work_dir / "data").mkdir(parents=True, exist_ok=True)
        (work_dir / "output").mkdir(parents=True, exist_ok=True)

        for name in [
            "aichip_us.json",
            "aichip_us_corpus.json",
            "aichip_us_id_map.json",
            "aichip_us_value_scores.json",
        ]:
            copy_if_exists(input_dir / name, work_dir / "data" / name)

        for name in [
            "aichip_us_queries.named_entity_output.tsv",
            "openie_aichip_us_results_ner_gpt-5.5_200.json",
        ]:
            copy_if_exists(input_dir / name, work_dir / "output" / name)

    extract_tar(artifact_archive, work_dir)

    patch_script = work_dir / "colbertv2_colab_runtime_patch.py"
    shutil.copy2(repo_dir / "scripts" / "colbertv2_colab_runtime_patch.py", patch_script)
    env = os.environ.copy()
    env["WORK_DIR"] = str(work_dir)
    run(["python", str(patch_script)], cwd=work_dir, env=env)
    run(
        [
            "python",
            "-m",
            "py_compile",
            "src/named_entity_extraction_parallel.py",
            "src/openie_with_retrieval_option_parallel.py",
            "src/colbertv2_indexing.py",
            "src/hipporag.py",
        ],
        cwd=work_dir,
    )

    print("Restore complete.")


if __name__ == "__main__":
    main()
