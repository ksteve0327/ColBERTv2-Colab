"""Scan this public release bundle for common credential patterns."""

from __future__ import annotations

import re
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PATTERNS = {
    "openai_key": re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(rb"(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}"),
    "huggingface_token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    "google_api_key": re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "slack_token": re.compile(rb"xox[baprs]-[0-9A-Za-z-]{20,}"),
    "openrouter_assignment": re.compile(
        rb"OPENROUTER_API_KEY\s*=\s*['\"]?(?!dummy|example|your-key)[A-Za-z0-9_-]{12,}",
        re.IGNORECASE,
    ),
    "openai_assignment": re.compile(
        rb"OPENAI_API_KEY\s*=\s*['\"]?(?!colab-dummy-key|codex-proxy|dummy|example|your-key)[A-Za-z0-9_-]{12,}",
        re.IGNORECASE,
    ),
}

TEXT_SUFFIXES = {
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".tsv",
    ".yml",
    ".yaml",
}


def scan_bytes(label: str, data: bytes) -> list[str]:
    findings: list[str] = []
    for name, pattern in PATTERNS.items():
        if pattern.search(data):
            findings.append(f"{label}: {name}")
    return findings


def read_limited(path: Path) -> bytes:
    return path.read_bytes()


def scan_regular_files() -> list[str]:
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or path.is_dir():
            continue
        if path.suffix == ".tar.gz":
            continue
        if path.suffix in TEXT_SUFFIXES or path.stat().st_size < 5_000_000:
            findings.extend(scan_bytes(str(path.relative_to(ROOT)), read_limited(path)))
    return findings


def scan_tarballs() -> list[str]:
    findings: list[str] = []
    for path in ROOT.rglob("*.tar.gz"):
        with tarfile.open(path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isdir():
                    continue
                fileobj = tar.extractfile(member)
                if fileobj is None:
                    continue
                data = fileobj.read()
                findings.extend(scan_bytes(f"{path.relative_to(ROOT)}:{member.name}", data))
    return findings


def main() -> None:
    findings = scan_regular_files() + scan_tarballs()
    if findings:
        print("SECURITY SCAN FAILED")
        for finding in findings:
            print(finding)
        raise SystemExit(1)

    print("SECURITY SCAN PASSED")
    print(f"Scanned root: {ROOT}")
    print("Patterns: " + ", ".join(sorted(PATTERNS)))


if __name__ == "__main__":
    main()
