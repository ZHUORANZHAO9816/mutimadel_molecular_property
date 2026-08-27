#!/usr/bin/env python3
"""Read-only audit of files that would enter the first public commit."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_PUBLIC_BYTES = 50 * 1024 * 1024
FORBIDDEN_PARTS = {".claude", "__pycache__", ".pytest_cache", "checkpoints"}
FORBIDDEN_SUFFIXES = {".pth", ".pt", ".ckpt", ".joblib", ".log"}
SENSITIVE_PATTERNS = {
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "personal macOS path": re.compile(r"/" + r"Users/[^/\s]+/"),
    "personal Linux path": re.compile(r"/" + r"home/[^/\s]+/"),
}


def _git_paths(*arguments: str) -> set[Path]:
    result = subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    )
    return {Path(line) for line in result.stdout.splitlines() if line}


def publication_candidates() -> list[Path]:
    tracked = _git_paths("ls-files")
    untracked = _git_paths("ls-files", "--others", "--exclude-standard")
    return sorted(path for path in tracked | untracked if (PROJECT_ROOT / path).is_file())


def _check_readme_links(errors: list[str]) -> int:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"!?\[[^]]*\]\(([^)]+)\)", readme)
    local_count = 0
    for raw_target in targets:
        target = raw_target.strip().strip("<>").split("#", 1)[0]
        if not target or re.match(r"^(?:https?|mailto):", target):
            continue
        local_count += 1
        if not (PROJECT_ROOT / target).exists():
            errors.append(f"README local link does not exist: {target}")
    return local_count


def main() -> None:
    errors: list[str] = []
    candidates = publication_candidates()
    scanned_text = 0
    for relative in candidates:
        path = PROJECT_ROOT / relative
        if path.stat().st_size > MAX_PUBLIC_BYTES:
            errors.append(f"oversized publication candidate: {relative} ({path.stat().st_size} bytes)")
        if FORBIDDEN_PARTS.intersection(relative.parts) or relative.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden artifact is a publication candidate: {relative}")
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scanned_text += 1
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(content):
                errors.append(f"{label} found in {relative}")

    trace_records = 0
    for metrics in (PROJECT_ROOT / "results" / "run_records").glob("**/metrics.json"):
        trace_records += 1
        for required in ("config.yaml", "environment.json"):
            if not (metrics.parent / required).is_file():
                errors.append(f"trace record missing {required}: {metrics.parent.relative_to(PROJECT_ROOT)}")

    local_links = _check_readme_links(errors)
    if errors:
        raise SystemExit("Release audit failed:\n- " + "\n- ".join(errors))
    print(
        f"Release audit passed: {len(candidates)} publication candidates, "
        f"{scanned_text} text files, {trace_records} trace records, "
        f"{local_links} README local links."
    )


if __name__ == "__main__":
    main()
