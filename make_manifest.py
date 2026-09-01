#!/usr/bin/env python3
"""Create a deterministic SHA-256 manifest for the release directory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "MANIFEST.sha256"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def release_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != TARGET
        and ".venv" not in path.parts
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and ".baiduyun." not in path.name.lower()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = release_files()
    if args.check:
        recorded = {}
        for line in TARGET.read_text(encoding="utf-8").splitlines():
            value, name = line.split("  ", 1)
            recorded[name] = value
        actual = {path.relative_to(ROOT).as_posix(): digest(path) for path in files}
        if recorded != actual:
            missing = sorted(set(recorded) - set(actual))
            extra = sorted(set(actual) - set(recorded))
            changed = sorted(
                name for name in set(recorded) & set(actual)
                if recorded[name] != actual[name]
            )
            raise SystemExit(
                f"manifest mismatch: missing={missing}, extra={extra}, changed={changed}"
            )
        print(f"PASS: {len(actual)} manifest entries verified")
        return
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    TARGET.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {TARGET.name}: {len(files)} files")


if __name__ == "__main__":
    main()
