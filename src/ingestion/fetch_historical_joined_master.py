#!/usr/bin/env python3
"""Fetch the canonical 2021-2025 joined MLB master from a GitHub Release asset.

This script intentionally refuses production use unless the manifest contains
an expected SHA-256 checksum and the downloaded asset matches it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REPO = "shanecolter1/mlb-model-v6"
DEFAULT_MANIFEST = Path("data/external/MLB_Game_Stats_Joined_2021_2025.manifest.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output-dir", type=Path, default=Path("data/external/cache"))
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    tag = manifest["release_tag"]
    asset = manifest["canonical_filename"]
    expected = manifest["integrity"].get("sha256_csv_gz")
    if not expected:
        raise SystemExit(
            "Manifest checksum is not populated. Upload the canonical release asset, "
            "calculate SHA-256, update the manifest, then retry."
        )

    url = f"https://github.com/{REPO}/releases/download/{tag}/{asset}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / asset

    with urlopen(url) as r, out.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    actual = sha256_file(out)
    if actual.lower() != expected.lower():
        out.unlink(missing_ok=True)
        raise SystemExit(f"SHA-256 mismatch: expected {expected}, got {actual}")

    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
