"""Check public release file hashes and Python syntax without importing research code."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "provenance/RELEASE_SHA256.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = root / name
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise SystemExit(f"Missing or invalid release file: {name}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise SystemExit(f"Release file hash mismatch: {name}")
        if path.suffix == ".py":
            ast.parse(data, filename=name)
    print(f"PASS: {len(manifest)} release file hashes and Python syntax checked.")


if __name__ == "__main__":
    main()

