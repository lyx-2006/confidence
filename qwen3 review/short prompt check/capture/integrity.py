from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SHORT_ROOT = Path(__file__).resolve().parent.parent
REVIEW_ROOT = SHORT_ROOT.parent
OUTPUT = SHORT_ROOT / "output" / "diagnostics" / "original_integrity.json"


def _eligible(path: Path) -> bool:
    parts = set(path.parts)
    return "short prompt check" not in parts and "__pycache__" not in parts and ".pytest_cache" not in parts and path.name != "active.pid"


def manifest() -> dict[str, dict[str, int | str]]:
    result = {}
    for path in sorted(REVIEW_ROOT.rglob("*")):
        if not path.is_file() or not _eligible(path): continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
        result[str(path.relative_to(REVIEW_ROOT))] = {"size": path.stat().st_size, "sha256": digest.hexdigest()}
    return result


def snapshot(path: Path = OUTPUT) -> dict:
    if path.exists(): raise FileExistsError(f"Integrity baseline already exists: {path}")
    payload = {"status": "baseline", "review_root": str(REVIEW_ROOT), "files": manifest()}
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "baseline", "file_count": len(payload["files"])}


def verify(path: Path = OUTPUT) -> dict:
    baseline = json.loads(path.read_text(encoding="utf-8")); current = manifest()
    missing = sorted(set(baseline["files"]) - set(current)); added = sorted(set(current) - set(baseline["files"]))
    changed = sorted(name for name in set(current) & set(baseline["files"]) if current[name] != baseline["files"][name])
    result = {"status": "complete" if not (missing or added or changed) else "failed", "file_count": len(current),
              "missing": missing, "added": added, "changed": changed}
    destination = path.with_name("original_integrity_verification.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if result["status"] != "complete": raise RuntimeError(f"Original qwen3 review tree changed: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("snapshot", "verify")); args = parser.parse_args()
    print(json.dumps(snapshot() if args.action == "snapshot" else verify(), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())

