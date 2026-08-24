from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any

from .io import atomic_json


class FormalRunLock:
    def __init__(self, output_dir: Path, *, enabled: bool = True, lock_path: Path | None = None) -> None:
        self.output_dir = output_dir
        self.enabled = enabled
        self.lock_path = lock_path or Path("/tmp/dp_sa_activation_patching_formal.lock")
        self.handle: Any | None = None
        self.pid_path = output_dir / "active.pid"

    def __enter__(self):
        if not self.enabled:
            return self
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.seek(0)
            owner = self.handle.read().strip()
            raise RuntimeError(f"Another formal delayed patching task is active: {owner}") from exc
        owner = {"pid": os.getpid(), "output_dir": str(self.output_dir.resolve())}
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps(owner))
        self.handle.flush()
        os.fsync(self.handle.fileno())
        atomic_json(self.pid_path, owner)
        return self

    def __exit__(self, *_args: Any) -> None:
        if not self.enabled:
            return
        if self.pid_path.exists():
            try:
                owner = json.loads(self.pid_path.read_text())
            except Exception:
                owner = {}
            if owner.get("pid") == os.getpid():
                self.pid_path.unlink()
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
