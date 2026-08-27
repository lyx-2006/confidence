"""Shared V2 generation runtime.

The legacy generators intentionally keep their own checkpoint formats.  V2
uses this module for the only cross-producer resource: a durable Qwen queue
whose jobs can be dispatched in real homogeneous batches.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


QUEUE_SCHEMA_VERSION = "shared_qwen_queue.v2"


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_isolated_root(new_root: Path, old_paths: Iterable[Path]) -> Path:
    """Reject equal/ancestor/descendant output paths before any write."""
    resolved = new_root.expanduser().resolve(strict=False)
    for raw in old_paths:
        old = Path(raw).expanduser().resolve(strict=False)
        if resolved == old or resolved in old.parents or old in resolved.parents:
            raise ValueError(
                f"V2 output root {resolved} overlaps legacy/output path {old}; "
                "choose a separate output directory"
            )
    return resolved


class PersistentQwenQueue:
    """A process-safe JSON queue for text and image jobs.

    Queue mutation is lock + atomic replace.  Job IDs are idempotent: a
    completed job is returned as-is and is never generated twice.
    """

    def __init__(self, path: str | Path, poll_seconds: float = 0.05):
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.poll_seconds = poll_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            atomic_write_json(self.path, {
                "schema_version": QUEUE_SCHEMA_VERSION,
                "next_sequence": 1,
                "last_dispatched_kind": None,
                "jobs": [],
            })

    def _mutate(self, callback: Any) -> Any:
        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if payload.get("schema_version") != QUEUE_SCHEMA_VERSION:
                    raise ValueError(
                        f"Unsupported Qwen queue schema: {payload.get('schema_version')!r}"
                    )
                result, changed = callback(payload)
                if changed:
                    payload["updated_at"] = time.time()
                    atomic_write_json(self.path, payload)
                return result
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _job_result(job: dict[str, Any]) -> dict[str, Any]:
        result = job.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"Qwen job {job.get('job_id')} has no result")
        return result

    def reset_running(self) -> None:
        def reset(payload: dict[str, Any]) -> tuple[None, bool]:
            changed = False
            for job in payload["jobs"]:
                if job.get("status") == "running":
                    job["status"] = "queued"
                    job["recovered_at"] = time.time()
                    changed = True
            return None, changed
        self._mutate(reset)

    def enqueue(self, job: dict[str, Any]) -> None:
        required = {"job_id", "kind", "prompt", "answer_classes", "metadata"}
        if not required.issubset(job):
            raise ValueError(f"Qwen job missing fields: {sorted(required - set(job))}")
        if job["kind"] not in {"text", "image"}:
            raise ValueError(f"Unsupported Qwen job kind: {job['kind']!r}")

        def add(payload: dict[str, Any]) -> tuple[None, bool]:
            existing = next((item for item in payload["jobs"] if item.get("job_id") == job["job_id"]), None)
            if existing is not None:
                if (
                    existing.get("kind") != job["kind"]
                    or existing.get("prompt") != job["prompt"]
                    or existing.get("input_sha256") != job.get("input_sha256")
                ):
                    raise ValueError(f"Qwen job ID collision: {job['job_id']}")
                return None, False
            sequence = int(payload.get("next_sequence", 1))
            payload["next_sequence"] = sequence + 1
            payload["jobs"].append({
                **job,
                "sequence": sequence,
                "status": "queued",
                "created_at": time.time(),
                "started_at": None,
                "completed_at": None,
                "result": None,
                "error": None,
            })
            return None, True
        self._mutate(add)

    def get(self, job_id: str) -> dict[str, Any] | None:
        return self._mutate(lambda payload: (
            next((dict(job) for job in payload["jobs"] if job.get("job_id") == job_id), None),
            False,
        ))

    def wait(self, job_id: str, timeout: float) -> dict[str, Any]:
        started = time.monotonic()
        while True:
            job = self.get(job_id)
            if job is None:
                raise RuntimeError(f"Qwen queue lost job {job_id}")
            if job.get("status") == "completed":
                return self._job_result(job)
            if job.get("status") == "failed":
                raise RuntimeError(f"Qwen job {job_id} failed: {job.get('error')}")
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"Timed out waiting for Qwen job {job_id}")
            time.sleep(self.poll_seconds)

    def submit_and_wait(self, job: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.enqueue(job)
        return self.wait(str(job["job_id"]), timeout)

    def claim_batch(self, batch_size: int, wait_ms: int) -> list[dict[str, Any]]:
        if batch_size <= 0 or wait_ms < 0:
            raise ValueError("batch_size must be positive and wait_ms non-negative")

        def claim(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
            queued = [job for job in payload["jobs"] if job.get("status") == "queued"]
            if not queued:
                return [], False
            queued.sort(key=lambda item: int(item.get("sequence", 0)))
            by_kind = {
                kind: [job for job in queued if job.get("kind") == kind]
                for kind in ("text", "image")
            }
            available = [kind for kind in ("text", "image") if by_kind[kind]]
            last = payload.get("last_dispatched_kind")
            full = [kind for kind in available if len(by_kind[kind]) >= batch_size]
            if full:
                candidates = [kind for kind in full if kind != last] or full
                kind = min(candidates, key=lambda value: by_kind[value][0]["sequence"])
            else:
                oldest = min(queued, key=lambda item: float(item.get("created_at", 0)))
                if (time.time() - float(oldest.get("created_at", 0))) * 1000 < wait_ms:
                    return [], False
                # With no full batch, dispatch the oldest waiting job's type;
                # fairness rotation is only needed when both types can fill a
                # real batch.
                kind = str(oldest["kind"])
            selected = by_kind[kind][:batch_size]
            for job in selected:
                job["status"] = "running"
                job["started_at"] = time.time()
            payload["last_dispatched_kind"] = kind
            return [dict(job) for job in selected], True
        return self._mutate(claim)

    def complete_batch(
        self,
        results: dict[str, dict[str, Any] | None],
        errors: dict[str, dict[str, str] | None] | None = None,
    ) -> None:
        errors = errors or {}

        def finish(payload: dict[str, Any]) -> tuple[None, bool]:
            changed = False
            for job in payload["jobs"]:
                job_id = str(job.get("job_id"))
                if job.get("status") != "running" or job_id not in results:
                    continue
                error = errors.get(job_id)
                job["status"] = "failed" if error else "completed"
                job["result"] = None if error else results[job_id]
                job["error"] = error
                job["completed_at"] = time.time()
                changed = True
            return None, changed
        self._mutate(finish)

    def has_unfinished(self) -> bool:
        return bool(self._mutate(lambda payload: (
            any(job.get("status") in {"queued", "running"} for job in payload["jobs"]), False
        )))

    def fail_unfinished(self, error: dict[str, str]) -> None:
        def fail(payload: dict[str, Any]) -> tuple[None, bool]:
            changed = False
            for job in payload["jobs"]:
                if job.get("status") in {"queued", "running"}:
                    job["status"] = "failed"
                    job["error"] = error
                    job["completed_at"] = time.time()
                    changed = True
            return None, changed
        self._mutate(fail)


class QwenBatchScheduler:
    """Single-owner scheduler around an already loaded inference object."""

    def __init__(self, queue: PersistentQwenQueue, inference: Any, batch_size: int = 8, wait_ms: int = 500):
        self.queue = queue
        self.inference = inference
        self.batch_size = int(batch_size)
        self.wait_ms = int(wait_ms)

    def run(self, stop_event: Any) -> None:
        self.queue.reset_running()
        while not stop_event.is_set() or self.queue.has_unfinished():
            batch = self.queue.claim_batch(self.batch_size, self.wait_ms)
            if not batch:
                time.sleep(self.queue.poll_seconds)
                continue
            results: dict[str, dict[str, Any] | None] = {}
            errors: dict[str, dict[str, str] | None] = {}
            try:
                batch_results = self.inference.generate_answer_with_metrics_batch(batch)
                if len(batch_results) != len(batch):
                    raise RuntimeError("Qwen batch returned a result count different from its jobs")
                for job, result in zip(batch, batch_results, strict=True):
                    results[str(job["job_id"])] = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            except Exception as exc:
                for job in batch:
                    errors[str(job["job_id"])] = {"type": type(exc).__name__, "message": str(exc)}
                    results[str(job["job_id"])] = None
            self.queue.complete_batch(results, errors)


def parse_shape_from_question(question: str) -> str | None:
    match = re.search(r"color\s+of\s+(?:the\s+)?(.+?)\?", question, re.IGNORECASE)
    return match.group(1).strip().casefold() if match else None
