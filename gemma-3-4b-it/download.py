#!/usr/bin/env python3
"""分片下载 google/gemma-3-4b-it —— ModelScope 源, 30 进程并发。

大文件按 CHUNK 切块, 30 个 worker 进程从任务队列取块, 用 HTTP Range 拉取后
直接 os.pwrite 写入预分配的目标文件 (不同 offset 互不干扰)。支持断点续传:
已完成的块在 models/.chunks/ 下留一个空标记文件, 重跑时自动跳过。
全部下载完毕后逐个校验 SHA256。

用法: python3 download.py [--workers 30] [--chunk-mb 32]
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path

import requests

MODEL_ID = "google/gemma-3-4b-it"
REVISION = "4b66b5f8d3bf66a468c1a755ff93f0ff374beaed"
ROOT = Path(__file__).resolve().parent
DEST = ROOT / "models"
CHUNK_DIR = DEST / ".chunks"

API = f"https://www.modelscope.cn/api/v1/models/{MODEL_ID}"
REPO = f"{API}/repo"

DEFAULT_WORKERS = 30
DEFAULT_CHUNK_MB = 32
# 小于该体积的文件一次性整包下载, 不值得切块
SINGLE_SHOT = 8 * 1024 * 1024
MAX_RETRY = 8
UA = {"User-Agent": "modelscope-downloader/1.0"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 元数据
# --------------------------------------------------------------------------- #
def list_files() -> list[dict]:
    """取仓库文件清单 (固定 revision)。"""
    r = requests.get(
        f"{API}/repo/files",
        params={"Revision": REVISION, "Recursive": "true"},
        headers=UA,
        timeout=60,
    )
    r.raise_for_status()
    files = [f for f in r.json()["Data"]["Files"] if f.get("Type") != "tree"]
    out = []
    for f in files:
        out.append(
            {
                "path": f["Path"],
                "size": int(f.get("Size") or 0),
                "sha256": (f.get("Sha256") or "").lower() or None,
            }
        )
    out.sort(key=lambda x: -x["size"])
    return out


def resolve_cdn(path: str, session: requests.Session) -> str:
    """把仓库路径解析为 CDN 直链, 避免每块都吃一次 302。"""
    r = session.get(
        REPO,
        params={"Revision": REVISION, "FilePath": path},
        headers=UA,
        allow_redirects=False,
        timeout=60,
    )
    if r.status_code in (301, 302, 303, 307, 308):
        return r.headers["Location"]
    if r.status_code == 200:
        # 没有跳转, 直接就是这个地址
        return r.url
    r.raise_for_status()
    raise RuntimeError(f"unreachable: {r.status_code}")


# --------------------------------------------------------------------------- #
# 下载原语
# --------------------------------------------------------------------------- #
class UrlExpired(RuntimeError):
    """CDN 直链过期, 需要重新解析。"""


def fetch_range(session: requests.Session, url: str, offset: int, length: int) -> bytes:
    """拉取 [offset, offset+length) 区间; 自己按块累积以容忍慢速连接。"""
    headers = dict(UA)
    headers["Range"] = f"bytes={offset}-{offset + length - 1}"
    last: Exception | None = None
    for attempt in range(MAX_RETRY):
        try:
            with session.get(url, headers=headers, stream=True, timeout=(20, 120)) as r:
                # 403/404 多半是签名过期, 重试无意义, 立刻交给上层重解析
                if r.status_code in (403, 404):
                    raise UrlExpired(f"HTTP {r.status_code}")
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                buf = bytearray()
                for piece in r.iter_content(1024 * 256):
                    if piece:
                        buf += piece
                if len(buf) != length:
                    raise RuntimeError(f"short read {len(buf)}/{length}")
                return bytes(buf)
        except UrlExpired:
            raise
        except Exception as e:  # noqa: BLE001 - 网络异常一律重试
            last = e
            if attempt == MAX_RETRY - 1:
                break
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"range {offset}+{length} failed after {MAX_RETRY} tries: {last}")


def download_whole(session: requests.Session, url: str, dest: Path, size: int) -> None:
    data = fetch_range(session, url, 0, size)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)


def download_chunk(session: requests.Session, url: str, offset: int, length: int, dest: Path) -> None:
    data = fetch_range(session, url, offset, length)
    # 预分配好的文件里按 offset 落盘, 各进程写各的区间
    fd = os.open(dest, os.O_WRONLY)
    try:
        os.pwrite(fd, data, offset)
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
def worker(tasks: mp.Queue, done_bytes, done_chunks, total_chunks, errors: mp.Queue) -> None:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    while True:
        try:
            task = tasks.get(timeout=10)
        except queue.Empty:
            if tasks.empty():
                return
            continue
        if task is None:  # 收工信号
            return
        kind, path, url, offset, length, dest_s, idx = task
        dest = Path(dest_s)
        try:
            if kind == "whole":
                download_whole(session, url, dest, length)
            else:
                try:
                    download_chunk(session, url, offset, length, dest)
                except UrlExpired:
                    # CDN 直链可能过期, 重新解析一次再试
                    url = resolve_cdn(path, session)
                    download_chunk(session, url, offset, length, dest)
            if idx is not None:
                mark_done(dest, idx)
                with done_chunks.get_lock():
                    done_chunks.value += 1
            with done_bytes.get_lock():
                done_bytes.value += length
        except Exception as e:  # noqa: BLE001
            # 记下失败块继续干别的; 剩下的块由重跑时的续传补齐
            errors.put(f"{path} @{offset}+{length}: {e}")
            continue


# --------------------------------------------------------------------------- #
# 断点续传记录
# --------------------------------------------------------------------------- #
def mark_done(dest: Path, idx: int) -> None:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    # 每个块一个标记文件, 免去多进程同时改同一个 json 的竞争
    (CHUNK_DIR / f"{dest.name}.{idx}.done").touch()


def collect_done(dest: Path, n_chunks: int) -> set[int]:
    prefix = f"{dest.name}."
    out = set()
    if not CHUNK_DIR.exists():
        return out
    for p in CHUNK_DIR.iterdir():
        name = p.name
        if name.startswith(prefix) and name.endswith(".done") and not name.endswith(".done.json"):
            mid = name[len(prefix) : -len(".done")]
            if mid.isdigit() and 0 <= int(mid) < n_chunks:
                out.add(int(mid))
    return out


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def sha256_of(path: Path, buf_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--chunk-mb", type=int, default=DEFAULT_CHUNK_MB)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    chunk_size = args.chunk_mb * 1024 * 1024
    DEST.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    log(f"model   : {MODEL_ID} @ {REVISION[:12]}")
    log(f"dest    : {DEST}")
    log(f"workers : {args.workers}  chunk: {args.chunk_mb} MB")

    files = list_files()
    total = sum(f["size"] for f in files)
    log(f"files   : {len(files)}  total {total / 1e9:.2f} GB")

    if args.verify_only:
        return verify(files)

    session = requests.Session()
    session.headers.update(UA)

    # 建任务: 小文件整包, 大文件切块
    tasks_list: list[tuple] = []
    chunk_counts: dict[str, int] = {}
    for f in files:
        dest = DEST / f["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = resolve_cdn(f["path"], session)
        size = f["size"]
        if size <= SINGLE_SHOT:
            if not (dest.exists() and dest.stat().st_size == size):
                tasks_list.append(("whole", f["path"], url, 0, size, str(dest), None))
            chunk_counts[f["path"]] = 0
            continue
        n = (size + chunk_size - 1) // chunk_size
        chunk_counts[f["path"]] = n
        if not (dest.exists() and dest.stat().st_size == size):
            # 预分配, 让各进程直接按 offset 写
            with dest.open("wb") as fh:
                fh.truncate(size)
        done = collect_done(dest, n)
        for i in range(n):
            if i in done:
                continue
            off = i * chunk_size
            ln = min(chunk_size, size - off)
            tasks_list.append(("chunk", f["path"], url, off, ln, str(dest), i))
        if done:
            log(f"resume  : {f['path']} 已完成 {len(done)}/{n} 块")

    if not tasks_list:
        log("nothing to download")
        return verify(files)

    remaining = sum(t[4] for t in tasks_list)
    log(f"todo    : {len(tasks_list)} tasks / {remaining / 1e9:.2f} GB")

    # 多进程跑
    ctx = mp.get_context("fork")
    tasks: mp.Queue = ctx.Queue()
    errors: mp.Queue = ctx.Queue()
    done_bytes = ctx.Value("q", 0)
    done_chunks = ctx.Value("q", 0)
    total_chunks = ctx.Value("q", sum(1 for t in tasks_list if t[0] == "chunk"))
    for t in tasks_list:
        tasks.put(t)
    for _ in range(args.workers):
        tasks.put(None)

    procs = [
        ctx.Process(
            target=worker,
            args=(tasks, done_bytes, done_chunks, total_chunks, errors),
            daemon=False,
        )
        for _ in range(args.workers)
    ]
    t0 = time.time()
    for p in procs:
        p.start()

    last = 0
    while any(p.is_alive() for p in procs):
        time.sleep(5)
        got = done_bytes.value
        el = max(time.time() - t0, 1e-6)
        speed = got / el / 1e6
        pct = 100 * got / max(remaining, 1)
        eta = (remaining - got) / max(got / el, 1)
        log(
            f"progress: {got / 1e9:5.2f}/{remaining / 1e9:.2f} GB "
            f"({pct:5.1f}%)  {speed:6.1f} MB/s  eta {eta / 60:5.1f} min  "
            f"chunks {done_chunks.value}/{total_chunks.value}"
        )
        last = got

    for p in procs:
        p.join()

    errs = []
    while not errors.empty():
        errs.append(errors.get())

    log(f"download finished in {(time.time() - t0) / 60:.1f} min")
    if errs:
        log(f"ERRORS ({len(errs)}):")
        for e in errs[:20]:
            log("  " + e)
        return 1

    # 清理分片标记
    for p in CHUNK_DIR.glob("*.done"):
        p.unlink()

    return verify(files)


def verify(files: list[dict]) -> int:
    log("verifying sha256 ...")
    bad = 0
    for f in files:
        dest = DEST / f["path"]
        if not dest.exists():
            log(f"  MISSING {f['path']}")
            bad += 1
            continue
        if dest.stat().st_size != f["size"]:
            log(f"  SIZE MISMATCH {f['path']} {dest.stat().st_size} != {f['size']}")
            bad += 1
            continue
        if not f["sha256"]:
            continue
        got = sha256_of(dest)
        if got != f["sha256"]:
            log(f"  SHA256 MISMATCH {f['path']}\n    got {got}\n    exp {f['sha256']}")
            bad += 1
        else:
            log(f"  ok {f['path']}")
    if bad:
        log(f"verify FAILED: {bad} bad file(s)")
        return 1
    log("verify OK - all files match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
