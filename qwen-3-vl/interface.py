#!/usr/bin/env python3
"""
Qwen3-VL-8B-Instruct 本地部署（Transformers 原生模式）。

本文件包含两部分：

1) 40 进程分片下载器
   把 Qwen/Qwen3-VL-8B-Instruct 的完整快照拉取到 ./model/。
   大文件按 HTTP Range 切片，由 40 个 worker 进程并发写入同一个文件的
   互不重叠偏移（os.pwrite），因此不需要「先下分片再合并」，
   峰值磁盘占用 = 模型体积本身。
   支持断点续传（每个文件一份 .done 记录，记录已完成的分片偏移）与
   sha256 校验（校验值取自 HF API 的 LFS 元数据）。

2) 推理接口
   文本 / 单图 / 多图推理，以及供研究使用的 hidden_states 前向分析。

用法:
  # 下载（默认 40 进程、32MB 分片）
  python interface.py download
  python interface.py download --workers 40 --chunk-mb 32
  python interface.py download --endpoint mirror     # 强制走 hf-mirror.com
  python interface.py download --endpoint hf         # 强制走 huggingface.co
  python interface.py download --status              # 只看进度，不下载

  # 校验
  python interface.py verify

  # 推理
  python interface.py infer --prompt "介绍一下你自己"
  python interface.py infer --image img.jpg --prompt "描述这张图片"
  python interface.py infer --image a.jpg --image b.jpg --prompt "两张图有何不同"
  python interface.py infer --image img.jpg --prompt "图中有什么？" --output outputs/result.json

  # 研究用：导出 hidden_states
  python interface.py forward --image img.jpg --prompt "描述这张图片" --output outputs/hidden.pt

环境变量:
  HF_TOKEN / HUGGING_FACE_HUB_TOKEN  —— 可选，私有仓库或提高限流阈值时使用
  http_proxy / https_proxy           —— 走 huggingface.co 时使用；
                                        走 hf-mirror.com 时会自动绕过代理
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "main"

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
STATE_DIRNAME = ".download_state"          # 断点续传记录（下载完成后自动清理）

ENDPOINT_OFFICIAL = "https://huggingface.co"
ENDPOINT_MIRROR = "https://hf-mirror.com"

DEFAULT_WORKERS = 40                        # 用户指定：40 个进程分片下载
DEFAULT_CHUNK_MB = 32
DEFAULT_TIMEOUT = 60                        # 两次数据之间的 socket 超时（秒）
CHUNK_DEADLINE_MIN = 300                    # 单个分片单次尝试的最短总时限（秒）
MIN_CHUNK_RATE = 32 * 1024                  # 低于此速率视为病态，超时重试（字节/秒）
MAX_ATTEMPTS = 6                            # 单个分片的最大尝试次数
USER_AGENT = "qwen-3-vl-interface/1.0"

# 抑制 do_sample=False 时无关的 temperature 警告
warnings.filterwarnings(
    "ignore",
    message=".*temperature.*do_sample.*",
    category=UserWarning,
)


def _log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def _set_model_dir(path: Path) -> None:
    """把模型目录写进模块全局。

    worker 进程通过 fork 继承父进程内存，因此在创建 Pool 之前调用本函数，
    子进程里的 _download_chunk / _verify_one 就能看到正确的目录。
    """
    global MODEL_DIR
    MODEL_DIR = path


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:6.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:6.2f} PB"


def _auth_headers() -> dict[str, str]:
    """构造请求头；有 token 就带上（公开模型不带也能下）。"""
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        if token_file.is_file():
            token = token_file.read_text(encoding="utf-8").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _build_opener(use_proxy: bool):
    """显式构造 opener。

    走官方站点时使用 http(s)_proxy 环境变量；走 hf-mirror 时强制直连，
    避免国内代理把镜像流量绕远。
    """
    if use_proxy:
        proxies = {}
        for scheme in ("http", "https"):
            value = os.environ.get(f"{scheme}_proxy") or os.environ.get(
                f"{scheme.upper()}_PROXY"
            )
            if value:
                proxies[scheme] = value
        return build_opener(ProxyHandler(proxies))
    return build_opener(ProxyHandler({}))


# ---------------------------------------------------------------------------
# 端点选择
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Endpoint:
    base: str
    use_proxy: bool
    label: str

    def resolve_url(self, rel_path: str) -> str:
        return f"{self.base}/{REPO_ID}/resolve/{REVISION}/{quote(rel_path)}"

    def api_url(self) -> str:
        return f"{self.base}/api/models/{REPO_ID}/revision/{REVISION}?blobs=true"


def _probe_endpoint(
    endpoint: Endpoint, probe_bytes: int = 4 << 20, timeout: float = 15.0
) -> float:
    """下载一小段数据测速，返回 MB/s；失败返回 -1。

    4MB 单次采样，socket 超时 15s：两个候选端点速度通常差好几倍，
    这点采样误差不影响选择，但不该让探测本身变成启动时的主要等待。
    """
    url = endpoint.resolve_url("model-00004-of-00004.safetensors")
    headers = _auth_headers()
    headers["Range"] = f"bytes=0-{probe_bytes - 1}"
    opener = _build_opener(endpoint.use_proxy)
    try:
        start = time.perf_counter()
        with opener.open(Request(url, headers=headers), timeout=timeout) as resp:
            got = 0
            while got < probe_bytes:
                block = resp.read(min(1 << 20, probe_bytes - got))
                if not block:
                    break
                got += len(block)
        elapsed = time.perf_counter() - start
        if got == 0 or elapsed <= 0:
            return -1.0
        return got / elapsed / 1e6
    except Exception as exc:                                  # noqa: BLE001
        _log("WARN", f"探测 {endpoint.label} 失败: {type(exc).__name__}: {exc}")
        return -1.0


def select_endpoint(mode: str) -> Endpoint:
    """mode: auto | hf | mirror。auto 模式下实测两者速度取快者。"""
    candidates = {
        "hf": Endpoint(ENDPOINT_OFFICIAL, True, "huggingface.co (proxy)"),
        "mirror": Endpoint(ENDPOINT_MIRROR, False, "hf-mirror.com (direct)"),
    }
    if mode in candidates:
        return candidates[mode]

    _log("INFO", "auto 模式：探测 huggingface.co 与 hf-mirror.com 的下载速度 ...")
    # 并发探测：慢的那个端点不会拖长总等待
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            key: executor.submit(_probe_endpoint, ep) for key, ep in candidates.items()
        }
        speeds = {key: future.result() for key, future in futures.items()}

    best: tuple[float, Endpoint] | None = None
    for key in ("hf", "mirror"):
        speed = speeds[key]
        if speed > 0:
            _log("INFO", f"  {candidates[key].label:32s} {speed:6.2f} MB/s")
            if best is None or speed > best[0]:
                best = (speed, candidates[key])
    if best is None:
        raise RuntimeError(
            "两个端点都不可用。请检查网络/代理设置，或用 --endpoint 手动指定。"
        )
    _log("INFO", f"选用端点: {best[1].label} ({best[0]:.2f} MB/s)")
    return best[1]


# ---------------------------------------------------------------------------
# 远端文件清单
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RemoteFile:
    rel_path: str
    size: int
    sha256: str | None       # 仅 LFS 大文件有；小文件为 None

    @property
    def is_lfs(self) -> bool:
        return self.sha256 is not None


def fetch_remote_files(endpoint: Endpoint) -> list[RemoteFile]:
    """读取仓库文件清单（含 LFS sha256）。"""
    opener = _build_opener(endpoint.use_proxy)
    req = Request(endpoint.api_url(), headers=_auth_headers())
    with opener.open(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    files: list[RemoteFile] = []
    for sibling in payload.get("siblings", []):
        size = sibling.get("size")
        if size is None:
            continue
        lfs = sibling.get("lfs") or {}
        files.append(
            RemoteFile(
                rel_path=sibling["rfilename"],
                size=int(size),
                sha256=lfs.get("sha256"),
            )
        )
    if not files:
        raise RuntimeError("远端文件清单为空，可能是仓库 ID 或网络有问题。")
    files.sort(key=lambda f: f.size)          # 小文件优先，processor 可尽早可用
    return files


# ---------------------------------------------------------------------------
# 分片下载
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkTask:
    rel_path: str
    offset: int
    length: int
    url: str
    use_proxy: bool


def _state_file(model_dir: Path, rel_path: str) -> Path:
    return model_dir / STATE_DIRNAME / f"{rel_path.replace('/', '__')}.done"


def _complete_marker(model_dir: Path, rel_path: str) -> Path:
    return model_dir / STATE_DIRNAME / f"{rel_path.replace('/', '__')}.complete"


def _load_done_offsets(model_dir: Path, rel_path: str) -> set[int]:
    state = _state_file(model_dir, rel_path)
    if not state.is_file():
        return set()
    done: set[int] = set()
    for line in state.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.isdigit():
            done.add(int(line))
    return done


def _pwrite_all(fd: int, data: bytes, offset: int) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        written += os.pwrite(fd, view[written:], offset + written)


def _read_exact(resp, count: int, deadline: float | None = None) -> bytes:
    """从响应中精确读取 count 字节，内存占用有界。

    deadline 是 perf_counter 时间戳。socket timeout 只管「两次数据之间」的间隔，
    一个持续滴数据但极慢的连接可以永远不触发它 —— 所以这里再加一道总时限，
    超时就断开重试（换条连接往往就快了）。
    """
    parts: list[bytes] = []
    got = 0
    while got < count:
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError(f"读取超时，已读 {got}/{count} 字节")
        block = resp.read(min(1 << 20, count - got))
        if not block:
            break
        parts.append(block)
        got += len(block)
    return b"".join(parts)


_OPENER_CACHE: dict[bool, Any] = {}


def _worker_opener(use_proxy: bool):
    opener = _OPENER_CACHE.get(use_proxy)
    if opener is None:
        opener = _build_opener(use_proxy)
        _OPENER_CACHE[use_proxy] = opener
    return opener


@dataclass(frozen=True)
class ChunkResult:
    rel_path: str
    offset: int
    written: int
    error: str | None = None


def _download_chunk(task: ChunkTask) -> ChunkResult:
    """下载单个分片并写入目标文件的对应偏移。

    失败不抛异常，而是回一个带 error 的 ChunkResult —— 单个分片重试到极限
    不该让整轮下载崩掉，父进程收集失败、最后统一报告，重跑即可补上。
    """
    model_dir = MODEL_DIR
    target = model_dir / task.rel_path
    state = _state_file(model_dir, task.rel_path)

    headers = _auth_headers()
    headers["Range"] = f"bytes={task.offset}-{task.offset + task.length - 1}"
    opener = _worker_opener(task.use_proxy)

    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            # 指数退避 + 抖动，避免 40 个进程同时重试打爆端点
            delay = min(2 ** attempt, 30) * (0.5 + random.random())
            time.sleep(delay)
        try:
            # 时限随分片大小走：32MB 分片按 32KB/s 也允许 ~17 分钟，
            # 免得把「慢但在推进」的连接掐掉重来（真卡住由 socket 超时兜底）
            deadline = time.perf_counter() + max(
                CHUNK_DEADLINE_MIN, task.length / MIN_CHUNK_RATE
            )
            with opener.open(Request(task.url, headers=headers), timeout=DEFAULT_TIMEOUT) as resp:
                status = resp.status
                if status == 200 and task.offset > 0:
                    # 服务端忽略了 Range：丢弃前缀，取我们这一段（内存有界）
                    discarded = 0
                    while discarded < task.offset:
                        if time.perf_counter() > deadline:
                            raise TimeoutError(f"丢弃前缀超时（{discarded}/{task.offset}）")
                        block = resp.read(min(1 << 20, task.offset - discarded))
                        if not block:
                            break
                        discarded += len(block)
                    if discarded != task.offset:
                        raise IOError(f"服务端忽略 Range 且数据不足（{discarded}/{task.offset}）")
                data = _read_exact(resp, task.length, deadline)
            if len(data) != task.length:
                raise IOError(
                    f"分片长度不符: 期望 {task.length}, 实际 {len(data)} (HTTP {status})"
                )

            fd = os.open(target, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                _pwrite_all(fd, data, task.offset)
            finally:
                os.close(fd)

            # 写完再记账（O_APPEND 小写入在 Linux 上是原子的，进程间安全）
            fd = os.open(state, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, f"{task.offset}\n".encode())
            finally:
                os.close(fd)
            return ChunkResult(task.rel_path, task.offset, len(data))
        except Exception as exc:                              # noqa: BLE001
            last_error = exc
    return ChunkResult(
        task.rel_path,
        task.offset,
        0,
        f"{type(last_error).__name__}: {last_error}",
    )


def plan_chunks(
    files: list[RemoteFile],
    endpoint: Endpoint,
    model_dir: Path,
    chunk_size: int,
) -> tuple[list[ChunkTask], list[RemoteFile], list[RemoteFile]]:
    """把待下载文件切成任务列表。

    返回 (tasks, 待下载文件, 已完整文件)。任务按文件轮转排列，
    让多个大文件同时推进，避免长尾。
    """
    per_file: list[list[ChunkTask]] = []
    pending: list[RemoteFile] = []
    finished: list[RemoteFile] = []

    for remote in files:
        target = model_dir / remote.rel_path
        marker = _complete_marker(model_dir, remote.rel_path)
        if (
            target.is_file()
            and target.stat().st_size == remote.size
            and marker.is_file()
        ):
            finished.append(remote)
            continue

        pending.append(remote)
        target.parent.mkdir(parents=True, exist_ok=True)
        done = _load_done_offsets(model_dir, remote.rel_path)

        url = endpoint.resolve_url(remote.rel_path)
        chunks: list[ChunkTask] = []
        for offset in range(0, max(remote.size, 1), chunk_size):
            length = min(chunk_size, remote.size - offset)
            if length <= 0 or offset in done:
                continue
            chunks.append(
                ChunkTask(
                    rel_path=remote.rel_path,
                    offset=offset,
                    length=length,
                    url=url,
                    use_proxy=endpoint.use_proxy,
                )
            )
        if chunks:
            per_file.append(chunks)
        elif remote.size == 0:
            # 空文件：直接创建
            target.touch()

    # 轮转合并：file1[0], file2[0], ..., file1[1], file2[1], ...
    tasks: list[ChunkTask] = []
    for index in range(max((len(c) for c in per_file), default=0)):
        for chunks in per_file:
            if index < len(chunks):
                tasks.append(chunks[index])
    return tasks, pending, finished


def _preallocate(model_dir: Path, files: list[RemoteFile]) -> None:
    """按最终大小预分配文件（稀疏），worker 直接按偏移写入。"""
    for remote in files:
        target = model_dir / remote.rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(target, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            os.ftruncate(fd, remote.size)
        finally:
            os.close(fd)


def _sha256_file(path: Path, block_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _verify_one(args: tuple[str, int, str | None]) -> tuple[str, bool, str]:
    rel_path, size, expected = args
    path = MODEL_DIR / rel_path
    if not path.is_file():
        return rel_path, False, "文件缺失"
    actual_size = path.stat().st_size
    if actual_size != size:
        return rel_path, False, f"大小不符 {actual_size} != {size}"
    if expected:
        actual = _sha256_file(path)
        if actual != expected:
            return rel_path, False, f"sha256 不符 ({actual[:12]}... != {expected[:12]}...)"
        return rel_path, True, "sha256 通过"
    return rel_path, True, "大小通过"


def download(args: argparse.Namespace) -> int:
    import multiprocessing as mp

    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / STATE_DIRNAME).mkdir(parents=True, exist_ok=True)
    _set_model_dir(model_dir)

    endpoint = select_endpoint(args.endpoint)
    files = fetch_remote_files(endpoint)
    total_size = sum(f.size for f in files)
    _log("INFO", f"仓库 {REPO_ID}@{REVISION}: {len(files)} 个文件, 共 {_human(total_size)}")

    chunk_size = int(args.chunk_mb * 1024 * 1024)
    tasks, pending, finished = plan_chunks(files, endpoint, model_dir, chunk_size)

    if finished:
        _log("INFO", f"已完成 {len(finished)} 个文件，跳过。")
    if not tasks:
        _log("INFO", "没有待下载的分片。")

    pending_bytes = sum(t.length for t in tasks)
    _log(
        "INFO",
        f"待下载 {len(tasks)} 个分片 / {_human(pending_bytes)}"
        f"（分片 {args.chunk_mb}MB, 进程 {args.workers}）",
    )

    if args.status:
        for remote in pending:
            done = _load_done_offsets(model_dir, remote.rel_path)
            have = min(len(done) * chunk_size, remote.size)
            pct = 100.0 * have / remote.size if remote.size else 100.0
            _log("INFO", f"  {remote.rel_path:44s} {pct:5.1f}%  {_human(have)}/{_human(remote.size)}")
        return 0

    started = time.perf_counter()
    if tasks:
        # 按最终大小预分配（稀疏文件），worker 直接按偏移写入，无需合并
        _preallocate(model_dir, pending)
        workers = max(1, min(args.workers, len(tasks)))
        _log("INFO", f"启动 {workers} 个进程开始下载 ...")

        done_bytes = 0
        done_chunks = 0
        failed: list[str] = []
        chunk_failures: list[tuple[str, int, str]] = []
        last_print = 0.0
        samples: deque[tuple[float, int]] = deque()   # (时刻, 累计字节)，滚动窗口

        ctx = mp.get_context("fork")
        pool = ctx.Pool(processes=workers)
        try:
            for result in pool.imap_unordered(_download_chunk, tasks, chunksize=1):
                done_chunks += 1
                if result.error:
                    # 单个分片失败不中断整轮：记录后继续，最后统一报告
                    chunk_failures.append(
                        (result.rel_path, result.offset, result.error)
                    )
                    continue
                done_bytes += result.written
                now = time.perf_counter()
                samples.append((now, done_bytes))
                # 只用最近 15s 算速度：开局建池、网络抖动的开销不会一直拖累读数
                while len(samples) > 2 and now - samples[0][0] > 15.0:
                    samples.popleft()

                if now - last_print >= 1.0 or done_chunks == len(tasks):
                    last_print = now
                    window = now - samples[0][0]
                    speed = (
                        (done_bytes - samples[0][1]) / window / 1e6
                        if window >= 1.0
                        else 0.0
                    )
                    overall = 100.0 * done_bytes / pending_bytes if pending_bytes else 100.0
                    eta = (
                        f"{(pending_bytes - done_bytes) / (speed * 1e6) / 60:5.1f} 分钟"
                        if speed > 0
                        else "  --  "
                    )
                    print(
                        f"\r[DOWN] {overall:5.1f}%  {_human(done_bytes)}/{_human(pending_bytes)}"
                        f"  {speed:6.2f} MB/s  分片 {done_chunks}/{len(tasks)}"
                        f"  剩余约 {eta}   ",
                        end="",
                        flush=True,
                    )
        except KeyboardInterrupt:
            _log("WARN", "\n收到中断，正在停止 worker ...")
            pool.terminate()
            pool.join()
            _log("INFO", "已停止。重新执行同一命令即可断点续传。")
            return 130
        except Exception as exc:                              # noqa: BLE001
            pool.terminate()
            pool.join()
            _log("ERROR", f"下载失败: {exc}")
            failed.append(str(exc))
        else:
            pool.close()
            pool.join()
        print()

        for message in failed:
            _log("ERROR", message)
        if chunk_failures:
            _log("ERROR", f"{len(chunk_failures)} 个分片重试耗尽仍未成功：")
            for rel_path, offset, error in chunk_failures[:10]:
                _log("ERROR", f"  {rel_path}@{offset}: {error}")
            if len(chunk_failures) > 10:
                _log("ERROR", f"  ...（其余 {len(chunk_failures) - 10} 个略）")

    # -- 校验并落 complete 标记 -------------------------------------------
    _log("INFO", "校验文件 ...")
    check_args = [(f.rel_path, f.size, f.sha256) for f in files]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(4, len(check_args) or 1)) as pool:
        results = pool.map(_verify_one, check_args)

    bad: list[str] = []
    for rel_path, ok, detail in results:
        if ok:
            _complete_marker(model_dir, rel_path).touch()
        else:
            bad.append(f"{rel_path}: {detail}")
            _log("ERROR", f"  {rel_path}: {detail}")

    # 清掉已完成文件的分片续传记录（.complete 标记保留，供下次运行跳过整个文件）
    for remote in files:
        state = _state_file(model_dir, remote.rel_path)
        if _complete_marker(model_dir, remote.rel_path).is_file() and state.is_file():
            state.unlink()

    if bad:
        _log("ERROR", f"{len(bad)} 个文件未通过校验，重新运行以下命令可续传修复：")
        _log("ERROR", f"  python {Path(__file__).name} download")
        return 1

    elapsed = time.perf_counter() - started
    _log("INFO", f"全部完成: {len(files)} 个文件 / {_human(total_size)}，用时 {elapsed / 60:.1f} 分钟")
    _log("INFO", f"模型目录: {model_dir}")
    return 0


def verify(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir).resolve()
    _set_model_dir(model_dir)
    endpoint = select_endpoint(args.endpoint)
    files = fetch_remote_files(endpoint)
    _log("INFO", f"校验 {len(files)} 个文件（大文件为 sha256，小文件为大小）...")

    import multiprocessing as mp

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(4, len(files))) as pool:
        results = pool.map(
            _verify_one, [(f.rel_path, f.size, f.sha256) for f in files]
        )

    failed = 0
    for rel_path, ok, detail in results:
        _log("INFO" if ok else "ERROR", f"  {rel_path:44s} {detail}")
        if not ok:
            failed += 1
    if failed:
        _log("ERROR", f"{failed} 个文件校验失败。")
        return 1
    _log("INFO", "全部通过。")
    return 0


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    model_path: str
    image_paths: list[str]
    prompt: str
    response: str
    generation_config: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "image_paths": self.image_paths,
            "prompt": self.prompt,
            "response": self.response,
            "generation_config": self.generation_config,
            "runtime": self.runtime,
        }


class Qwen3VLInference:
    """加载一次 Qwen3-VL，之后可反复调用。

    Parameters
    ----------
    model_path : str
        本地模型目录（即 ./model）。
    min_pixels / max_pixels : int or None
        视觉预处理的像素范围。默认 None —— 沿用模型自带的
        preprocessor_config.json，不覆盖官方默认值。
    attn_implementation : str
        "sdpa"（默认，快）或 "eager"（需要导出 attention 时用）。
    dtype : str
        "auto" / "bfloat16" / "float16"。
    """

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        attn_implementation: str = "sdpa",
        dtype: str = "auto",
    ):
        import torch
        from transformers import AutoProcessor

        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.attn_implementation = attn_implementation
        self.attn_impl_used = attn_implementation

        if not Path(model_path).is_dir():
            raise FileNotFoundError(
                f"模型目录不存在: {model_path}\n请先运行: python {Path(__file__).name} download"
            )

        processor_kwargs: dict[str, Any] = {"local_files_only": True}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels

        _log("INFO", f"加载 processor: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)

        _log("INFO", f"加载模型: {model_path} (attn={attn_implementation})")
        self.model, self.dtype, self.dtype_name = self._load_model(dtype)
        self.model.eval()
        _log("INFO", f"模型加载完成: dtype={self.dtype_name}, device_map=auto")
        self._report_device_map()

    # -- 内部工具 ---------------------------------------------------------

    def _load_model(self, dtype: str):
        import torch
        from transformers import Qwen3VLForConditionalGeneration

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        if dtype not in dtype_map:
            raise ValueError(f"未知 dtype: {dtype}（可选 auto/bfloat16/float16）")

        weight_dtype = dtype_map[dtype]
        # 先按用户指定的 attention 实现加载；失败则退回 sdpa
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=weight_dtype,
                device_map="auto",
                attn_implementation=self.attn_implementation,
                local_files_only=True,
            )
        except Exception as exc:                              # noqa: BLE001
            if self.attn_implementation == "sdpa":
                raise
            _log("WARN", f"attn_implementation={self.attn_implementation} 加载失败: {exc}")
            _log("WARN", "回退到 sdpa ...")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=weight_dtype,
                device_map="auto",
                attn_implementation="sdpa",
                local_files_only=True,
            )
            self.attn_impl_used = "sdpa"

        # 模型自带的 generation_config 是给采样准备的
        # （do_sample=true, temperature=0.7, top_p=0.8, top_k=20），
        # 而本接口固定贪心解码，这些值既不生效、又会让 transformers 每次
        # 报一条 "generation flags are not valid" 的警告。
        # 这里把它们置回中性默认值。注意不能直接删属性 ——
        # GenerationConfig 内部会无条件读取这些字段，删了会 AttributeError。
        generation_config = model.generation_config
        generation_config.do_sample = False
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
        generation_config.top_k = 50

        resolved = getattr(model, "dtype", weight_dtype)
        name = str(resolved).replace("torch.", "") if hasattr(resolved, "__str__") else str(dtype)
        return model, resolved, name

    def _report_device_map(self) -> None:
        if hasattr(self.model, "hf_device_map"):
            devices = set(str(v) for v in self.model.hf_device_map.values())
            _log("INFO", f"device_map 覆盖设备: {sorted(devices)}")
        else:
            try:
                _log("INFO", f"全部参数位于: {next(self.model.parameters()).device}")
            except StopIteration:
                _log("WARN", "无法确定模型设备。")

    def _input_device(self):
        import torch

        try:
            return next(self.model.model.embed_tokens.parameters()).device
        except (AttributeError, StopIteration):
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_messages(self, prompt: str, image_paths: list[str]) -> list[dict[str, Any]]:
        from PIL import Image

        content: list[dict[str, Any]] = []
        for path in image_paths:
            image_path = Path(path)
            if not image_path.is_file():
                raise FileNotFoundError(f"图片不存在: {path}")
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as exc:                          # noqa: BLE001
                raise ValueError(f"图片无法读取: {path} — {exc}") from exc
            content.append({"type": "image", "image": str(image_path.resolve())})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _build_inputs(self, messages: list[dict[str, Any]]):
        """优先用 transformers 原生模板；旧版回退到 qwen_vl_utils。"""
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except (TypeError, ValueError, KeyError) as exc:
            _log("WARN", f"apply_chat_template 原生路径失败({type(exc).__name__})，回退 qwen_vl_utils")
            from qwen_vl_utils import process_vision_info

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            return self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

    # -- 公开接口 ---------------------------------------------------------

    def generate(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        max_new_tokens: int = 256,
    ) -> GenerationResult:
        import torch

        image_paths = list(image_paths or [])
        started = time.perf_counter()
        messages = self._build_messages(prompt, image_paths)
        inputs = self._build_inputs(messages)
        inputs = inputs.to(self._input_device())

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        trimmed = [out[input_len:] for out in generated]
        response = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        elapsed = time.perf_counter() - started
        new_tokens = int(trimmed[0].shape[0])
        peak_gpu = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

        return GenerationResult(
            model_path=self.model_path,
            image_paths=image_paths,
            prompt=prompt,
            response=response,
            generation_config=dict(gen_kwargs),
            runtime={
                "device": str(self._input_device()),
                "dtype": self.dtype_name,
                "attn_implementation": self.attn_impl_used,
                "elapsed_seconds": round(elapsed, 3),
                "new_tokens": new_tokens,
                "tokens_per_second": round(new_tokens / elapsed, 2) if elapsed > 0 else 0.0,
                "peak_gpu_memory_gb": round(peak_gpu, 3),
                "num_images": len(image_paths),
            },
        )

    def forward_analysis(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
    ) -> dict[str, Any]:
        """单次前向，返回 logits / hidden_states / attentions。

        供研究使用（PANL、置信度探针等）。注意 output_attentions 需要
        eager attention 且显存开销大，默认关闭。
        """
        import torch

        if output_attentions and self.attn_impl_used != "eager":
            raise RuntimeError(
                f"output_attentions=True 需要 eager attention，当前为 {self.attn_impl_used}。"
                "请用 attn_implementation='eager' 构造。"
            )

        image_paths = list(image_paths or [])
        messages = self._build_messages(prompt, image_paths)
        inputs = self._build_inputs(messages).to(self._input_device())

        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                use_cache=False,
            )

        result: dict[str, Any] = {
            "logits": outputs.logits,
            "hidden_states": list(outputs.hidden_states) if output_hidden_states else None,
            "attentions": list(outputs.attentions) if output_attentions else None,
            "shapes": {
                "input_ids": tuple(inputs["input_ids"].shape),
                "logits": tuple(outputs.logits.shape),
                "num_hidden_states": len(outputs.hidden_states or []),
                "hidden_state_shape": tuple(outputs.hidden_states[0].shape)
                if output_hidden_states and outputs.hidden_states
                else None,
            },
            "metadata": {
                "model_path": self.model_path,
                "prompt": prompt,
                "image_paths": image_paths,
                "dtype": self.dtype_name,
                "attn_implementation": self.attn_impl_used,
            },
        }
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-dir",
        default=str(MODEL_DIR),
        help=f"模型存放目录（默认: {MODEL_DIR}）",
    )
    parser.add_argument(
        "--endpoint",
        choices=["auto", "hf", "mirror"],
        default="auto",
        help="下载端点: auto 自动测速选择 / hf 官方站 / mirror hf-mirror.com",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-8B-Instruct 本地部署：40 进程分片下载 + Transformers 推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- download --
    p_dl = sub.add_parser("download", help="40 进程分片下载模型到 model/")
    _add_common(p_dl)
    p_dl.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"并发下载进程数（默认 {DEFAULT_WORKERS}）",
    )
    p_dl.add_argument(
        "--chunk-mb", type=int, default=DEFAULT_CHUNK_MB,
        help=f"分片大小 MB（默认 {DEFAULT_CHUNK_MB}）",
    )
    p_dl.add_argument(
        "--status", action="store_true", help="只显示各文件进度，不下载",
    )
    p_dl.set_defaults(func=download)

    # -- verify --
    p_vf = sub.add_parser("verify", help="校验已下载文件的 sha256/大小")
    _add_common(p_vf)
    p_vf.set_defaults(func=verify)

    # -- infer --
    p_in = sub.add_parser("infer", help="文本/图片推理")
    _add_common(p_in)
    p_in.add_argument("--prompt", required=True, help="文本提示词")
    p_in.add_argument(
        "--image", action="append", default=None,
        help="图片路径，可重复传入以做多图推理",
    )
    p_in.add_argument("--max-new-tokens", type=int, default=256, help="最大生成 token 数")
    p_in.add_argument("--min-pixels", type=int, default=None, help="视觉预处理最小像素（默认沿用模型配置）")
    p_in.add_argument("--max-pixels", type=int, default=None, help="视觉预处理最大像素（默认沿用模型配置）")
    p_in.add_argument(
        "--attn-implementation", choices=["sdpa", "eager", "flash_attention_2"],
        default="sdpa", help="注意力实现（默认 sdpa）",
    )
    p_in.add_argument(
        "--dtype", choices=["auto", "bfloat16", "float16"], default="auto",
        help="模型精度（默认 auto）",
    )
    p_in.add_argument("--output", default=None, help="把结果 JSON 写到该路径")
    p_in.set_defaults(func=run_infer)

    # -- forward --
    p_fw = sub.add_parser("forward", help="单次前向，导出 hidden_states（研究用）")
    _add_common(p_fw)
    p_fw.add_argument("--prompt", required=True, help="文本提示词")
    p_fw.add_argument("--image", action="append", default=None, help="图片路径，可重复传入")
    p_fw.add_argument("--min-pixels", type=int, default=None)
    p_fw.add_argument("--max-pixels", type=int, default=None)
    p_fw.add_argument(
        "--attn-implementation", choices=["sdpa", "eager", "flash_attention_2"],
        default="eager", help="默认 eager（便于导出 attention）",
    )
    p_fw.add_argument("--dtype", choices=["auto", "bfloat16", "float16"], default="auto")
    p_fw.add_argument("--attentions", action="store_true", help="同时导出 attention（显存开销大）")
    p_fw.add_argument("--output", required=True, help="保存 .pt 文件的路径")
    p_fw.set_defaults(func=run_forward)

    return parser.parse_args(argv)


def _print_result(result: GenerationResult) -> None:
    print()
    print("=" * 66)
    print(f"PROMPT: {result.prompt}")
    for path in result.image_paths:
        print(f"IMAGE:  {path}")
    print(f"RESPONSE:\n{result.response}")
    print("=" * 66)
    runtime = result.runtime
    print(f"耗时: {runtime['elapsed_seconds']:.2f}s | "
          f"生成 {runtime['new_tokens']} tokens | {runtime['tokens_per_second']:.1f} tok/s")
    print(f"显存峰值: {runtime['peak_gpu_memory_gb']:.2f} GB | "
          f"dtype: {runtime['dtype']} | attn: {runtime['attn_implementation']}")
    print(f"设备: {runtime['device']}")
    print("=" * 66)


def run_infer(args: argparse.Namespace) -> int:
    import torch
    from PIL import Image

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        _log("ERROR", f"模型目录不存在: {model_dir}")
        _log("ERROR", f"请先运行: python {Path(__file__).name} download")
        return 1

    try:
        engine = Qwen3VLInference(
            model_path=str(model_dir),
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            attn_implementation=args.attn_implementation,
            dtype=args.dtype,
        )
    except Exception as exc:                                  # noqa: BLE001
        _log("ERROR", f"初始化失败: {exc}")
        return 1

    try:
        result = engine.generate(
            prompt=args.prompt,
            image_paths=args.image,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as exc:                                  # noqa: BLE001
        _log("ERROR", f"推理失败: {exc}")
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                _log("ERROR", f"  已分配显存: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
                _log("ERROR", f"  已保留显存: {torch.cuda.memory_reserved() / 1e9:.1f} GB")
            for path in args.image or []:
                try:
                    with Image.open(path) as img:
                        _log("ERROR", f"  图片 {path} 尺寸: {img.size}")
                except Exception:                             # noqa: BLE001
                    pass
            _log("ERROR", "  建议: 降低 --max-pixels、减少图片数量，或调小 --max-new-tokens")
        return 1

    _print_result(result)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _log("INFO", f"结果已保存: {out}")
    return 0


def run_forward(args: argparse.Namespace) -> int:
    import torch

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        _log("ERROR", f"模型目录不存在: {model_dir}")
        return 1

    try:
        engine = Qwen3VLInference(
            model_path=str(model_dir),
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            attn_implementation=args.attn_implementation,
            dtype=args.dtype,
        )
    except Exception as exc:                                  # noqa: BLE001
        _log("ERROR", f"初始化失败: {exc}")
        return 1

    try:
        analysis = engine.forward_analysis(
            prompt=args.prompt,
            image_paths=args.image,
            output_hidden_states=True,
            output_attentions=args.attentions,
        )
    except Exception as exc:                                  # noqa: BLE001
        _log("ERROR", f"前向失败: {exc}")
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "logits": analysis["logits"].detach().cpu(),
        "hidden_states": [h.detach().cpu() for h in analysis["hidden_states"]]
        if analysis["hidden_states"]
        else None,
        "attentions": [a.detach().cpu() for a in analysis["attentions"]]
        if analysis["attentions"]
        else None,
        "shapes": analysis["shapes"],
        "metadata": analysis["metadata"],
    }
    torch.save(payload, out)

    _log("INFO", f"已保存: {out}")
    for key, value in analysis["shapes"].items():
        _log("INFO", f"  {key}: {value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        _log("WARN", "已中断。")
        return 130
    except Exception as exc:                                  # noqa: BLE001
        _log("ERROR", f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
