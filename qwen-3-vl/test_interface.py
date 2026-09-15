#!/usr/bin/env python3
"""interface.py 的下载器测试。

重点验证「多进程按偏移写同一个文件」这件事本身是对的 ——
分片写错偏移在 sha256 校验前不会暴露，所以这里用 curl 作为独立的
参照实现，逐字节比对。

运行:
  pytest test_interface.py -v
  pytest test_interface.py -v -m "not network"     # 跳过联网用例
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interface as I  # noqa: E402

BIG_FILE = "model-00004-of-00004.safetensors"

# 需要联网的用例（本机需能访问 hf-mirror.com），其余用例离线也能跑
network = pytest.mark.network


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def endpoint() -> I.Endpoint:
    ep = I.Endpoint(I.ENDPOINT_MIRROR, False, "hf-mirror.com (direct)")
    try:
        if I._probe_endpoint(ep, 1 << 20) <= 0:
            pytest.skip("hf-mirror.com 不可达")
    except Exception as exc:                                  # noqa: BLE001
        pytest.skip(f"网络不可用: {exc}")
    return ep


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    target = tmp_path / "model"
    (target / I.STATE_DIRNAME).mkdir(parents=True)
    I._set_model_dir(target)
    yield target
    I._set_model_dir(I.ROOT / "model")


def curl_range(ep: I.Endpoint, rel_path: str, offset: int, length: int) -> bytes:
    """用 curl 独立取一段字节，作为比对基准。"""
    cmd = [
        "curl", "-sL", "--max-time", "120", "--noproxy", "*",
        "-r", f"{offset}-{offset + length - 1}",
        ep.resolve_url(rel_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return proc.stdout


def run_pool(tasks: list[I.ChunkTask], workers: int = 8) -> list:
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        return list(pool.imap_unordered(I._download_chunk, tasks, chunksize=1))


# ---------------------------------------------------------------------------
# 分片写偏移的正确性
# ---------------------------------------------------------------------------

@network
def test_sharded_write_offsets_match_reference(endpoint, model_dir):
    """把 2MB 拆成 8 个 256KB 分片乱序写入，结果应与 curl 取的整段逐字节一致。"""
    total = 2 << 20
    chunk = 256 << 10
    relative = BIG_FILE
    target = model_dir / relative

    # 先预分配，模拟真实流程
    fd = os.open(target, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        os.ftruncate(fd, total)
    finally:
        os.close(fd)

    offsets = list(range(0, total, chunk))
    tasks = [
        I.ChunkTask(relative, off, chunk, endpoint.resolve_url(relative), endpoint.use_proxy)
        for off in offsets
    ]
    # 乱序下发，确保偏移不是靠「顺序写入」蒙对的
    import random

    shuffled = tasks[:]
    random.Random(0).shuffle(shuffled)

    results = run_pool(shuffled, workers=8)
    assert len(results) == len(tasks)
    assert all(r.error is None for r in results), [r.error for r in results if r.error]
    assert sum(r.written for r in results) == total

    reference = curl_range(endpoint, relative, 0, total)
    assert len(reference) == total
    assert target.read_bytes() == reference, "分片写入结果与参照实现不一致"


@network
def test_offset_is_actually_used(endpoint, model_dir):
    """同一分片写到不同偏移，内容应不同 —— 防止「写到了 0」这类错位。"""
    relative = BIG_FILE
    length = 1 << 20
    target = model_dir / relative
    fd = os.open(target, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        os.ftruncate(fd, 8 << 20)
    finally:
        os.close(fd)

    url = endpoint.resolve_url(relative)
    run_pool(
        [
            I.ChunkTask(relative, 0, length, url, endpoint.use_proxy),
            I.ChunkTask(relative, 4 << 20, length, url, endpoint.use_proxy),
        ],
        workers=2,
    )

    blob = target.read_bytes()
    head = blob[:length]
    far = blob[4 << 20 : 5 << 20]
    assert head == curl_range(endpoint, relative, 0, length)
    assert far == curl_range(endpoint, relative, 4 << 20, length)
    assert head != far


# ---------------------------------------------------------------------------
# 失败处理
# ---------------------------------------------------------------------------

def test_failing_chunk_reports_error_instead_of_raising(tmp_path, monkeypatch):
    """单个分片彻底失败时应回一个带 error 的结果，而不是抛异常炸掉整轮。"""
    import multiprocessing as mp

    model_dir = tmp_path / "model"
    (model_dir / I.STATE_DIRNAME).mkdir(parents=True)
    monkeypatch.setattr(I, "MODEL_DIR", model_dir)
    monkeypatch.setattr(I, "MAX_ATTEMPTS", 1)          # 别真等 6 轮退避

    target = model_dir / "x.bin"
    fd = os.open(target, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        os.ftruncate(fd, 100)
    finally:
        os.close(fd)

    # 指向本机必然拒绝连接的端口，快速失败
    task = I.ChunkTask("x.bin", 0, 100, "http://127.0.0.1:9/nope", False)
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=1) as pool:
        results = list(pool.imap_unordered(I._download_chunk, [task]))

    assert len(results) == 1
    assert results[0].error is not None, "失败应体现在 error 字段里"
    assert results[0].written == 0
    # 失败不该留下续传记账，否则会误以为分片已完成
    assert not I._state_file(model_dir, "x.bin").exists()


# ---------------------------------------------------------------------------
# 计划 / 续传
# ---------------------------------------------------------------------------

def test_plan_resumes_and_skips_completed(tmp_path, endpoint):
    model_dir = tmp_path / "model"
    (model_dir / I.STATE_DIRNAME).mkdir(parents=True)
    I._set_model_dir(model_dir)

    files = [
        I.RemoteFile("a.bin", 100, None),
        I.RemoteFile("b.bin", 250, "deadbeef"),
    ]
    tasks, pending, finished = I.plan_chunks(files, endpoint, model_dir, chunk_size=100)
    assert len(tasks) == 1 + 3          # a.bin 1 片；b.bin 3 片
    assert finished == []

    # b.bin 的前两片记为已完成
    state = I._state_file(model_dir, "b.bin")
    state.write_text("0\n100\n", encoding="utf-8")

    tasks2, pending2, _ = I.plan_chunks(files, endpoint, model_dir, chunk_size=100)
    offsets = sorted(t.offset for t in tasks2 if t.rel_path == "b.bin")
    assert offsets == [200], f"续传应只补最后一片，实际 {offsets}"
    assert {t.rel_path for t in tasks2} == {"a.bin", "b.bin"}

    # 完整 + complete 标记 → 整文件跳过
    for name, size in (("a.bin", 100), ("b.bin", 250)):
        (model_dir / name).write_bytes(b"x" * size)
        I._complete_marker(model_dir, name).touch()
    tasks3, _, finished3 = I.plan_chunks(files, endpoint, model_dir, chunk_size=100)
    assert tasks3 == []
    assert {f.rel_path for f in finished3} == {"a.bin", "b.bin"}

    I._set_model_dir(I.ROOT / "model")


def test_tasks_are_interleaved_across_files(tmp_path, endpoint):
    """轮转排列：大文件应同时推进，而不是一个下完再下一个。"""
    model_dir = tmp_path / "model"
    (model_dir / I.STATE_DIRNAME).mkdir(parents=True)
    files = [I.RemoteFile("a.bin", 1000, None), I.RemoteFile("b.bin", 1000, None)]
    tasks, _, _ = I.plan_chunks(files, endpoint, model_dir, chunk_size=100)
    assert [t.rel_path for t in tasks[:4]] == ["a.bin", "b.bin", "a.bin", "b.bin"]


def test_chunk_task_is_picklable():
    import pickle

    task = I.ChunkTask("f.bin", 128, 64, "https://example.com/x", True)
    assert pickle.loads(pickle.dumps(task)) == task


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def test_verify_detects_corruption(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setattr(I, "MODEL_DIR", model_dir)

    good = b"hello world" * 100
    (model_dir / "good.bin").write_bytes(good)
    (model_dir / "bad.bin").write_bytes(b"corrupted" + good[9:])
    (model_dir / "short.bin").write_bytes(b"abc")

    digest = hashlib.sha256(good).hexdigest()
    assert I._verify_one(("good.bin", len(good), digest))[1] is True
    assert I._verify_one(("bad.bin", len(good), digest))[1] is False
    assert I._verify_one(("short.bin", len(good), digest))[1] is False
    assert I._verify_one(("missing.bin", 5, None))[1] is False
    assert I._verify_one(("good.bin", len(good), None))[1] is True   # 只校验大小


# ---------------------------------------------------------------------------
# download() 端到端（只跑小文件，避免真的下 17GB）
# ---------------------------------------------------------------------------

@network
def test_download_end_to_end_small_files(tmp_path, endpoint, monkeypatch, capsys):
    model_dir = tmp_path / "model"
    all_files = I.fetch_remote_files(endpoint)
    small = [f for f in all_files if f.size < 1 << 20]
    assert small, "没有小文件可用于测试"

    monkeypatch.setattr(I, "fetch_remote_files", lambda ep: small)
    args = argparse.Namespace(
        model_dir=str(model_dir),
        endpoint="mirror",
        workers=8,
        chunk_mb=1,
        status=False,
    )
    assert I.download(args) == 0

    for remote in small:
        path = model_dir / remote.rel_path
        assert path.is_file(), f"{remote.rel_path} 未下载"
        assert path.stat().st_size == remote.size
        assert I._complete_marker(model_dir, remote.rel_path).is_file()

    # 分片续传记录应清理掉，完成标记要保留（下次运行靠它跳过整个文件）
    state_dir = model_dir / I.STATE_DIRNAME
    assert not list(state_dir.glob("*.done")), "完成文件的 .done 记录应被清理"

    # 再跑一次应直接跳过，不再下载
    assert I.download(args) == 0
    assert "已完成" in capsys.readouterr().out


@network
def test_download_status_is_read_only(tmp_path, endpoint, monkeypatch):
    model_dir = tmp_path / "model"
    small = [f for f in I.fetch_remote_files(endpoint) if f.size < 1 << 20]
    monkeypatch.setattr(I, "fetch_remote_files", lambda ep: small)

    args = argparse.Namespace(
        model_dir=str(model_dir), endpoint="mirror", workers=8, chunk_mb=1, status=True,
    )
    assert I.download(args) == 0
    for remote in small:
        assert not (model_dir / remote.rel_path).exists(), "--status 不应写任何数据"


def test_cli_parses():
    args = I.parse_args(["download"])
    assert args.workers == I.DEFAULT_WORKERS == 40
    assert args.chunk_mb == I.DEFAULT_CHUNK_MB
    assert args.endpoint == "auto"

    args = I.parse_args(["infer", "--prompt", "hi", "--image", "a.jpg", "--image", "b.jpg"])
    assert args.image == ["a.jpg", "b.jpg"]
    assert args.func is I.run_infer
