#!/usr/bin/env python3
"""Standalone, low-concurrency DeepSeek connectivity diagnostic.

This script does not load Qwen, does not call the colour-pool generator, and
does not write checkpoints. It deliberately disables SDK retries so the
observed latency and error belong to one HTTP request.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


MODEL = "glm5.2"


def _load_config() -> dict:
    """Load API configuration from the project root api_config.json."""
    config_path = Path(__file__).resolve().parent / "api_config.json"
    if not config_path.exists():
        print(f"[DeepSeekTest] missing config file: {config_path}", file=sys.stderr)
        print("[DeepSeekTest] create api_config.json with 'api_key' and 'base_url' fields", file=sys.stderr)
        sys.exit(2)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose DeepSeek API timeout and concurrency")
    parser.add_argument(
        "--prompt",
        default='Reply with JSON only: {"ok": true, "message": "connection test"}',
        help="Test prompt; keep it short when diagnosing timeout",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request HTTP timeout in seconds")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=1, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests; use 1 first")
    return parser.parse_args()


def one_request(client, prompt: str, model: str, max_tokens: int, index: int) -> dict[str, object]:
    started = time.perf_counter()
    print(f"[DeepSeekTest] request={index} started", flush=True)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        elapsed = time.perf_counter() - started
        print(
            f"[DeepSeekTest] request={index} succeeded elapsed={elapsed:.2f}s "
            f"response_chars={len(content)}",
            flush=True,
        )
        print(f"[DeepSeekTest] response={content[:500]!r}", flush=True)
        return {"ok": True, "elapsed": elapsed, "error": None}
    except Exception as exc:
        elapsed = time.perf_counter() - started
        error_type = type(exc).__name__
        print(
            f"[DeepSeekTest] request={index} failed elapsed={elapsed:.2f}s "
            f"error_type={error_type}: {exc}",
            flush=True,
        )
        return {"ok": False, "elapsed": elapsed, "error": f"{error_type}: {exc}"}


def main() -> int:
    args = parse_args()
    if args.timeout <= 0 or args.max_tokens <= 0 or args.repeat <= 0 or args.concurrency <= 0:
        print("[DeepSeekTest] timeout, max-tokens, repeat, and concurrency must be positive", file=sys.stderr)
        return 2

    config = _load_config()
    api_key = config.get("api_key", "")
    base_url = config.get("base_url", "")

    if not api_key or not base_url:
        print("[DeepSeekTest] missing api_key or base_url in api_config.json", file=sys.stderr)
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print("[DeepSeekTest] missing Python package: openai", file=sys.stderr)
        return 2

    print(
        f"[DeepSeekTest] base_url={base_url} model={args.model} "
        f"timeout={args.timeout}s repeat={args.repeat} concurrency={args.concurrency}",
        flush=True,
    )
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=args.timeout,
        max_retries=0,
    )

    request_args = (client, args.prompt, args.model, args.max_tokens)
    if args.concurrency == 1:
        results = [one_request(*request_args, index) for index in range(1, args.repeat + 1)]
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            results = list(
                executor.map(
                    lambda index: one_request(*request_args, index),
                    range(1, args.repeat + 1),
                )
            )

    succeeded = sum(bool(result["ok"]) for result in results)
    failed = len(results) - succeeded
    print(f"[DeepSeekTest] summary succeeded={succeeded} failed={failed}", flush=True)
    if failed and args.concurrency > 1:
        print("[DeepSeekTest] compare with --concurrency 1; failures only under concurrency indicate rate limiting or service capacity", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
