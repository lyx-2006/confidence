#!/usr/bin/env python3
"""
Smoke test for Qwen2.5-VL-7B-Instruct deployment.

Verifies:
  1. CUDA available
  2. Processor loads (via QwenVLInference)
  3. Model loads (via QwenVLInference — only ONCE)
  4. Model dtype
  5. Model device map
  6. Pure text inference
  7. Image-text inference (auto-generates test image if --image not provided)
  8. Output is non-empty
  9. do_sample=False is respected
 10. Peak GPU memory recorded

Usage:
  python test_inference.py
  python test_inference.py --image /path/to/test.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def create_test_image(output_path: str) -> str:
    """Create a simple test image: white background with a red square."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([150, 150, 362, 362], fill="red")
    img.save(output_path)
    print(f"[TEST] Created test image: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL smoke test")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a test image. If not provided, one will be auto-generated.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/autodl-tmp/qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct",
        help="Path to local model checkpoint directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("Qwen2.5-VL-7B-Instruct Smoke Test")
    print("=" * 60)

    results: dict = {"tests": {}, "summary": {}}

    # --- 1. Check CUDA ---
    print("\n[1/10] Checking CUDA ...")
    if not torch.cuda.is_available():
        print("  FAIL: CUDA is not available.")
        results["tests"]["cuda"] = "FAIL"
        return 1
    print(f"  OK: CUDA available. GPU: {torch.cuda.get_device_name(0)}")
    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  GPU total memory: {gpu_total:.1f} GB")
    results["tests"]["cuda"] = "OK"

    # --- 2-3. Load model ONCE via QwenVLInference ---
    print("\n[2-3/10] Loading processor + model (single load) ...")
    from inference import QwenVLInference

    try:
        inference = QwenVLInference(
            model_path=args.model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        print("  OK: Processor and model loaded.")
        results["tests"]["processor_load"] = "OK"
        results["tests"]["model_load"] = "OK"
    except Exception as e:
        print(f"  FAIL: {e}")
        results["tests"]["processor_load"] = f"FAIL: {e}"
        results["tests"]["model_load"] = f"FAIL: {e}"
        return 1

    # --- 4. Model dtype ---
    print("\n[4/10] Checking model dtype ...")
    results["model_dtype"] = inference.dtype_name
    print(f"  OK: Model dtype = {inference.dtype_name}")

    # --- 5. Device map ---
    print("\n[5/10] Checking device map ...")
    model = inference.model
    if hasattr(model, "hf_device_map"):
        dm = model.hf_device_map
        # Summarize: count per device
        from collections import Counter

        device_counts = Counter(dm.values())
        device_summary = {str(k): v for k, v in device_counts.items()}
        print(f"  device_map summary (layer counts per device): {device_summary}")
        results["device_map"] = str(device_summary)

        offload_devices = {"cpu", "disk"}
        offloaded = {k: v for k, v in dm.items() if v in offload_devices}
        if offloaded:
            print(f"  WARN: {len(offloaded)} layers offloaded to CPU/disk!")
            print(f"  Offloaded: {list(offloaded.keys())[:5]}...")  # first 5 only
            results["device_map_warning"] = f"{len(offloaded)} layers offloaded"
        else:
            print("  OK: All layers on GPU(s).")
            results["device_map_warning"] = None
    else:
        dev = next(model.parameters()).device
        print(f"  All parameters on: {dev}")
        results["device_map"] = str(dev)
        results["device_map_warning"] = None

    # --- 6. Pure text inference ---
    print("\n[6/10] Pure text inference ...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        text_result = inference.generate(
            prompt="用一句话介绍你自己。",
            image_path=None,
            max_new_tokens=64,
        )
        text_elapsed = time.perf_counter() - t0
        print(f"  Prompt: 用一句话介绍你自己。")
        print(f"  Response: {text_result.response}")
        print(f"  Time: {text_elapsed:.2f}s")
        print(f"  OK: Text inference completed.")
        results["tests"]["text_inference"] = "OK"
        results["text_response"] = text_result.response
        results["text_elapsed"] = round(text_elapsed, 3)
    except Exception as e:
        print(f"  FAIL: {e}")
        results["tests"]["text_inference"] = f"FAIL: {e}"

    # --- 7. Image inference ---
    print("\n[7/10] Image-text inference ...")
    test_image = args.image
    if not test_image:
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        test_image = create_test_image(str(output_dir / "generated_test_image.png"))

    t0_img = time.perf_counter()
    try:
        img_result = inference.generate(
            prompt="图中正方形是什么颜色？只回答颜色。",
            image_path=test_image,
            max_new_tokens=32,
        )
        img_elapsed = time.perf_counter() - t0_img
        print(f"  Prompt: 图中正方形是什么颜色？只回答颜色。")
        print(f"  Image: {test_image}")
        print(f"  Response: {img_result.response}")
        print(f"  Time: {img_elapsed:.2f}s")
        print(f"  OK: Image inference completed.")
        results["tests"]["image_inference"] = "OK"
        results["image_response"] = img_result.response
        results["image_elapsed"] = round(img_elapsed, 3)
    except Exception as e:
        print(f"  FAIL: {e}")
        results["tests"]["image_inference"] = f"FAIL: {e}"

    # --- 8. Check output non-empty ---
    print("\n[8/10] Checking output non-empty ...")
    text_ok = (
        results.get("tests", {}).get("text_inference") == "OK"
        and len(results.get("text_response", "")) > 0
    )
    image_ok = (
        results.get("tests", {}).get("image_inference") == "OK"
        and len(results.get("image_response", "")) > 0
    )
    print(f"  Text output non-empty: {text_ok}")
    print(f"  Image output non-empty: {image_ok}")
    results["tests"]["output_non_empty"] = "OK" if (text_ok and image_ok) else "FAIL"

    # --- 9. Verify do_sample=False ---
    print("\n[9/10] Verifying do_sample=False ...")
    gen_config = text_result.generation_config
    is_deterministic = gen_config.get("do_sample") == False  # noqa: E712
    has_temp = any(
        k in gen_config for k in ("temperature", "top_p", "top_k")
    )
    if is_deterministic and not has_temp:
        print("  OK: do_sample=False, no temperature/top_p/top_k set.")
        results["tests"]["do_sample"] = "OK"
    else:
        print(f"  FAIL: gen_config = {gen_config}")
        results["tests"]["do_sample"] = "FAIL"

    # --- 10. Peak GPU memory ---
    print("\n[10/10] Peak GPU memory ...")
    peak_gpu = 0.0
    if torch.cuda.is_available():
        peak_gpu = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  Peak GPU memory: {peak_gpu:.2f} GB")
    results["peak_gpu_memory_gb"] = round(peak_gpu, 3)

    # --- Summary ---
    print("\n" + "=" * 60)
    all_passed = all(
        v == "OK" for v in results.get("tests", {}).values()
    )
    results["summary"]["all_passed"] = all_passed
    results["summary"]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        for name, status in results["tests"].items():
            if status != "OK":
                print(f"  FAILED: {name} — {status}")

    print(f"Peak GPU memory: {peak_gpu:.2f} GB")
    print("=" * 60)

    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = output_dir / "smoke_test.json"
    smoke_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[INFO] Smoke test results saved to {smoke_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
