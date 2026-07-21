#!/usr/bin/env python3
"""
Environment check script for Qwen2.5-VL-7B-Instruct deployment.

Verifies Python, PyTorch, CUDA, GPU, and required package imports.
Run before any inference to confirm the environment is ready.
"""

import sys
import platform


def check_python() -> dict:
    version = sys.version.split()[0]
    return {"Python": version}


def check_pytorch() -> dict:
    import torch

    result = {}
    result["PyTorch"] = torch.__version__
    result["CUDA available"] = torch.cuda.is_available()
    result["PyTorch CUDA"] = torch.version.cuda or "N/A"

    if torch.cuda.is_available():
        result["GPU"] = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        result["GPU memory"] = f"{total_mem:.0f} GB"
        free_mem = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated(0)
        ) / (1024**3)
        result["GPU free memory"] = f"{free_mem:.1f} GB"
    else:
        result["GPU"] = "N/A"
        result["GPU memory"] = "N/A"
        result["GPU free memory"] = "N/A"

    result["BF16 supported"] = (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    return result


def check_transformers() -> dict:
    result = {}
    try:
        import transformers

        result["Transformers"] = transformers.__version__
    except ImportError as e:
        result["Transformers"] = f"NOT INSTALLED: {e}"
    return result


def check_qwen_import() -> dict:
    result = {}
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401

        result["Qwen2_5_VLForConditionalGeneration import"] = "OK"
    except ImportError as e:
        result["Qwen2_5_VLForConditionalGeneration import"] = f"FAIL: {e}"

    try:
        from transformers import AutoProcessor  # noqa: F401

        result["AutoProcessor import"] = "OK"
    except ImportError as e:
        result["AutoProcessor import"] = f"FAIL: {e}"
    return result


def check_qwen_vl_utils() -> dict:
    result = {}
    try:
        import qwen_vl_utils  # noqa: F401

        result["qwen_vl_utils import"] = "OK"
    except ImportError as e:
        result["qwen_vl_utils import"] = f"FAIL: {e}"

    try:
        from qwen_vl_utils import process_vision_info  # noqa: F401

        result["process_vision_info import"] = "OK"
    except ImportError as e:
        result["process_vision_info import"] = f"FAIL: {e}"
    return result


def check_other_packages() -> dict:
    result = {}
    packages = ["accelerate", "huggingface_hub", "safetensors", "PIL"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            result[pkg] = ver
        except ImportError:
            result[pkg] = "NOT INSTALLED"
    return result


def main():
    print("=" * 60)
    print("Qwen2.5-VL-7B-Instruct Environment Check")
    print("=" * 60)

    all_ok = True

    checks = [
        check_python,
        check_pytorch,
        check_transformers,
        check_qwen_import,
        check_qwen_vl_utils,
        check_other_packages,
    ]

    for check_fn in checks:
        print()
        section = check_fn.__doc__ or check_fn.__name__
        print(f"--- {section} ---")
        for key, value in check_fn().items():
            print(f"  {key}: {value}")
            if "FAIL" in str(value) or "NOT INSTALLED" in str(value):
                all_ok = False

    print()
    print("=" * 60)
    if all_ok:
        print("All checks PASSED. Environment is ready for inference.")
    else:
        print("Some checks FAILED. Please resolve issues above before proceeding.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
