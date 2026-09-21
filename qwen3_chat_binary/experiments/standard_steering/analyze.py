"""Organized entrypoint for the standard steering summary."""

from qwen3_chat_binary.summarize_results import *  # noqa: F401,F403
from qwen3_chat_binary.summarize_results import main


if __name__ == "__main__":
    raise SystemExit(main())
