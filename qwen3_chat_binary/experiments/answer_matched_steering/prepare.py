"""Organized entrypoint for answer-matched preparation."""

from qwen3_chat_binary.prepare_answer_matched import *  # noqa: F401,F403
from qwen3_chat_binary.prepare_answer_matched import main


if __name__ == "__main__":
    raise SystemExit(main())
