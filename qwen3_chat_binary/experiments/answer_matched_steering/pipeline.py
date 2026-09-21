"""Organized entrypoint for answer-matched smoke/pipeline execution."""

from qwen3_chat_binary.run_answer_matched_pipeline import *  # noqa: F401,F403
from qwen3_chat_binary.run_answer_matched_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
