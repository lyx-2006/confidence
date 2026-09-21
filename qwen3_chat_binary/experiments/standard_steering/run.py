"""Organized entrypoint for the standard paired steering runner."""

from qwen3_chat_binary.steering import *  # noqa: F401,F403
from qwen3_chat_binary.steering import main


if __name__ == "__main__":
    raise SystemExit(main())
