"""Organized entrypoint for counterfactual construction."""

from qwen3_chat_binary.prepare_counterfactual import *  # noqa: F401,F403
from qwen3_chat_binary.prepare_counterfactual import main


if __name__ == "__main__":
    raise SystemExit(main())
