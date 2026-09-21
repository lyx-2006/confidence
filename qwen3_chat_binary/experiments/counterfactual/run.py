"""Organized entrypoint for counterfactual four-cell measurement."""

from qwen3_chat_binary.run_counterfactual import *  # noqa: F401,F403
from qwen3_chat_binary.run_counterfactual import main


if __name__ == "__main__":
    raise SystemExit(main())
