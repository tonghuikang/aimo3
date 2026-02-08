"""
List all action_ids from data/alternate/<prefix>-*/<state_id>/<action_id>.jsonl, comma-separated.

uv run python -m data.scripts.list_action_ids <prefix>
"""

import sys
from pathlib import Path


def list_action_ids(prefix: str) -> None:
    base = Path(__file__).parent.parent / "alternate"
    action_ids = sorted(p.stem for p in base.glob(f"{prefix}*/*/*.jsonl"))
    print(",".join(action_ids))


if __name__ == "__main__":
    list_action_ids(sys.argv[1])
