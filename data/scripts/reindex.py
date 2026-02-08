"""
Reindex problems.jsonl by scanning the raw/ directory structure.

uv run python -m data.reindex
"""

import json
from pathlib import Path

from data.store_types import Entry


def reindex(states_dir: Path, output_path: Path) -> None:
    """Rebuild problems.jsonl from the raw/ directory structure.

    Scans all problem directories, reads states.txt for each, and loads
    state data from the directory hierarchy. Sorts by timestamp, latest last.
    """
    entries: list[tuple[str, Entry]] = []  # (timestamp, entry)

    # Iterate through all problem directories
    for problem_dir in sorted(states_dir.iterdir()):
        if not problem_dir.is_dir():
            continue

        problem_id = problem_dir.name
        states_txt = problem_dir / "states.txt"

        if not states_txt.exists():
            continue

        # Read state hashes from states.txt
        state_hashes = [h for h in states_txt.read_text().strip().split("\n") if h]

        for state_hash in state_hashes:
            try:
                entry = Entry.load(str(states_dir), problem_id, state_hash)
                # Skip states without actions
                if not entry.state.actions:
                    continue
                # Get latest timestamp from actions
                timestamps = [a.time_generated for a in entry.state.actions if a.time_generated]
                entry.timestamp = max(timestamps) if timestamps else ""
                entries.append((entry.timestamp, entry))
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Failed to load {problem_id}/{state_hash}: {e}")

    # Sort by timestamp, latest last
    entries.sort(key=lambda x: x[0])

    # Write all entries to problems.jsonl
    content = "\n".join(entry.to_json() for _, entry in entries) + "\n"
    output_path.write_text(content)

    print(f"Reindexed from {states_dir}: {len(entries)} states")


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent
    reindex(data_dir / "raw", data_dir / "problems.jsonl")
