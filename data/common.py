"""Common utilities for data processing scripts."""

from pathlib import Path

RAW_DIR = Path("data/raw")
ANNOTATE_DIR = Path("data/annotate")
LABEL_DIR = Path("data/label")
ALTERNATE_DIR = Path("data/alternate")
LABELS_JSONL = Path("data/labels.jsonl")


def find_action_location(
    action_hash: str,
    problem_id: str | None = None,
    state_hash: str | None = None,
) -> tuple[str, str] | None:
    """Search for action_hash in data/raw and return (problem_id, state_hash).

    If problem_id and/or state_hash are provided, narrows the search.
    """
    if problem_id and state_hash:
        # Direct check - no iteration needed
        action_dir = RAW_DIR / problem_id / state_hash / action_hash
        if action_dir.is_dir():
            return problem_id, state_hash
        return None
    
    print(f"Finding action location {action_hash=}")

    problem_dirs = [RAW_DIR / problem_id] if problem_id else RAW_DIR.iterdir()
    for problem_dir in problem_dirs:
        if not problem_dir.is_dir():
            continue
        state_dirs = [problem_dir / state_hash] if state_hash else problem_dir.iterdir()
        for state_dir in state_dirs:
            if not state_dir.is_dir():
                continue
            action_dir = state_dir / action_hash
            if action_dir.is_dir():
                return problem_dir.name, state_dir.name
    return None


def find_state_location(
    state_hash: str,
    problem_id: str | None = None,
) -> str | None:
    """Search for state_hash in data/raw and return problem_id.

    If problem_id is provided, verifies the state exists in that problem.
    """
    if problem_id:
        # Direct check - no iteration needed
        state_dir = RAW_DIR / problem_id / state_hash
        if state_dir.is_dir():
            return problem_id
        return None

    for problem_dir in RAW_DIR.iterdir():
        if not problem_dir.is_dir():
            continue
        state_dir = problem_dir / state_hash
        if state_dir.is_dir():
            return problem_dir.name
    return None
