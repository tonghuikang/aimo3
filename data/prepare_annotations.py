"""Prepare annotation files from alternates.

Combines print text with alternates to create annotation templates.
Preserves existing labels from data/label when regenerating.

Usage:
    uv run python -m data.prepare_annotations                       # all problems with alternates
    uv run python -m data.prepare_annotations <action_hash>
    uv run python -m data.prepare_annotations <state_hash>
    uv run python -m data.prepare_annotations <problem_id>
    uv run python -m data.prepare_annotations --problem <problem_id>
    uv run python -m data.prepare_annotations --state <state_hash>
    uv run python -m data.prepare_annotations --action <action_hash>
"""

import json
from dataclasses import dataclass
from pathlib import Path

from openai_harmony import HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding

from data.store_types import State


@dataclass
class LabelData:
    """Label data with better/neutral/worse lists."""
    better: list[str]
    neutral: list[str]
    worse: list[str]
    reason: str

def format_list(items: list[str]) -> str:
    """Format a list of completion IDs for annotation file."""
    return ", ".join(items)


from data.common import (
    ALTERNATE_DIR,
    ANNOTATE_DIR,
    LABEL_DIR,
    RAW_DIR,
    find_action_location,
)

# Global encoding
harmony_encoding: HarmonyEncoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def load_alternates(jsonl_path: Path) -> dict[int, list[dict]]:
    """Load alternates from JSONL file.

    Returns dict mapping position (int) to list of {hash: tokens}.
    Handles multiple lines for the same position by extending and deduplicating.
    """
    alternates: dict[int, list[dict]] = {}
    seen_hashes: dict[int, set[str]] = {}
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                for pos_str, alts in data.items():
                    pos = int(pos_str)
                    if pos not in alternates:
                        alternates[pos] = []
                        seen_hashes[pos] = set()
                    for alt in alts:
                        alt_hash = next(iter(alt))
                        if alt_hash not in seen_hashes[pos]:
                            alternates[pos].append(alt)
                            seen_hashes[pos].add(alt_hash)
    return alternates


def load_labels(jsonl_path: Path) -> dict[str, LabelData]:
    """Load existing labels from JSONL file.

    Returns dict mapping start position (str) to LabelData.
    Keys in label files use span format "start-end".
    """
    labels: dict[str, LabelData] = {}
    if not jsonl_path.exists():
        return labels
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                for key, value in data.items():
                    # Extract start position from span key "start-end"
                    if "-" not in key:
                        raise ValueError(f"Invalid span key format: {key}")
                    start_pos = key.split("-")[0]
                    labels[start_pos] = LabelData(
                        better=value["better"],
                        neutral=value["neutral"],
                        worse=value["worse"],
                        reason=value["reason"],
                    )
    return labels


def get_paragraphs_at_positions(
    full_tokens: list[int],
    action_start: int,
    sorted_positions: list[int],
) -> list[tuple[str, int]]:
    """Get paragraph text and end positions for multiple positions at once.

    Position is the first token containing \n\n, so we need to skip
    any prefix before \n\n in that token, then extract until the next \n\n.

    Returns list of (paragraph_text, rel_end_pos) matching input order.
    """
    action_tokens = full_tokens[action_start:]
    results: list[tuple[str, int]] = []

    for rel_pos, next_rel_pos in zip(sorted_positions, sorted_positions[1:] + [len(action_tokens)]):
        tokens_from_pos = action_tokens[rel_pos:next_rel_pos]
        if 200012 in tokens_from_pos:
            end_idx = tokens_from_pos.index(200012)
            next_rel_pos = rel_pos + end_idx + 1
            tokens_from_pos = action_tokens[rel_pos:next_rel_pos]
        paragraph_text = harmony_encoding.decode(tokens_from_pos).strip()
        if "\n\n" in paragraph_text:
            new_tokens_from_pos = []
            for cur_pos in range(rel_pos, next_rel_pos):
                new_tokens_from_pos = action_tokens[rel_pos:cur_pos + 1]
                paragraph_text = harmony_encoding.decode(new_tokens_from_pos)
                if "\n\n" in paragraph_text:
                    next_rel_pos = cur_pos + 1
                    tokens_from_pos = action_tokens[rel_pos:next_rel_pos]
                    paragraph_text = harmony_encoding.decode(tokens_from_pos).strip()
                    while "\n\n" in paragraph_text:
                        paragraph_text = paragraph_text.replace("\n\n", "\n")
                    break
        assert "\n\n" not in paragraph_text, paragraph_text
        results.append((paragraph_text, next_rel_pos))

    return results


def prepare_annotation(
    action_hash: str,
    problem_id: str | None = None,
    state_hash: str | None = None,
):
    """Prepare annotation file for an action."""
    # Find location if not provided
    if problem_id is None or state_hash is None:
        result = find_action_location(action_hash)
        if result is None:
            print(f"Action {action_hash} not found")
            return
        problem_id, state_hash = result
        print(f"Found action in {problem_id}/{state_hash}")

    # Check if alternates exist
    alt_path = ALTERNATE_DIR / problem_id / state_hash / f"{action_hash}.jsonl"
    if not alt_path.exists():
        print(f"Alternates not found: {alt_path}")
        return

    # Load state and action
    state = State.load_from_dir(str(RAW_DIR / problem_id / state_hash))
    action = None
    for a in state.actions:
        if a.hash == action_hash:
            action = a
            break

    if action is None:
        print(f"Action {action_hash} not found in state")
        return

    # Generate print files into annotate directory
    annotate_state_dir = ANNOTATE_DIR / problem_id / state_hash
    annotate_state_dir.mkdir(parents=True, exist_ok=True)

    # Write state tokens as state.txt
    state_print_path = annotate_state_dir / "state.txt"
    state_print_path.write_text(harmony_encoding.decode(state.tokens))

    # Write action tokens as {action_hash}-original.txt
    action_print_path = annotate_state_dir / f"{action_hash}-original.txt"
    action_print_path.write_text(harmony_encoding.decode(action.tokens))

    # Initialize {action_hash}-reflections.txt
    reflections_path = annotate_state_dir / f"{action_hash}-reflections.txt"
    if not reflections_path.exists():
        reflections_path.touch()
        print(f"Created {reflections_path}")

    # Load alternates
    alternates = load_alternates(alt_path)
    print(f"Loaded {len(alternates)} positions with alternates")

    # Load existing labels from data/label if available
    label_path = LABEL_DIR / problem_id / state_hash / f"{action_hash}.jsonl"
    existing_labels = load_labels(label_path)
    # Count entries with actual label content
    labeled_count = sum(
        1 for label in existing_labels.values()
        if label.better or label.neutral or label.worse or label.reason
    )
    if labeled_count:
        print(f"Loaded {labeled_count} existing labels from {label_path}")

    # Combine tokens
    full_tokens = state.tokens + action.tokens
    action_start = len(state.tokens)

    # Build annotation content
    lines = []

    # Sort positions
    sorted_positions = sorted(alternates.keys())

    # Get all paragraphs at once (batched for efficiency)
    paragraphs = get_paragraphs_at_positions(full_tokens, action_start, sorted_positions)

    for rel_pos, (original_text, rel_end_pos) in zip(sorted_positions, paragraphs):
        alts = alternates[rel_pos]

        lines.append(f"Token {rel_pos}-{rel_end_pos} original: {original_text}")

        # Add alternates (first token already included in stored tokens)
        for alt_dict in alts:
            for alt_hash, alt_tokens in alt_dict.items():
                alt_text = harmony_encoding.decode(alt_tokens).strip()
                lines.append(f"Token {rel_pos} alternate {alt_hash}: {alt_text}")

        # Use existing labels if available, otherwise leave empty
        pos_str = str(rel_pos)
        if pos_str in existing_labels:
            label_data = existing_labels[pos_str]
            better = format_list(label_data.better)
            neutral = format_list(label_data.neutral)
            worse = format_list(label_data.worse)
            reason = label_data.reason
        else:
            better = ""
            neutral = ""
            worse = ""
            reason = ""
        lines.append(f"Token {rel_pos} better: {better}".strip())
        lines.append(f"Token {rel_pos} neutral: {neutral}".strip())
        lines.append(f"Token {rel_pos} worse: {worse}".strip())
        lines.append(f"Token {rel_pos} reason: {reason}".strip())
        lines.append("")  # Blank line between positions

    # Write annotation file
    output_dir = ANNOTATE_DIR / problem_id / state_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{action_hash}.txt"

    output_path.write_text("\n".join(lines))
    print(f"Saved to {output_path}")


def prepare_for_state(problem_id: str, state_hash: str):
    """Prepare annotations for all actions in a state."""
    alt_dir = ALTERNATE_DIR / problem_id / state_hash
    if not alt_dir.exists():
        print(f"No alternates found for state {state_hash}")
        return
    for alt_file in alt_dir.glob("*.jsonl"):
        action_hash = alt_file.stem
        prepare_annotation(action_hash, problem_id, state_hash)


def prepare_for_problem(problem_id: str):
    """Prepare annotations for all states/actions in a problem."""
    problem_dir = ALTERNATE_DIR / problem_id
    if not problem_dir.exists():
        print(f"No alternates found for problem {problem_id}")
        return

    # Create findings.txt for AI to write key insights on the solution
    findings_dir = ANNOTATE_DIR / problem_id
    findings_dir.mkdir(parents=True, exist_ok=True)
    findings_path = findings_dir / "findings.txt"
    if not findings_path.exists():
        findings_path.touch()
        print(f"Created {findings_path}")

    # Create correct_answer.txt from state.json
    correct_answer_path = findings_dir / "correct_answer.txt"
    if not correct_answer_path.exists():
        # Find any state directory to load the problem answer
        raw_problem_dir = RAW_DIR / problem_id
        if raw_problem_dir.exists():
            state_dirs = [d for d in raw_problem_dir.iterdir() if d.is_dir()]
            if state_dirs:
                # Load state to get the correct answer
                state = State.load_from_dir(str(state_dirs[0]))
                correct_answer_path.write_text(str(state.problem.answer))
                print(f"Created {correct_answer_path} with answer: {state.problem.answer}")

    for state_dir in problem_dir.iterdir():
        if state_dir.is_dir():
            prepare_for_state(problem_id, state_dir.name)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare annotation files")
    parser.add_argument("identifier", nargs="?", help="problem_id, state_hash, or action_hash")
    parser.add_argument("--problem", "-p", help="Process all actions for a problem")
    parser.add_argument("--state", "-s", help="Process all actions for a state")
    parser.add_argument("--action", "-a", help="Process a specific action")
    args = parser.parse_args()

    if args.action:
        prepare_annotation(
            action_hash=args.action,
            problem_id=args.problem,
            state_hash=args.state,
        )
    elif args.problem:
        prepare_for_problem(args.problem)
    elif args.state:
        # Find the problem for this state
        for problem_dir in ALTERNATE_DIR.iterdir():
            if problem_dir.is_dir():
                state_dir = problem_dir / args.state
                if state_dir.exists():
                    prepare_for_state(problem_dir.name, args.state)
                    return
        print(f"State {args.state} not found")
    elif args.identifier:
        # Try to detect what the identifier is
        # Check if it's a problem_id
        if (ALTERNATE_DIR / args.identifier).is_dir():
            prepare_for_problem(args.identifier)
        else:
            # Check if it's a state_hash
            for problem_dir in ALTERNATE_DIR.iterdir():
                if problem_dir.is_dir():
                    state_dir = problem_dir / args.identifier
                    if state_dir.is_dir():
                        prepare_for_state(problem_dir.name, args.identifier)
                        return
            # Otherwise treat as action_hash
            prepare_annotation(action_hash=args.identifier)
    else:
        # No argument: prepare for all problems with alternates
        if not ALTERNATE_DIR.exists():
            print(f"No alternates directory found: {ALTERNATE_DIR}")
            return
        problems = sorted(p.name for p in ALTERNATE_DIR.iterdir() if p.is_dir())
        if not problems:
            print("No problems with alternates found")
            return
        print(f"Preparing annotations for {len(problems)} problems...")
        for problem_id in problems:
            print(f"\n=== Problem {problem_id} ===")
            prepare_for_problem(problem_id)


if __name__ == "__main__":
    main()
