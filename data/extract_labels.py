"""Extract labels from completed annotations.

Parses annotation files and extracts performance labels to data/label.

Usage:
    uv run python -m data.extract_labels                        # all problems
    uv run python -m data.extract_labels <problem_id>           # specific problem
    uv run python -m data.extract_labels --problem <problem_id>
    uv run python -m data.extract_labels --state <state_hash>
    uv run python -m data.extract_labels --action <action_hash>
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from data.common import (
    ALTERNATE_DIR,
    ANNOTATE_DIR,
    LABEL_DIR,
    LABELS_JSONL,
    RAW_DIR,
    find_action_location,
)


@dataclass
class LabelData:
    better: list[str] = field(default_factory=list)  # Completion IDs that are better
    neutral: list[str] = field(default_factory=list)  # Completion IDs that are neutral
    worse: list[str] = field(default_factory=list)  # Completion IDs that are worse
    reason: str = ""


@dataclass
class EntryStats:
    problem_id: str
    state_hash: str
    action_hash: str
    better: int
    mixed: int  # original is neutral but there are other better/worse alternatives
    worse: int
    neutral: int
    unlabeled: int
    action_reward: float
    learnable: int
    action_length: int
    reflection_exists: bool


def parse_completion_list(text: str) -> list[str]:
    """Parse a comma-separated completion list like 'original, abc123'.

    Returns list of completion IDs (deduplicated, preserving order).
    """
    text = text.strip()
    if not text:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in text.split(","):
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_labels_from_annotation(
    annotation_path: Path,
) -> tuple[dict[int, LabelData], dict[int, int]]:
    """Parse annotation file and extract labels.

    We assume there is only one span per starting token.

    Returns:
        tuple of (labels, spans) where:
        - labels: dict mapping position (int) to LabelData
        - spans: dict mapping start position (int) to end position (int)
    """
    labels: dict[int, LabelData] = {}
    spans: dict[int, int] = {}
    content = annotation_path.read_text()

    # Valid keywords after "Token N" or "Token N-M"
    valid_keywords = {"original", "alternate", "better", "neutral", "worse", "reason"}

    # Pattern: Token {pos} or Token {pos}-{end} followed by {keyword}:
    keyword_pattern = r"Token (\d+)(?:-\d+)? (\w+):"

    # Check for invalid/misspelled keywords
    lines = content.splitlines()
    for match in re.finditer(keyword_pattern, content):
        keyword = match.group(2)
        if keyword not in valid_keywords:
            line_num = content[: match.start()].count("\n") + 1
            full_line = lines[line_num - 1] if line_num <= len(lines) else ""
            raise ValueError(
                f"{annotation_path}:{line_num} Invalid keyword: '{keyword}'\n{full_line}"
            )

    # Pattern: Token {pos}-{end} original: (extract span from original lines)
    span_pattern = r"Token (\d+)-(\d+) original:"
    for match in re.finditer(span_pattern, content):
        start = int(match.group(1))
        end = int(match.group(2))
        spans[start] = end

    # Pattern: Token {pos} better: [...] (capture rest of line)
    better_pattern = r"Token (\d+) better:[ \t]*(.*)$"

    # Extract better lists
    for match in re.finditer(better_pattern, content, re.MULTILINE):
        pos = int(match.group(1))
        better_text = match.group(2).strip()
        better_list = parse_completion_list(better_text)
        if pos not in labels:
            labels[pos] = LabelData()
        labels[pos].better = better_list

    # Pattern: Token {pos} neutral: [...] (capture rest of line)
    neutral_pattern = r"Token (\d+) neutral:[ \t]*(.*)$"

    # Extract neutral lists
    for match in re.finditer(neutral_pattern, content, re.MULTILINE):
        pos = int(match.group(1))
        neutral_text = match.group(2).strip()
        neutral_list = parse_completion_list(neutral_text)
        if pos not in labels:
            labels[pos] = LabelData()
        labels[pos].neutral = neutral_list

    # Pattern: Token {pos} worse: [...] (capture rest of line)
    worse_pattern = r"Token (\d+) worse:[ \t]*(.*)$"

    # Extract worse lists
    for match in re.finditer(worse_pattern, content, re.MULTILINE):
        pos = int(match.group(1))
        worse_text = match.group(2).strip()
        worse_list = parse_completion_list(worse_text)
        if pos not in labels:
            labels[pos] = LabelData()
        labels[pos].worse = worse_list

    # Pattern: Token {pos} reason: {reason} (capture rest of line)
    # Use [ \t]* instead of \s* to avoid matching across newlines
    reason_pattern = r"Token (\d+) reason:[ \t]*(.*)$"

    # Extract reasons
    for match in re.finditer(reason_pattern, content, re.MULTILINE):
        pos = int(match.group(1))
        reason = match.group(2).strip()
        if pos not in labels:
            labels[pos] = LabelData()
        labels[pos].reason = reason

    return labels, spans


def get_alternates_by_position(
    problem_id: str, state_hash: str, action_hash: str
) -> dict[int, set[str]]:
    """Get alternate IDs for each position from alternate file."""
    alternate_path = ALTERNATE_DIR / problem_id / state_hash / f"{action_hash}.jsonl"
    result: dict[int, set[str]] = {}
    if alternate_path.exists():
        for line in alternate_path.read_text().strip().split("\n"):
            if line:
                data = json.loads(line)
                for pos_str, alts in data.items():
                    pos = int(pos_str)
                    if pos not in result:
                        result[pos] = set()
                    for alt_obj in alts:
                        for alt_id in alt_obj.keys():
                            result[pos].add(alt_id)
    return result


def get_all_positions(problem_id: str, state_hash: str, action_hash: str) -> set[int]:
    """Get all positions from alternate file."""
    return set(get_alternates_by_position(problem_id, state_hash, action_hash).keys())


def _get_contiguous_label_blocks(
    labeled_positions: set[int], all_positions: set[int]
) -> list[tuple[int, int]]:
    """Get contiguous blocks of labeled positions within all positions.

    "Contiguous" means consecutive in the sorted order of all_positions,
    not consecutive integer values.

    Returns list of (start_index, size) tuples for each contiguous block,
    where start_index is the index in the sorted all_positions list.
    """
    if not labeled_positions or not all_positions:
        return []

    sorted_all = sorted(all_positions)
    pos_to_index = {pos: i for i, pos in enumerate(sorted_all)}

    # Get indices of labeled positions in sorted order
    labeled_indices = sorted(pos_to_index[pos] for pos in labeled_positions)

    blocks = []
    block_start = labeled_indices[0]
    block_size = 1

    for i in range(1, len(labeled_indices)):
        if labeled_indices[i] == labeled_indices[i - 1] + 1:
            # Contiguous in order - extend current block
            block_size += 1
        else:
            # Gap - save current block, start new one
            blocks.append((block_start, block_size))
            block_start = labeled_indices[i]
            block_size = 1

    # Save final block
    blocks.append((block_start, block_size))
    return blocks


def extract_labels(
    action_hash: str,
    problem_id: str | None = None,
    state_hash: str | None = None,
    raise_error: bool = False,
) -> tuple[str, str, int] | None:
    """Extract labels from annotation file.

    Returns (problem_id, state_hash, num_labels) on success, None on failure.
    """

    if action_hash.endswith("-original") or action_hash.endswith("-reflections"):
        return None

    # Find location if not provided
    if problem_id is None or state_hash is None:
        result = find_action_location(action_hash)
        if result is None:
            print(f"Action {action_hash} not found")
            return None
        problem_id, state_hash = result
        print(f"Found action in {problem_id}/{state_hash}")

    # Check if annotation exists
    annotation_path = ANNOTATE_DIR / problem_id / state_hash / f"{action_hash}.txt"
    if not annotation_path.exists():
        print(f"Annotation not found: {annotation_path}")
        return None

    # Extract labels and spans from annotation
    labels, spans = extract_labels_from_annotation(annotation_path)

    # Check if all completions are the same and not all neutral - reclassify to neutral
    # Track positions with all-same non-neutral classifications
    all_same_positions: list[tuple[int, str]] = []  # (pos, category)
    for pos, label_data in labels.items():
        all_completions = label_data.better + label_data.neutral + label_data.worse
        if all_completions:
            # All in better only (not neutral)
            if label_data.better and not label_data.neutral and not label_data.worse:
                all_same_positions.append((pos, "better"))
                print(
                    f"WARNING: All completions at position {pos} are 'better'. "
                    f"Reclassifying to neutral: {label_data.better}"
                )
                label_data.neutral = label_data.better
                label_data.better = []
            # All in worse only (not neutral)
            elif label_data.worse and not label_data.better and not label_data.neutral:
                all_same_positions.append((pos, "worse"))
                print(
                    f"WARNING: All completions at position {pos} are 'worse'. "
                    f"Reclassifying to neutral: {label_data.worse}"
                )
                label_data.neutral = label_data.worse
                label_data.worse = []

    # Get alternates by position and filter out invalid hashes
    alternates_by_pos = get_alternates_by_position(problem_id, state_hash, action_hash)
    all_positions = set(alternates_by_pos.keys())

    # Filter out hashes that don't exist at their position
    for pos, label_data in labels.items():
        valid_ids = alternates_by_pos[pos] | {"original"} if pos in alternates_by_pos else {"original"}

        invalid_better = [h for h in label_data.better if h not in valid_ids]
        invalid_neutral = [h for h in label_data.neutral if h not in valid_ids]
        invalid_worse = [h for h in label_data.worse if h not in valid_ids]

        if invalid_better or invalid_neutral or invalid_worse:
            all_invalid = invalid_better + invalid_neutral + invalid_worse
            print(f"Removing invalid hashes at position {pos}: {all_invalid}")
            label_data.better = [h for h in label_data.better if h in valid_ids]
            label_data.neutral = [h for h in label_data.neutral if h in valid_ids]
            label_data.worse = [h for h in label_data.worse if h in valid_ids]

    # Save labels as JSONL (create file even if empty)
    # Keys use span format "start-end" where end comes from annotation file
    output_dir = LABEL_DIR / problem_id / state_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{action_hash}.jsonl"

    with open(output_path, "w") as f:
        for pos in sorted(all_positions):
            if pos not in spans:
                raise ValueError(
                    f"Missing span for position {pos} in {annotation_path}"
                )
            key = f"{pos}-{spans[pos]}"

            if pos in labels:
                label_data = labels[pos]
                line = json.dumps(
                    {
                        key: {
                            "better": label_data.better,
                            "neutral": label_data.neutral,
                            "worse": label_data.worse,
                            "reason": label_data.reason,
                        }
                    }
                )
            else:
                line = json.dumps(
                    {key: {"better": [], "neutral": [], "worse": [], "reason": ""}}
                )
            f.write(line + "\n")

    # Log extraction results - count positions with actual label content
    # "labeled" = has better or worse (meaningful labels)
    # "neutral" = has neutral/reason but no better/worse
    # "unlabeled" = no labels at all
    labeled_count = sum(1 for label in labels.values() if label.better or label.worse)
    neutral_count = sum(
        1
        for label in labels.values()
        if (label.neutral or label.reason) and not label.better and not label.worse
    )
    unlabeled = len(all_positions) - labeled_count - neutral_count
    print(
        f"{output_path}: {labeled_count} labeled, "
        f"{neutral_count} neutral, {unlabeled} unlabeled"
    )

    # Check if current action has significantly more annotations than other actions in same state
    if raise_error:
        current_labeled_count = sum(
            1
            for label in labels.values()
            if label.better or label.worse or label.neutral
        )
        # Get annotation counts for other actions in the same state
        state_annotate_dir = ANNOTATE_DIR / problem_id / state_hash
        other_actions_counts: dict[str, int] = {}
        for other_annotation in state_annotate_dir.glob("*.txt"):
            other_action_hash = other_annotation.stem
            if other_action_hash == action_hash:
                continue
            # Count labeled positions in the other action's label file
            other_label_file = LABEL_DIR / problem_id / state_hash / f"{other_action_hash}.jsonl"
            if other_label_file.exists():
                other_count = 0
                for line in other_label_file.read_text().strip().split("\n"):
                    if line:
                        data = json.loads(line)
                        for label_info in data.values():
                            if (
                                label_info["better"]
                                or label_info["worse"]
                                or label_info["neutral"]
                            ):
                                other_count += 1
                other_actions_counts[other_action_hash] = other_count
            else:
                other_actions_counts[other_action_hash] = 0

        # Warn if current action has 20+ more annotations than ALL other actions
        if other_actions_counts:
            all_have_fewer = all(
                current_labeled_count >= other_count + 20
                for other_count in other_actions_counts.values()
            )
            if all_have_fewer:
                sorted_others = sorted(other_actions_counts.items(), key=lambda x: x[1])
                least_annotated = sorted_others[0][0]
                under_annotated = [
                    f"{ah} ({c} labels)" for ah, c in sorted_others
                ]
                raise ValueError(
                    f"Current action {action_hash} has {current_labeled_count} annotations, "
                    f"which is 20+ more than all other actions in this state.\n"
                    f"Other actions needing annotations: {', '.join(under_annotated)}\n"
                    f"Please annotate action {least_annotated} next."
                )

            # Recommend stopping if all actions have enough labels each
            all_counts = [current_labeled_count] + list(other_actions_counts.values())
            if all(count > 40 for count in all_counts):
                action_labels = [f"{action_hash} ({current_labeled_count} labels)"] + [
                    f"{ah} ({c} labels)" for ah, c in other_actions_counts.items()
                ]
                raise ValueError(
                    f"All actions in this state have more than enough labels each.\n"
                    f"Actions: {', '.join(action_labels)}\n"
                    f"Recommend stopping annotation, and terminating your session now."
                )

        labeled_positions = {
            pos
            for pos, label in labels.items()
            if label.better or label.worse or label.neutral
        }
        if labeled_positions and all_positions:
            blocks = _get_contiguous_label_blocks(labeled_positions, all_positions)
            label_ratio = len(labeled_positions) / len(all_positions)

            if (
                len(blocks) > 0
                and blocks[0][0] == 0  # First block starts at index 0 (first span)
                and blocks[0][1] >= 5  # First block size having a certain size
                and len(blocks) < 2  # Less than a certain number of total blocks
                and label_ratio < 0.1  # Less than 10% of positions labeled
            ):
                raise ValueError(
                    f"Labels appear sequential from start: first block of {blocks[0][1]} "
                    f"consecutive spans starting from the beginning, only {len(blocks)} "
                    f"blocks total, {label_ratio:.1%} of positions labeled. "
                    f"Please search for answer values and divergence points instead of "
                    f"labeling sequentially. You may keep the current annotations."
                )

        # Raise error if all-same non-neutral found
        if all_same_positions:
            positions_str = ", ".join(
                f"{pos} ({cat})" for pos, cat in all_same_positions
            )
            raise ValueError(
                f"All completions have same non-neutral classification at positions: "
                f"{positions_str}"
                f"Please only award 'better' or 'worse' if the completion is better or worse than the other completions."
                f"Do not award award 'better' or 'worse' if all completions are equally 'good' or 'bad'."
            )

        # Warn if many labels and all are neutral (no better or worse)
        labels_with_content = [
            label for label in labels.values()
            if label.better or label.neutral or label.worse
        ]
        labels_with_ranking = [
            label for label in labels.values()
            if label.better or label.worse
        ]
        if len(labels_with_content) > 18 and len(labels_with_ranking) == 0:
            raise ValueError(
                f"There are ({len(labels_with_content)}) labels but all are neutral.\n"
                f"Please identify some lines where completions are better or worse than others.\n"
                "You are encouraged to revisit previous labels."
            )

    return problem_id, state_hash, len(labels)


def extract_for_state(problem_id: str, state_hash: str):
    """Extract labels for all actions in a state."""
    annotate_dir = ANNOTATE_DIR / problem_id / state_hash
    if not annotate_dir.exists():
        raise ValueError(f"State not found: {state_hash}")
    for annotation_file in annotate_dir.glob("*.txt"):
        action_hash = annotation_file.stem
        if action_hash.endswith("-original") or action_hash.endswith("-reflections"):
            continue
        if action_hash == "state":
            continue
        result = extract_labels(action_hash, problem_id, state_hash)
        if result:
            regenerate_labels_jsonl(problem_id, state_hash, action_hash)


def extract_for_problem(problem_id: str):
    """Extract labels for all states/actions in a problem."""
    problem_dir = ANNOTATE_DIR / problem_id
    for state_dir in problem_dir.iterdir():
        if state_dir.is_dir():
            extract_for_state(problem_id, state_dir.name)


def extract_all():
    """Extract labels for all problems."""
    for problem_dir in ANNOTATE_DIR.iterdir():
        if problem_dir.is_dir():
            extract_for_problem(problem_dir.name)


def _compute_entry_stats(
    problem_id: str, state_hash: str, action_hash: str
) -> EntryStats:
    """Compute stats for a single action.

    Counts positions where:
    - better: "original" is in better list (original is better than some alternates)
    - mixed: "original" is in neutral list but there are other better/worse alternatives
    - worse: "original" is in worse list (original is worse than some alternates)
    - neutral: "original" is in neutral list with no other better/worse alternatives
    - unlabeled: position has no annotation (empty lists and no reason)
    """
    label_file = LABEL_DIR / problem_id / state_hash / f"{action_hash}.jsonl"

    better = 0
    mixed = 0
    worse = 0
    neutral = 0
    unlabeled = 0
    learnable = 0

    if label_file.exists():
        for line in label_file.read_text().strip().split("\n"):
            if line:
                data = json.loads(line)
                for label_info in data.values():
                    better_list = label_info["better"]
                    neutral_list = label_info["neutral"]
                    worse_list = label_info["worse"]
                    reason = label_info["reason"]

                    learnable += len(better_list)

                    if "original" in better_list:
                        better += 1
                    elif "original" in worse_list:
                        worse += 1
                    elif "original" in neutral_list:
                        # Check if there are other better/worse alternatives
                        has_better_alt = any(alt != "original" for alt in better_list)
                        has_worse_alt = any(alt != "original" for alt in worse_list)
                        if has_better_alt or has_worse_alt:
                            mixed += 1
                        else:
                            neutral += 1
                    elif better_list or neutral_list or worse_list or reason:
                        # Has annotation but original not explicitly listed
                        # If there are better/worse alternatives, count as mixed
                        if better_list or worse_list:
                            mixed += 1
                        else:
                            neutral += 1
                    else:
                        # No annotation at all
                        unlabeled += 1

    # Read action reward from raw data
    action_file = RAW_DIR / problem_id / state_hash / action_hash / "action.json"
    action_data = json.loads(action_file.read_text())
    action_reward: float = action_data["reward"]
    action_length: int = len(action_data["tokens"])

    # Check if reflection file exists and is non-empty
    reflection_path = (
        ANNOTATE_DIR / problem_id / state_hash / f"{action_hash}-reflections.txt"
    )
    reflection_exists = (
        reflection_path.exists() and reflection_path.stat().st_size > 0
    )

    return EntryStats(
        problem_id=problem_id,
        state_hash=state_hash,
        action_hash=action_hash,
        better=better,
        mixed=mixed,
        worse=worse,
        neutral=neutral,
        unlabeled=unlabeled,
        action_reward=action_reward,
        learnable=learnable,
        action_length=action_length,
        reflection_exists=reflection_exists,
    )


def _update_labels_jsonl_entry(problem_id: str, state_hash: str, action_hash: str):
    """Incrementally update a single entry in labels.jsonl."""
    entries: dict[tuple[str, str, str], EntryStats] = {}
    if LABELS_JSONL.exists():
        for line in LABELS_JSONL.read_text().strip().split("\n"):
            if line:
                data = json.loads(line)
                data.setdefault("reflection_exists", False)
                entry = EntryStats(**data)
                key = (entry.problem_id, entry.state_hash, entry.action_hash)
                entries[key] = entry

    new_entry = _compute_entry_stats(problem_id, state_hash, action_hash)
    key = (problem_id, state_hash, action_hash)
    entries[key] = new_entry

    sorted_entries = sorted(
        entries.values(),
        key=lambda e: (e.problem_id, e.state_hash, e.action_hash),
    )
    with LABELS_JSONL.open("w") as f:
        for entry in sorted_entries:
            f.write(json.dumps(asdict(entry)) + "\n")


def _regenerate_all_labels_jsonl():
    """Regenerate all entries in labels.jsonl from scratch."""
    entries: list[EntryStats] = []

    for problem_dir in sorted(LABEL_DIR.iterdir()):
        if not problem_dir.is_dir() or problem_dir.name.startswith("."):
            continue

        problem_id = problem_dir.name

        for state_dir in sorted(problem_dir.iterdir()):
            if not state_dir.is_dir() or state_dir.name.startswith("."):
                continue

            state_hash = state_dir.name

            for label_file in sorted(state_dir.iterdir()):
                if label_file.suffix != ".jsonl":
                    continue

                action_hash = label_file.stem
                if action_hash.endswith("-original") or action_hash.endswith("-reflections"):
                    continue
                if action_hash == "state":
                    continue

                entry = _compute_entry_stats(problem_id, state_hash, action_hash)
                entries.append(entry)

    with LABELS_JSONL.open("w") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry)) + "\n")

    print(f"Generated {len(entries)} entries -> {LABELS_JSONL}")


def regenerate_labels_jsonl(
    problem_id: str | None = None,
    state_hash: str | None = None,
    action_hash: str | None = None,
):
    """Update the consolidated data/labels.jsonl file.

    If all three parameters are provided, performs an incremental update.
    Otherwise, regenerates all entries from scratch.
    """
    if problem_id and state_hash and action_hash:
        _update_labels_jsonl_entry(problem_id, state_hash, action_hash)
    else:
        _regenerate_all_labels_jsonl()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract labels from annotations")
    parser.add_argument("identifier", nargs="?", help="problem_id")
    parser.add_argument("--problem", "-p", help="Process all actions for a problem")
    parser.add_argument("--state", "-s", help="Process all actions for a state")
    parser.add_argument("--action", "-a", help="Process a specific action")
    args = parser.parse_args()

    if args.action:
        result = extract_labels(
            action_hash=args.action,
            problem_id=args.problem,
            state_hash=args.state,
            raise_error=True,
        )
        if result:
            problem_id, state_hash, _ = result
            regenerate_labels_jsonl(problem_id, state_hash, args.action)
        else:
            raise ValueError(f"Action not found: {args.action}")
        return
    elif args.problem or args.identifier:
        problem_id = args.problem or args.identifier
        if not (ANNOTATE_DIR / problem_id).is_dir():
            raise ValueError(f"Problem not found: {problem_id}")
        extract_for_problem(problem_id)
        return
    elif args.state:
        # Find the problem for this state
        for problem_dir in ANNOTATE_DIR.iterdir():
            if problem_dir.is_dir():
                state_dir = problem_dir / args.state
                if state_dir.exists():
                    extract_for_state(problem_dir.name, args.state)
                    return
        raise ValueError(f"State not found: {args.state}")
    else:
        # No arguments - extract all
        extract_all()

    # Regenerate all entries
    regenerate_labels_jsonl()


if __name__ == "__main__":
    main()
