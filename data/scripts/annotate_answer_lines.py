#!/usr/bin/env python3
"""Auto-annotate alternates with correct boxed answers in wrong solution traces.

For wrong solutions (traces where original ends with wrong boxed answer),
this script finds tokens where an alternate contains \\boxed{correct_answer}
and auto-annotates: alternate as `better`, original as `worse`.

Usage: uv run python -m data.scripts.annotate_answer_lines
"""

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from data.common import ANNOTATE_DIR, LABEL_DIR
from data.extract_labels import regenerate_labels_jsonl


@dataclass
class AlternateData:
    """Data for a single alternate completion."""

    text: str
    boxed: list[str]  # All boxed values found


@dataclass
class PositionData:
    """Data for a single token position."""

    span: str | None
    original_text: str | None
    original_boxed: list[str]  # All boxed values found
    alternates: dict[str, AlternateData]


def load_answers() -> dict[str, str]:
    """Load problem_id -> answer mapping from problems.csv."""
    answers = {}
    problems_csv = Path("data/problems.csv")
    with open(problems_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            answers[row["problem_id"]] = row["answer"]
    return answers


def extract_all_boxed(text: str) -> list[str]:
    """Extract all values inside \\boxed{...} from text.

    Handles nested braces properly.
    Returns list of non-empty boxed values.
    """
    results = []
    for match in re.finditer(r"\\boxed\{", text):
        start = match.end()
        depth = 1
        i = start

        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1

        if depth == 0:
            content = text[start : i - 1].strip()
            if content:
                results.append(content)
    return results


def is_integer(value: str) -> bool:
    """Check if a string is an integer."""
    return bool(re.match(r"^-?\d+$", value))


def has_integer_answer(boxed_list: list[str]) -> bool:
    """Check if any boxed value in the list is an integer."""
    return any(is_integer(v) for v in boxed_list)


def get_integer_answers(boxed_list: list[str]) -> list[str]:
    """Get all integer answers from a list of boxed values."""
    return [v for v in boxed_list if is_integer(v)]


def mentions_answer(text: str, answer: str) -> bool:
    """Check if text mentions the answer as a standalone number."""
    # Match answer as a whole word (not part of another number)
    pattern = rf"\b{re.escape(answer)}\b"
    return bool(re.search(pattern, text.replace(",", "")))


def count_answer_mentions(text: str, answer: str) -> int:
    """Count lines where both 'answer' and the answer value appear."""
    pattern = rf"\b{re.escape(answer)}\b"
    count = 0
    for line in text.split("\n"):
        if "answer" in line.lower() and re.search(pattern, line):
            count += 1
    return count


def parse_annotate_file(content: str) -> dict[str, PositionData]:
    """Parse annotation file content to extract positions with boxed values.

    Handles multiline content - text continues until the next Token line.

    Returns: {
        "82": PositionData(span="82-155", original_boxed="69035", ...),
        ...
    }
    """
    positions: dict[str, PositionData] = {}

    # Pattern for original with span: Token N-M original: <text>
    original_pattern = re.compile(r"^Token (\d+)-(\d+) original:\s*(.*)$")
    # Pattern for alternate: Token N alternate HASH: <text>
    alternate_pattern = re.compile(r"^Token (\d+) alternate (\w+):\s*(.*)$")
    # Pattern for any Token line (to detect end of multiline content)
    token_line_pattern = re.compile(r"^Token \d+")

    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        orig_match = original_pattern.match(line)
        if orig_match:
            start_pos = orig_match.group(1)
            end_pos = orig_match.group(2)
            text_parts = [orig_match.group(3)]

            # Collect continuation lines until next Token line
            j = i + 1
            while j < len(lines) and not token_line_pattern.match(lines[j]):
                text_parts.append(lines[j])
                j += 1

            text = "\n".join(text_parts)
            span = f"{start_pos}-{end_pos}"
            boxed_vals = extract_all_boxed(text)

            if start_pos not in positions:
                positions[start_pos] = PositionData(
                    span=span,
                    original_text=text,
                    original_boxed=boxed_vals,
                    alternates={},
                )
            else:
                positions[start_pos].span = span
                positions[start_pos].original_text = text
                positions[start_pos].original_boxed = boxed_vals

            i = j
            continue

        alt_match = alternate_pattern.match(line)
        if alt_match:
            pos = alt_match.group(1)
            alt_id = alt_match.group(2)
            text_parts = [alt_match.group(3)]

            # Collect continuation lines until next Token line
            j = i + 1
            while j < len(lines) and not token_line_pattern.match(lines[j]):
                text_parts.append(lines[j])
                j += 1

            text = "\n".join(text_parts)
            boxed_vals = extract_all_boxed(text)

            if pos not in positions:
                positions[pos] = PositionData(
                    span=None,
                    original_text=None,
                    original_boxed=[],
                    alternates={},
                )
            positions[pos].alternates[alt_id] = AlternateData(
                text=text, boxed=boxed_vals
            )

            i = j
            continue

        # Non-matching line, advance
        i += 1

    return positions


def compute_labels(
    position_data: PositionData, correct_answer: str, history_text: str
) -> tuple[list[str], list[str], list[str], str] | None:
    """Compute (better, neutral, worse, reason) for a position.

    Returns None if no actionable boxed answers found at this position.

    Logic:
    - If original has correct boxed answer
      AND answer has been mentioned with 'answer' at least 5 times in history:
      - better: [original, alternates with correct answer boxed]
      - worse: [alternates without correct answer boxed]
    - If original has wrong integer boxed answer:
      - worse: [original, alternates with any wrong integer answer boxed]
      - neutral: [alternates that mention the wrong answer boxed or unboxed]
      - better: [other alternates]
    - If original has no integer boxed, but alternate has correct answer boxed
        AND answer has been mentioned with 'answer' at least 5 times in history:
      - better: [alternates with correct answer]
      - neutral: [original, alternates without correct answer]
    - Skip if no integer boxed answers exist, or if no alternate has correct.
    """
    original_boxed = position_data.original_boxed
    alternates = position_data.alternates

    if not has_integer_answer(original_boxed):
        # Check if any alternate has the correct answer boxed
        any_alt_correct = any(
            correct_answer in alt.boxed for alt in alternates.values()
        )
        if not any_alt_correct:
            return None
        if count_answer_mentions(history_text, correct_answer) < 10:
            all_ids = ["original"] + list(alternates.keys())
            reason = f"Alternate has correct boxed answer {correct_answer} but answer has not been sufficiently repeated."
            return ([], all_ids, [], reason)
        better: list[str] = []
        neutral: list[str] = ["original"]
        for alt_id, alt_data in alternates.items():
            if correct_answer in alt_data.boxed:
                better.append(alt_id)
            else:
                neutral.append(alt_id)
        reason = f"Alternate has correct boxed answer {correct_answer}, original has no integer boxed. Correct answer has been sufficiently repeated in history."
        return (better, neutral, [], reason)

    better: list[str] = []
    neutral: list[str] = []
    worse: list[str] = []
    reason: str = ""

    if correct_answer in original_boxed:
        # Only label if answer mentioned with 'answer' at least 5 times in history
        if count_answer_mentions(history_text, correct_answer) < 5:
            all_ids = ["original"] + list(alternates.keys())
            reason = f"Original has correct boxed answer {correct_answer} but answer has not been sufficiently repeated."
            return ([], all_ids, [], reason)
        # Original has correct answer - mark as better
        better.append("original")
        # Alternates with correct answer are also better, others are worse
        for alt_id, alt_data in alternates.items():
            if correct_answer in alt_data.boxed:
                better.append(alt_id)
            else:
                worse.append(alt_id)
        reason = f"Original has correct boxed answer {correct_answer}. Correct answer has been sufficiently repeated in history."
    else:
        # Original has wrong integer boxed answer(s) - mark as worse
        wrong_answers = get_integer_answers(original_boxed)
        worse.append("original")
        # Classify alternates
        for alt_id, alt_data in alternates.items():
            alt_integers = get_integer_answers(alt_data.boxed)
            # Check if any boxed integer is wrong
            has_wrong_integer_boxed = any(ans != correct_answer for ans in alt_integers)
            if has_wrong_integer_boxed:
                # Has wrong integer boxed - worse
                worse.append(alt_id)
            elif any(mentions_answer(alt_data.text, ans) for ans in wrong_answers):
                # Mentions original's wrong answer in text - neutral
                neutral.append(alt_id)
            else:
                # Other alternates - better
                better.append(alt_id)
        reason = f"Original has wrong boxed answer {wrong_answers[0]}, correct answer is {correct_answer}. Not boxing the answer provides a chance to move on from the wrong answer."

    total_labeled = len(better) + len(neutral) + len(worse)
    if (
        len(better) == total_labeled
        or len(worse) == total_labeled
    ):
        return [], better + worse, [], reason

    return (better, neutral, worse, reason)


def load_label_file(label_path: Path) -> dict[str, dict]:
    """Load existing labels from JSONL file.

    Returns dict mapping span key "start-end" to label data.
    """
    labels: dict[str, dict] = {}
    if not label_path.exists():
        return labels
    with open(label_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                for key, value in data.items():
                    labels[key] = value
    return labels


def update_label_file(
    label_path: Path,
    span_key: str,
    better: list[str],
    neutral: list[str],
    worse: list[str],
    reason: str,
    force: bool = False,
) -> bool:
    """Update a single entry in label JSONL. Returns True if modified."""
    labels = load_label_file(label_path)

    if span_key not in labels:
        # Span doesn't exist in label file - create it
        labels[span_key] = {"better": [], "neutral": [], "worse": [], "reason": ""}

    existing = labels[span_key]

    # Check if already labeled (non-empty better, neutral, or worse)
    if not force and (existing["better"] or existing["neutral"] or existing["worse"]):
        return False  # Skip, already has labels

    # Update the entry
    labels[span_key] = {
        "better": better,
        "neutral": neutral,
        "worse": worse,
        "reason": reason,
    }

    # Write back to file (preserving order by sorting keys)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort spans by start position numerically
    def span_sort_key(span: str) -> tuple[int, int]:
        parts = span.split("-")
        return (int(parts[0]), int(parts[1]))

    sorted_spans = sorted(labels.keys(), key=span_sort_key)

    with open(label_path, "w") as f:
        for span in sorted_spans:
            f.write(json.dumps({span: labels[span]}) + "\n")

    return True


def process_annotate_file(
    annotate_path: Path, correct_answer: str, force: bool = False
) -> tuple[int, int]:
    """Process a single annotation file.

    Returns (modified_count, skipped_count).
    """
    # Derive label path from annotate path
    # data/annotate/problem/state/action.txt -> data/label/problem/state/action.jsonl
    rel_path = annotate_path.relative_to(ANNOTATE_DIR)
    label_path = LABEL_DIR / rel_path.with_suffix(".jsonl")

    # Read annotation file
    content = annotate_path.read_text()

    # Parse positions with boxed values
    positions = parse_annotate_file(content)

    modified = 0
    skipped = 0

    sorted_positions = sorted(positions.keys(), key=int)
    history_parts: list[str] = []

    for pos in sorted_positions:
        pos_data = positions[pos]
        if pos_data.span is None:
            continue

        history_text = "\n".join(history_parts)
        result = compute_labels(pos_data, correct_answer, history_text)
        if pos_data.original_text is not None:
            history_parts.append(pos_data.original_text)
        if result is None:
            continue

        better, neutral, worse, reason = result

        if update_label_file(
            label_path, pos_data.span, better, neutral, worse, reason, force
        ):
            modified += 1
            print(
                f"  Updated {pos_data.span}: better={better}, neutral={neutral}, worse={worse}"
            )
        else:
            skipped += 1

    return modified, skipped


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Auto-annotate boxed answer lines")
    parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite existing labels"
    )
    args = parser.parse_args()

    print("Loading answers from problems.csv...")
    answers = load_answers()
    print(f"Loaded {len(answers)} problem answers")
    if args.force:
        print("Force mode: will overwrite existing labels")

    total_modified = 0
    total_skipped = 0
    files_processed = 0

    # Process all annotation files
    for annotate_file in sorted(ANNOTATE_DIR.rglob("*.txt")):
        # Skip findings.txt
        if annotate_file.name == "findings.txt":
            continue

        # Extract problem_id from path
        # data/annotate/problem_id/state_hash/action.txt
        rel_path = annotate_file.relative_to(ANNOTATE_DIR)
        parts = rel_path.parts
        if len(parts) < 3:
            continue

        problem_id = parts[0]

        if problem_id not in answers:
            print(f"Warning: No answer for problem {problem_id}")
            continue

        correct_answer = answers[problem_id]

        print(f"\nProcessing {annotate_file.relative_to(ANNOTATE_DIR)}...")
        modified, skipped = process_annotate_file(
            annotate_file, correct_answer, args.force
        )

        if modified > 0:
            print(f"  Modified: {modified}, Skipped (already labeled): {skipped}")
            total_modified += modified
            files_processed += 1
        elif skipped > 0:
            total_skipped += skipped

    print("\n=== Summary ===")
    print(f"Files with modifications: {files_processed}")
    print(f"Total labels added: {total_modified}")
    print(f"Total skipped (already labeled): {total_skipped}")

    # Regenerate consolidated labels.jsonl
    regenerate_labels_jsonl()

    if total_modified > 0:
        print(
            "\nRun 'uv run python -m data.prepare_annotations' to regenerate txt files."
        )


if __name__ == "__main__":
    main()
