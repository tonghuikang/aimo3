#!/usr/bin/env python3
"""Dump all non-empty reflections to data/reflections.csv.

Columns: problem_id, answer, correct_answer, reflection_text, problem_statement, solution_trace

Usage: uv run python -m data.scripts.dump_reflections
"""

import csv
import re
from pathlib import Path

from data.common import ANNOTATE_DIR


def main() -> None:
    output_path = Path("data/reflections.csv")
    rows: list[dict[str, str]] = []

    # Walk all reflection files
    for reflection_path in sorted(ANNOTATE_DIR.rglob("*-reflections.txt")):
        content = reflection_path.read_text().strip()
        if not content:
            continue

        # Path structure: data/annotate/<problem_id>/<state_hash>/<action_hash>-reflections.txt
        action_hash = reflection_path.name.removesuffix("-reflections.txt")
        state_dir = reflection_path.parent
        problem_id = state_dir.parent.name

        # Extract answer from reflection "## Answer: correct/wrong (VALUE)"
        answer_match = re.search(r"^## Answer:\s*\w+\s*\(([^)]+)\)", content, re.MULTILINE)
        answer = answer_match.group(1).strip() if answer_match else ""

        # Load correct answer from correct_answer.txt
        correct_answer_path = state_dir.parent / "correct_answer.txt"
        correct_answer = ""
        if correct_answer_path.exists():
            correct_answer = correct_answer_path.read_text().strip()

        # Load problem statement from state.txt
        state_path = state_dir / "state.txt"
        problem_statement = ""
        if state_path.exists():
            problem_statement = state_path.read_text().strip()

        # Load solution trace from <action_hash>-original.txt
        original_path = state_dir / f"{action_hash}-original.txt"
        solution_trace = ""
        if original_path.exists():
            solution_trace = original_path.read_text().strip()

        rows.append(
            {
                "problem_id": problem_id,
                "answer": answer,
                "correct_answer": correct_answer,
                "reflection_text": content,
                "problem_statement": problem_statement,
                "solution_trace": solution_trace,
                "action_hash": action_hash,
            }
        )

    # Write CSV
    fieldnames = [
        "problem_id",
        "answer",
        "correct_answer",
        "reflection_text",
        "problem_statement",
        "solution_trace",
        "action_hash",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} reflections to {output_path}")


if __name__ == "__main__":
    main()
