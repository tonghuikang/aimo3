#!/usr/bin/env python3
"""
Create training corpus from labeled data.

Extracts "better" traces from labeled data and builds:
- data/corpus.csv - Main CSV with columns: problem_id, prompt, completion,
  character_count, token_count, annotation_reason
- data/corpus_full.jsonl - Same as corpus.csv but prompt and completion are token IDs
- data/corpus.jsonl - Lightweight index for UI
- data/corpus/<problem_id>/<state>/<action>/<spanstart-spanend>/<trace_id>.jsonl - Individual traces

Execution:
uv run python3 -m data.create_corpus
"""

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import tiktoken
import random

random.seed(42)

@dataclass
class LabelEntry:
    problem_id: str
    state_hash: str
    action_hash: str
    action_reward: float = 0.0
    reflection_exists: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "LabelEntry":
        return cls(
            problem_id=data["problem_id"],
            state_hash=data["state_hash"],
            action_hash=data["action_hash"],
            action_reward=data["action_reward"],
            reflection_exists=data["reflection_exists"],
        )


@dataclass
class CorpusEntry:
    problem_id: str
    problem_source: str
    state_hash: str
    action_hash: str
    span: str
    trace_id: str
    prompt: str
    completion: str
    prompt_token_count: int
    completion_token_count: int
    character_count: int
    included: bool
    annotation_reason: str
    prompt_tokens: list[int] | None = None
    completion_tokens: list[int] | None = None
    reflection_exists: bool = False
    correct_trace: bool = False
    has_boxed: bool = False
    file_path: str = ""

    @property
    def token_count(self) -> int:
        return self.prompt_token_count + self.completion_token_count

    def to_full_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "problem_source": self.problem_source,
            "prompt": self.prompt_tokens,
            "completion": self.completion_tokens,
            "character_count": self.character_count,
            "token_count": self.token_count,
            "included": self.included,
            "annotation_reason": self.annotation_reason,
            "correct_trace": self.correct_trace,
            "has_boxed": self.has_boxed,
        }

    def to_index_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "state_hash": self.state_hash,
            "action_hash": self.action_hash,
            "span": self.span,
            "trace_id": self.trace_id,
            "prompt_token_count": self.prompt_token_count,
            "completion_token_count": self.completion_token_count,
            "character_count": self.character_count,
            "included": self.included,
            "correct_trace": self.correct_trace,
            "has_boxed": self.has_boxed,
        }

    def to_csv_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "problem_source": self.problem_source,
            "prompt": self.prompt,
            "completion": self.completion,
            "character_count": self.character_count,
            "token_count": self.token_count,
            "included": self.included,
            "annotation_reason": self.annotation_reason,
            "correct_trace": self.correct_trace,
            "has_boxed": self.has_boxed,
        }


@dataclass
class SpanLabel:
    better: list[str]
    neutral: list[str]
    worse: list[str]
    reason: str

    @classmethod
    def from_dict(cls, data: dict) -> "SpanLabel":
        return cls(
            better=data["better"],
            neutral=data["neutral"],
            worse=data["worse"],
            reason=data["reason"],
        )




def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file as list of dicts."""
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_labels(path: Path) -> dict[str, SpanLabel]:
    """Load label file. Returns dict mapping span key to SpanLabel."""
    labels: dict[str, SpanLabel] = {}
    for entry in load_jsonl(path):
        for span_key, label_data in entry.items():
            labels[span_key] = SpanLabel.from_dict(label_data)
    return labels


def load_alternates(path: Path) -> dict[int, dict[str, list[int]]]:
    """Load alternates file. Returns dict mapping position to {alt_id: tokens}."""
    alternates: dict[int, dict[str, list[int]]] = {}
    for entry in load_jsonl(path):
        for pos_key, alts_list in entry.items():
            pos = int(pos_key)
            if pos not in alternates:
                alternates[pos] = {}
            for alt_obj in alts_list:
                for alt_id, tokens in alt_obj.items():
                    alternates[pos][alt_id] = tokens
    return alternates


def load_action_tokens(path: Path) -> list[int]:
    """Load action.json and return tokens."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data["tokens"]


def load_state_tokens(path: Path) -> list[int]:
    """Load state.json and return tokens."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data["tokens"]


SPECIAL_TOKENS = {
    199998: "<|startoftext|>",
    199999: "<|endoftext|>",
    200002: "<|return|>",
    200003: "<|constrain|>",
    200005: "<|channel|>",
    200006: "<|start|>",
    200007: "<|end|>",
    200008: "<|message|>",
    200012: "<|call|>",
}


def decode_tokens(enc: tiktoken.Encoding, token_ids: list[int]) -> str:
    """Decode tokens to text, handling special tokens."""
    result = ""
    regular_tokens: list[int] = []
    for token_id in token_ids:
        if token_id in SPECIAL_TOKENS:
            if regular_tokens:
                result += enc.decode(regular_tokens)
                regular_tokens = []
            result += SPECIAL_TOKENS[token_id]
        else:
            regular_tokens.append(token_id)
    if regular_tokens:
        result += enc.decode(regular_tokens)
    return result


def get_problem_source(problem_id: str) -> str:
    """Get problem source from problem_id prefix."""
    if problem_id.startswith("aimo"):
        return "AIMO"
    elif problem_id.startswith("imo"):
        return "IMO"
    elif problem_id.startswith("pe"):
        return "Project Euler"
    return "Unknown"


skipped_problem_ids: set[str] = {
    "aimo3-q9-m0",
    "aimo3-q9-m1",
    "aimo3-q9-m2",
    "aimo3-q10-g0",
    "aimo3-q10-g0-g4M",
    "aimo3-q10-g44636594",
}

def choose_entries_to_include(
    entries: list[CorpusEntry],
) -> None:

    random.shuffle(entries)
    action_span_to_trace_selected: dict[tuple[str,str], tuple[int, str]] = {}
    action_hash_to_count: dict[str, int] = Counter()
    for e in entries:
        # general filters
        if e.completion_token_count <= 10:
            continue
        if not e.reflection_exists:
            continue
        if e.problem_id in skipped_problem_ids:
            continue

        # one training data per trace
        if (e.action_hash, e.span) not in action_span_to_trace_selected:
            action_span_to_trace_selected[e.action_hash, e.span] = e.completion_token_count, e.trace_id
        else:
            if not e.correct_trace:
                # for wrong completions, choose the longest trace
                existing_completion_length, _ = action_span_to_trace_selected[e.action_hash, e.span]
                if e.completion_token_count >= existing_completion_length:
                    action_span_to_trace_selected[e.action_hash, e.span] = e.completion_token_count, e.trace_id

    for e in entries:
        if (e.action_hash, e.span) not in action_span_to_trace_selected:
            continue
        _, selected_trace_id = action_span_to_trace_selected[e.action_hash, e.span]
        if selected_trace_id != e.trace_id:
            continue
        action_hash_to_count[e.action_hash] += 1
        if not e.problem_id.startswith("aimo3"):
            if action_hash_to_count[e.action_hash] > 4:
                continue
        else:
            if action_hash_to_count[e.action_hash] > 20:
                continue
        e.included = True

    entries.sort(key = lambda entry : (entry.problem_id, entry.state_hash, entry.action_hash, int(entry.span.split("-")[0]), entry.trace_id))


def main(token_limit: int = 32768, character_limit: int = 100000):
    data_dir = Path(__file__).parent
    corpus_dir = data_dir / "corpus"

    # Clean and recreate corpus directory
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True)

    # Load tiktoken encoder
    enc = tiktoken.get_encoding("o200k_base")

    # Load labels index
    labels_index_path = data_dir / "labels.jsonl"
    labels_index = load_jsonl(labels_index_path)

    # Prepare CSV, full JSONL, and index
    csv_path = data_dir / "corpus.csv"
    full_path = data_dir / "corpus_full.jsonl"
    index_path = data_dir / "corpus.jsonl"

    entries: list[CorpusEntry] = []

    # Parse into LabelEntry objects
    all_entries = [LabelEntry.from_dict(e) for e in labels_index]

    for entry in all_entries:

        # Load label file
        label_path = (
            data_dir
            / "label"
            / entry.problem_id
            / entry.state_hash
            / f"{entry.action_hash}.jsonl"
        )
        labels = load_labels(label_path)

        # Load alternates file
        alt_path = (
            data_dir
            / "alternate"
            / entry.problem_id
            / entry.state_hash
            / f"{entry.action_hash}.jsonl"
        )
        alternates = load_alternates(alt_path)

        # Load state tokens
        state_path = (
            data_dir / "raw" / entry.problem_id / entry.state_hash / "state.json"
        )
        state_tokens = load_state_tokens(state_path)

        # Load original action tokens
        action_path = (
            data_dir
            / "raw"
            / entry.problem_id
            / entry.state_hash
            / entry.action_hash
            / "action.json"
        )
        original_tokens = load_action_tokens(action_path)

        # Process each span
        for span_key, span_label in labels.items():
            if not span_label.better:
                # only include traces marked better
                continue

            annotation_reason = span_label.reason

            # Parse span
            parts = span_key.split("-")
            span_start = int(parts[0])
            span_end = int(parts[1]) if len(parts) > 1 else span_start

            # Build full prompt = state tokens + action tokens up to span_start
            full_prompt_tokens = state_tokens + original_tokens[:span_start]

            # Get alternates for this position
            pos_alternates = alternates.get(span_start, {})

            # Check if any completion at this position has \boxed{
            any_boxed = "\\boxed{" in decode_tokens(
                enc, original_tokens[span_start:span_end]
            )
            if not any_boxed:
                for alt_tokens in pos_alternates.values():
                    if "\\boxed{" in decode_tokens(enc, alt_tokens):
                        any_boxed = True
                        break

            for trace_id in span_label.better:
                # Get tokens for this trace (completion)
                if trace_id == "original":
                    completion_tokens = original_tokens[span_start:span_end]
                else:
                    if trace_id not in pos_alternates:
                        print(
                            f"Warning: trace_id {trace_id} not found in alternates for "
                            f"{entry.problem_id}/{entry.state_hash}/{entry.action_hash} "
                            f"at position {span_start}"
                        )
                        continue
                    completion_tokens = pos_alternates[trace_id]

                if not completion_tokens:
                    continue
                
                # Check token limit - Tinker infra limitations
                total_tokens = len(full_prompt_tokens) + len(completion_tokens)
                if total_tokens >= token_limit:
                    continue

                # Decode texts for CSV
                prompt_text = decode_tokens(enc, full_prompt_tokens)
                completion_text = decode_tokens(enc, completion_tokens)

                # Check character limit - Corpus prize compliance
                character_count = len(prompt_text) + len(completion_text)
                if character_count > character_limit:
                    continue

                # Create output path
                span_str = f"{span_start}-{span_end}"
                trace_dir = (
                    corpus_dir
                    / entry.problem_id
                    / entry.state_hash
                    / entry.action_hash
                    / span_str
                )
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_path = trace_dir / f"{trace_id}.jsonl"

                # Write trace file with both prompt and completion tokens
                with open(trace_path, "w") as f:
                    json.dump(
                        {
                            "prompt_tokens": full_prompt_tokens,
                            "tokens": completion_tokens,
                        },
                        f,
                    )
                    f.write("\n")

                entries.append(
                    CorpusEntry(
                        problem_id=entry.problem_id,
                        problem_source=get_problem_source(entry.problem_id),
                        state_hash=entry.state_hash,
                        action_hash=entry.action_hash,
                        span=span_str,
                        trace_id=trace_id,
                        prompt=prompt_text,
                        completion=completion_text,
                        prompt_token_count=len(full_prompt_tokens),
                        completion_token_count=len(completion_tokens),
                        character_count=character_count,
                        included=False,  # to be overwritten
                        annotation_reason=annotation_reason,
                        prompt_tokens=full_prompt_tokens,
                        completion_tokens=completion_tokens,
                        reflection_exists=entry.reflection_exists,
                        correct_trace=entry.action_reward == 1.0,
                        has_boxed=any_boxed,
                        file_path=str(trace_path.relative_to(corpus_dir)),
                    )
                )

    choose_entries_to_include(entries)

    # Write CSV
    if entries:
        fieldnames = [
            "problem_id",
            "problem_source",
            "prompt",
            "completion",
            "character_count",
            "token_count",
            "included",
            "annotation_reason",
            "correct_trace",
            "has_boxed",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in entries:
                writer.writerow(e.to_csv_dict())

    # Write full JSONL (same as CSV but with tokens instead of text)
    with open(full_path, "w") as f:
        for e in entries:
            json.dump(e.to_full_dict(), f)
            f.write("\n")

    # Write index
    with open(index_path, "w") as f:
        for e in entries:
            json.dump(e.to_index_dict(), f)
            f.write("\n")

    # Calculate totals and maximums
    total_prompt_tokens = sum(e.prompt_token_count for e in entries)
    total_completion_tokens = sum(e.completion_token_count for e in entries)
    included_count = sum(1 for e in entries if e.included)
    max_token_count = max(e.token_count for e in entries) if entries else 0
    max_character_count = max(e.character_count for e in entries) if entries else 0

    print(f"Created {len(entries)} corpus entries ({included_count} included)")
    print(f"Total prompt tokens: {total_prompt_tokens:,}")
    print(f"Total completion tokens: {total_completion_tokens:,}")
    print(f"Max token count: {max_token_count:,}")
    print(f"Max character count: {max_character_count:,}")
    print(f"CSV: {csv_path}")
    print(f"Full (tokens): {full_path}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create training corpus from labeled data"
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=32768,
        help="Maximum total tokens (prompt + completion). Default: 32768",
    )
    parser.add_argument(
        "--character-limit",
        type=int,
        default=100000,
        help="Maximum total characters (prompt + completion). Default: 100000",
    )
    args = parser.parse_args()
    main(token_limit=args.token_limit, character_limit=args.character_limit)
