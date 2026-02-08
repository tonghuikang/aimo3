"""Shared utilities for training scripts."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import tinker

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")


class CorpusEntry(TypedDict):
    """Entry from corpus.jsonl."""

    problem_id: str
    state_hash: str
    action_hash: str
    span: str
    trace_id: str
    prompt_token_count: int
    completion_token_count: int
    included: bool
    correct_trace: bool
    has_boxed: bool


class TrainingData(TypedDict):
    """Entry from training JSONL files."""

    prompt_tokens: list[int]
    tokens: list[int]


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_corpus_entries() -> list[CorpusEntry]:
    """Load corpus.jsonl and return typed entries."""
    return cast(list[CorpusEntry], load_jsonl(DATA_DIR / "corpus.jsonl"))


def load_training_data(path: Path) -> list[TrainingData]:
    """Load training JSONL file and return typed entries."""
    return cast(list[TrainingData], load_jsonl(path))


@dataclass
class TrainingExample:
    """Represents a single training example with pre-tokenized data."""

    problem_id: str
    state_hash: str
    action_hash: str
    span: str
    trace_id: str
    prompt_token_count: int
    completion_token_count: int

    @classmethod
    def from_dict(cls, entry: CorpusEntry) -> "TrainingExample":
        """Create a TrainingExample from a corpus entry dict."""
        return cls(
            problem_id=entry["problem_id"],
            state_hash=entry["state_hash"],
            action_hash=entry["action_hash"],
            span=entry["span"],
            trace_id=entry["trace_id"],
            prompt_token_count=entry["prompt_token_count"],
            completion_token_count=entry["completion_token_count"],
        )

    def get_training_file_path(self) -> Path:
        """Get path to the training JSONL file."""
        return (
            DATA_DIR
            / "corpus"
            / self.problem_id
            / self.state_hash
            / self.action_hash
            / self.span
            / f"{self.trace_id}.jsonl"
        )

    def load_tokens(self) -> tuple[list[int], list[int]]:
        """Load prompt and completion tokens from the training file."""
        training_path = self.get_training_file_path()
        training_data = load_training_data(training_path)
        entry = training_data[0]
        return entry["prompt_tokens"], entry["tokens"]


def build_datum(
    prompt_tokens: list[int],
    completion_tokens: list[int],
    max_length: int = 32768,
) -> tinker.Datum | None:
    """Build a Tinker Datum from prompt and completion tokens."""
    full_tokens = prompt_tokens + completion_tokens

    if len(full_tokens) > max_length:
        full_tokens = full_tokens[:max_length]
        completion_len = len(full_tokens) - len(prompt_tokens)
        if completion_len <= 0:
            return None
    else:
        completion_len = len(completion_tokens)

    prompt_len = len(full_tokens) - completion_len

    model_input = tinker.ModelInput(
        chunks=[tinker.types.EncodedTextChunk(tokens=full_tokens[:-1])]
    )
    target_tokens = full_tokens[1:]
    weights = [0.0] * (prompt_len - 1) + [1.0] * completion_len
    weights = weights[: len(target_tokens)]

    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "weights": tinker.TensorData(
                data=weights,
                dtype="float32",
                shape=[len(weights)],
            ),
            "target_tokens": tinker.TensorData(
                data=target_tokens,
                dtype="int64",
                shape=[len(target_tokens)],
            ),
        },
    )
