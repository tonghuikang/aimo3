"""Store types for managing problem states and actions."""

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Self

BASE36_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"


def tokens_to_base36_hash(token_ids: list[int], length: int = 6) -> str:
    """Generate a base36 hash of specified length from token IDs."""
    token_bytes = b"".join(t.to_bytes(4, "big") for t in token_ids)
    digest = hashlib.sha256(token_bytes).digest()
    num = int.from_bytes(digest[:8], "big")
    result = ""
    while num and len(result) < length:
        result = BASE36_CHARS[num % 36] + result
        num //= 36
    return result.zfill(length)[:length]


def update_txt_file(filepath: str, hash_value: str) -> None:
    """Add hash to txt file if not already present. Creates file if needed."""
    path = Path(filepath)
    existing_hashes: set[str] = set()
    if path.exists():
        existing_hashes = set(path.read_text().strip().split("\n")) - {""}
    if hash_value not in existing_hashes:
        existing_hashes.add(hash_value)
        path.write_text(
            "\n".join(sorted(existing_hashes)) + "\n" if existing_hashes else ""
        )


@dataclass
class Problem:
    id: str
    statement: str
    answer: int


@dataclass
class Action:
    tokens: list[int]
    reward: float
    model_version: str
    time_generated: str
    answer: int | None = None

    @property
    def hash(self) -> str:
        return tokens_to_base36_hash(self.tokens)

    def to_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "reward": self.reward,
            "model_version": self.model_version,
            "time_generated": self.time_generated,
            "answer": self.answer,
        }

    def save_to_dir(self, dir_path: str) -> None:
        """Save action.json to the specified directory."""
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "action.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_dir(cls, dir_path: str) -> Self:
        """Load action from the specified directory."""
        path = Path(dir_path)

        with open(path / "action.json") as f:
            data = json.load(f)

        return cls(
            tokens=data["tokens"],
            reward=data["reward"],
            model_version=data["model_version"],
            time_generated=data["time_generated"],
            answer=data["answer"],
        )


@dataclass
class State:
    """
    How to Remove a State:

    Required cleanup:
    1. data/problems.jsonl - remove entry with matching state_hash
    2. data/raw/{problem_id}/{state_hash}/ - remove state directory
    3. data/raw/{problem_id}/states.txt - remove hash from file
    4. data/annotate/{problem_id}/{state_hash}/state.txt - remove print file
    5. data/annotate/{problem_id}/{state_hash}/ - remove annotation directory
    6. data/label/{problem_id}/{state_hash}/ - remove label directory
    7. data/labels.jsonl - remove entries with matching state_hash
    """

    problem: Problem
    tokens: list[int]
    actions: list[Action]
    reward: float
    model_version: str

    @property
    def hash(self) -> str:
        return tokens_to_base36_hash(self.tokens)

    def extract_code_from_tokens(self) -> list[str]:
        """Extract python code blocks from tokens using regex on detokenized text."""
        import re

        from openai_harmony import HarmonyEncodingName, load_harmony_encoding

        harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        text = harmony_encoding.decode(self.tokens)

        # Pattern: to=python code<|message|>CODE<|call|>
        pattern = r"to=python code<\|message\|>(.*?)<\|call\|>"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]

    def to_dict(self) -> dict:
        """Return dict without actions (they're stored separately)."""
        return {
            "problem": asdict(self.problem),
            "tokens": self.tokens,
            "model_version": self.model_version,
        }

    def to_dict_full(self) -> dict:
        """Return full dict including actions (for backward compatibility)."""
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict_full(), indent=2)

    def save_to_dir(self, dir_path: str) -> None:
        """Save state.json and actions to the specified directory."""
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)

        # Save state.json
        with open(path / "state.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # Update actions.txt (append new actions without removing existing ones)
        actions_txt = path / "actions.txt"
        if self.actions:
            for action in self.actions:
                update_txt_file(str(actions_txt), action.hash)
                action.save_to_dir(str(path / action.hash))
        elif not actions_txt.exists():
            actions_txt.write_text("")

    @classmethod
    def load_from_dir(cls, dir_path: str) -> Self:
        """Load state and its actions from the specified directory."""
        path = Path(dir_path)

        # Load state.json
        with open(path / "state.json") as f:
            data = json.load(f)

        # Load actions
        actions: list[Action] = []
        actions_txt = path / "actions.txt"
        if actions_txt.exists():
            action_hashes = actions_txt.read_text().strip().split("\n")
            action_hashes = [h for h in action_hashes if h]  # Filter empty
            for action_hash in action_hashes:
                actions.append(Action.load_from_dir(str(path / action_hash)))

        # Calculate reward as average of action rewards
        reward = sum(a.reward for a in actions) / len(actions) if actions else 0.0

        return cls(
            problem=Problem(
                id=data["problem"]["id"],
                statement=data["problem"]["statement"],
                answer=data["problem"]["answer"],
            ),
            tokens=data["tokens"],
            actions=actions,
            reward=reward,
            model_version=data["model_version"],
        )

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            problem=Problem(
                id=data["problem"]["id"],
                statement=data["problem"]["statement"],
                answer=data["problem"]["answer"],
            ),
            tokens=data["tokens"],
            actions=[
                Action(
                    tokens=a["tokens"],
                    reward=a["reward"],
                    model_version=a["model_version"],
                    time_generated=a["time_generated"],
                    answer=a["answer"],
                )
                for a in data["actions"]
            ],
            reward=data["reward"],
            model_version=data["model_version"],
        )

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        return cls.from_dict(json.loads(json_str))


@dataclass
class Entry:
    timestamp: str
    state: State
    states_dir: str = "data/raw"
    index_path: str = "data/problems.jsonl"

    @property
    def filename(self) -> str:
        return f"{self.problem_id}-{self.timestamp}.json"

    @property
    def problem_id(self) -> str:
        return self.state.problem.id

    @property
    def reward(self) -> float:
        return self.state.reward

    @property
    def completion_rewards(self) -> list[float]:
        return [a.reward for a in self.state.actions]

    @property
    def prompt_tokens(self) -> int:
        return len(self.state.tokens)

    @property
    def completion_tokens(self) -> int:
        return sum(len(a.tokens) for a in self.state.actions)

    @property
    def num_completions(self) -> int:
        return len(self.state.actions)

    @property
    def model_version(self) -> str:
        return self.state.model_version

    @property
    def correct_lengths(self) -> list[int]:
        return [
            len(a.tokens)
            for a in self.state.actions
            if a.reward >= 1.0 and len(a.tokens) > 0
        ]

    @property
    def shortest(self) -> int | None:
        """Token length of the shortest correct completion, or None if none correct."""
        return min(self.correct_lengths) if self.correct_lengths else None

    @property
    def longest(self) -> int | None:
        """Token length of the longest correct completion, or None if none correct."""
        return max(self.correct_lengths) if self.correct_lengths else None

    @property
    def has_findings(self) -> bool:
        """Check if a non-empty findings.txt exists for this problem."""
        annotate_dir = Path(self.states_dir).parent / "annotate"
        findings_path = annotate_dir / self.problem_id / "findings.txt"
        return findings_path.exists() and findings_path.stat().st_size > 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "problem_id": self.problem_id,
                "state_hash": self.state.hash,
                "action_hashes": [a.hash for a in self.state.actions],
                "completion_rewards": self.completion_rewards,
                "prompt_tokens": self.prompt_tokens,
                "timestamp": self.timestamp,
                "shortest": self.shortest,
                "longest": self.longest,
                "has_findings": self.has_findings,
            }
        )

    def save(self) -> str:
        """Save state to raw/{problem_id}/{state_hash}/ hierarchy. Returns state_hash."""
        problem_dir = Path(self.states_dir) / self.problem_id
        state_dir = problem_dir / self.state.hash

        # Save state and its actions
        self.state.save_to_dir(str(state_dir))

        # Update states.txt in problem directory
        problem_dir.mkdir(parents=True, exist_ok=True)
        update_txt_file(str(problem_dir / "states.txt"), self.state.hash)

        # Append to problems.jsonl with only the new actions (frontend merges entries)
        index_path = Path(self.index_path)
        with open(index_path, "a") as f:
            f.write(self.to_json() + "\n")

        return self.state.hash

    @classmethod
    def load(cls, states_dir: str, problem_id: str, state_hash: str) -> Self:
        """Load entry from directory structure."""
        state_dir = Path(states_dir) / problem_id / state_hash
        state = State.load_from_dir(str(state_dir))
        return cls(timestamp="", state=state, states_dir=states_dir)

    @classmethod
    def from_dict(cls, data: dict, state: State) -> Self:
        return cls(timestamp=data["timestamp"], state=state)
