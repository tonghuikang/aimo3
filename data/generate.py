"""Generate State objects for RL training by running rollouts on math problems.

Usage: uv run python3 -m data.generate
"""

import json
import os
import random
import resource
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Increase file descriptor limit (equivalent to ulimit -n 4096)
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))

import pandas as pd

# Reuse functions from notebook.py
from notebook import (
    SYSTEM_CONTENT,
    LocalJupyterSession,
    build_prompt_token_ids,
    execute_python_code,
    extract_boxed_text,
    harmony_encoding,
    is_valid_answer_string,
    rollout_given_state,
    starter_code,
)

from data.store_types import Action, Entry, Problem, State

# Configuration
DATA_DIR = "data"
STATES_DIR = f"{DATA_DIR}/raw"
INDEX_PATH = f"{DATA_DIR}/problems.jsonl"


def should_generate_for_problem(problem_id: str, num_correct: int, num_wrong: int) -> bool:
    # Check basic filters
    if problem_id not in ("pe-616"):
        return False
    return True


def run_single_completion(
    problem: Problem,
    prompt_token_ids: list[int],
    initial_prompt_length: int,
    completion_idx: int,
    num_completions: int,
) -> Action:
    """Run a single completion for a problem. Thread-safe."""
    print(f"\n--- [{problem.id}] Completion {completion_idx + 1}/{num_completions} ---")

    # Run rollout with fresh Jupyter session
    all_token_ids = prompt_token_ids.copy()
    jupyter_session = None
    try:
        jupyter_session = LocalJupyterSession()
        execute_python_code(jupyter_session, starter_code)
        final_token_ids = rollout_given_state(
            all_token_ids=all_token_ids,
            jupyter_session=jupyter_session,
        )
    except RuntimeError as e:
        print(f"[{problem.id}] rollout error: {e}")
        final_token_ids = prompt_token_ids  # reset intended
    finally:
        if jupyter_session is not None:
            print(f"[{problem.id}] Cleaning up Jupyter session")
            jupyter_session.close()

    # Split into prompt tokens and completion tokens
    completion_token_ids = final_token_ids[initial_prompt_length:]

    # Decode and extract information
    detokenized_text = harmony_encoding.decode(final_token_ids)
    boxed_text = extract_boxed_text(detokenized_text)

    # Extract answer and compute reward (1 if correct, 0 otherwise)
    reward = 0.0
    extracted_answer: int | None = None
    if is_valid_answer_string(boxed_text):
        extracted_answer = int(boxed_text)
        if extracted_answer == problem.answer:
            reward = 1.0
            print(f"[{problem.id}] CORRECT! Answer: {extracted_answer}")
        else:
            print(
                f"[{problem.id}] INCORRECT. Got {extracted_answer}, expected {problem.answer}"
            )
    else:
        print(f"[{problem.id}] No valid answer extracted. Expected: {problem.answer}")

    return Action(
        tokens=completion_token_ids,
        reward=reward,
        model_version="gpt-oss-120b",
        time_generated=datetime.now().strftime("%m-%d-%H-%M-%S"),
        answer=extracted_answer,
    )


def generate_state_for_problem(
    problem: Problem, num_completions: int = 2
) -> tuple[State, str]:
    """Generate a State object for a given problem with parallel completions.

    Returns tuple of (state, timestamp) where timestamp is when generation started.
    """
    timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
    print(f"\n{'=' * 60}")
    print(f"Processing problem: {problem.id} ({num_completions} completions)")
    print(f"{'=' * 60}")

    # Build initial prompt (system instructions + problem statement)
    prompt_token_ids: list[int] = build_prompt_token_ids(
        system_content=SYSTEM_CONTENT,
        user_content=problem.statement,
    )
    initial_prompt_length = len(prompt_token_ids)

    # Run completions in parallel
    actions: list[Action] = [None] * num_completions  # type: ignore
    with ThreadPoolExecutor(max_workers=num_completions) as executor:
        futures = {
            executor.submit(
                run_single_completion,
                problem,
                prompt_token_ids,
                initial_prompt_length,
                idx,
                num_completions,
            ): idx
            for idx in range(num_completions)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                actions[idx] = future.result()
            except (RuntimeError, ValueError, OSError) as e:
                print(f"[{problem.id}] Completion {idx + 1} failed: {e}")
                # Create a failed action with 0 reward
                actions[idx] = Action(
                    tokens=[],
                    reward=0.0,
                    model_version="gpt-oss-120b",
                    time_generated=datetime.now().strftime("%m-%d-%H-%M-%S"),
                    answer=None,
                )

    # Compute average reward across all completions
    total_reward = sum(action.reward for action in actions)
    avg_reward = total_reward / num_completions if num_completions > 0 else 0.0

    # Build State object
    state = State(
        problem=problem,
        tokens=prompt_token_ids,
        actions=[action for action in actions if action.tokens],
        reward=avg_reward,
        model_version="gpt-oss-120b",
    )

    return state, timestamp


def save_state(state: State, timestamp: str) -> str:
    """Save state to detailed/ and append summary to problems.jsonl."""
    entry = Entry(
        timestamp=timestamp, state=state, states_dir=STATES_DIR, index_path=INDEX_PATH
    )
    filename = entry.save()
    print(f"Saved state to {STATES_DIR}/{filename}")
    return filename


def load_problems(csv_path: str = "data/problems.csv") -> list[Problem]:
    """Load problems from CSV file."""
    df = pd.read_csv(csv_path)
    problems = []
    for _, row in df.iterrows():
        # Use 0 as placeholder if answer is missing
        answer = row.get("answer", 0)
        if answer is None or (isinstance(answer, float) and pd.isna(answer)):
            answer = 0
        problems.append(
            Problem(
                id=str(row["problem_id"]),
                statement=str(row["problem"]),
                answer=int(answer),
            )
        )
    return problems


def load_completion_stats() -> dict[str, dict[str, int]]:
    """Load completion statistics from problems.jsonl.

    Returns dict mapping problem_id to {'correct': count, 'wrong': count}.
    """
    stats: dict[str, dict[str, int]] = {}
    if not os.path.exists(INDEX_PATH):
        return stats

    with open(INDEX_PATH) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                problem_id = data["problem_id"]
                if problem_id not in stats:
                    stats[problem_id] = {"correct": 0, "wrong": 0}

                # Count correct vs wrong completions based on rewards
                for reward in data["completion_rewards"]:
                    if reward >= 1.0:
                        stats[problem_id]["correct"] += 1
                    else:
                        stats[problem_id]["wrong"] += 1

    return stats


def process_and_save_problem(problem: Problem, num_completions: int) -> str | None:
    """Generate state for a problem and save it. Returns the filename or None if failed."""
    state, timestamp = generate_state_for_problem(
        problem, num_completions=num_completions
    )

    # Check if any completions succeeded (have tokens)
    has_successful_completion = any(len(action.tokens) > 0 for action in state.actions)
    if not has_successful_completion:
        print(f"[{problem.id}] All completions failed, not saving state")
        return None

    return save_state(state, timestamp)


def main(
    num_completions: int,
    max_parallel_problems: int,
):
    """Main entry point."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Create empty problems.jsonl if needed
    if not os.path.exists(INDEX_PATH):
        open(INDEX_PATH, "w").close()

    # Load problems and shuffle them
    problems = load_problems()
    random.shuffle(problems)

    # Load existing completion statistics
    completion_stats = load_completion_stats()

    # Filter problems based on should_generate_for_problem
    original_count = len(problems)
    filtered_problems = []
    for problem in problems:
        # Check if we should generate more completions for this problem
        stats = completion_stats.get(problem.id, {"correct": 0, "wrong": 0})
        num_correct = stats["correct"]
        num_wrong = stats["wrong"]

        if should_generate_for_problem(problem.id, num_correct, num_wrong):
            filtered_problems.append(problem)

    problems = filtered_problems
    skipped_count = original_count - len(problems)

    print(f"Loaded {original_count} problems, skipping {skipped_count} based on completion stats")
    print(
        f"Processing {len(problems)} problems with {num_completions} completions each"
    )
    print(f"Running up to {max_parallel_problems} problems in parallel")

    # Generate states for problems in parallel
    with ThreadPoolExecutor(max_workers=max_parallel_problems) as executor:
        futures = {
            executor.submit(process_and_save_problem, problem, num_completions): problem
            for problem in problems
        }
        for future in as_completed(futures):
            problem = futures[future]
            try:
                filename = future.result()
                if filename is not None:
                    print(f"Completed problem {problem.id}: {filename}")
                else:
                    print(f"Problem {problem.id} skipped (no successful completions)")
            except (RuntimeError, ValueError, OSError) as e:
                print(f"Problem {problem.id} failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate states for RL training")
    parser.add_argument(
        "-c",
        "--num-completions",
        type=int,
        default=1,
        help="Number of completions per problem (default: 2)",
    )
    parser.add_argument(
        "-p",
        "--max-parallel-problems",
        type=int,
        default=4,
        help="Max problems to run in parallel (default: 1)",
    )
    args = parser.parse_args()
    main(
        num_completions=args.num_completions,
        max_parallel_problems=args.max_parallel_problems,
    )
