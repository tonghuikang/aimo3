"""Generate alternate completions at paragraph boundaries.

Finds positions after double newlines in assistant turns and generates
alternate continuations until the next double newline or end of turn.

Usage: uv run python3 -m data.roll_alternates <problem_id>
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai_harmony import HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding
import httpx
from openai import OpenAI, APIConnectionError, APIError, APIStatusError, APITimeoutError

from data.store_types import Action, State, tokens_to_base36_hash
from kaggle_secrets import UserSecretsClient
from data.prepare_annotations import prepare_annotation
from data.extract_labels import extract_labels, regenerate_labels_jsonl
from data.common import (
    ALTERNATE_DIR,
    RAW_DIR,
    find_action_location,
    find_state_location,
)

secrets = UserSecretsClient()
fetch_secret = getattr(secrets, "get_secret")

# Model configuration (Modal inference)
MODEL_NAME = "vllm-model"
INFERENCE_URL = fetch_secret("MODAL_INFERENCE_URL")

client = OpenAI(
    base_url=INFERENCE_URL,
    api_key="not-used",
)

# Global encoding
harmony_encoding: HarmonyEncoding = load_harmony_encoding(
    HarmonyEncodingName.HARMONY_GPT_OSS
)
stop_token_ids: list[int] = list(harmony_encoding.stop_tokens_for_assistant_actions())


def find_double_newline_positions(tokens: list[int]) -> list[int]:
    """Find token indices right after double newlines.

    Returns indices of tokens that immediately follow a token containing \n\n.
    """
    positions = []
    for i, token in enumerate(tokens):
        if "\n\n" in harmony_encoding.decode([token]):
            if i + 1 < len(tokens):
                positions.append(i + 1)
    return positions


def find_next_paragraph_end(tokens: list[int], start_idx: int) -> int:
    """Find where the next paragraph ends (next \n\n or end of tokens).

    Returns the token index where generation should stop.
    """
    if start_idx >= len(tokens):
        return len(tokens)

    text = harmony_encoding.decode(tokens[start_idx:])
    pos = text.find("\n\n")

    if pos == -1:
        return len(tokens)

    # Map character position back to token index
    running_len = 0
    for token_idx, token in enumerate(tokens[start_idx:]):
        token_text = harmony_encoding.decode([token])
        running_len += len(token_text)
        if running_len >= pos:
            return start_idx + token_idx + 1

    return len(tokens)


def is_in_assistant_turn(tokens: list[int], position: int) -> bool:
    """Check if position is within an assistant turn (not in tool/user turn)."""
    # Decode up to position and check for turn markers
    text = harmony_encoding.decode(tokens[:position])

    # Find last turn marker
    last_assistant = text.rfind("<|start|>assistant")
    last_user = text.rfind("<|start|>user")
    last_tool = text.rfind("<|start|>python")

    # Position is in assistant turn if assistant marker is most recent
    if last_assistant == -1:
        return False

    return last_assistant > max(last_user, last_tool)


def generate_single_alternate(
    prompt_tokens: list[int],
    max_tokens: int = 2048,
) -> list[int] | None:
    """Generate a single alternate completion until \n\n or stop token.

    Returns the generated tokens (not including prompt), or None on connection error.
    """
    try:
        stream = client.completions.create(
            model=MODEL_NAME,
            prompt=prompt_tokens,
            max_tokens=max_tokens,
            temperature=1.0,
            stream=True,
            stop="\n\n",
            extra_body={
                "return_token_ids": True,
                # "include_stop_str_in_output": True,
            },
        )

        generated_tokens: list[int] = []
        generated_text = ""

        for chunk in stream:
            chunk_token_ids = getattr(chunk.choices[0], "token_ids", None)
            if chunk_token_ids:
                for token_id in chunk_token_ids:
                    generated_tokens.append(token_id)
                    if token_id in stop_token_ids:
                        stream.close()
                        return generated_tokens

            # Get text directly if available
            chunk_text = chunk.choices[0].text
            if chunk_text:
                generated_text += chunk_text
                # Stop on \n\n (paragraph boundary)
                if "\n\n" in generated_text:
                    stream.close()
                    return generated_tokens

            finish_reason = chunk.choices[0].finish_reason
            if finish_reason:
                stream.close()
                break

        return generated_tokens
    except (
        APIConnectionError,
        APIError,
        APIStatusError,
        APITimeoutError,
        httpx.HTTPError,
    ) as e:
        print(f"    Error, skipping: {type(e).__name__}")
        return None


def roll_alternates_streaming(
    state: State,
    action_tokens: list[int],
    output_path: Path,
    num_alternates: int = 3,
    max_position: int = 32768,
    span: int | None = None,
) -> None:
    """Generate alternates and stream results to JSONL file."""
    # Combine state and action tokens
    full_tokens = state.tokens + action_tokens

    # Find paragraph boundaries in action (offset by state length)
    action_start = len(state.tokens)

    beyond_positions: list[int] = []

    if span is not None:
        # Use single specified position
        assistant_positions = [action_start + span]
        print(f"Using specified span position: {span}")
    else:
        # Auto-detect paragraph boundaries
        positions = find_double_newline_positions(full_tokens)

        # Filter to positions within action tokens
        action_positions = [p for p in positions if p >= action_start]

        # Filter to positions in assistant turns
        assistant_positions = [
            p for p in action_positions if is_in_assistant_turn(full_tokens, p)
        ]

        # Split by max_position limit
        beyond_positions = [
            p for p in assistant_positions if (p - action_start) > max_position
        ]
        assistant_positions = [
            p for p in assistant_positions if (p - action_start) <= max_position
        ]

        print(
            f"Found {len(assistant_positions)=} and {len(beyond_positions)=}"
        )

    # Prepare tasks for parallel execution, skipping empty/invalid paragraphs
    tasks = []
    skipped = 0
    for i, pos in enumerate(assistant_positions):
        end_pos = find_next_paragraph_end(full_tokens, pos)
        original_paragraph = harmony_encoding.decode(full_tokens[pos:end_pos])
        rel_pos = pos - action_start

        # Skip empty paragraphs (consecutive \n\n) and paragraphs crossing tool call boundaries
        paragraph_text = original_paragraph.strip()
        if not paragraph_text:
            skipped += 1
            continue

        # Build prompt
        prompt_tokens = full_tokens[:pos]

        print(
            f"  [{i + 1 - skipped}/{len(assistant_positions) - skipped}] Position {rel_pos}: {len(paragraph_text)} chars"
        )

        for alt_idx in range(num_alternates):
            tasks.append((rel_pos, alt_idx, prompt_tokens))

    if skipped:
        print(f"  Skipped {skipped} empty or cross-turn positions")

    # Load existing alternates from file
    existing_alternates: dict[str, int] = {}  # position -> count of existing alternates
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    for pos_key, alts in data.items():
                        if pos_key in existing_alternates:
                            existing_alternates[pos_key] += len(alts)
                        else:
                            existing_alternates[pos_key] = len(alts)
        print(f"  Found {len(existing_alternates)} existing positions in file")

    # Write original tokens for positions beyond max_position (no API calls)
    if beyond_positions:
        written = 0
        with open(output_path, "a") as f:
            for pos in beyond_positions:
                rel_pos = pos - action_start
                pos_key = str(rel_pos)
                if pos_key in existing_alternates:
                    continue
                end_pos = find_next_paragraph_end(full_tokens, pos)
                original_tokens = full_tokens[pos:end_pos]
                paragraph_text = harmony_encoding.decode(original_tokens).strip()
                if not paragraph_text:
                    continue
                line = json.dumps({pos_key: []})
                f.write(line + "\n")
                written += 1
        if written:
            print(f"  Wrote {written} original-only positions beyond max_position={max_position}")

    # Filter out tasks based on existing alternates
    filtered_tasks = []
    for rel_pos, alt_idx, prompt_tokens in tasks:
        pos_key = str(rel_pos)
        existing_count = existing_alternates[pos_key] if pos_key in existing_alternates else 0
        # Only add task if we need more alternates at this position
        if alt_idx >= existing_count and alt_idx < num_alternates:
            filtered_tasks.append((rel_pos, alt_idx, prompt_tokens))
    tasks = filtered_tasks

    if not tasks:
        if span is not None:
            pos_key = str(span)
            existing = existing_alternates[pos_key] if pos_key in existing_alternates else 0
            print(f"  Position {span} already has {existing} alternates (need {num_alternates})")
        else:
            print("  All positions already have enough alternates")
        return

    # Track alternates per position
    positions_to_process = {str(t[0]) for t in tasks}
    alternates: dict[str, list] = {pos: [] for pos in positions_to_process}
    completed_positions: set[str] = set()

    # Generate alternates in parallel, streaming to file as positions complete
    print(f"  Generating {len(tasks)} alternates in parallel...")
    with open(output_path, "a") as f:
        with ThreadPoolExecutor(max_workers=min(len(tasks), 36)) as executor:
            futures = {
                executor.submit(generate_single_alternate, prompt_tokens): (
                    rel_pos,
                    alt_idx,
                )
                for rel_pos, alt_idx, prompt_tokens in tasks
            }
            for future in as_completed(futures):
                rel_pos, alt_idx = futures[future]
                alt_tokens = future.result()
                if alt_tokens is None:
                    continue
                alt_hash = tokens_to_base36_hash(alt_tokens)
                alternates[str(rel_pos)].append({alt_hash: alt_tokens})
                alt_text = harmony_encoding.decode(alt_tokens)
                print(
                    f"    Position {rel_pos} alt {alt_idx + 1}: {len(alt_tokens)} tokens, {len(alt_text)} chars"
                )

                # Check if all alternates for this position are done
                if (
                    len(alternates[str(rel_pos)]) == num_alternates
                    and str(rel_pos) not in completed_positions
                ):
                    completed_positions.add(str(rel_pos))
                    line = json.dumps({str(rel_pos): alternates[str(rel_pos)]})
                    f.write(line + "\n")
                    f.flush()
                    print(f"    -> Wrote position {rel_pos} to file")


def process_single_completion(
    action_hash: str,
    problem_id: str | None = None,
    state_hash: str | None = None,
    num_alternates: int = 1,
    max_position: int = 32768,
    span: int | None = None,
):
    """Process a single completion and generate alternates."""
    # Search for problem_id and state_hash if not provided
    if problem_id is None or state_hash is None:
        result = find_action_location(action_hash)
        if result is None:
            print(f"Action {action_hash} not found in any problem")
            return
        problem_id, state_hash = result
        print(f"Found action in {problem_id}/{state_hash}")

    state_dir = RAW_DIR / problem_id / state_hash
    state = State.load_from_dir(str(state_dir))

    # Find the action
    action = None
    for a in state.actions:
        if a.hash == action_hash:
            action = a
            break

    if action is None:
        print(f"Action {action_hash} not found")
        return

    print(f"Processing {problem_id}/{state_hash}/{action_hash}")
    print(f"  State tokens: {len(state.tokens)}")
    print(f"  Action tokens: {len(action.tokens)}")

    # Setup output file for streaming
    output_dir = ALTERNATE_DIR / problem_id / state_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{action_hash}.jsonl"

    roll_alternates_streaming(
        state,
        action.tokens,
        output_path=output_path,
        num_alternates=num_alternates,
        max_position=max_position,
        span=span,
    )

    print(f"Saved to {output_path}")

    # Generate annotation file
    prepare_annotation(action_hash, problem_id, state_hash)

    # Extract labels from annotations (if any exist)
    result = extract_labels(action_hash, problem_id, state_hash)
    if result:
        _, _, num_labels = result
        if num_labels > 0:
            regenerate_labels_jsonl(problem_id, state_hash, action_hash)


def select_actions(
    actions: list[Action],
    max_correct: int,
    max_incorrect: int,
) -> list[Action]:
    """Select actions based on correctness limits.

    For correct actions (reward == 1): if more than max_correct, keep longest.
    For incorrect actions (reward == 0): if more than max_incorrect, keep shortest.
    Pass -1 to ignore the constraint for that category.
    """
    correct = [a for a in actions if a.reward == 1]
    incorrect = [a for a in actions if a.reward != 1]

    # For correct: keep shortest (sort ascending by token count)
    if max_correct != -1 and len(correct) > max_correct:
        correct = sorted(correct, key=lambda a: len(a.tokens))[:max_correct]

    # For incorrect: keep shortest (sort ascending by token count)
    if max_incorrect != -1 and len(incorrect) > max_incorrect:
        incorrect = sorted(incorrect, key=lambda a: len(a.tokens))[:max_incorrect]

    return correct + incorrect


def process_problem(
    problem_id: str,
    num_alternates: int = 3,
    max_correct_action: int = 2,
    max_incorrect_action: int = 2,
    max_position: int = 32768,
):
    """Process all states and actions in a problem directory."""
    problem_dir = RAW_DIR / problem_id
    for state_dir in problem_dir.iterdir():
        if not state_dir.is_dir():
            continue
        state_hash = state_dir.name
        state = State.load_from_dir(str(state_dir))

        # Select actions based on correctness limits
        selected_actions = select_actions(
            state.actions,
            max_correct=max_correct_action,
            max_incorrect=max_incorrect_action,
        )

        rewards = [int(a.reward) for a in selected_actions]
        print(
            f"\nState {state_hash}: selected {len(selected_actions)}/{len(state.actions)} actions, rewards={rewards}"
        )

        for action in selected_actions:
            print(f"\n=== {problem_id}/{state_hash}/{action.hash} ===")
            process_single_completion(
                action_hash=action.hash,
                problem_id=problem_id,
                state_hash=state_hash,
                num_alternates=num_alternates,
                max_position=max_position,
            )


def process_state(
    problem_id: str, state_hash: str, num_alternates: int = 3, max_position: int = 32768
):
    """Process all actions in a state directory."""
    state_dir = RAW_DIR / problem_id / state_hash
    state = State.load_from_dir(str(state_dir))
    for action in state.actions:
        print(f"\n=== {problem_id}/{state_hash}/{action.hash} ===")
        process_single_completion(
            action_hash=action.hash,
            problem_id=problem_id,
            state_hash=state_hash,
            num_alternates=num_alternates,
            max_position=max_position,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate alternate completions")
    parser.add_argument(
        "--num-alternates", type=int, default=2, help="Alternates per position"
    )
    parser.add_argument("-s", "--state", help="State hash(es), comma-separated")
    parser.add_argument("-a", "--action", help="Action hash(es), comma-separated")
    parser.add_argument(
        "--max-correct-action",
        type=int,
        default=2,
        help="Max correct actions per state (-1 for unlimited, only for problem_id)",
    )
    parser.add_argument(
        "--max-incorrect-action",
        type=int,
        default=2,
        help="Max incorrect actions per state (-1 for unlimited, only for problem_id)",
    )
    parser.add_argument(
        "--max-position",
        type=int,
        default=32768 // 2,
        help="Max position in action tokens to generate alternates for",
    )
    parser.add_argument(
        "--span",
        type=int,
        help="Specific position to generate alternates at (overrides auto-detection)",
    )
    parser.add_argument("-p", "--problem", help="Problem ID(s), comma-separated")
    parser.add_argument(
        "-pp", "--problem-prefix", help="Process all problems matching prefix"
    )
    parser.add_argument(
        "problem_id", nargs="?", help="Problem ID to process (positional)"
    )
    args = parser.parse_args()

    # Use this to find problems
    """
    uv run python -c "import json,sys; p=sys.argv[1] if len(sys.argv)>1 else ''; print(','.join(d['problem_id'] for line in open('data/problems.jsonl') if (d := json.loads(line)) and d['problem_id'].startswith(p) and 0 < sum(d['completion_rewards'])/len(d['completion_rewards']) < 1))" "pe-7"
    uv run python -c "import json,sys; p=sys.argv[1] if len(sys.argv)>1 else ''; print(','.join([d['problem_id'] for line in open('data/problems.jsonl') if (d := json.loads(line)) and d['problem_id'].startswith(p) and sum(d['completion_rewards']) >= 2][1::3]))" "pe-"
    """

    # Allow -p or positional argument for problem_id
    problem_id = args.problem or args.problem_id

    # Parse comma-separated values
    actions = [a.strip() for a in args.action.split(",")] if args.action else []
    states = [s.strip() for s in args.state.split(",")] if args.state else []
    problems = [p.strip() for p in problem_id.split(",")] if problem_id else []

    if args.problem_prefix:
        # Find all problems matching prefix
        matching = [
            d.name
            for d in RAW_DIR.iterdir()
            if d.is_dir() and d.name.startswith(args.problem_prefix)
        ]
        matching.sort()
        print(f"Found {len(matching)} problems matching prefix '{args.problem_prefix}'")
        for pid in matching:
            print(f"\n{'=' * 60}\nProcessing {pid}\n{'=' * 60}")
            process_problem(
                pid,
                args.num_alternates,
                max_correct_action=args.max_correct_action,
                max_incorrect_action=args.max_incorrect_action,
                max_position=args.max_position,
            )
    elif actions:
        # Process specific action(s)
        for action_hash in actions:
            result = find_action_location(action_hash)
            if result:
                process_single_completion(
                    action_hash=action_hash,
                    problem_id=result[0],
                    state_hash=result[1],
                    num_alternates=args.num_alternates,
                    max_position=args.max_position,
                    span=args.span,
                )
            else:
                print(f"Action {action_hash} not found")
    elif states:
        # Process all actions in state(s)
        for state_hash in states:
            pid = problems[0] if problems else find_state_location(state_hash)
            if pid:
                process_state(pid, state_hash, args.num_alternates, args.max_position)
            else:
                print(f"State {state_hash} not found")
    elif problems:
        # Process all states/actions in problem(s)
        for pid in problems:
            process_problem(
                pid,
                args.num_alternates,
                max_correct_action=args.max_correct_action,
                max_incorrect_action=args.max_incorrect_action,
                max_position=args.max_position,
            )
    else:
        print(
            "Provide problem_id, -p problem, --problem-prefix prefix, -s state, or -a action"
        )


if __name__ == "__main__":
    main()
