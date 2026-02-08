#!/usr/bin/env python3
"""Train on specific examples and compare logprobs before/after.

Space-separated = different microbatches (separate backprop)
Comma-separated = same batch

uv run python -m train_sample 'aimo3-21818/*/*/7022-7081/*-lyskfk' --validation_batch 'aimo3-21818/*/*/7022-7081/lyskfk' --learning-rate 1e-4 --logpath 1e-4
"""

import argparse
import asyncio
import json
import logging
import shutil
from pathlib import Path

import tinker
from tinker import types

from train_common import TrainingExample, build_datum, load_corpus_entries

logger = logging.getLogger(__name__)

TRAINING_DIR = Path("training")


async def get_logprobs_async(
    sampling_client: tinker.SamplingClient,
    token_ids: list[int],
) -> list[float]:
    """Get logprobs for token sequence using Tinker."""
    model_input = types.ModelInput.from_ints(token_ids)
    logprobs = await sampling_client.compute_logprobs_async(model_input)
    return [lp for lp in logprobs if lp is not None]


def parse_example_spec(spec: str, examples: list[TrainingExample]) -> list[TrainingExample]:
    """Parse problem_id/state/action/span/trace_id and find matching examples."""
    parts = spec.strip().split("/")
    if len(parts) != 5:
        raise ValueError(f"Expected problem_id/state/action/span/trace_id, got: {spec}")
    problem_id, state_hash, action_hash, span, trace_id = parts

    result = []
    for example in examples:
        if (
            example.problem_id == problem_id
            and (example.state_hash == state_hash or state_hash == "*")
            and (example.action_hash == action_hash or action_hash == "*")
            and (example.span == span or span == "*")
        ):
            should_include = False
            if example.trace_id == trace_id:
                should_include = True
            if trace_id == "*":
                should_include = True
            if trace_id.startswith("*-") and example.trace_id != trace_id[2:]:
                should_include = True
            if should_include:
                result.append(example)

    if len(result) == 0:
        raise ValueError(f"No matching entry found for: {spec}")
    return result


async def main():
    parser = argparse.ArgumentParser(
        description="Train on specific examples and compare logprobs"
    )
    parser.add_argument(
        "batches",
        nargs="+",
        help="Batches of examples (space=different batch, comma=same batch)",
    )
    parser.add_argument(
        "--model-name",
        default="openai/gpt-oss-120b",
        help="Model name for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank",
    )
    parser.add_argument(
        "--validation_batch",
        help="Validation examples (comma-separated, no backprop)",
    )
    parser.add_argument(
        "--logpath",
        default="default",
        help="Subdirectory name for saving logs (default: 'default')",
    )
    args = parser.parse_args()

    # Clear training/sample/{logpath} directory at start
    sample_dir = TRAINING_DIR / "sample" / args.logpath
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Index file at training/sample/{logpath}/index.jsonl
    summary_file = sample_dir / "index.jsonl"

    corpus = [TrainingExample.from_dict(e) for e in load_corpus_entries()]

    # Parse batches structure
    batches: list[list[TrainingExample]] = []
    all_examples: list[TrainingExample] = []
    example_batch_indices: list[int] = []  # batch index for each example

    for batch_idx, batch_str in enumerate(args.batches):
        specs = batch_str.split(",")
        batch_examples = []
        for spec in specs:
            batch_examples.extend(parse_example_spec(spec, corpus))
        batches.append(batch_examples)
        all_examples.extend(batch_examples)
        example_batch_indices.extend([batch_idx] * len(batch_examples))

    logger.info(f"Loaded {len(all_examples)} examples across {len(batches)} batch(es)")

    # Parse validation examples
    validation_examples: list[TrainingExample] = []
    if args.validation_batch:
        specs = args.validation_batch.split(",")
        for spec in specs:
            validation_examples.extend(parse_example_spec(spec, corpus))
        logger.info(f"Loaded {len(validation_examples)} validation examples")

    all_tokens = [ex.load_tokens() for ex in all_examples]
    validation_tokens = [ex.load_tokens() for ex in validation_examples]

    service_client = tinker.ServiceClient()

    # Get base model logprobs BEFORE training
    base_sampling_client = await service_client.create_sampling_client_async(
        base_model=args.model_name
    )
    base_logprobs = [
        await get_logprobs_async(base_sampling_client, p + c) for p, c in all_tokens
    ]
    validation_base_logprobs = [
        await get_logprobs_async(base_sampling_client, p + c)
        for p, c in validation_tokens
    ]

    # Train each batch separately
    training_client = await service_client.create_lora_training_client_async(
        base_model=args.model_name,
        rank=args.lora_rank,
    )

    example_idx = 0
    for batch_idx, batch_examples in enumerate(batches):
        batch_tokens = all_tokens[example_idx : example_idx + len(batch_examples)]
        example_idx += len(batch_examples)

        data = [build_datum(p, c) for p, c in batch_tokens]
        data = [d for d in data if d is not None]

        fwdbwd_future = await training_client.forward_backward_async(
            data, loss_fn="cross_entropy"
        )
        optim_future = await training_client.optim_step_async(
            types.AdamParams(learning_rate=args.learning_rate)
        )
        await fwdbwd_future.result_async()
        await optim_future.result_async()
        logger.info(
            f"Trained batch {batch_idx + 1}/{len(batches)} with {len(data)} examples"
        )

    # Get trained model logprobs AFTER all batches
    trained_sampling_client = (
        await training_client.save_weights_and_get_sampling_client_async(
            name="trained-checkpoint"
        )
    )
    trained_logprobs = [
        await get_logprobs_async(trained_sampling_client, p + c) for p, c in all_tokens
    ]
    validation_trained_logprobs = [
        await get_logprobs_async(trained_sampling_client, p + c)
        for p, c in validation_tokens
    ]

    # Save logprobs and compare
    with open(summary_file, "w") as summary_f:
        for i, example in enumerate(all_examples):
            prompt_len = len(all_tokens[i][0])
            base_sum = sum(base_logprobs[i][prompt_len - 1 :])
            trained_sum = sum(trained_logprobs[i][prompt_len - 1 :])

            # Save per-example logprobs
            example_dir = (
                sample_dir
                / example.problem_id
                / example.state_hash
                / example.action_hash
                / example.span
                / example.trace_id
            )
            example_dir.mkdir(parents=True, exist_ok=True)

            with open(example_dir / "initial.jsonl", "w") as f:
                f.write(json.dumps({"logprobs": base_logprobs[i]}) + "\n")

            with open(example_dir / "tuned.jsonl", "w") as f:
                f.write(json.dumps({"logprobs": trained_logprobs[i]}) + "\n")

            # Write summary
            completion_len = len(all_tokens[i][1])
            summary = {
                "logpath": args.logpath,
                "problem_id": example.problem_id,
                "state_hash": example.state_hash,
                "action_hash": example.action_hash,
                "span": example.span,
                "trace_id": example.trace_id,
                "base_sum": round(base_sum, 2),
                "trained_sum": round(trained_sum, 2),
                "completion_len": completion_len,
                "split": "train",
                "batch": example_batch_indices[i],
            }
            summary_f.write(json.dumps(summary) + "\n")

            logger.info(
                f"{example.trace_id}:\tbase={base_sum:.2f}\ttrained={trained_sum:.2f}\t"
                f"diff={trained_sum - base_sum:.2f}\tratio={1 - trained_sum / (-0.01 + base_sum):.3f}"
            )

        # Write validation examples
        for i, example in enumerate(validation_examples):
            prompt_len = len(validation_tokens[i][0])
            base_sum = sum(validation_base_logprobs[i][prompt_len - 1 :])
            trained_sum = sum(validation_trained_logprobs[i][prompt_len - 1 :])

            # Save per-example logprobs
            example_dir = (
                sample_dir
                / example.problem_id
                / example.state_hash
                / example.action_hash
                / example.span
                / example.trace_id
            )
            example_dir.mkdir(parents=True, exist_ok=True)

            with open(example_dir / "initial.jsonl", "w") as f:
                f.write(json.dumps({"logprobs": validation_base_logprobs[i]}) + "\n")

            with open(example_dir / "tuned.jsonl", "w") as f:
                f.write(json.dumps({"logprobs": validation_trained_logprobs[i]}) + "\n")

            # Write summary
            completion_len = len(validation_tokens[i][1])
            summary = {
                "logpath": args.logpath,
                "problem_id": example.problem_id,
                "state_hash": example.state_hash,
                "action_hash": example.action_hash,
                "span": example.span,
                "trace_id": example.trace_id,
                "base_sum": round(base_sum, 2),
                "trained_sum": round(trained_sum, 2),
                "completion_len": completion_len,
                "split": "validation",
            }
            summary_f.write(json.dumps(summary) + "\n")

            logger.info(
                f"[val] {example.trace_id}:\tbase={base_sum:.2f}\ttrained={trained_sum:.2f}\t"
                f"diff={trained_sum - base_sum:.2f}\tratio={1 - trained_sum / (-0.01 + base_sum):.3f}"
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main())
