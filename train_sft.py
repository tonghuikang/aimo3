#!/usr/bin/env python3
"""
SFT Training on gpt-oss-120b with Tinker.

rm -r training/sft/
uv run python3 -m train_sft
"""

import argparse
import asyncio
import logging
import random

import chz
import tinker
from tinker_cookbook.supervised.train import Config, main as train_main
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder

from train_common import TrainingExample, build_datum, load_corpus_entries

logger = logging.getLogger(__name__)


def filter_training_examples(examples: list[TrainingExample]) -> list[TrainingExample]:
    # handwritten filter logic
    filtered_examples: list[TrainingExample] = []
    for example in examples:
        # if not example.problem_id.startswith("aimo3-q9"):
        #     print("filtering for question")
        #     continue
        filtered_examples.append(example)
    return filtered_examples


class TokenDataset(SupervisedDataset):
    """Dataset that loads pre-tokenized training data."""

    def __init__(
        self,
        examples: list[TrainingExample],
        batch_size: int,
        max_length: int = 32768,
    ):
        self.examples = examples
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffled_indices = list(range(len(examples)))

    def _build_datum(self, example: TrainingExample) -> tinker.Datum | None:
        """Build a Datum from a training example using raw token IDs."""
        prompt_tokens, completion_tokens = example.load_tokens()
        datum = build_datum(prompt_tokens, completion_tokens, self.max_length)
        if datum is None:
            logger.warning(
                f"Skipping example {example.trace_id}: prompt exceeds max_length"
            )
        return datum

    def get_batch(self, index: int) -> list[tinker.Datum]:
        """Get a batch of Datum objects."""
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.shuffled_indices))

        data = []
        for i in range(start_idx, end_idx):
            example_idx = self.shuffled_indices[i]
            example = self.examples[example_idx]
            datum = self._build_datum(example)
            if datum is not None:
                data.append(datum)

        return data

    def set_epoch(self, seed: int = 0):
        """Shuffle examples for new epoch."""
        rng = random.Random(seed)
        self.shuffled_indices = list(range(len(self.examples)))
        rng.shuffle(self.shuffled_indices)

    def __len__(self) -> int:
        return len(self.examples) // self.batch_size


@chz.chz
class TokenDatasetBuilder(SupervisedDatasetBuilder):
    """Builder for the token-based dataset."""

    batch_size: int = 16
    max_length: int = 32768

    def __call__(self) -> tuple[SupervisedDataset, None]:
        """Build training dataset."""
        entries = load_corpus_entries()
        entries = [e for e in entries if e["included"]]
        examples = [TrainingExample.from_dict(e) for e in entries]
        logger.info(f"Loaded {len(examples)} examples")
        examples = filter_training_examples(examples)
        logger.info(f"Processing {len(examples)} examples after filtering")

        train_dataset = TokenDataset(
            examples=examples,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

        logger.info(
            f"Created training dataset with {len(examples)} examples "
            f"({len(train_dataset)} batches)"
        )

        return train_dataset, None


async def main():
    parser = argparse.ArgumentParser(description="SFT training on gpt-oss-120b")
    parser.add_argument(
        "--log-path",
        default="./training/sft/default/",
        help="Path to save logs and checkpoints",
    )
    parser.add_argument(
        "--model-name",
        default="openai/gpt-oss-120b",
        help="Model name for training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=32768,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save checkpoint every N steps (0 to disable)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run evaluation every N steps (0 to disable)",
    )
    args = parser.parse_args()

    config = Config(
        log_path=args.log_path,
        model_name=args.model_name,
        dataset_builder=TokenDatasetBuilder(
            batch_size=args.batch_size,
            max_length=args.max_length,
        ),
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,
        save_every=args.save_every,
        eval_every=args.eval_every,
        infrequent_eval_every=0,
    )

    logger.info(f"Starting SFT: model={config.model_name}, lr={config.learning_rate}")
    await train_main(config)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main())
