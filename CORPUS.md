This is my entry for the [Math Corpus Prize](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/overview/prizes) in AIMO3.

- Dataset: [link](https://www.kaggle.com/datasets/huikang/aimo3-corpus-prize-submission)
- Models: [link](https://www.kaggle.com/models/huikang/gpt-oss-120b-aimo3/Transformers/160a/9)
- Discussion post: [link](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/672528)
- Comment in submission thread: [link](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/663330#3403621)
- Github: [link](https://github.com/tonghuikang/aimo3)

---

## Summary

In my submission, I demonstrate how I annotate solution traces and train the model.

My submission contains:

Performance on Project Euler problems
- This informs which problems have space for improvement, and to understand the failure modes.
- This is `problems.jsonl`, `problems.csv` in the Kaggle dataset.
- You can visualize the problems at [aimo.huikang.dev/problems](https://aimo.huikang.dev/problems)

Annotations on thinking traces
- Each solution trace is an attempt to solve the problem by the original gpt-oss-120b
- I annotate solution traces to different extents
- This is `reflections.csv` in the Kaggle dataset.
- You can visualize the annotations at: [aimo.huikang.dev/annotations](https://aimo.huikang.dev/annotations)

Training corpus
- Each entry is a prompt-completion pair.
- Metadata for each entry includes problem source, character counts, and annotation comments.
- This is `corpus.csv` and `corpus.jsonl` in the Kaggle dataset.
- You can visualize the corpus at [aimo.huikang.dev/corpus](https://aimo.huikang.dev/corpus)

Trained models
- I have published the model ([huikang/gpt-oss-120b-aimo3/Transformers/160a/9](https://www.kaggle.com/models/huikang/gpt-oss-120b-aimo3/Transformers/160a/9)) on Kaggle. My notebook [streaming-inference-tool-calling](https://www.kaggle.com/code/huikang/streaming-inference-tool-calling?scriptVersionId=296461526) shows an example how this model is used.


---


## Motivation

Think of how you learn from experience.

You tried to solve a math problem in a competition.

Each attempt involves brainstorming, calculation, and verification.

After solving the math problem, you find out the correct answer. You may read the reference solution there.

Now you need to learn.

If you get a problem correct, does this really mean that everything you did is correct?
If you get a problem wrong, is it really true that all the actions you have made are wrong?

What you do is to reflect exactly which actions are good and bad.

Good actions and bad actions are also relative.
You want to focus on learning actions that can make a difference, instead of imitating what you would have done anyway.

You also want to learn only actions that you could make.
There is not much value in learning from actions that someone else could make, or from actions that could only be made with privileged information.

In my corpus prize submission, I demonstrate how I write reflections, make annotations, and train the model.


---

## Corpus Creation Procedure

### Step 1: Problem Selection and Triage

I curate the problem set from three sources:
- AIMO3 reference problems
- Project Euler problems
- IMO problems

I selected these three sources because they have verified correct answers.

I make minor modifications to the problem statements.
- For AIMO3 and IMO problems, I change numerical variables to create parametric variants.
- For Project Euler problems, I request `mod 99991` as long as it does not confuse the answer requirement.

The problem set is in `problems.csv`.

[View problems](https://aimo.huikang.dev/problems)

### Step 2: Generate Solution Traces

For each problem, I generate multiple solution attempts using gpt-oss-120b with the tool calling notebook `notebook.py`.

The script is `data/generate.py` which calls `notebook.py`.

[View example trace](https://aimo.huikang.dev/problems.html?problem=aimo3-q10-g44636594&state=nt96yb&action=rh2htj&scrollTo=better&pos=874)

### Step 3: Generate Alternative Continuations

At each paragraph boundary in a solution trace, I generate one to five alternative continuations.

The script is `data/roll_alternates.py`.

[View example alternates](https://aimo.huikang.dev/annotations.html?problem=aimo3-q10-g44636594&state=nt96yb&action=rh2htj&scrollTo=better&pos=874)

### Step 4: Write Reflections

Based on the solution traces, I write reflections.

I annotate critical points of the solution trace:
- `[breakthrough]` — The trace arrives at a key insight. This is an idea that meaningfully advances the solution.
- `[insight]` - This is the precursor leading to the `[breakthrough]`.
- `[mistake]` — The trace makes a concrete mathematical or logical error.
- `[questionable statement]` - This is the precursor leading to the `[mistake]`.
- `[missed opportunity]` — The trace had the information or context to take a better path but didn't.
- `[wasted]` — A span of tokens (use `Token NNNN-MMMM`) where the trace spends effort on something unproductive.
- `[verification]` - Productive rigorous check on a certain hypothesis.
- `[final answer]` — Where the trace commits to its final answer value.

I research and write the reflections with AI coding agents.

I record the reflection guidelines in [data/REFLECTION_GUIDELINES.md](https://github.com/tonghuikang/aimo3/blob/master/data/REFLECTION_GUIDELINES.md).

[View example reflection](https://aimo.huikang.dev/annotations.html?problem=aimo3-q10-g44636594&state=nt96yb&action=rh2htj)

### Step 5: Annotate Comparisons

Based on the reflections, I annotate the thinking traces.
I also research and make the annotations with AI coding agents.

At each paragraph boundary, I compare the original continuation against the generated alternatives and classify each completion as:
- `better` — More mathematically accurate, makes better progress toward the solution, provides clearer reasoning, or avoids errors present in other completions. An alternate does not need to be correct to be "better" — it just needs to show significantly better reasoning than the others.
- `neutral` — No meaningful difference between completions, or all completions make the same error. If all completions pursue the same wrong approach, they are all neutral.
- `worse` — Contains mathematical errors that others avoid, makes less progress, or misses key insights that other completions capture. A completion is only "worse" relative to a "better" alternative — if no better alternative exists, all completions are neutral.

I focus annotations on lines identified in the reflections, where a difference could have occurred.
The most valuable labels go to alternates that could have provided a better result.

I record the annotation guidelines in [data/ANNOTATION_GUIDELINES.md](https://github.com/tonghuikang/aimo3/blob/master/data/ANNOTATION_GUIDELINES.md).

[View example annotation](https://aimo.huikang.dev/annotations.html?problem=aimo3-q10-g44636594&state=nt96yb&action=rh2htj&scrollTo=better&pos=874)

### Step 6: Build Training Corpus

I select the traces for training.

Even though there are 6,096 entries, I include only [665](https://aimo.huikang.dev/corpus.html?included=true) in training.
I chose a small selection because of my limited budget on Tinker, and also to more reliably show that I can improve on the AIMO3 reference problems.
Toggle the included column to see which entries are included.

[View corpus](https://aimo.huikang.dev/corpus)

---

## Model Training Procedure

### Step 7: Training with LoRA

I use [Tinker](https://thinkingmachines.ai/) from Thinking Machines Lab to train the model via LoRA (Low-Rank Adaptation).

The batch size chosen is 16, the LoRA rank chosen is 32. I trained for one epoch over the 665 training entries.

I [chose](https://x.com/giffmana/status/2020595919960567893) a starting learning rate of 2e-4 among [1e-4](http://aimo.huikang.dev/training_sample.html?logpath=1e-4), [2e-4](http://aimo.huikang.dev/training_sample.html?logpath=2e-4), [3e-4](http://aimo.huikang.dev/training_sample.html?logpath=3e-4), [4e-4](http://aimo.huikang.dev/training_sample.html?logpath=4e-4) and [5e-4](http://aimo.huikang.dev/training_sample.html?logpath=5e-4). 
I chose the learning rate to optimize the average logprob improvement on the completion text, after training on the same completion text.

### Step 8: Merge Adapter with LoRA

After training, I merge the LoRA adapter weights back into the base model with `merge_adapter.py`.

The merge handles mixed precision:
- BF16 for attention layers (q/k/v/o projections)
- MXFP4 for MoE expert layers (128 experts per layer)

I calculated the merge statistics.
Among the MXFP4 MoE weights, approximately 5% of the weights changed.
Notably, the share of MXFP4 weights with absolute value 0.5 has increased from 16% to 23%.

### Step 9: Upload to Kaggle

I upload the merged model to Kaggle via `upload_model.py` using the Kaggle API.

### Step 10: Run Kaggle notebook and submit

I run the [notebook](https://www.kaggle.com/code/huikang/streaming-inference-tool-calling?scriptVersionId=296461526) by just replacing the reference to the [original model](https://www.kaggle.com/models/danielhanchen/gpt-oss-120b) with [my model](https://www.kaggle.com/models/huikang/gpt-oss-120b-aimo3/Transformers/160a/9).


---

## Math Corpus Prize Requirements

I cite the requirements and I write my responses.

> The dataset must be publicly released either on Kaggle or on HuggingFace prior to February 9, 2026 11:59 PM UTC. Teams must create a Kaggle Discussion post until this date that explicitly links to their dataset and tags it as an entry for the Math Corpus Prize. All eligible datasets will be judged according to the criteria below, and the highest scoring dataset will be awarded the prize.

- Kaggle dataset: [link](https://www.kaggle.com/datasets/huikang/aimo3-corpus-prize-submission)
- Kaggle discussion post: [link](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/672528)
- Comment in submission thread: [link](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/663330#3403621)

> Your Kaggle Discussion post will be used to judge your dataset submission based on the criteria below, and your claims will be independently verified by the Host Team.

This document serves as the detailed write-up for the discussion post.

> The dataset must be in English.

All content is in English.

> The earliest timestamped version of the dataset will be used for evaluation (later modifications will not be considered)

I will release any subsequent improvements as separate datasets.

> The dataset is not allowed to exceed 5M datapoints, which is an upper bound on common datasets used by prior AIMO winners to fine-tune their models. Each datapoint is not allowed to exceed more than 100k characters.

The corpus contains:
- 6,096 prompt-completion pairs (only 665 of them are included in training)
- All entries are under 100K characters (prompt + completion)
- Metadata includes token counts and trace identifiers

> Each datapoint from the submitted dataset must come with an open source license that allows free dissemination of data.

Licensing

- My original contributions — including the annotation methodology, reflection guidelines, training pipeline, code, merge scripts, and all associated tooling — are released under Apache 2.0 and may be freely redistributed, modified, and used for any purpose, including commercial use.
- Content derived from Project Euler: Subject to Project Euler's CC BY-NC-SA 4.0 license. These entries are identified by the problem_source field.
- Content derived from AIMO3 reference problems: Subject to AIMO3's CC BY-SA 4.0 as noted in the [competition data](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/data)
- Content derived from IMO: Users should verify the licensing terms of the original problem sources before redistribution.

I encourage free use of this dataset for research and model training. The problem_source metadata field allows users to filter by source and apply the appropriate license terms.

---

## Math Corpus Prize Evaluation Criteria

### Data Novelty (25 points)

> Is the dataset distinctly unique from other datasets found on the internet?

What is different from other datasets:
- Performance evaluation of gpt-oss-120b
- Modification of some existing problems
- Reflection on solution traces and reflection on solution lines
- Very easily navigable data format ([aimo.huikang.dev/annotations](https://aimo.huikang.dev/annotations))
- I have proven that the included traces improve performance on the original version of the problem

I decided not to invent problems because I need to verify them. I am satisfied with the current problem set until I can effectively train a model to solve them perfectly.


### Format (25 points)

> Does the dataset comes in a format that makes it easy to handle and train a model?
> Does each datapoint contain rich metadata?

I provide the dataset in two formats:

1. `corpus.csv` with `problem_id`, `problem_source`, `prompt`, `completion`, `character_count`, `token_count`, `annotation_reason`
2. `corpus_full.jsonl` - I provide `prompt` and `completion` in token IDs, to avoid retokenization drift.


I also provide HTML visualizations for browsing the corpus data at [aimo.huikang.dev/corpus](https://aimo.huikang.dev/corpus).

### Performance (50 points)

> Does the dataset improve mathematical reasoning and aid in improving model performance during the AIMO3. How so and how much?

I actually finetuned the frontier OSS model and made submissions.
You may use my scripts to repeat my end-to-end process.

I trained the model on solution traces solving the variants of the [Q9](https://aimo.huikang.dev/annotations.html?problem=aimo3-q9*) and [Q10](https://aimo.huikang.dev/annotations.html?problem=aimo3-q10*) AIMO3 problems.
For Q9, the shifty function problem, instead of a support size of 8, I trained on problems with a support size other than 8.
For Q10, the Norwegian number problem, I trained on problems requesting a small subset of the terms.

Training on variants improves solve rates on the original problems.
The answer to the original problems (160, 8687) does not appear in the training corpus.

I also show that my training does not degrade the performance of the model on the public leaderboard.
Unlike the highest scoring public notebooks, the notebook I am submitting for public leaderboard performance only considers the earliest boxed answer.

This setup allows me to:
- Evaluate the model without tuning the time constraint parameters
- Observe the differences in time usage
- Saving Kaggle GPU quota because it takes faster to evaluate
- More easily observe improvements that generalize to the public leaderboard

| Model              | default gpt-oss-120b                                                                                                  | finetuned gpt-oss-120b-aimo3/160a/v9                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Q9 performance     | [22/50 in 2h 24m](https://www.kaggle.com/code/huikang/streaming-inference-tool-calling/log?scriptVersionId=296131552) | [35/50 in 3h 51m](https://www.kaggle.com/code/huikang/streaming-inference-tool-calling/log?scriptVersionId=296490634)  |
|                    |                                                                                                                       | [31/50 in 3h 51m](https://www.kaggle.com/code/huikang/streaming-inference-tool-calling/log?scriptVersionId=296605502)  |
| Q10 performance    | [7/50 in 6h 14m](https://www.kaggle.com/code/huikang/streaming-inference-tool-calling/log?scriptVersionId=296192788)  | [19/50 in 7h 42m](https://www.kaggle.com/code/huikang/streaming-inference-tool-calling/log?scriptVersionId=296488461)  |
|                    |                                                                                                                       | [20/50 in 7h 57m](https://www.kaggle.com/code/huikang/streaming-inference-tool-calling/log?scriptVersionId=296605521)  |
| Public leaderboard | [29/50 in 3h 37m](https://www.kaggle.com/code/huikang/streaming-inference-tool-calling?scriptVersionId=296123901)     | [32/50 in 4h 4m](https://www.kaggle.com/code/huikang/streaming-inference-tool-calling?scriptVersionId=296461526)       |
|                    | [27/50 3h 27m](https://www.kaggle.com/code/huikang/streaming-inference-tool-calling?scriptVersionId=292989026)        | [33/50 in 3h 51m](https://www.kaggle.com/code/huikang/streaming-inference-tool-calling?scriptVersionId=296605502)      |


My submission fits into several steps of a larger progression:
- Training on a sequence increases the probability of the sequence.
- Training on a sequence increases the probability of similar sequences. (I briefly looked at [this](https://aimo.huikang.dev/training_sample))
- Training on the problem increases the probability of success on the problem.
- Training on very similar problems increases the probability of success on the problem (I did this).
- Training on problems using similar techniques increases the probability of success on the problem.
- Training on half of the Project Euler problem set improves performance on the other half of the Project Euler problem set.
- Getting a significantly better score in AIMO3 in a first-submission format (I have shown that my solution does not degrade public leaderboard performance.)
- Getting a significantly better score in AIMO3 on optimized notebooks.

I hope my corpus prize submission can immediately serve you by:
- Displaying a list of Project Euler problems where gpt-oss-120b could have done better.
- Showing how you can modify problem statements and use them to augment the problems.
- Proving that you can improve on a problem with very similar problems, without degradation in overall performance.
- Providing reflections and annotations (and the process to generate the reflections and annotations) to help you teach your model.

---

Thank you for reading.

I would like to have much more GPU resources to test out my ideas, do reach out if you can help!

I would like to thank Modal, Tinker, Fireworks (and Han Chung Lee) for providing compute thus far.
