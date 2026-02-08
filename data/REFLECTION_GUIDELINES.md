# Reflection Guidelines

Your task is to write a reflection on the traces.

You must read the entire trace at once (state.txt, and <action_id>-original.txt)

## File Structure

```
<current-working-directory>/data/
├── problems.csv                      # Problem statements and answers
├── alternate/<problem_id>/           # Alternate statements (tokenized JSONL - do not use/edit directly)
├── annotate/<problem_id>/            # Annotation files with decoded original + alternates for each line
│   ├── correct_answer.txt            # Correct answer for problem_id
│   ├── findings.txt                  # Findings about the problem solution that you will discover and document
│   └── <state_id>/
│       ├── state.txt                      # Full decoded state (problem statement)
│       ├── <action_id>-original.txt       # Full decoded solution trace
│       ├── <action_id>-reflections.txt    # Your reflections on the trace (you write this)
│       └── <action_id>.txt                # Annotation file for understanding token positions
└── label/<problem_id>/               # Parsed annotations (auto-regenerated on edit, do not use/edit directly)
```

## Workflow

### Step 1: Understand the problem
- Read the problem.
- Note the correct answer provided. The correct answer is available in correct_answer.txt in `annotate/<problem_id>/`

### Step 2: Identify which traces are correct vs wrong
Before reading traces in detail, quickly identify which traces get correct vs wrong answers:
- Group traces by their final answer (correct answer, wrong answer A, wrong answer B, etc.)
- Different wrong answers often indicate different root cause errors

### Step 3: Understand the solutions
- Read a trace that arrives at the **correct answer** - understand the key approach
- Read a trace that arrives at a **wrong answer** - identify WHERE it diverges from the correct approach

### Step 4: Write reflection (do not write all at once, write one at a time)
- Write reflections in `annotate/<problem_id>/<state_id>/<action_id>-reflections.txt` using this template:

```
## Answer: <correct/wrong> (<value>)

## Key moments
- Token NNNN [insight]: <description>
- Token NNNN [breakthrough]: <description>
- Token NNNN [questionable statement]: <description>
- Token NNNN [mistake]: <description>
- Token NNNN [missed opportunity]: <description>
- Token NNNN-MMMM [wasted]: <description>
- Token NNNN [verification]: <description>
- Token NNNN [final answer]: <description>
```

### Tag definitions

- `[breakthrough]` — The trace arrives at a key insight, correct reduction, or strategic idea that meaningfully advances the solution (e.g., identifying the right formula, reducing the problem to a simpler form).
- `[insight]` - This is the earliest source leading to the `[breakthrough]`. This may be a hypothesis, or a discovery that is not finalized.
- `[mistake]` — The trace makes a concrete mathematical or logical error (wrong computation, flawed deduction, misapplied theorem). Cite what went wrong specifically.
- `[questionable statement]` - This is the source of the mistake. This may be some questionable statement even though the model has yet to commit to the mistake.
- `[missed opportunity]` — The trace had the information or context to take a better path but didn't. This is a failure to capitalize on something available (e.g., not noticing a pattern, skipping a simpler approach).
- `[wasted]` — A span of tokens (use `Token NNNN-MMMM`) where the trace spends effort on something unproductive: redundant re-derivation, exploring a dead-end that was already ruled out, or circular reasoning.
- `[verification]` - Productive rigorous check on certain hypothesis. If the process is inefficient, or the check is unnecessary, consider this `[wasted]`. If the answer is wrong, use `[missed opportunity]` instead on what could have been realized.
- `[final answer]` — Where the trace commits to its final answer value. There should be exactly one of these per reflection. Please write if the answer is correct.

### Formatting rules

List entries chronologically (in token order) as you read through the trace.
Do not include problem-level knowledge (correct approach, solution formulas) — that belongs in findings.txt.
Use bullet points, not prose paragraphs. Always cite token numbers.

IMPORTANT: Cover the FULL trace, not just the beginning. Entries must span early, middle, and late portions.
- You are advised to write down the most obvious tags first, then fill in the gaps later.
- You are advised to work on one file first before moving to another file.

Specifically document:
- Where the trace commits to its final answer
- Verification/checking steps (or lack thereof)
- Late reconsiderations or course corrections
- Where the trace could have caught its errors before finalizing

### Step 5: Verify reflection (do not write all at once, write one at a time)

What to check
- The maximum tokens between each tag should be at most 5000 ~ 10000
  - If the key moments are spaced evenly apart, it means you are not really searching correctly
- If the final answer (or critical intermediate result) is wrong, there must be multiple `[mistake]` somewhere
- If there is a `[mistake]` but the final answer (or critical intermediate result) is correct, there must be a `[breakthrough]` somewhere in between.
- If one solution is taking many more tokens than another solution there must be something wasted somewhere
- For each `[mistake]`, find out exactly where is the source of the mistake. Tag the location of the source with `[questionable statement]`.
- For each `[breakthrough]`, find out exactly where is the source of the breakthrough. Tag the location of the source with `[insight]`.
- `data/annotate/<problem_id>/findings.txt` is updated

Complete one reflection file first before moving to other reflection file.
Do NOT write all the files at once.

### Step 6: Update Findings

You are also responsible to update `data/annotate/<problem_id>/findings.txt`
- Note that `findings.txt` may not be entirely correct. It is your job to fix it.
- Document: correct answer, correct approach, wrong approaches and why they fail
- **Do NOT include completion-specific data** (e.g., token ranges, line numbers, annotation progress) — findings.txt is shared among all completions, only include problem-level insights
- **Document both correct AND wrong approaches**:
  - Correct approach: formula, key values, why it works
  - Wrong approach(es): what formula/values they use, what wrong answer they produce

### Note on existing annotations
If you notice errors in existing annotations (`data/annotate/<state_id>/<action_id>.txt`), you may fix them, but it is not required — your primary task is writing reflections and findings.

