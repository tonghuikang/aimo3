# Data Pipeline

## Directory Structure

| Directory | Contents |
|-----------|----------|
| `problems.csv` | Problem statements and answers |
| `raw/` | Raw solution traces (tokens, actions, states) |
| `alternate/` | Alternative statements at each token position |
| `annotate/` | Annotation files for labeling + readable solution traces |
| `label/` | Extracted labels (JSONL format) |

## Pipeline Steps

### 1. Generate raw traces
```
./generate.sh
```
Reads `problems.csv` and generates solution traces in `raw/`.

### 2. Generate alternates
```
uv run python -m data.roll_alternates <problem_id>
```
At paragraph boundaries in `raw/`, generates alternative continuations and saves to `alternate/`.

### 3. Prepare annotation templates
```
uv run python -m data.prepare_annotations [problem_id]
```
Combines `raw/`, `alternate/`, and existing `label/` to create annotation templates in `annotate/`.
Also generates readable traces (`state.txt`, `*-original.txt`) in `annotate/`.

If no argument is provided, prepares annotations for all problems with alternates.

### 4. Annotate (human or AI)
Review `state.txt` and `*-original.txt` for context, then fill in `performance:` and `reason:` fields in `annotate/` files.

See `./ANNOTATION_GUIDELINES.md` for annotation instructions.

### 5. Extract labels
```
uv run python -m data.extract_labels
```
Parses completed `annotate/` files and extracts labels to `label/`.

### 6. Generate training data (not yet done)
Combines `raw/`, `alternate/`, and `label/` to produce final training data.
