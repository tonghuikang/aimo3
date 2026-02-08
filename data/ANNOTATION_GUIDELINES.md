# Annotation Guidelines

Your task is to hand annotate solution traces by comparing the original span against generated alternatives, labeling each as `better`, `neutral`, or `worse`.

Do not use heuristics or keyword rules to decide labels; please read the text and judge by content.

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
│       ├── <action_id>-reflections.txt    # Your reflections on the trace
│       └── <action_id>.txt                # Annotation file you will edit
└── label/<problem_id>/               # Parsed annotations (auto-regenerated on edit, do not use/edit directly)
```

**To view alternates for a token**: Open the annotation file at `data/annotate/<problem_id>/<state_id>/<action_id>.txt` and search for "Token N". This file contains the decoded text for both original and all alternates.

## Annotation Format

```
Token N-M original: <original statement>
Token N alternate abc123: <alternate 1>
Token N alternate def456: <alternate 2>
Token N alternate ghi789: <alternate 3>
Token N better: <completion_ids>
Token N neutral: <completion_ids>
Token N worse: <completion_ids>
Token N reason: <your reason here>
```

The original line shows the span `N-M` where N is the start token and M is the end token (exclusive). This span represents the actual paragraph content, ending at `\n\n` or a tool call boundary.

Completion IDs include `original` and alternate IDs (e.g., `abc123`).

## Annotation Labels

### When to mark completions as better
A completion is better when it:
- Is more mathematically accurate or correct
- Makes better progress toward the solution
- Provides clearer or more precise reasoning
- Avoids errors or misconceptions present in other completions
- Introduces key insights (e.g., problem reductions, winning strategies)

**Alternates don't need to be correct to be "better"**: An alternate is "better" if it shows *significantly better reasoning* than the others, even if all completions are wrong. Examples:
- More careful/rigorous analysis (even within a wrong framework)
- Questions assumptions that other completions blindly accept
- Explores more possibilities before committing
- Is less prematurely committed to a wrong direction

If you think "all completions pursue the same wrong approach" - look closer. Within a wrong approach, there are often differences in reasoning quality.

### When to mark completions as worse
A completion is worse when it:
- Contains mathematical errors that other completions avoid
- Makes less progress than other completions
- Is vague while other completions are specific
- Misses key insights that other completions capture
- Leads reasoning in a wrong direction

**IMPORTANT**:
- If all completions (original + alternates) make the same error, they are all neutral - don't mark any as worse just because the trace arrives at a wrong answer.
- If you mark any completion as "worse", at least one other completion must be "better". Worse is relative - a completion can only be worse *than* something.
- Similarly, if you mark any completion as "better", at least one other completion must be "worse".
- Don't label a completion as "worse" just because it's bad (e.g., wrong answer). A completion is only "worse" if a better alternative exists among the completions. No good alternative = all neutral.

## Workflow

### Step 1: Understand the problem and trace
- Read the problem.
- Note the correct answer in `annotate/<problem_id>/correct_answer.txt`
- Read `annotate/<problem_id>/findings.txt` for the correct approach and known wrong approaches
- Read the reflections file (`<action_id>-reflections.txt`) for the trace you are annotating — this tells you the answer, key moments, mistakes, and missed opportunities

Reflections and findings are assumed to already exist. If they are missing, write them first following the [Reflection Guidelines](REFLECTION_GUIDELINES.md).

### Step 2: Annotate the solution

Use the reflections file to guide your annotation. The reflections identify key moments (breakthroughs, mistakes, missed opportunities) — these are where alternates are most likely to make a difference. **Do not annotate every token mechanically**; focus your effort on tokens where differentiation matters.

1. **Start from the reflections file**
   - Read the `[breakthrough]`, `[mistake]`, and `[missed opportunity]` tokens listed in reflections
   - These are the tokens most likely to have meaningful alternate differences
   - Go to each of these tokens in the annotation file and read the alternates carefully

2. **Hunt for decision points where alternates diverge**
   - Where does the trace commit to an approach? (e.g., "Thus we use formula X...")
   - Are there alternates that question this choice or suggest different approaches?
   - Alternates asking "could there be a better approach?" are often valuable
   - Where does the trace declare specific values? Do alternates propose different values?

3. **Label with differentiation in mind**
   - If all completions follow the same approach, they're likely all neutral
   - Look for alternates that QUESTION the approach - these often deserve "better"
   - Look for alternates that COMMIT to wrong approaches - these may be "worse"

4. **Skip tokens that don't matter**
   - Only annotate tokens where you find meaningful differentiation between completions
   - Leave mechanical tokens (setup, computation, transitions) unannotated — reading them for context is fine, but do not spend effort annotating them

## Examples

### Example 1: Original worse (wrong answer guess)
```
Token 801-892 original: For k=3, likely f(3)=5? We need to compute exactly.
Token 801 alternate gorajo: One might think for k=3, maybe f(3)=3? But we need to examine constraints.
Token 801 better: gorajo
Token 801 neutral:
Token 801 worse: original
Token 801 reason: The correct answer is f(3)=3. The original guesses 5 (wrong), while gorajo correctly hypothesizes 3.
```

### Example 2: Original worse (alternate has key insight)
```
Token 731-819 original: First, this is reminiscent of a search problem like "snake and monsters".
Token 731 alternate 00cqyy: This resembles search problem with unknown obstacles. But we have constraints: exactly one monster per intermediate row, in distinct columns. This is like trying to find a "safe column" that passes all rows.
Token 731 better: 00cqyy
Token 731 neutral:
Token 731 worse: original
Token 731 reason: The alternate identifies the crucial "safe column" insight that directly leads to the solution strategy. The original only makes a vague analogy.
```

### Example 3: Original worse (alternate suggests correct strategy)
```
Token 2940-3012 original: But you can also succeed without complete identification: you can also attempt to "snake" down the grid, adjusting left/right to avoid monsters.
Token 2940 alternate ry5x3r: Idea: Strategy: Use attempts to locate the safe column using a binary search style: you can test row by row on two columns per attempt? For k=3, maybe you can guarantee success within 3 attempts.
Token 2940 better: ry5x3r
Token 2940 neutral:
Token 2940 worse: original
Token 2940 reason: The alternate suggests the correct answer (3 attempts) and a strategic approach, while the original discusses a less efficient strategy.
```

### Example 4: Original better (key problem reduction)
```
Token 765-831 original: Given constraints: 3 rows with monsters, each in distinct columns among 4. So exactly one column is completely safe in those rows. So if the snail can discover which column is safe, then it can simply go straight down that column and succeed. So the core is to identify a monster-free column.
Token 765 alternate ah5klc: We have a 5x4 grid. There is exactly one monster in each of rows 1,2,3, each in a distinct column.
Token 765 better: original
Token 765 neutral:
Token 765 worse: ah5klc
Token 765 reason: The original identifies the core problem reduction (find the safe column), while the alternate just restates facts.
```

### Example 5: Original better (avoids wrong answer)
```
Token 1483-1521 original: So, find optimal minimal attempts.
Token 1483 alternate 8ix08n: So we need to find f(3). Possibly answer is 5? Let's think.
Token 1483 better: original
Token 1483 neutral:
Token 1483 worse: 8ix08n
Token 1483 reason: The alternate incorrectly guesses f(3)=5. The original stays neutral.
```

### Example 6: Neutral (transition statement)
```
Token 819-856 original: Let's analyze comprehensively.
Token 819 alternate njeucy: Approach: Let's attempt to determine lower bounds and upper bounds.
Token 819 better:
Token 819 neutral: original, njeucy
Token 819 worse:
Token 819 reason: Both are transition statements that don't substantively advance the solution.
```

### Example 7: Neutral (same exploration)
```
Token 2003-2089 original: If the safe column is unknown, you might try path down a column until you hit a monster... worst-case attempts = 4.
Token 2003 alternate upf91f: Potentially the worst-case number of attempts equals number of monsters + 1 = 4?
Token 2003 better:
Token 2003 neutral: original, upf91f
Token 2003 worse:
Token 2003 reason: Both discuss the same bound (4 attempts) with similar reasoning. Neither provides decisive insight.
```

### Example 8: Alternate better (questions the approach)
```
Token 2511-2598 original: But also consider possibility where three distinct divisors do not include X but include others: could produce smaller X? Let's think.
Token 2511 alternate e6c1eo: But must we also consider other triples of divisors not containing X? ... the extreme case is using the two largest proper divisors of X, perhaps X/p and maybe X/q. If those are close to X, X could be reduced further beyond the case X + d + 1.
Token 2511 alternate xiolnu: So indeed minimal X should involve X.
Token 2511 better: e6c1eo
Token 2511 neutral: original
Token 2511 worse: xiolnu
Token 2511 reason: e6c1eo explores the correct alternative approach (using X/p and X/q as divisors) which leads to the correct solution. xiolnu dismisses alternatives and commits to the wrong approach.
```

### Example 9: All neutral (all commit to same wrong approach)
```
Token 5039-5124 original: Now N = M + c, and M = 3^{2025!}. Since M is odd...
Token 5039 alternate oswvyg: Thus f(N) = N - 1 - (N-1)/t_min.
Token 5039 alternate yci63m: Thus f(N) = (N-1) * (t-1)/t = (N-1) - (N-1)/t.
Token 5039 better:
Token 5039 neutral: original, oswvyg, yci63m, mxmubg, 2ppo7z, 7ol3zg
Token 5039 worse:
Token 5039 reason: All completions commit to the same formula based on the {1,d,X} approach. None explore the alternative {X,X/u,X/v} approach. Since all make the same error, they are all neutral relative to each other.
```

### Example 10: Neutral (alternate starts function call)

```
Token 15297-15382 original: We need to compute 3^L mod (D-1). The exponent L is huge; we can compute L mod φ(D-1) using factorial modulo φ(D-1).
Token 15297 alternate d6jlti: We need to compute 3^L mod (D-1). The exponent L is huge; we can compute L mod φ(D-1) using factorial modulo φ(D-1). Compute via Python.<|end|><|start|>assistant<|channel|>analysis to=python code<|message|>import sympy as sp
Token 15297 better:
Token 15297 neutral: original, d6jlti
Token 15297 worse:
Token 15297 reason: Both completions are similar
```

Note: `<|end|><|start|>assistant<|channel|>analysis to=python code<|message|>` should not be considered corrupted output, this is the format to start a function call. Similarly for `<|call|>`, which denotes the end of a function call.


## Tips

### Verify findings.txt before trusting it
- findings.txt may have been written by someone who misunderstood the problem
- Always verify by checking against a trace that gets the CORRECT answer
- If findings.txt describes an approach that leads to a wrong answer, update it

### Identify correct vs wrong traces first
Use the Grep tool (not bash grep) to find which traces get which answers:
- Search for `boxed` or answer values in `data/annotate/<problem_id>/`
- Different wrong answers often mean different errors

### Look for alternates that question the approach
The most valuable "better" labels often go to alternates that:
- Ask "could there be a better approach?"
- Question assumptions ("is it always optimal to use X?")
- Explore alternatives before dismissing them

Conversely, "worse" labels often go to alternates that:
- Prematurely dismiss alternatives ("So we must use X")
- Commit to an approach without considering alternatives

### Avoid heuristic labeling
- Do not auto-label based on keyword hits (e.g., "answer", numbers, "delta", etc.).
- Do not use scripts or regex heuristics to decide better/neutral/worse.
- Heuristics can be used only to *locate* candidate lines for review; the final labels must be decided by reading the text.

### Handling large files
Some annotation files exceed the token limit. Use the Grep tool (not bash grep) to navigate:
- Search for answer values or `boxed` in the `-original.txt` file
- Search for `thus|therefore|conclude|hence` to find key mathematical conclusions
- Search for `reason: $` to find unannotated tokens
- Use the Read tool with offset/limit to read specific sections

### Quick editing pattern
```
old_string:
Token N better:
Token N neutral:
Token N worse:
Token N reason:

new_string:
Token N better: [completion_ids]
Token N neutral: [completion_ids]
Token N worse: [completion_ids]
Token N reason: <reason>
```

### Prioritizing traces
When multiple traces exist for a problem, annotate in this order:
1. Start with traces that have the **wrong answer** (more learning signal from errors)
2. Then annotate traces with the **correct answer** (to capture good reasoning)
3. Focus on traces with diverse approaches
