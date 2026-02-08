#!/usr/bin/env python3
"""Clear better/neutral/worse for lines committed before cutoff, keep reason."""

import glob
import json
import subprocess
import re
import sys
from datetime import datetime

CUTOFF = datetime(2026, 2, 6, 14, 0, 0)

files = sorted(glob.glob("data/label/aimo3-21818*/*/*"))
print(f"Found {len(files)} files")

modified_count = 0
for fpath in files:
    result = subprocess.run(
        ["git", "blame", "--porcelain", fpath],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"SKIP {fpath}: git blame failed", file=sys.stderr)
        continue

    # Parse porcelain output: track date by commit hash, then map lines to hashes
    commit_dates = {}
    line_commits = {}
    current_hash = None
    current_line = None
    for blame_line in result.stdout.splitlines():
        m = re.match(r'^([0-9a-f]{40}) \d+ (\d+)', blame_line)
        if m:
            current_hash = m.group(1)
            current_line = int(m.group(2))
            line_commits[current_line] = current_hash
        elif blame_line.startswith("author-time "):
            ts = int(blame_line.split()[1])
            dt = datetime.fromtimestamp(ts)
            if current_hash is not None:
                commit_dates[current_hash] = dt

    # Read file lines
    with open(fpath) as f:
        original_lines = f.readlines()

    new_lines = []
    changed = False
    for i, line in enumerate(original_lines, start=1):
        commit_hash = line_commits.get(i)
        line_date = commit_dates.get(commit_hash) if commit_hash else None
        if line_date and line_date < CUTOFF:
            try:
                obj = json.loads(line)
                for key in obj:
                    inner = obj[key]
                    if any(inner[k] for k in ("better", "neutral", "worse")):
                        inner["better"] = []
                        inner["neutral"] = []
                        inner["worse"] = []
                        changed = True
                new_line = json.dumps(obj, ensure_ascii=False) + "\n"
                new_lines.append(new_line)
            except json.JSONDecodeError:
                new_lines.append(line)
        else:
            new_lines.append(line)

    if changed:
        with open(fpath, "w") as f:
            f.writelines(new_lines)
        modified_count += 1
        print(f"MODIFIED {fpath}")

print(f"\nDone. Modified {modified_count}/{len(files)} files.")
