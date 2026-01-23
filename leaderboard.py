# uv run modal deploy leaderboard.py
# uv run modal app logs leaderboard-monitor

import os
from datetime import datetime

import modal

app = modal.App("leaderboard-monitor")

image = modal.Image.debian_slim(python_version="3.11").pip_install("kaggle")

# Persistent Dict storage
positions_dict = modal.Dict.from_name("leaderboard-positions", create_if_missing=True)
history_dict = modal.Dict.from_name("leaderboard-history", create_if_missing=True)


def parse_leaderboard_csv(csv_content: str) -> list[dict]:
    """Parse kaggle leaderboard CSV output into structured data."""
    import csv
    import io

    entries = []
    reader = csv.DictReader(io.StringIO(csv_content))

    for row in reader:
        entries.append(
            {
                "team_id": row.get("TeamId", row.get("teamId", "")),
                "team_name": row.get("TeamName", row.get("teamName", "")),
                "submission_date": row.get(
                    "LastSubmissionDate",
                    row.get("SubmissionDate", row.get("submissionDate", "")),
                ),
                "score": row.get("Score", row.get("score", "")),
                "submission_count": row.get(
                    "SubmissionCount", row.get("submissionCount", "")
                ),
            }
        )
    return entries


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("kaggle")],
    schedule=modal.Cron("* * * * *"),  # Every minute
    timeout=20 * 60,
)
def check_leaderboard():
    import subprocess
    from pathlib import Path

    # Setup kaggle credentials from KAGGLE_API_TOKEN
    kaggle_token = os.environ.get("KAGGLE_API_TOKEN")
    if kaggle_token:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        kaggle_json = kaggle_dir / "kaggle.json"
        kaggle_json.write_text(kaggle_token)
        kaggle_json.chmod(0o600)

    now = datetime.utcnow()
    print(f"\n{'=' * 60}")
    print(f"Leaderboard check at {now.isoformat()} UTC")
    print(f"{'=' * 60}\n")

    # Download full leaderboard to a temp file (clean up first to avoid stale files)
    download_dir = Path("/tmp/leaderboard")
    import shutil

    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(exist_ok=True)

    result = subprocess.run(
        [
            "kaggle",
            "competitions",
            "leaderboard",
            "ai-mathematical-olympiad-progress-prize-3",
            "--download",
            "--path",
            str(download_dir),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return result.stdout

    # Extract ZIP if present, then read CSV
    import zipfile

    for zip_file in download_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(download_dir)

    csv_files = list(download_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV file found in {download_dir}")
        print(f"Files: {list(download_dir.iterdir())}")
        return ""
    csv_file = csv_files[0]

    csv_content = csv_file.read_text()
    entries = parse_leaderboard_csv(csv_content)

    # Build leaderboard_positions: list of [team_id, team_name, position, submission_count]
    leaderboard_positions = []
    new_submissions = 0

    for position, entry in enumerate(entries, start=1):
        team_id = entry["team_id"]
        if not team_id:
            continue

        team_name = entry["team_name"]
        score = entry["score"]
        submission_count = entry["submission_count"]
        submission_date_str = entry["submission_date"]

        # Add to positions list
        leaderboard_positions.append([team_id, team_name, position, submission_count])

        # Parse submission time to create date key and calculate minutes_taken
        if submission_date_str:
            # Format: "2024-12-11 01:23:45.123456" -> "12-11"
            date_str_clean = submission_date_str.split(".")[0]
            sub_time = datetime.strptime(date_str_clean, "%Y-%m-%d %H:%M:%S")
            date_key = sub_time.strftime("%m-%d")
            submission_time_formatted = sub_time.strftime("%Y-%m-%d-%H-%M")

            # Calculate minutes_taken as time since submission (relative to cron run time)
            delta = now - sub_time
            minutes_taken = int(delta.total_seconds() / 60)

            # Get existing history for this team
            existing_entries = history_dict.get(team_id, {})

            # Check if this submission_time is new
            is_new = True
            for existing_date_key, existing_data in existing_entries.items():
                if existing_data.get("submission_time") == submission_time_formatted:
                    is_new = False
                    break

            if is_new:
                # Add new entry under the date key
                # If there's already an entry for this date, use a unique key
                final_date_key = date_key
                if date_key in existing_entries:
                    # Append a counter to make it unique
                    counter = 1
                    new_key = f"{date_key}-{counter}"
                    while new_key in existing_entries:
                        counter += 1
                        new_key = f"{date_key}-{counter}"
                    final_date_key = new_key

                existing_entries[final_date_key] = {
                    "score": score,
                    "minutes_taken": minutes_taken,
                    "submission_time": submission_time_formatted,
                }
                history_dict[team_id] = existing_entries
                new_submissions += 1

    # Store positions
    positions_dict["positions"] = leaderboard_positions

    # Cache the full API response for fast retrieval
    all_history = dict(history_dict)
    positions_dict["cached_response"] = {
        "positions": leaderboard_positions,
        "history": all_history,
    }

    print(f"Stored {len(leaderboard_positions)} teams in positions")
    print(f"New submissions recorded: {new_submissions}")

    # Print top 10 for quick reference
    print("\nTop 10:")
    print(f"{'Pos':>4} {'Team':<25} {'Score':>6} {'Subs':>5}")
    print(f"{'-' * 4} {'-' * 25} {'-' * 6} {'-' * 5}")
    for pos in leaderboard_positions[:10]:
        team_id, team_name, position, sub_count = pos
        # Get latest score from history
        team_hist = history_dict.get(team_id, {})
        latest_score = ""
        if team_hist:
            latest_key = max(team_hist.keys())
            latest_score = team_hist[latest_key].get("score", "")
        print(f"{position:>4} {team_name[:25]:<25} {latest_score:>6} {sub_count:>5}")

    return result.stdout


@app.function(max_containers=1)
@modal.fastapi_endpoint(method="GET")
def get_history():
    """Return leaderboard positions and historical scores as JSON.

    Format:
    {
        "positions": [[team_id, team_name, position, submission_count], ...],
        "history": {team_id: {date: {score, minutes_taken, submission_time}}}
    }
    """
    from fastapi.responses import JSONResponse

    # Single lookup - cached by cron job
    cached = positions_dict.get("cached_response", {"positions": [], "history": {}})

    return JSONResponse(
        content=cached,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )
