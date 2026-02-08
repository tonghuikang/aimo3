"""
Upload merged model from Modal volume directly to Kaggle.

Usage:
    uv run modal run --detach upload_model.py

Prerequisites:
    1. KAGGLE_API_TOKEN in env.json
    2. Merged model at Modal volume /merged/model (from merge_adapter.py)

The script automatically:
    1. Creates the model instance if it doesn't exist (with non-shard files only for speed)
    2. Creates a new version with all files (including shards)
"""

import json
import os
import shutil
import tempfile

import modal
from requests.exceptions import HTTPError

kaggle_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "kaggle>=1.6.0"
)

merged_model_vol = modal.Volume.from_name("merged-model", create_if_missing=True)

app = modal.App("upload-to-kaggle")


MODEL_DIR = "/merged/model"
DEFAULT_INSTANCE = "huikang/gpt-oss-120b-aimo3/Transformers/160a"


@app.function(
    image=kaggle_image,
    volumes={"/merged": merged_model_vol},
    timeout=3 * 60 * 60,
)
def upload_to_kaggle(kaggle_api_token: str):
    """Upload model from Modal volume to Kaggle."""
    # Write credentials before importing kaggle (it auto-authenticates on import)
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    with open(os.path.join(kaggle_dir, "access_token"), "w") as f:
        f.write(kaggle_api_token)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print("Kaggle API authenticated")

    # Verify source directory exists and has files
    if not os.path.exists(MODEL_DIR):
        raise ValueError(f"Source directory not found: {MODEL_DIR}")

    files = os.listdir(MODEL_DIR)
    print(f"Found {len(files)} files in {MODEL_DIR}:")
    for fname in sorted(files):
        size = os.path.getsize(os.path.join(MODEL_DIR, fname))
        print(
            f"  {fname}: {size / 1e9:.2f} GB"
            if size > 1e9
            else f"  {fname}: {size / 1e6:.2f} MB"
        )

    # Parse instance string: owner/model/framework/slug
    parts = DEFAULT_INSTANCE.split("/")
    owner, model_slug, framework, instance_slug = parts[0], parts[1], parts[2], parts[3]

    # Check if instance exists
    def instance_exists() -> bool:
        try:
            api.model_instance_get(DEFAULT_INSTANCE)
            return True
        except HTTPError:
            return False

    # Create instance if it doesn't exist (with non-shard files only for speed)
    if not instance_exists():
        print(f"\nInstance {DEFAULT_INSTANCE} does not exist, creating...")

        upload_dir = tempfile.mkdtemp()
        for fname in files:
            if not (fname.startswith("model-") and fname.endswith(".safetensors")):
                shutil.copy(os.path.join(MODEL_DIR, fname), upload_dir)
        print(f"Copied {len(os.listdir(upload_dir))} non-shard files to temp dir")

        metadata = {
            "ownerSlug": owner,
            "modelSlug": model_slug,
            "instanceSlug": instance_slug,
            "framework": framework,
            "licenseName": "Apache 2.0",
            "overview": "Fine-tuned GPT-OSS-120B for AIMO3",
        }
        metadata_path = os.path.join(upload_dir, "model-instance-metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        print(f"Created metadata: {metadata}")

        api.model_instance_create(upload_dir, dir_mode="skip")
        print("Instance created")
    else:
        print(f"\nInstance {DEFAULT_INSTANCE} already exists")

    # Always create new version with all files
    print(f"\nUploading new version to {DEFAULT_INSTANCE} with all files...")
    api.model_instance_version_create(DEFAULT_INSTANCE, MODEL_DIR, dir_mode="skip")
    print("Version created with all files")

    print("\nUpload complete!")
    return "Success"


@app.local_entrypoint()
def main():
    """Upload merged model to Kaggle."""
    print(f"Uploading to Kaggle model instance: {DEFAULT_INSTANCE}")
    with open("env.json") as f:
        env = json.load(f)
        KAGGLE_API_TOKEN = env["KAGGLE_API_TOKEN"]

    result = upload_to_kaggle.remote(KAGGLE_API_TOKEN)
    print(result)
