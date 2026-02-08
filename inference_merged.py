# uv run modal deploy inference_merged.py
# Serves the merged model from the merged-model volume

import os
import modal

# Container setup
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
)

# Model configuration - merged model from volume
MODEL_PATH = "/merged/model"

# Volume setup for caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
flashinfer_cache_vol = modal.Volume.from_name(
    "flashinfer-cache", create_if_missing=True
)
merged_model_vol = modal.Volume.from_name("merged-model", create_if_missing=True)

vllm_passphrase = str(os.environ.get("VLLM_PASSPHRASE")) + "-merged"

# App configuration
app = modal.App(f"example-gpt-oss-inference{vllm_passphrase}")
N_GPU = 1
MAX_INPUTS = 32  # could differ from Kaggle
MAX_MODEL_LEN = 131072
MINUTES = 60
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H100!:{N_GPU}",
    scaledown_window=5 * MINUTES,
    timeout=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/root/.cache/flashinfer": flashinfer_cache_vol,
        "/merged": merged_model_vol,
    },
    max_containers=1,
)
@modal.concurrent(max_inputs=MAX_INPUTS * 8)  # additional allowance for /metrics calls
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_PATH,
        "--served-model-name",
        "vllm-model",
        "--tensor-parallel-size",
        str(N_GPU),
        "--max-num-seqs",
        f"{MAX_INPUTS}",
        "--gpu-memory-utilization",
        "0.95",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "auto",
        "--async-scheduling",
        "--attention-backend",
        "FLASH_ATTN",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        "8192",
    ]

    print(cmd)
    subprocess.Popen(" ".join(cmd), shell=True)
