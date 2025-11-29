# uv run modal deploy inference.py
# See https://modal.com/docs/examples/gpt_oss_inference

import modal

# Container setup
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.2",
        "huggingface_hub[hf_transfer]==0.35.0",
    )
)

# Model configuration
MODEL_NAME = "openai/gpt-oss-120b"
MODEL_REVISION = "main"

# Volume setup for caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
flashinfer_cache_vol = modal.Volume.from_name(
    "flashinfer-cache", create_if_missing=True
)


# App configuration
app = modal.App("example-gpt-oss-inference")
N_GPU = 1
MAX_INPUTS = 32  # differs from Kaggle
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
    },
    env={"VLLM_ATTENTION_BACKEND": "TRITON_ATTN"},
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        "vllm-model",
        "--tensor-parallel-size",
        str(N_GPU),
        "--max-num-seqs",
        f"{MAX_INPUTS}",
        "--gpu-memory-utilization",
        "0.96",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "auto",
        "--max-model-len",
        str(MAX_MODEL_LEN),
    ]

    print(cmd)
    subprocess.Popen(" ".join(cmd), shell=True)
