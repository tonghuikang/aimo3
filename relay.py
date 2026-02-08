"""Local relay server that proxies OpenAI-compatible completions API to Tinker's native SamplingClient.

Tinker's OpenAI-compatible endpoint doesn't support token IDs as prompt input.
This relay accepts token IDs, calls Tinker's native API, and streams back the response.

Usage:
    uv run python tinker_relay.py

Then point your OpenAI client to http://localhost:8100/v1
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import tinker
from tinker import ServiceClient, SamplingClient, types
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai_harmony import HarmonyEncodingName, load_harmony_encoding

# Load tokenizer for decoding
harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Load secrets
with open("env.json") as f:
    secrets = json.load(f)

TINKER_API_KEY = secrets["TINKER_API_KEY"]
TINKER_MODEL_NAME = secrets["TINKER_MODEL_NAME"]

# Hardcoded stop tokens for Harmony model
STOP_TOKEN_IDS = [200002, 200012]

# Global sampling client (initialized on startup)
sampling_client: SamplingClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Tinker client on startup."""
    global sampling_client
    print(f"Initializing Tinker SamplingClient for model: {TINKER_MODEL_NAME}")
    os.environ["TINKER_API_KEY"] = TINKER_API_KEY
    service_client = ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=TINKER_MODEL_NAME)
    print("Tinker SamplingClient ready")
    yield
    print("Shutting down")


app = FastAPI(lifespan=lifespan)


class CompletionRequest(BaseModel):
    model_config = {"extra": "allow"}  # Allow extra fields like extra_body

    model: str
    prompt: str | list[int] | list[str]
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: str | list[str] | list[int] | None = None


def create_completion_chunk(
    completion_id: str,
    model: str,
    token_ids: list[int],
    finish_reason: str | None = None,
) -> dict:
    """Create an OpenAI-compatible completion chunk."""
    # Decode tokens to text
    text = harmony_encoding.decode(token_ids) if token_ids else ""
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "token_ids": token_ids,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


async def stream_tokens(
    completion_id: str,
    model: str,
    tokens: list[int],
):
    """Stream tokens one by one as SSE events (token IDs only, no decoding)."""
    for token_id in tokens:
        chunk = create_completion_chunk(
            completion_id=completion_id,
            model=model,
            token_ids=[token_id],
        )
        yield f"data: {json.dumps(chunk)}\n\n"

    # Send final chunk with finish_reason
    final_chunk = create_completion_chunk(
        completion_id=completion_id,
        model=model,
        token_ids=[],
        finish_reason="stop",
    )
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def get_prompt_logprobs(
    model: str,
    model_input: types.ModelInput,
    prompt_tokens: list[int],
) -> dict:
    """Compute logprobs for prompt tokens using Tinker's compute_logprobs API."""

    def compute():
        assert sampling_client is not None
        return sampling_client.compute_logprobs(model_input).result()

    try:
        logprobs_list = await asyncio.get_event_loop().run_in_executor(None, compute)
    except tinker.TinkerError as e:
        raise HTTPException(status_code=500, detail=f"Tinker API error: {e}")

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": "",
                "logprobs": {
                    "token_logprobs": logprobs_list,
                    "tokens": prompt_tokens,
                    "top_logprobs": None,
                },
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": 0,
            "total_tokens": len(prompt_tokens),
        },
    }


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI-compatible completions endpoint that proxies to Tinker."""
    if sampling_client is None:
        raise HTTPException(status_code=503, detail="Sampling client not initialized")

    # Validate prompt is a list of token IDs
    if isinstance(request.prompt, str):
        raise HTTPException(
            status_code=400,
            detail="String prompts not supported by relay. Use token IDs.",
        )
    if not all(isinstance(x, int) for x in request.prompt):
        raise HTTPException(
            status_code=400,
            detail="Prompt must be a list of integers (token IDs)",
        )
    prompt_tokens: list[int] = request.prompt  # type: ignore[assignment]
    model_input = types.ModelInput.from_ints(prompt_tokens)
    prompt_logprobs = getattr(request, "prompt_logprobs", None)

    if prompt_logprobs is not None and prompt_logprobs >= 1:
        assert 0 <= request.max_tokens <= 1
        return await get_prompt_logprobs(request.model, model_input, prompt_tokens)

    # Build Tinker request
    # Use hardcoded stop tokens, fallback to request stop if provided
    stop_tokens = request.stop if request.stop else STOP_TOKEN_IDS
    sampling_params = types.SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=stop_tokens,
    )

    # Call Tinker's native API in a thread pool to avoid blocking the event loop
    def call_tinker():
        assert sampling_client is not None
        future = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        return future.result()

    try:
        loop = asyncio.get_event_loop()
        response: types.SampleResponse = await loop.run_in_executor(None, call_tinker)
    except tinker.TinkerError as e:
        raise HTTPException(status_code=500, detail=f"Tinker API error: {e}")

    # Extract generated tokens
    if not response.sequences:
        raise HTTPException(status_code=500, detail="No sequences in response")

    generated_tokens = response.sequences[0].tokens
    stop_reason = response.sequences[0].stop_reason

    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"

    if request.stream:
        return StreamingResponse(
            stream_tokens(
                completion_id=completion_id,
                model=request.model,
                tokens=generated_tokens,
            ),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming response (token IDs only, no text decoding)
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "text": "",  # No text decoding in relay
                    "token_ids": generated_tokens,
                    "logprobs": None,
                    "finish_reason": str(stop_reason),
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(generated_tokens),
                "total_tokens": len(prompt_tokens) + len(generated_tokens),
            },
        }


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": TINKER_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "tinker",
            }
        ],
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "sampling_client_ready": sampling_client is not None}


@app.get("/metrics")
async def metrics():
    """Stub metrics endpoint for compatibility with notebook's get_gpu_kv_cache_usage."""
    # Return -1 for KV cache usage (parsed as -100% after *100 in notebook)
    return "vllm:kv_cache_usage_perc -0.01"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
