# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# References
# - https://www.kaggle.com/code/huikang/arc-agi-2-code-approach
# - https://www.kaggle.com/code/huikang/r1-distill-qwen-tir
#
# ```
# uv run python3 kaggle.py
# ```

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Configuration

# %% [code] {"jupyter":{"outputs_hidden":false}}
run_all_questions_on_kaggle = False  # ignored for submissions
replication_count_for_commit_runs = 2

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os
import time
from collections.abc import Callable

from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()


start_time = time.time()
final_cutoff_time = start_time + (4 * 60 + 50) * 60  # 5 hours from start time


def is_on_kaggle_commit() -> bool:
    return os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Batch" and not bool(
        os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    )


def is_on_kaggle_interactive() -> bool:
    return os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Interactive" and not bool(
        os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    )


def is_on_kaggle() -> bool:
    return bool(os.getenv("KAGGLE_KERNEL_RUN_TYPE"))


REMOTE_VLLM_URL = "NOT_AVAILABLE"
if not is_on_kaggle():
    REMOTE_VLLM_URL = secrets.get_secret("REMOTE_VLLM_URL")


# Some debugger warning on Kaggle
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# %% [code] {"_kg_hide-output":true,"jupyter":{"outputs_hidden":false}}
# print settings
print(f"{is_on_kaggle()=}")
print(f"{is_on_kaggle_interactive()=}")
print(f"{is_on_kaggle_commit()=}")
print(f"{run_all_questions_on_kaggle=}")
print(f"{REMOTE_VLLM_URL[::-1][:13][::-1]=}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Setup

# %% [code] {"_kg_hide-output":true,"jupyter":{"outputs_hidden":false}}
import subprocess

if is_on_kaggle():
    subprocess.run(
        [
            "pip",
            "uninstall",
            "--yes",
            "tensorflow",
            "matplotlib",
            "keras",
            "scikit-learn",
        ]
    )


# %% [code] {"_kg_hide-output":true,"jupyter":{"outputs_hidden":false}}
import os
import subprocess


def cache_model(
    path, exts=(".bin", ".pt", ".safetensors"), num_workers=None, chunk_mb=256
):
    """
    Pre-read model weight files into the OS page cache to speed up later loads.

    Args:
        path        : Directory containing model files, or a single file path.
        exts        : File extensions treated as model weight files.
        num_workers : Number of threads (default = min(CPU cores, 8)).
        chunk_mb    : Size of each read chunk in MB.
    Returns:
        Total bytes read (int).
    """
    import os
    import multiprocessing
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def warmup_file(fpath):
        """Sequentially read an entire file in chunks."""
        chunk_size = chunk_mb * 1024 * 1024
        total = 0
        with open(fpath, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                total += len(data)
        return fpath, total

    # Collect files to read
    if os.path.isdir(path):
        files = [
            os.path.join(root, name)
            for root, _, names in os.walk(path)
            for name in names
            if name.endswith(exts)
        ]
        files.sort()
    else:
        files = [path]

    if not files:
        raise ValueError(f"No model files found under: {path}")

    # Decide number of worker threads
    if num_workers is None:
        try:
            num_workers = min(multiprocessing.cpu_count(), 8)
        except Exception:
            num_workers = 4

    print(f"[cache_model] {len(files)} file(s), {num_workers} worker(s)")

    t0 = time.time()
    total_bytes = 0

    # Read files in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(warmup_file, f): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            fpath, n = fut.result()
            total_bytes += n
            print(f"[{i}/{len(files)}] cached {os.path.basename(fpath)}")

    elapsed = time.time() - t0
    gb = total_bytes / 1024**3
    speed = gb / elapsed if elapsed > 0 else 0
    print(f"[cache_model] total read ≈ {gb:.2f} GB")
    print(f"[cache_model] elapsed {elapsed:.2f} s, ~{speed:.2f} GB/s")

    return total_bytes


if is_on_kaggle():
    cache_model(
        "/kaggle/input/gpt-oss-120b/transformers/default/1",
        num_workers=16,
        chunk_mb=1024,
    )

# %% [code] {"jupyter":{"outputs_hidden":false}}
import torch
import numpy as np


cutoff_times = [
    int(x) for x in np.linspace(final_cutoff_time, start_time + 30 * 60, 50 + 1)
]  # generous allowance at the start
cutoff_times.pop()

from datetime import datetime

# Create timestamped run directory
os.makedirs("runs", exist_ok=True)
RUN_DIR = f"runs/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
SOLUTIONS_DIR = f"{RUN_DIR}/solutions"
os.makedirs(SOLUTIONS_DIR, exist_ok=True)

# Initialize stats CSV file
from collections import Counter

import pandas as pd

STATS_CSV_PATH = "stats.csv"  # saved in root directory
STATS_COLUMNS = [
    "question_id",
    "final_answer",
    "time_taken",
    "time_available",
    "total_length",
    "solver_to_length",
    "solver_to_answer",
]

if __name__ == "__main__":
    pd.DataFrame(columns=pd.Index(STATS_COLUMNS)).to_csv(STATS_CSV_PATH, index=False)


def save_stats(
    question_id: str,
    final_answer: int,
    time_taken: float,
    time_available: float,
    solver_to_length: dict[int, int],
    solver_to_answer: dict[int, int],
) -> None:
    """Append stats for a question to the CSV file."""
    row_data = {
        "question_id": question_id,
        "final_answer": final_answer,
        "time_taken": round(time_taken, 1),
        "time_available": round(time_available, 1),
        "total_length": sum(solver_to_length.values()),
        "solver_to_length": str(solver_to_length),
        "solver_to_answer": str(solver_to_answer),
    }
    assert list(row_data.keys()) == STATS_COLUMNS, (
        f"Column mismatch: {list(row_data.keys())} != {STATS_COLUMNS}"
    )
    row = pd.DataFrame([row_data])
    row.to_csv(STATS_CSV_PATH, mode="a", header=False, index=False)


if is_on_kaggle():
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Serve vLLM

# %% [code] {"execution":{"iopub.status.busy":"2025-11-25T11:14:04.106282Z","iopub.execute_input":"2025-11-25T11:14:04.106474Z","iopub.status.idle":"2025-11-25T11:14:04.121482Z","shell.execute_reply.started":"2025-11-25T11:14:04.106461Z","shell.execute_reply":"2025-11-25T11:14:04.12108Z"},"jupyter":{"outputs_hidden":false}}
if is_on_kaggle():
    subprocess.run(["ls", "/kaggle/usr/lib/pip_install_aimo3_1/tiktoken_encodings"])

# %% [code] {"execution":{"iopub.status.busy":"2025-11-25T11:23:00.401889Z","iopub.execute_input":"2025-11-25T11:23:00.402121Z","iopub.status.idle":"2025-11-25T11:23:00.405096Z","shell.execute_reply.started":"2025-11-25T11:23:00.402105Z","shell.execute_reply":"2025-11-25T11:23:00.404686Z"},"jupyter":{"outputs_hidden":false}}
with open("a-vllm.log", "w") as f:
    f.write("")

# %% [code] {"execution":{"iopub.status.busy":"2025-11-24T08:13:52.188688Z","iopub.execute_input":"2025-11-24T08:13:52.188811Z","iopub.status.idle":"2025-11-24T08:13:52.194435Z","shell.execute_reply.started":"2025-11-24T08:13:52.188802Z","shell.execute_reply":"2025-11-24T08:13:52.194061Z"},"jupyter":{"outputs_hidden":false}}
import subprocess

num_generations = 6
max_model_len = 131072


def start_vllm_server() -> subprocess.Popen[bytes]:
    """Start vLLM server in the background"""
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["TRANSFORMERS_NO_FLAX"] = "1"
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html#troubleshooting
    os.environ["TIKTOKEN_ENCODINGS_BASE"] = (
        "/kaggle/usr/lib/pip_install_aimo3_1/tiktoken_encodings"
    )

    command: list[str] = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "/kaggle/input/gpt-oss-120b/transformers/default/1",
        "--served-model-name",
        "vllm-model",
        "--tensor-parallel-size",
        "1",
        "--max-num-seqs",
        f"{num_generations}",
        "--gpu-memory-utilization",
        "0.96",  # any higher may not have enough for graph capture
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "auto",
        "--max-model-len",
        f"{max_model_len}",
    ]

    # Start the process in the background
    with open("/kaggle/working/a-vllm.log", "w") as logfile:
        process: subprocess.Popen[bytes] = subprocess.Popen(
            command, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True
        )

    print("Logs: /kaggle/working/a-vllm.log")
    return process


# Start the server
if is_on_kaggle():
    vllm_process: subprocess.Popen[bytes] = start_vllm_server()

# %% [code] {"jupyter":{"outputs_hidden":false}}
import time


def await_client(printing: bool = False):
    for _ in range(15 * 60):
        time.sleep(1)
        try:
            model_list = client.models.list()
            if printing:
                print(model_list)
        except NameError:
            raise  # maybe you did not run the cell initializing client
        except Exception:
            continue
        break
    else:
        raise


if is_on_kaggle():
    # inference server needs to start within 15 minutes
    # we should be able to await now with cache_model()
    await_client()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Code execution


# %% [code] {"jupyter":{"outputs_hidden":false}}
class LocalJupyterSession:
    """Stateful helper that proxies execution through a local Jupyter kernel.
    Extracted from gpt_oss.tools.python_docker.docker_tool.
    Thread-safe: creates its own ZMQ context for use within a single thread.
    """

    def __init__(self, timeout: float = 10.0) -> None:
        import zmq
        from jupyter_client.blocking.client import BlockingKernelClient
        from jupyter_client.manager import KernelManager

        self._default_timeout = timeout
        # Create a dedicated ZMQ context for this session (thread-safe)
        self._zmq_context = zmq.Context()
        self._km = KernelManager(context=self._zmq_context)
        self._km.start_kernel()
        self._client: BlockingKernelClient = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=None)
        # Disable colors in IPython tracebacks
        self._client.execute("%colors NoColor", store_history=False)
        # Track msg_id of a timed-out execution that may still be running
        self._pending_msg_id: str | None = None

    def _drain_pending_output(self) -> str:
        """Drain output from a previous timed-out execution. Interrupts if still running."""
        if self._pending_msg_id is None:
            return ""

        msg_id = self._pending_msg_id
        self._pending_msg_id = None
        client = self._client

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        execution_finished = False

        # Drain any available output without blocking long
        while True:
            try:
                msg = client.get_iopub_msg(timeout=0.1)
            except queue.Empty:
                break

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                traceback_data = content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                execution_finished = True
                break

        # If still running, interrupt it
        if not execution_finished:
            self._km.interrupt_kernel()
            # Collect interrupt traceback
            while True:
                try:
                    msg = client.get_iopub_msg(timeout=1.0)
                except queue.Empty:
                    break
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                msg_type = msg.get("msg_type")
                content = msg.get("content", {})
                if msg_type == "stream":
                    text = content.get("text", "")
                    if content.get("name") == "stdout":
                        stdout_parts.append(text)
                    else:
                        stderr_parts.append(text)
                elif msg_type == "error":
                    traceback_data = content.get("traceback")
                    if traceback_data:
                        stderr_parts.append("\n".join(traceback_data))
                elif msg_type == "status" and content.get("execution_state") == "idle":
                    break

        # Drain shell channel
        while True:
            try:
                reply = client.get_shell_msg(timeout=0.1)
                if reply.get("parent_header", {}).get("msg_id") == msg_id:
                    break
            except queue.Empty:
                break

        # Combine output
        output = "".join(stdout_parts)
        if stderr_parts:
            output = (
                f"{output.rstrip()}\n{''.join(stderr_parts)}"
                if output
                else "".join(stderr_parts)
            )

        if output.strip():
            end_marker = (
                "[End previous output]"
                if execution_finished
                else "[End previous output - interrupted]"
            )
            return f"[Previous execution output]\n{output.rstrip()}\n{end_marker}\n"
        return ""

    def execute(self, code: str, continue_executing_on_timeout: bool = False) -> str:
        """Execute code in the kernel, returning combined stdout/stderr output."""
        # Drain any pending output from previous timed-out execution
        pending_output = self._drain_pending_output()

        client = self._client
        effective_timeout = self._default_timeout
        msg_id = client.execute(
            code, store_history=True, allow_stdin=False, stop_on_error=False
        )

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        while True:
            try:
                msg = client.get_iopub_msg(timeout=effective_timeout)
            except queue.Empty:
                if continue_executing_on_timeout:
                    # Deferred interruption: let kernel continue, interrupt on next execute()
                    self._pending_msg_id = msg_id
                    error_msg = "[TIMEOUT] Execution still running. Will drain remaining output on next call."
                else:
                    # Immediate interruption
                    self._km.interrupt_kernel()
                    error_msg = "[TIMEOUT] Execution interrupted."
                # Return partial output with timeout message
                partial_output = "".join(stdout_parts)
                if stderr_parts:
                    partial_output = (
                        f"{partial_output.rstrip()}\n{''.join(stderr_parts)}"
                        if partial_output
                        else "".join(stderr_parts)
                    )
                result = f"{partial_output.rstrip()}\n{error_msg}".lstrip()
                return f"{pending_output}{result}" if pending_output else result

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                traceback_data = content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = content.get("ename", "")
                    evalue = content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        # Drain the shell channel to capture final execution status
        while True:
            try:
                reply = client.get_shell_msg(timeout=effective_timeout)
            except queue.Empty:
                if continue_executing_on_timeout:
                    # Shell channel timeout - use deferred interruption
                    self._pending_msg_id = msg_id
                    error_msg = "[TIMEOUT] Execution still running. Will drain remaining output on next call."
                else:
                    # Immediate interruption
                    self._km.interrupt_kernel()
                    error_msg = "[TIMEOUT] Execution interrupted."
                partial_output = "".join(stdout_parts)
                if stderr_parts:
                    partial_output = (
                        f"{partial_output.rstrip()}\n{''.join(stderr_parts)}"
                        if partial_output
                        else "".join(stderr_parts)
                    )
                result = f"{partial_output.rstrip()}\n{error_msg}".lstrip()
                return f"{pending_output}{result}" if pending_output else result

            if reply.get("parent_header", {}).get("msg_id") == msg_id:
                break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            if stdout:
                stdout = f"{stdout.rstrip()}\n{stderr}"
            else:
                stdout = stderr

        if not stdout.strip():
            stdout = "[WARN] No output available. Use print() to output anything to stdout to receive the output"

        return f"{pending_output}{stdout}" if pending_output else stdout

    def close(self) -> None:
        import contextlib

        # Stop client channels first (closes ZMQ sockets)
        with contextlib.suppress(Exception):
            self._client.stop_channels()

        # Shutdown kernel process
        with contextlib.suppress(Exception):
            self._km.shutdown_kernel(now=True)

        # Cleanup kernel manager resources
        with contextlib.suppress(Exception):
            self._km.cleanup_resources()

        # Destroy ZMQ context - use destroy() instead of term() for immediate cleanup
        with contextlib.suppress(Exception):
            self._zmq_context.destroy(linger=0)

    def __del__(self) -> None:
        # Guard against Python shutdown (when sys.meta_path is None)
        import sys

        if sys.meta_path is not None:
            self.close()


def execute_python_code(session: LocalJupyterSession, script: str) -> str:
    """Execute Python code in a stateful Jupyter session."""
    try:
        return session.execute(script)
    except TimeoutError as exc:
        return f"[ERROR] {exc}"


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Token processing

# %% [code] {"execution":{"iopub.status.busy":"2025-11-24T08:13:52.195997Z","iopub.execute_input":"2025-11-24T08:13:52.196102Z","iopub.status.idle":"2025-11-24T08:14:00.409293Z","shell.execute_reply.started":"2025-11-24T08:13:52.196092Z","shell.execute_reply":"2025-11-24T08:14:00.408863Z"},"jupyter":{"outputs_hidden":false}}
import os
from openai import OpenAI, Stream
from openai.types import Completion

# Point the client to vLLM server (local on Kaggle, Modal otherwise)
if is_on_kaggle():
    os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:8000/v1"
else:
    os.environ["OPENAI_API_BASE"] = REMOTE_VLLM_URL
os.environ["OPENAI_API_KEY"] = "sk-local"  # any non-empty string

client: OpenAI = OpenAI(
    base_url=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"],
)

# Initialize openai-harmony encoding for GPT-OSS models
from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    SystemContent,
    load_harmony_encoding,
)

harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
stop_token_ids: list[int] = list(harmony_encoding.stop_tokens_for_assistant_actions())

# Python tool configuration for gpt-oss (extracted from gpt_oss.tools.python_docker.docker_tool)
# Using dangerously_use_local_jupyter backend - stateful execution via Jupyter kernel
from openai_harmony import Author, TextContent, ToolNamespaceConfig
import queue

# Stateful Python tool instruction (matches how the model was trained)
PYTHON_TOOL_INSTRUCTION = """
Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. Internet access for this session is disabled.
""".strip()

python_tool_config = ToolNamespaceConfig(
    name="python", description=PYTHON_TOOL_INSTRUCTION, tools=[]
)


def make_python_tool_response(output: str, channel: str | None = None) -> Message:
    """Create a tool response message for the Python tool."""
    content = TextContent(text=output)
    author = Author(role=Role.TOOL, name="python")
    message = Message(author=author, content=[content]).with_recipient("assistant")
    if channel:
        message = message.with_channel(channel)
    return message


def build_prompt_token_ids(
    system_content: str,
    user_content: str,
) -> list[int]:
    """Convert system and user content to token IDs using harmony format."""
    system_content_obj = SystemContent.new().with_reasoning_effort(ReasoningEffort.HIGH)
    system_content_obj = system_content_obj.with_tools(python_tool_config)
    system_message = Message.from_role_and_content(Role.SYSTEM, system_content_obj)
    developer_message = Message.from_role_and_content(
        Role.DEVELOPER, DeveloperContent.new().with_instructions(system_content)
    )
    user_message = Message.from_role_and_content(Role.USER, user_content)
    convo = Conversation.from_messages(
        [system_message, developer_message, user_message]
    )
    return list(
        harmony_encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    )


def append_user_turn_token_ids(
    all_token_ids: list[int], user_content: str
) -> list[int]:
    """Append a new user turn to the token IDs."""
    new_user_message = Message.from_role_and_content(Role.USER, user_content)
    user_tokens = list(
        harmony_encoding.render_conversation_for_completion(
            Conversation.from_messages([new_user_message]), Role.ASSISTANT
        )
    )
    return all_token_ids + user_tokens


import time


def append_tool_response_token_ids(
    all_token_ids: list[int], tool_response: Message
) -> list[int]:
    """Append a tool response to the token IDs."""
    tool_tokens = list(
        harmony_encoding.render_conversation_for_completion(
            Conversation.from_messages([tool_response]), Role.ASSISTANT
        )
    )
    return all_token_ids + tool_tokens


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T11:23:07.544491Z","iopub.execute_input":"2025-11-25T11:23:07.544919Z","iopub.status.idle":"2025-11-25T11:23:07.550311Z","shell.execute_reply.started":"2025-11-25T11:23:07.5449Z","shell.execute_reply":"2025-11-25T11:23:07.549876Z"}}
from cachetools import cached, TTLCache
import os
import time
import requests


@cached(cache=TTLCache(maxsize=50, ttl=20))
def get_gpu_kv_cache_usage(question_id: str | None) -> float:
    # question_id is used as cache key
    # Parse vLLM /metrics endpoint using configured base URL
    try:
        base_url = os.environ["OPENAI_API_BASE"]
        # Remove /v1 suffix to get metrics endpoint
        metrics_url = base_url.replace("/v1", "/metrics")
        resp = requests.get(metrics_url, timeout=5)
        for line in resp.text.split("\n"):
            # vllm:kv_cache_usage_perc is the metric for KV cache usage
            if line.startswith("vllm:kv_cache_usage_perc"):
                value = float(line.split()[-1])
                return value * 100  # convert to percentage
    except (requests.RequestException, ValueError, IndexError) as e:
        print(f"get_gpu_kv_cache_usage: {e}")
        pass
    return -1


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:22:55.289753Z","iopub.execute_input":"2025-11-24T08:22:55.289878Z","iopub.status.idle":"2025-11-24T08:23:00.176618Z","shell.execute_reply.started":"2025-11-24T08:22:55.289861Z","shell.execute_reply":"2025-11-24T08:23:00.176183Z"}}
if is_on_kaggle_interactive():
    test_prompt_ids = build_prompt_token_ids(
        system_content="Reply your answer in \\boxed{}",
        user_content="How many r are there in strawberry?",
    )
    resp: Completion = client.completions.create(
        model="vllm-model",
        prompt=test_prompt_ids,
        max_tokens=1024,
        temperature=1.0,
        extra_body=dict(
            min_p=0.02,
            stop_token_ids=stop_token_ids,
            return_token_ids=True,
        ),
    )

    print("Token IDs:", resp.choices[0].token_ids)  # type: ignore[attr-defined]

    print(resp.choices[0].text)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Text processing


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:23:00.326586Z","iopub.execute_input":"2025-11-24T08:23:00.326712Z","iopub.status.idle":"2025-11-24T08:23:00.333663Z","shell.execute_reply.started":"2025-11-24T08:23:00.326702Z","shell.execute_reply":"2025-11-24T08:23:00.333234Z"}}
def extract_boxed_text(text: str) -> str:
    """Extract text inside \\boxed{} from LaTeX-formatted text"""
    import re

    pattern: str = r"oxed{(.*?)}"
    matches: list[str] = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""


def is_valid_answer_string(text: str) -> bool:
    try:
        if int(text) == float(text):
            if 0 <= int(text) <= 99_999:
                # now AIMO answers no longer need modulo
                return True
    except Exception:
        pass
    return False


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:23:00.334105Z","iopub.execute_input":"2025-11-24T08:23:00.334228Z","iopub.status.idle":"2025-11-24T08:23:00.341278Z","shell.execute_reply.started":"2025-11-24T08:23:00.334218Z","shell.execute_reply":"2025-11-24T08:23:00.340907Z"}}

completed_question_ids: set[str] = set()
question_id_to_solver_to_length: dict[str, dict[int, int]] = {"": {}}
question_id_to_solver_to_answer: dict[str, dict[int, int]] = {"": {}}


def vote_answer(question_id: str, force_answer: bool = False) -> int | None:
    """Majority voting with larger answer as tiebreaker."""
    solver_to_answer = question_id_to_solver_to_answer[question_id]
    answer_counter = Counter(solver_to_answer.values())
    if force_answer and not answer_counter:
        print(f"Current GPU usage {get_gpu_kv_cache_usage(question_id):.1f}")
        print("force_answer=True but no answer recorded")
        completed_question_ids.add(question_id)
        return 12453

    # Sort by (count, value) - higher count wins, larger value breaks ties
    sorted_answers = sorted(
        answer_counter.items(), key=lambda x: (x[1], x[0]), reverse=True
    )

    if force_answer:
        print(f"Votes over {sum(answer_counter.values())} attempts")
        print(f"Current GPU usage {get_gpu_kv_cache_usage(question_id):.1f}")
        for value, count in sorted_answers:
            print(f"{value:10}   {count:8d}")
        return sorted_answers[0][0]

    # Mark as completed as long as there's any answer
    if sorted_answers:
        completed_question_ids.add(question_id)

    return None


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Generate solution


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:26:53.213762Z","iopub.execute_input":"2025-11-24T08:26:53.214218Z","iopub.status.idle":"2025-11-24T08:26:53.221603Z","shell.execute_reply.started":"2025-11-24T08:26:53.214205Z","shell.execute_reply":"2025-11-24T08:26:53.221218Z"}}
def rollout_given_state(
    all_token_ids: list[int],
    jupyter_session: LocalJupyterSession,
    should_stop: Callable[[], bool] = lambda: False,
) -> list[int]:
    """
    Rollout function that generates tokens and handles tool calls given existing state.

    Args:
        all_token_ids: Initial prompt token IDs
        jupyter_session: Jupyter session for executing Python code (with existing state)
        should_stop: Callback to check if generation should stop early

    Returns:
        Final token IDs after rollout completes
    """
    tool_call_count = 0

    for iteration in range(64):
        # Loop until we get an answer
        while True:
            # Loop to handle tool calls within each iteration
            text_response = ""
            breaking = False

            if len(all_token_ids) > max_model_len - 8192 * 3:
                return all_token_ids

            # Use streaming with completions API
            stream: Stream[Completion] = client.completions.create(
                model="vllm-model",
                prompt=all_token_ids,
                max_tokens=max_model_len - len(all_token_ids) - 8192 * 2,
                temperature=1.0,
                stream=True,
                extra_body=dict(
                    min_p=0.02,
                    stop_token_ids=stop_token_ids,
                    return_token_ids=True,
                ),
            )

            # Use StreamableParser to process streaming tokens
            stream_parser = StreamableParser(harmony_encoding, role=Role.ASSISTANT)

            for chunk in stream:
                # Get token IDs from the chunk (vLLM extension)
                chunk_token_ids = getattr(chunk.choices[0], "token_ids", None)
                if chunk_token_ids:
                    # Process tokens through harmony parser for text
                    for token_id in chunk_token_ids:
                        all_token_ids.append(token_id)
                        stream_parser.process(token_id)

                # Also get text directly if available
                chunk_text = chunk.choices[0].text
                if chunk_text:
                    text_response += chunk_text

                # Check finish_reason to see if generation completed naturally
                finish_reason = chunk.choices[0].finish_reason
                if finish_reason:
                    break

                if should_stop():
                    breaking = True
                    break

                # Stop early if we found a valid boxed answer
                if (
                    chunk_text
                    and "}" in chunk_text
                    and is_valid_answer_string(extract_boxed_text(text_response))
                ):
                    break

            # Append response token IDs to prompt for multi-turn
            stream.close()

            if breaking:
                break

            # Check if the last parsed message is a tool call
            # After streaming, parser.messages contains the parsed Message objects
            parsed_messages = stream_parser.messages
            if parsed_messages:
                last_message = parsed_messages[-1]
                if (
                    last_message.recipient is not None
                    and last_message.recipient.startswith("python")
                ):
                    tool_call_count += 1
                    # Extract Python code from the message content
                    python_code = ""
                    if last_message.content:
                        first_block = last_message.content[0]
                        if isinstance(first_block, TextContent):
                            python_code = first_block.text
                    if python_code:
                        print(
                            f"iteration {iteration:01d} tool {tool_call_count:02d} token {len(all_token_ids):05d}",
                            flush=True,
                        )
                        # Execute the code using stateful Jupyter session
                        output = execute_python_code(jupyter_session, python_code)
                        if len(output) > 12_000:
                            output = output[:5000] + "(truncated)" + output[-5000:]

                        # Create python tool response message
                        tool_response = make_python_tool_response(
                            output, channel=last_message.channel
                        )
                        # Append tool response tokens
                        all_token_ids = append_tool_response_token_ids(
                            all_token_ids, tool_response
                        )
                        continue

            # Exit inner loop
            break

        if breaking:
            break

        boxed_text = extract_boxed_text(text_response)
        user_follow_up = None
        print(
            f"iteration {iteration:01d} tool {tool_call_count:02d} token {len(all_token_ids):05d}"
        )
        if not is_valid_answer_string(boxed_text):
            print("follow-up - ask boxed answer")
            user_follow_up = "The answer is expected to be an integer between 0 and 99999 inclusive. Place your final answer in \\boxed{}. Do not guess the answer."
        else:
            # answer found, no issues detected, proceed to answering
            break

        if user_follow_up:
            # Append response and user follow-up as token IDs
            all_token_ids = append_user_turn_token_ids(all_token_ids, user_follow_up)

    return all_token_ids


starter_code = """
import random
import numpy as np
import sympy as sp
random.seed(42)
np.random.seed(42)
"""


def rollout(
    all_token_ids: list[int],
    should_stop: Callable[[], bool] = lambda: False,
) -> list[int]:
    """
    Rollout function that creates session state and generates tokens.

    Args:
        all_token_ids: Initial prompt token IDs
        should_stop: Callback to check if generation should stop early

    Returns:
        Final token IDs after rollout completes
    """
    jupyter_session = None
    try:
        jupyter_session = LocalJupyterSession()
        execute_python_code(jupyter_session, starter_code)
        return rollout_given_state(
            all_token_ids=all_token_ids,
            jupyter_session=jupyter_session,
            should_stop=should_stop,
        )
    except RuntimeError as e:
        print(f"rollout: {e}")
        return all_token_ids
    finally:
        if jupyter_session is not None:
            print("Cleaning up Jupyter session")
            jupyter_session.close()


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:26:53.213762Z","iopub.execute_input":"2025-11-24T08:26:53.214218Z","iopub.status.idle":"2025-11-24T08:26:53.221603Z","shell.execute_reply.started":"2025-11-24T08:26:53.214205Z","shell.execute_reply":"2025-11-24T08:26:53.221218Z"}}
def solve_single(
    question_text: str, question_id: str = "", solver_index: int = 0
) -> str:
    """
    Single solution attempt with side effects (file saving, voting).
    """
    if question_id in completed_question_ids:
        return ""
    if time.time() >= cutoff_times[-1]:
        return ""

    # Build initial prompt as token IDs with Python tool enabled
    all_token_ids: list[int] = build_prompt_token_ids(
        system_content="You will solve the problem and return the final answer in \\boxed{}. The answer is expected to be an integer between 0 and 99999, inclusive. Do not guess the answer, unless specifically given permission to.",
        user_content=question_text,
    )

    # Define stop condition callback
    def should_stop() -> bool:
        if question_id in completed_question_ids:
            return True
        if time.time() >= cutoff_times[-1]:
            return True
        solver_index_to_gpu_threshold = [95, 90, 85, 80, 75, 70]
        solver_index_mod = solver_index % len(solver_index_to_gpu_threshold)
        gpu_kv_cache_usage = get_gpu_kv_cache_usage(question_id)
        if gpu_kv_cache_usage > solver_index_to_gpu_threshold[solver_index_mod]:
            print(f"Terminated Solver {solver_index} at {gpu_kv_cache_usage:.1f}")
            return True
        return False

    # Run the rollout
    all_token_ids = rollout(
        all_token_ids=all_token_ids,
        should_stop=should_stop,
    )

    # Process results and handle side effects
    detokenized_text = harmony_encoding.decode(all_token_ids)
    boxed_text = extract_boxed_text(detokenized_text)

    if question_id and all_token_ids:
        answer_suffix = "NA"
        if is_valid_answer_string(boxed_text):
            answer_suffix = f"{boxed_text}"
        total_tokens = len(all_token_ids)
        question_id_to_solver_to_length[question_id][solver_index] = total_tokens
        # Count tool calls from the token stream
        tool_call_count = detokenized_text.count("to=python code")
        base_path = f"{SOLUTIONS_DIR}/{question_id}/{solver_index:02d}-{total_tokens:05d}-{tool_call_count:02d}-{answer_suffix}"
        # Save full stream as token IDs (one token ID per line)
        with open(f"{base_path}-tokens.txt", "w") as f:
            for token_id in all_token_ids:
                f.write(f"{token_id}\n")
        # Save detokenized full stream for readability
        with open(f"{base_path}-text.txt", "w") as f:
            f.write(detokenized_text)

    if is_valid_answer_string(boxed_text):
        question_id_to_solver_to_answer[question_id][solver_index] = int(boxed_text)
        vote_answer(question_id)

    return boxed_text


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:26:54.553208Z","iopub.execute_input":"2025-11-24T08:26:54.553692Z","iopub.status.idle":"2025-11-24T08:27:03.475341Z","shell.execute_reply.started":"2025-11-24T08:26:54.553671Z","shell.execute_reply":"2025-11-24T08:27:03.474837Z"}}
if is_on_kaggle_interactive():
    solve_single("What is 1+1?")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:27:06.296266Z","iopub.execute_input":"2025-11-24T08:27:06.296867Z","iopub.status.idle":"2025-11-24T08:27:06.300859Z","shell.execute_reply.started":"2025-11-24T08:27:06.29685Z","shell.execute_reply":"2025-11-24T08:27:06.300371Z"}}
import threading


def solve(question_text: str, question_id: str = "") -> int:
    print(f"Processing {question_id}")
    question_start_time = time.time()
    time_available = cutoff_times[-1] - question_start_time if cutoff_times else 0
    print(f"time_available {time_available:.1f}s")
    os.makedirs(f"{SOLUTIONS_DIR}/{question_id}", exist_ok=True)
    question_id_to_solver_to_length[question_id] = {}
    question_id_to_solver_to_answer[question_id] = {}
    completed_question_ids.discard(question_id)  # just in case question_id collides

    if question_id and time.time() > cutoff_times[-1]:
        print("timeout did not solve")
        return 12314

    get_gpu_kv_cache_usage(
        question_id
    )  # run once to prevent running in the first batch of execution

    # Start solver threads
    # I suspect that init LocalJupyterSession(timeout=10.0) can stall
    for solver_index in range(num_generations):
        threading.Thread(
            target=solve_single,
            args=(question_text, question_id, solver_index),
        ).start()

    # Poll every second until completed or time runs out
    for _ in range(int(time_available) + 1):
        if question_id in completed_question_ids:
            break
        time.sleep(1)
    else:
        print("Solve timeout - continuing to submission")

    final_answer = vote_answer(question_id, force_answer=True)
    assert final_answer is not None

    # Save stats
    time_taken = time.time() - question_start_time
    save_stats(
        question_id=question_id,
        final_answer=final_answer,
        time_taken=time_taken,
        time_available=time_available,
        solver_to_length=question_id_to_solver_to_length[question_id],
        solver_to_answer=question_id_to_solver_to_answer[question_id],
    )
    print(f"Submitting {final_answer} for {question_id} in {time_taken:.1f}s")

    return final_answer


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:27:06.481067Z","iopub.execute_input":"2025-11-24T08:27:06.481501Z","iopub.status.idle":"2025-11-24T08:27:19.404564Z","shell.execute_reply.started":"2025-11-24T08:27:06.481486Z","shell.execute_reply":"2025-11-24T08:27:19.404121Z"}}
if is_on_kaggle_interactive():
    solve("What is 1+1?")

if not is_on_kaggle() and __name__ == "__main__":
    # stack trace without inference_server is easier to debug
    print("solving")

    question_id = "dd7f5e"
    question_text = """
Let $\\mathcal{F}$ be the set of functions $\\alpha \\colon \\mathbb{Z}\\to \\mathbb{Z}$ for which there are only finitely many $n \\in \\mathbb{Z}$ such that $\\alpha(n) \\neq 0$. 

For two functions $\\alpha$ and $\\beta$ in $\\mathcal{F}$, define their product $\\alpha\\star\\beta$ to be $\\sum\\limits_{n\\in\\mathbb{Z}} \\alpha(n)\\cdot \\beta(n)$. Also, for $n\\in\\mathbb{Z}$, define a shift operator $S_n \\colon \\mathcal{F}\\to \\mathcal{F}$ by $S_n(\\alpha)(t)=\\alpha(t+n)$ for all $t \\in \\mathbb{Z}$.

A function $\\alpha \\in \\mathcal{F}$ is called \\emph{shifty} if 
\\begin{itemize}
    \\item $\\alpha(m)=0$ for all integers $m<0$ and $m>8$ and
    \\item There exists $\\beta \\in \\mathcal{F}$ and integers $k \\neq l$ such that for all $n \\in \\mathbb{Z}$
    \\begin{equation*}
        S_n(\\alpha)\\star\\beta =
        \\begin{cases}
            1 & n \\in \\{k,l\\} \\\\
            0 & n \\not \\in \\{k,l\\}
        \\end{cases}
        \\; .
    \\end{equation*}
\\end{itemize}
How many shifty functions are there in $\\mathcal{F}$?
""".strip()

    #     question_id = "92ba6a"
    #     question_text = """
    # Alice and Bob are each holding some integer number of sweets. Alice says to Bob: ``If we each added the number of sweets we're holding to our (positive integer) age, my answer would be double yours. If we took the product, then my answer would be four times yours.'' Bob replies: ``Why don't you give me five of your sweets because then both our sum and product would be equal.'' What is the product of Alice and Bob's ages?
    # """.strip()

    #     question_id = "641659"
    #     question_text = """
    # Let $ABC$ be a triangle with $AB \\neq AC$, circumcircle $\\Omega$, and incircle $\\omega$. Let the contact points of $\\omega$ with $BC$, $CA$, and $AB$ be $D$, $E$, and $F$, respectively. Let the circumcircle of $AFE$ meet $\\Omega$ at $K$ and let the reflection of $K$ in $EF$ be $K'$. Let $N$ denote the foot of the perpendicular from $D$ to $EF$. The circle tangent to line $BN$ and passing through $B$ and $K$ intersects $BC$ again at $T \\neq B$.

    # Let sequence $(F_n)_{n \\geq 0}$ be defined by $F_0 = 0$, $F_1 = 1$ and for $n \\geq 2$, $F_n = F_{n-1} + F_{n-2}$. Call $ABC$ $n$\\emph{-tastic} if $BD = F_n$, $CD = F_{n+1}$, and $KNK'B$ is cyclic. Across all $n$-tastic triangles, let $a_n$ denote the maximum possible value of $\\frac{CT \\cdot NB}{BT \\cdot NE}$. Let $\\alpha$ denote the smallest real number such that for all sufficiently large $n$, $a_{2n} < \\alpha$. Given that $\\alpha = p + \\sqrt{q}$ for rationals $p$ and $q$, what is the remainder when $\\left\\lfloor p^{q^p} \\right\\rfloor$ is divided by $99991$?
    # """.strip()

    #     question_id = "86e8e5"
    #     question_text = """
    # Let $n \\geq 6$ be a positive integer. We call a positive integer $n$-Norwegian if it has three distinct positive divisors whose sum is equal to $n$. Let $f(n)$ denote the smallest $n$-Norwegian positive integer. Let $M=3^{2025!}$ and for a non-negative integer $c$ define
    # \\begin{equation*}
    #     g(c)=\\frac{1}{2025!}\\left\\lfloor \\frac{2025! f(M+c)}{M}\\right\\rfloor.
    # \\end{equation*}
    # We can write
    # \\begin{equation*}
    #     g(0)+g(4M)+g(1848374)+g(10162574)+g(265710644)+g(44636594)=\\frac{p}{q}
    # \\end{equation*}
    # where $p$ and $q$ are coprime positive integers. What is the remainder when $p+q$ is divided by $99991$?
    # """.strip()

    os.makedirs(f"{SOLUTIONS_DIR}/{question_id}", exist_ok=True)
    solve(question_text, question_id)
    exit()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Submission server

# %% [code] {"_kg_hide-output":true,"_kg_hide-input":false,"jupyter":{"outputs_hidden":false},"execution":{"execution_failed":"2025-11-24T02:04:57.769Z"}}
import os

import kaggle_evaluation.aimo_3_inference_server
import pandas as pd
import polars as pl

if is_on_kaggle():
    df = pd.read_csv(
        "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
    ).drop("answer", axis=1)

    replication_count = min(replication_count_for_commit_runs, 50 // len(df))

    dfs = []
    for replication_idx in range(replication_count):
        df_copy = df.copy()
        df_copy["id"] = df_copy["id"] + f"_{replication_idx}"
        dfs.append(df_copy)
    pd.concat(dfs, ignore_index=True).to_csv("reference.csv", index=False)


# %% [code] {"_kg_hide-output":true,"_kg_hide-input":false,"jupyter":{"outputs_hidden":false},"execution":{"execution_failed":"2025-11-24T02:04:57.769Z"}}


# Replace this function with your inference code.
# The function should return a single integer between 0 and 99999, inclusive.
def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # Unpack values
    question_id: str = id_.item(0)
    question_text: str = problem.item(0)

    if not run_all_questions_on_kaggle:  # should be ignored for submissions
        if is_on_kaggle_commit():
            # to only run for hard problems
            if not (
                "shifty" in question_text  # noqa: E713
                or "tournament" in question_text
                or "KNK" in question_text
                or "Norwe" in question_text
                or "Alice" in question_text
            ):
                print("on kaggle commit, skipping question")
                # not popping cutoff_times
                return pl.DataFrame({"id": id_, "answer": 12315})

    # Make a prediction
    prediction = solve(question_text, question_id=question_id)
    completed_question_ids.add(question_id)
    cutoff_times.pop()
    return pl.DataFrame({"id": id_, "answer": prediction})


inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(
    predict  # type: ignore[arg-type]
)

print("Starting submission server")
if __name__ == "__main__":
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(("reference.csv",))

# %% [code] {"_kg_hide-input":false,"jupyter":{"outputs_hidden":false}}
