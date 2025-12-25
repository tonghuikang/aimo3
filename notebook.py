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
serve_vllm_on_kaggle = True
run_all_questions = False  # ignored for submissions

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os
import time
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
if not serve_vllm_on_kaggle or not is_on_kaggle():
    REMOTE_VLLM_URL = secrets.get_secret("REMOTE_VLLM_URL")


# Some debugger warning on Kaggle
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# %% [code] {"_kg_hide-output":true,"jupyter":{"outputs_hidden":false}}
# print settings
print(f"{is_on_kaggle()=}")
print(f"{is_on_kaggle_interactive()=}")
print(f"{is_on_kaggle_commit()=}")
print(f"{serve_vllm_on_kaggle=}")
print(f"{run_all_questions=}")
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

# %% [code] {"jupyter":{"outputs_hidden":false}}
import torch
import numpy as np


cutoff_times = [
    int(x) for x in np.linspace(final_cutoff_time, start_time + 30 * 60, 50 + 1)
]  # generous allowance at the start
cutoff_times.pop()

from datetime import datetime

RUN_DIR = f"runs/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
SOLUTIONS_DIR = f"{RUN_DIR}/solutions"
os.makedirs(SOLUTIONS_DIR, exist_ok=True)

if is_on_kaggle():
    if serve_vllm_on_kaggle:
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() == 1
    else:
        # Check internet access is available when using remote inference
        import urllib.request
        from urllib.error import URLError

        try:
            urllib.request.urlopen("https://modal.com", timeout=5)
            print("Internet access confirmed")
        except (URLError, TimeoutError) as e:
            raise RuntimeError(
                "Internet access required when serve_vllm_on_kaggle=False"
            ) from e

        # Check that you are not wasting Kaggle GPUs
        assert not torch.cuda.is_available()
        assert torch.cuda.device_count() == 0

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
if is_on_kaggle() and serve_vllm_on_kaggle:
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


if is_on_kaggle_interactive():
    # cannot await client on submission
    # because inference server needs to start within 15 minutes
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
        self._client.wait_for_ready(timeout=self._default_timeout)
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

    def execute(self, code: str, timeout: float | None = None) -> str:
        """Execute code in the kernel, returning combined stdout/stderr output."""
        # Drain any pending output from previous timed-out execution
        pending_output = self._drain_pending_output()

        client = self._client
        effective_timeout = timeout or self._default_timeout
        msg_id = client.execute(
            code, store_history=True, allow_stdin=False, stop_on_error=False
        )

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        while True:
            try:
                msg = client.get_iopub_msg(timeout=effective_timeout)
            except queue.Empty:
                # Deferred interruption: let kernel continue, interrupt on next execute()
                self._pending_msg_id = msg_id
                # Return partial output with timeout message
                partial_output = "".join(stdout_parts)
                if stderr_parts:
                    partial_output = (
                        f"{partial_output.rstrip()}\n{''.join(stderr_parts)}"
                        if partial_output
                        else "".join(stderr_parts)
                    )
                error_msg = "[TIMEOUT] Execution still running. Will drain remaining output on next call."
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
                # Shell channel timeout - also use deferred interruption
                self._pending_msg_id = msg_id
                partial_output = "".join(stdout_parts)
                if stderr_parts:
                    partial_output = (
                        f"{partial_output.rstrip()}\n{''.join(stderr_parts)}"
                        if partial_output
                        else "".join(stderr_parts)
                    )
                error_msg = "[TIMEOUT] Execution still running. Will drain remaining output on next call."
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


def execute_python_code(
    session: LocalJupyterSession, script: str, timeout: float = 10.0
) -> str:
    """Execute Python code in a stateful Jupyter session."""
    try:
        return session.execute(script, timeout=timeout)
    except TimeoutError as exc:
        return f"[ERROR] {exc}"


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Token processing

# %% [code] {"execution":{"iopub.status.busy":"2025-11-24T08:13:52.195997Z","iopub.execute_input":"2025-11-24T08:13:52.196102Z","iopub.status.idle":"2025-11-24T08:14:00.409293Z","shell.execute_reply.started":"2025-11-24T08:13:52.196092Z","shell.execute_reply":"2025-11-24T08:14:00.408863Z"},"jupyter":{"outputs_hidden":false}}
import os
from openai import OpenAI, Stream
from openai.types import Completion

# Point the client to vLLM server (local on Kaggle, Modal otherwise)
if is_on_kaggle() and serve_vllm_on_kaggle:
    os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:8000/v1"
else:
    os.environ["OPENAI_API_BASE"] = REMOTE_VLLM_URL
    if is_on_kaggle():
        # openai_harmony uses TIKTOKEN_ENCODINGS_BASE to read pre-downloaded files
        os.environ["TIKTOKEN_ENCODINGS_BASE"] = (
            "/kaggle/usr/lib/pip_install_aimo3_1/tiktoken_encodings"
        )
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
    message = Message(
        author=author,
        content=[content],
    ).with_recipient("assistant")
    if channel:
        message = message.with_channel(channel)
    return message


def build_prompt_token_ids(
    system_content: str,
    user_content: str,
    reasoning_effort: ReasoningEffort,
    enable_python_tool: bool = False,
) -> list[int]:
    """Convert system and user content to token IDs using harmony format."""
    system_content_obj = SystemContent.new().with_reasoning_effort(reasoning_effort)
    if enable_python_tool:
        # Enable Python tool using with_tools() for stateless mode
        system_content_obj = system_content_obj.with_tools(python_tool_config)
    system_message = Message.from_role_and_content(
        Role.SYSTEM,
        system_content_obj,
    )
    developer_message = Message.from_role_and_content(
        Role.DEVELOPER,
        DeveloperContent.new().with_instructions(system_content),
    )
    user_message = Message.from_role_and_content(
        Role.USER,
        user_content,
    )
    convo = Conversation.from_messages(
        [system_message, developer_message, user_message]
    )
    return list(
        harmony_encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    )


def append_user_turn_token_ids(
    prompt_ids: list[int], response_ids: list[int], user_content: str
) -> list[int]:
    """Append response token IDs and a new user turn to the prompt."""
    all_tokens = prompt_ids + response_ids
    # Build new user message and render to tokens
    new_user_message = Message.from_role_and_content(Role.USER, user_content)
    user_tokens = list(
        harmony_encoding.render_conversation_for_completion(
            Conversation.from_messages([new_user_message]), Role.ASSISTANT
        )
    )
    # Combine: previous prompt + response + user turn tokens
    return all_tokens + user_tokens


import time


def append_tool_response_token_ids(
    prompt_ids: list[int], response_ids: list[int], tool_response: Message
) -> list[int]:
    """Append response token IDs and a tool response to the prompt."""
    all_tokens = prompt_ids + response_ids
    # Render tool response message to tokens
    tool_tokens = list(
        harmony_encoding.render_conversation_for_completion(
            Conversation.from_messages([tool_response]), Role.ASSISTANT
        )
    )
    return all_tokens + tool_tokens


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T11:23:07.544491Z","iopub.execute_input":"2025-11-25T11:23:07.544919Z","iopub.status.idle":"2025-11-25T11:23:07.550311Z","shell.execute_reply.started":"2025-11-25T11:23:07.5449Z","shell.execute_reply":"2025-11-25T11:23:07.549876Z"}}
from cachetools import cached, TTLCache
import os
import time
import requests


@cached(cache=TTLCache(maxsize=50, ttl=20))
def get_gpu_kv_cache_usage(question_id: str | None = None) -> float:
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
    except (requests.RequestException, ValueError, IndexError):
        pass
    return -1


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:22:55.289753Z","iopub.execute_input":"2025-11-24T08:22:55.289878Z","iopub.status.idle":"2025-11-24T08:23:00.176618Z","shell.execute_reply.started":"2025-11-24T08:22:55.289861Z","shell.execute_reply":"2025-11-24T08:23:00.176183Z"}}
if is_on_kaggle_interactive():
    test_prompt_ids = build_prompt_token_ids(
        system_content="Reply your answer in \\boxed{}",
        user_content="How many r are there in strawberry?",
        reasoning_effort=ReasoningEffort.HIGH,
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
from collections import Counter

completed_question_ids: set[str] = set()
question_id_to_counter: dict[str, Counter] = {"": Counter()}


import math
from collections import Counter


def vote_answer(question_id: str, force_answer: bool = False) -> int | None:
    # reads counter from global
    counter = question_id_to_counter[question_id]
    if force_answer and not counter:
        print(f"Current GPU usage {get_gpu_kv_cache_usage()}")
        print("force_answer=True but no answer recorded")
        completed_question_ids.add(question_id)
        return 12453

    # voting mechanism
    modified_counter: dict[int, float] = {}
    for value, count in counter.items():
        # re-weighted because smaller answers seems to be wrong
        # "1.25 +" because log(1) = 0
        modified_counter[value] = (
            modified_counter.get(value, 0.0) + math.log(1.25 + abs(value)) * count
        )

    total_score = sum(modified_counter.values())
    score_list = sorted(
        (score, counter[value], value) for value, score in modified_counter.items()
    )
    if force_answer:
        print(f"score_list | {total_score:8.1f} over {sum(counter.values())} attempts")
        print(f"Current GPU usage {get_gpu_kv_cache_usage()}")
        for score, count, value in score_list[::-1]:
            print(f"{value:10}   {score:8.1f} {count:8d}")
        return score_list[-1][-1]
    if score_list[-1][0] > max(1, total_score / (2 + math.log(1 + total_score))):
        if len(score_list) == 1:
            completed_question_ids.add(question_id)
        else:
            if score_list[-1][0] - score_list[-2][0] > 1:
                # win by a certain number of points at least
                completed_question_ids.add(question_id)
    return None


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Generate solution


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:26:53.213762Z","iopub.execute_input":"2025-11-24T08:26:53.214218Z","iopub.status.idle":"2025-11-24T08:26:53.221603Z","shell.execute_reply.started":"2025-11-24T08:26:53.214205Z","shell.execute_reply":"2025-11-24T08:26:53.221218Z"}}
def generate_solution(
    question_text: str, question_id: str = "", solution_index: int = 0
) -> str:
    if question_id in completed_question_ids:
        return ""
    if time.time() >= cutoff_times[-1]:
        return ""

    # Create a dedicated Jupyter session for this solution attempt
    # Each generate_solution call gets its own isolated session
    jupyter_session: LocalJupyterSession | None = None

    try:
        # Build initial prompt as token IDs with Python tool enabled
        prompt_ids: list[int] = build_prompt_token_ids(
            system_content="You will solve the problem and return the final answer in \\boxed{}. The answer is expected to be an integer between 0 and 99999, inclusive. Do not guess the answer, unless specifically given permission to.",
            user_content=question_text,
            reasoning_effort=ReasoningEffort.HIGH,
            enable_python_tool=True,  # Enable Python tool for code execution
        )

        all_token_ids: list[int] = prompt_ids.copy()
        generation_idx = 0
        tool_call_count = 0

        for iteration in range(3):  # guess at 90, guess at 30
            # Inner loop to handle tool calls within each iteration
            while True:
                response_ids: list[int] = []
                text_response = ""
                breaking = False

                # Use streaming with completions API
                stream: Stream[Completion] = client.completions.create(
                    model="vllm-model",
                    prompt=prompt_ids,
                    max_tokens=32768,
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
                    generation_idx += 1
                    # Get token IDs from the chunk (vLLM extension)
                    chunk_token_ids = getattr(chunk.choices[0], "token_ids", None)
                    if chunk_token_ids:
                        response_ids.extend(chunk_token_ids)
                        # Process tokens through harmony parser for text
                        for token_id in chunk_token_ids:
                            stream_parser.process(token_id)

                    # Also get text directly if available
                    chunk_text = chunk.choices[0].text
                    if chunk_text:
                        text_response += chunk_text

                    # Check finish_reason to see if generation completed naturally
                    finish_reason = chunk.choices[0].finish_reason
                    if finish_reason:
                        break

                    if question_id in completed_question_ids:
                        # stop generating if we have finalized on an answer
                        breaking = True
                    if time.time() >= cutoff_times[-1]:
                        breaking = True
                    if (
                        get_gpu_kv_cache_usage(question_id) > 70
                        and int(get_gpu_kv_cache_usage(question_id) + solution_index)
                        % num_generations
                        == 0
                    ):
                        print("terminated to prevent excessive GPU usage")
                        breaking = True
                    if breaking:
                        break
                    # instead of breaking = True, so we want to inject instructions for these conditions
                    if (
                        chunk_text
                        and "}" in chunk_text
                        and is_valid_answer_string(extract_boxed_text(text_response))
                    ):
                        break
                    if iteration == 0 and cutoff_times[-1] - time.time() < 90:
                        break
                    if iteration == 1 and cutoff_times[-1] - time.time() < 30:
                        break

                # Append response token IDs to prompt for multi-turn
                all_token_ids.extend(response_ids)
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
                                f"solution {solution_index:01d} iteration {iteration:01d} tool {tool_call_count:02d} token {len(all_token_ids):05d}"
                            )
                            # Lazily create the Jupyter session on first tool call
                            if jupyter_session is None:
                                jupyter_session = LocalJupyterSession(timeout=10.0)
                            # Execute the code using stateful Jupyter session
                            output = execute_python_code(
                                jupyter_session, python_code, timeout=10
                            )
                            # Create tool response message
                            tool_response = make_python_tool_response(
                                output, channel=last_message.channel
                            )
                            # Append tool response to prompt
                            new_prompt_ids = append_tool_response_token_ids(
                                prompt_ids, response_ids, tool_response
                            )
                            # Track the new tokens added (tool response portion)
                            added_tokens = new_prompt_ids[
                                len(prompt_ids) + len(response_ids) :
                            ]
                            all_token_ids.extend(added_tokens)
                            prompt_ids = new_prompt_ids
                            # Continue the inner loop to get next generation
                            continue

                # Exit inner loop
                break

            if breaking:
                break

            boxed_text = extract_boxed_text(text_response)
            user_follow_up = None
            print(
                f"solution {solution_index:01d} iteration {iteration:01d} tool {tool_call_count:02d} token {len(all_token_ids):05d}"
            )
            if not is_valid_answer_string(extract_boxed_text(text_response)):
                if iteration == 0 and cutoff_times[-1] - time.time() < 90:
                    print("follow-up - guess answer soon")
                    user_follow_up = "The answer is expected to be an integer between 0 and 99999 inclusive. Please make an educated guess (e.g. lower bound, upper bound, current best answer, ...) and put your your final answer in \\boxed{}."
                elif iteration == 1 and cutoff_times[-1] - time.time() < 30:
                    print("follow-up - guess answer now")
                    user_follow_up = "The answer is expected to be an integer between 0 and 99999 inclusive. Please guess a reasonable answer and put in \\boxed{} as soon as possible."
                else:
                    print("follow-up - ask boxed answer")
                    user_follow_up = "The answer is expected to be an integer between 0 and 99999 inclusive. Place your final answer in \\boxed{}. Do not guess the answer."
            elif int(boxed_text) <= 10:
                print("follow-up - are you sure")
                user_follow_up = (
                    "Are you sure that is the answer? Do not guess the answer."
                )
            elif iteration == 0 and len(all_token_ids) < 3200:
                print("follow-up - have you verified")
                user_follow_up = "Have you verified your answer?"
            else:
                # answer found, no issues detected, proceed to answering
                break

            if user_follow_up:
                # Append response and user follow-up as token IDs
                new_prompt_ids = append_user_turn_token_ids(
                    prompt_ids, response_ids, user_follow_up
                )
                # Track the new tokens added (user follow-up portion)
                added_tokens = new_prompt_ids[len(prompt_ids) + len(response_ids) :]
                all_token_ids.extend(added_tokens)
                prompt_ids = new_prompt_ids

        detokenized_text = harmony_encoding.decode(all_token_ids)
        boxed_text = extract_boxed_text(detokenized_text)

        if question_id and all_token_ids:
            answer_suffix = "NA"
            if is_valid_answer_string(boxed_text):
                answer_suffix = f"{boxed_text}"
            total_tokens = len(all_token_ids)
            base_path = f"{SOLUTIONS_DIR}/{question_id}/{solution_index:02d}-{total_tokens:05d}-{tool_call_count:02d}-{answer_suffix}"
            # Save full stream as token IDs (one token ID per line)
            with open(f"{base_path}-tokens.txt", "w") as f:
                for token_id in all_token_ids:
                    f.write(f"{token_id}\n")
            # Save detokenized full stream for readability
            with open(f"{base_path}-text.txt", "w") as f:
                f.write(detokenized_text)

        if is_valid_answer_string(boxed_text):
            question_id_to_counter[question_id][int(boxed_text)] += 1
            vote_answer(question_id)

        return boxed_text

    finally:
        # Always clean up the Jupyter session when done
        if jupyter_session is not None:
            jupyter_session.close()


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:26:54.553208Z","iopub.execute_input":"2025-11-24T08:26:54.553692Z","iopub.status.idle":"2025-11-24T08:27:03.475341Z","shell.execute_reply.started":"2025-11-24T08:26:54.553671Z","shell.execute_reply":"2025-11-24T08:27:03.474837Z"}}
if is_on_kaggle_interactive():
    generate_solution("What is 1+1?")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:27:06.296266Z","iopub.execute_input":"2025-11-24T08:27:06.296867Z","iopub.status.idle":"2025-11-24T08:27:06.300859Z","shell.execute_reply.started":"2025-11-24T08:27:06.29685Z","shell.execute_reply":"2025-11-24T08:27:06.300371Z"}}
import concurrent.futures
from collections import Counter


def solve(question_text: str, question_id: str = "") -> int:
    print(f"processing {question_id}")
    await_client()
    print("client connected")
    os.makedirs(f"{SOLUTIONS_DIR}/{question_id}", exist_ok=True)
    question_id_to_counter[question_id] = Counter()
    completed_question_ids.discard(question_id)  # just in case question_id collides

    if question_id and time.time() > cutoff_times[-1]:
        print("timeout did not solve")
        return 12314

    get_gpu_kv_cache_usage(
        question_id
    )  # run once to prevent running in the first batch of execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_generations) as executor:
        # run in parallel
        results = executor.map(
            generate_solution,
            [question_text] * num_generations,
            [question_id] * num_generations,
            list(range(num_generations)),
        )
        list(results)

    final_answer = vote_answer(question_id, force_answer=True)
    assert final_answer is not None
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
    pd.read_csv(
        "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
    ).drop("answer", axis=1).to_csv("reference.csv", index=False)


# Replace this function with your inference code.
# The function should return a single integer between 0 and 99999, inclusive.
def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # Unpack values
    question_id: str = id_.item(0)
    question_text: str = problem.item(0)

    if not run_all_questions:
        if is_on_kaggle_commit():
            if serve_vllm_on_kaggle:
                # to conserve Kaggle H100 quota
                if not ("Norwegian" in question_text or "Alice" in question_text):
                    print(
                        "on kaggle commit, skipping question"
                    )  # not popping cutoff_times
                    return pl.DataFrame({"id": id_, "answer": 12315})
            else:
                # to get quicker feedback
                if not ("Norwegian" in question_text or "Alice" in question_text):
                    print(
                        "on kaggle commit, skipping question"
                    )  # not popping cutoff_times
                    return pl.DataFrame({"id": id_, "answer": 12315})

        if not is_on_kaggle():
            # if you want to debug a particular question locally
            if "Norwegian" not in question_text:
                print("not on kaggle, skipping question")  # not popping cutoff_times
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
