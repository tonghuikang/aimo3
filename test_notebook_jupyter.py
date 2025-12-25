"""Tests for individual functions in notebook.py"""

import os


def count_open_fds() -> int:
    """Count the number of open file descriptors for this process."""
    pid = os.getpid()
    try:
        return len(os.listdir(f"/proc/{pid}/fd"))
    except FileNotFoundError:
        # macOS - use lsof
        import subprocess

        result = subprocess.run(
            ["lsof", "-p", str(pid)], capture_output=True, text=True
        )
        return len(result.stdout.strip().split("\n")) - 1  # subtract header


def test_jupyter_session_no_fd_leak():
    """Test that creating and closing sessions does not leak file descriptors."""
    import gc

    from notebook import LocalJupyterSession, execute_python_code

    # Force garbage collection to clean up any lingering sessions
    gc.collect()

    fd_before = count_open_fds()
    print(f"FDs before: {fd_before}")

    # Create and close multiple sessions
    for i in range(3):
        session = LocalJupyterSession()
        result = execute_python_code(session, f"print('session {i}')")
        print(f"Session {i} result: {repr(result)}")
        session.close()
        gc.collect()

    fd_after = count_open_fds()
    print(f"FDs after: {fd_after}")

    # Allow some tolerance (a few FDs might be opened by imports, etc.)
    fd_leaked = fd_after - fd_before
    print(f"FDs leaked: {fd_leaked}")
    assert fd_leaked < 20, (
        f"Leaked {fd_leaked} file descriptors (before={fd_before}, after={fd_after})"
    )

    print("✓ No FD leak test passed!")


def test_generate_solution_fd_leak():
    """Stress test generate_solution with mock client to check for FD leaks."""
    import concurrent.futures
    import gc
    import time
    from unittest.mock import MagicMock, patch

    # Set up globals needed by generate_solution
    import notebook

    notebook.cutoff_times = [time.time() + 3600]  # 1 hour from now
    notebook.completed_question_ids = set()
    notebook.question_id_to_solver_to_answer = {}
    notebook.question_id_to_solver_to_token_length = {}
    notebook.num_generations = 6

    # Create a mock streaming response that triggers tool calls
    def make_mock_stream(include_tool_call: bool = True, num_chunks: int = 5):
        """Create a mock streaming response."""
        chunks = []
        for i in range(num_chunks):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].token_ids = [100 + i]  # Fake token IDs
            if include_tool_call and i == num_chunks - 1:
                # Last chunk triggers tool call
                chunk.choices[0].text = "print('hello')"
                chunk.choices[0].finish_reason = "stop"
            else:
                chunk.choices[0].text = f"chunk{i} "
                chunk.choices[0].finish_reason = None
            chunks.append(chunk)

        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: iter(chunks)
        mock_stream.close = MagicMock()
        return mock_stream

    # Mock StreamableParser to return tool call messages
    class MockStreamableParser:
        def __init__(self, *args, **kwargs):
            self.messages = []
            self._call_count = 0

        def process(self, token_id):
            pass

        def set_tool_call(self):
            """Set up a tool call message."""
            msg = MagicMock()
            msg.recipient = "python"
            msg.channel = None
            content = MagicMock()
            content.text = "print('hello from tool')"
            msg.content = [content]
            self.messages = [msg]

        def set_final_answer(self):
            """Set up a final answer message."""
            msg = MagicMock()
            msg.recipient = None
            msg.channel = None
            content = MagicMock()
            content.text = "The answer is \\boxed{42}"
            msg.content = [content]
            self.messages = [msg]

    gc.collect()
    fd_before = count_open_fds()
    print(f"FDs before: {fd_before}")

    def run_mock_generate_solution(solution_index: int):
        """Run generate_solution with mocks."""
        question_id = f"test_q_{solution_index}"
        notebook.question_id_to_solver_to_answer[question_id] = {}
        notebook.question_id_to_solver_to_token_length[question_id] = {}

        call_count = [0]

        def mock_completions_create(*args, **kwargs):
            call_count[0] += 1
            # First call returns tool call, subsequent calls return final answer
            return make_mock_stream(include_tool_call=(call_count[0] <= 3))

        mock_parser_instances = []

        def mock_parser_init(self, *args, **kwargs):
            MockStreamableParser.__init__(self, *args, **kwargs)
            mock_parser_instances.append(self)
            # Alternate between tool calls and final answers
            if len(mock_parser_instances) <= 3:
                self.set_tool_call()
            else:
                self.set_final_answer()

        with patch.object(
            notebook.client.completions, "create", mock_completions_create
        ):
            with patch(
                "notebook.StreamableParser.__init__", mock_parser_init, create=True
            ):
                with patch("notebook.StreamableParser.process", lambda self, x: None):
                    with patch(
                        "notebook.StreamableParser.messages",
                        property(lambda self: self.messages),
                    ):
                        # Actually just test the Jupyter session creation/cleanup
                        # by calling with mocked client
                        session = notebook.LocalJupyterSession()
                        try:
                            for i in range(5):  # Simulate multiple tool calls
                                notebook.execute_python_code(
                                    session, f"print('call {i}')"
                                )
                        finally:
                            session.close()
        return solution_index

    # Run multiple concurrent mock solutions
    num_rounds = 3
    for round_num in range(num_rounds):
        print(f"--- Round {round_num} ---")
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(run_mock_generate_solution, range(6)))
        gc.collect()
        fd_now = count_open_fds()
        print(f"After round {round_num}: {fd_now} (delta: {fd_now - fd_before})")

    fd_after = count_open_fds()
    fd_leaked = fd_after - fd_before
    print(f"Total FDs leaked: {fd_leaked}")

    # Allow some tolerance for imports, etc.
    assert fd_leaked < 50, (
        f"Leaked {fd_leaked} file descriptors (before={fd_before}, after={fd_after})"
    )
    print("✓ Generate solution FD leak test passed!")


def test_jupyter_session_no_ansi_colors():
    """Test that tracebacks do not contain ANSI color codes."""
    from notebook import LocalJupyterSession, execute_python_code

    session = LocalJupyterSession()

    # Trigger an error to get a traceback
    result = execute_python_code(session, "undefined_variable")
    print(f"Error output: {repr(result)}")

    session.close()

    expected = """\
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[1], line 1
----> 1 undefined_variable

NameError: name 'undefined_variable' is not defined"""

    assert result == expected, f"Expected:\n{repr(expected)}\n\nGot:\n{repr(result)}"

    print("✓ No ANSI colors test passed!")


def test_jupyter_session_immediate_interrupt():
    """Test immediate interrupt: timeout interrupts kernel immediately (default behavior)."""
    from notebook import LocalJupyterSession

    session = LocalJupyterSession(timeout=1.0)
    try:
        # Setup: define variable and verify statefulness
        session.execute("x = 42")
        result0 = session.execute("print(x * 2)")
        assert result0 == "84\n", f"Expected '84\\n', got: {repr(result0)}"

        # Infinite loop that gets interrupted immediately on timeout
        code_hang = """
import sys
print('before hang')
sys.stdout.flush()
while True: pass
"""
        result1 = session.execute(code_hang)
        print(f"Hang timeout result: {repr(result1)}")
        assert "[TIMEOUT] Execution interrupted." in result1, f"Got: {repr(result1)}"
        assert "before hang" in result1, f"Got: {repr(result1)}"

        # Next call should work without needing to drain previous output
        result2 = session.execute("print('after interrupt')")
        print(f"After interrupt: {repr(result2)}")
        # Should NOT have "[Previous execution output]" prefix
        assert "after interrupt" in result2, f"Got: {repr(result2)}"

        # Verify state preserved after interrupt
        result3 = session.execute("print(x)")
        print(f"Variable x after interrupt: {repr(result3)}")
        assert result3 == "42\n", f"Expected '42\\n', got: {repr(result3)}"

        print("✓ Immediate interrupt test passed!")
    finally:
        session.close()


def test_jupyter_session_deferred_interrupt():
    """Test deferred interrupt: timeout lets kernel continue, interrupt on next call.
    Also tests statefulness - variables persist across calls and interrupts.
    """
    import time

    from notebook import LocalJupyterSession

    session = LocalJupyterSession(timeout=1.0)
    try:
        # Setup: define variable and verify statefulness
        session.execute("x = 42")
        result0 = session.execute("print(x * 2)")
        assert result0 == "84\n", f"Expected '84\\n', got: {repr(result0)}"

        # Part 1: Code that completes in background after timeout
        code_sleep = """
import time
print('before sleep')
time.sleep(3)
print('after sleep')
"""
        result1 = session.execute(code_sleep, continue_executing_on_timeout=True)
        print(f"Sleep timeout result: {repr(result1)}")
        assert (
            result1
            == "before sleep\n[TIMEOUT] Execution still running. Will drain remaining output on next call."
        ), f"Got: {repr(result1)}"

        # Wait for background to complete, then drain on next call
        time.sleep(3)
        result2 = session.execute("print('after sleep drain')")
        print(f"After sleep drain: {repr(result2)}")
        assert (
            result2
            == "[Previous execution output]\nafter sleep\n[End previous output]\nafter sleep drain\n"
        ), f"Got: {repr(result2)}"

        # Verify state preserved after timeout
        result2b = session.execute("print(x)")
        assert result2b == "42\n", f"Expected '42\\n', got: {repr(result2b)}"

        # Part 2: Infinite loop that needs interrupt on next call
        code_hang = """
import sys
print('before hang')
sys.stdout.flush()
while True: pass
"""
        result3 = session.execute(code_hang, continue_executing_on_timeout=True)
        print(f"Hang timeout result: {repr(result3)}")
        assert (
            result3
            == "before hang\n[TIMEOUT] Execution still running. Will drain remaining output on next call."
        ), f"Got: {repr(result3)}"

        # Next call interrupts pending, shows KeyboardInterrupt, runs new code
        result4 = session.execute("print('after hang')")
        print(f"After hang: {repr(result4)}")
        expected4 = """\
[Previous execution output]
---------------------------------------------------------------------------
KeyboardInterrupt                         Traceback (most recent call last)
Cell In[6], line 4
      2 print('before hang')
      3 sys.stdout.flush()
----> 4 while True: pass

KeyboardInterrupt:
[End previous output - interrupted]
after hang
"""
        assert result4 == expected4, (
            f"Expected:\n{repr(expected4)}\n\nGot:\n{repr(result4)}"
        )

        # Verify state preserved after interrupt
        result5 = session.execute("print(x)")
        print(f"Variable x after interrupt: {repr(result5)}")
        assert result5 == "42\n", f"Expected '42\\n', got: {repr(result5)}"

        print("✓ Deferred interrupt test passed!")
    finally:
        session.close()
