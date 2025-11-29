For changes involving html files, please use MCP pupeeteer to test.

Package Management
- ONLY use uv, NEVER pip
- Installation: uv add package
- Running tools: uv run tool
- Upgrading: uv add --dev package --upgrade-package package
- FORBIDDEN: uv pip install, @latest syntax

Formatting
- Format: uvx ruff format *.py
- Check: uvx ruff check *.py
- Fix: uvx ruff check *.py --fix
- Sort imports: uvx ruff check --select I *.py --fix
- Type checking (mypy): uvx mypy *.py
- Type checking (pyright): uvx pyright *.py

When searching for vLLM library code
- Download vLLM library code by running `uv pip install vllm==0.11.2 --no-deps`