"""Mock kaggle_secrets for local development. Reads secrets from env.json."""

import json
import os
import warnings


class UserSecretsClient:
    """Mock UserSecretsClient that reads from env.json."""

    def __init__(self) -> None:
        if not os.path.exists("env.json"):
            warnings.warn(
                "\n\nenv.json not found.\n"
                "If run locally with remote GPUs, env.json should contain something like\n"
                '{ "REMOTE_VLLM_URL": "https://<handle>--example-gpt-oss-inference-serve.modal.run/v1" }\n\n'
                "Run `uv run modal deploy inference.py` to get the URL\n",
                stacklevel=2,
            )
            self._secrets: dict[str, str] = {}
        else:
            with open("env.json") as f:
                self._secrets = json.load(f)

    def get_secret(self, key: str) -> str:
        return self._secrets[key]
