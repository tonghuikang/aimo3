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
                "If run locally with remote inference, env.json should contain:\n"
                '{ "MODAL_INFERENCE_URL": "...", "FIREWORKS_API_KEY": "..." }\n',
                stacklevel=2,
            )
            self._secrets: dict[str, str] = {}
        else:
            with open("env.json") as f:
                self._secrets = json.load(f)

    def get_secret(self, key: str) -> str:
        return self._secrets[key]
