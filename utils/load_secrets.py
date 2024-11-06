import re
import os
from pathlib import Path


def load_env_file(env_path: "str | None" = None):
    try:
        if env_path is None:
            env_path = (Path.home() / ".kscope.env").as_posix()

            for line in open(env_path, "r").read().splitlines():
                match = re.match(r"^(export)? (\w+)=\"?([^\"]+)\"?$", line)
                if match:
                    _, key, value = match.groups()
                    os.environ[key] = value

        assert os.environ.get("OPENAI_API_KEY") is not None
        assert os.environ.get("OPENAI_BASE_URL") is not None

    except Exception as err:
        print(
            "Could not read your OpenAI key. ""Please make sure this is available in plain text under your home directory "f"in ~/.kscope.env: {err}"
        )

