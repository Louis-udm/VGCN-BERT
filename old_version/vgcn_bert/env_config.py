import os
from pathlib import Path

from dotenv import load_dotenv


class EnvConfig:
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    GLOBAL_SEED = int(os.environ.get("GLOBAL_SEED", 44))
    TRANSFORMERS_OFFLINE = int(os.environ.get("TRANSFORMERS_OFFLINE", 0))
    HUGGING_LOCAL_MODEL_FILES_PATH = os.environ.get(
        "HUGGING_LOCAL_MODEL_FILES_PATH", "."
    )


env_config = EnvConfig()
