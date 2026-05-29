"""
config.py — Environment and secrets management.

Loads variables from a .env file (local) or Colab userdata (Google Colab).
"""

import os


def load_env(in_colab: bool = False) -> None:
    """Load environment variables from .env file or Colab secrets.

    Detects the runtime environment automatically:
    - **Google Colab**: reads secrets via ``google.colab.userdata``.
    - **Local / CI**: reads a ``.env`` file using ``python-dotenv``.

    After calling this function, tokens are available via ``os.environ``.
    """
    try:
        # Running in Google Colab
        if in_colab:
            from google.colab import userdata  # type: ignore

            os.environ.setdefault("HF_TOKEN", userdata.get("HF_TOKEN"))
            os.environ.setdefault("WANDB_API_KEY", userdata.get("WANDB_API_KEY"))
            os.environ.setdefault("GITHUB_TOKEN", userdata.get("GITHUB_TOKEN") or "")
            print("✅ Secrets loaded from Colab userdata.")
        else:
            # Running locally or in CI
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
            print("✅ Secrets loaded from .env file.")
    except Exception as e:
        # Running locally
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        print(f"⚠️ Warning: Error loading secrets: {e}")
        print("Trying to load from .env file as fallback...")
        print("Some secrets may not be loaded. Ensure python-dotenv is installed and .env file exists.")


def get_hf_token() -> str:
    """Return the Hugging Face token, raising if missing."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN not found. Set it in .env or Colab secrets."
        )
    return token


def get_wandb_key() -> str:
    """Return the W&B API key, raising if missing."""
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise EnvironmentError(
            "WANDB_API_KEY not found. Set it in .env or Colab secrets."
        )
    return key
