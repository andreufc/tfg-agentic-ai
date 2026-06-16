"""
config.py — Environment and secrets management.

Loads variables from a .env file (local), Colab userdata (Google Colab),
or Kaggle Secrets (Kaggle kernels).
"""

import os


def load_env(environment: str = "local") -> None:
    """Load environment variables from the appropriate secrets backend.

    Parameters
    ----------
    environment : str
        One of ``"colab"``, ``"kaggle"``, or ``"local"`` (the default).

    Backends
    --------
    - **colab**: reads secrets via ``google.colab.userdata``.
    - **kaggle**: reads secrets via ``kaggle_secrets.UserSecretsClient``.
    - **local** (default): reads a ``.env`` file using ``python-dotenv``.

    After calling this function, tokens are available via ``os.environ``.
    """
    try:
        if environment == "colab":
            from google.colab import userdata  # type: ignore

            os.environ.setdefault("HF_TOKEN", userdata.get("HF_TOKEN"))
            os.environ.setdefault("WANDB_API_KEY", userdata.get("WANDB_API_KEY"))
            os.environ.setdefault("GITHUB_TOKEN", userdata.get("GITHUB_TOKEN") or "")
            print("✅ Secrets loaded from Colab userdata.")

        elif environment == "kaggle":
            from kaggle_secrets import UserSecretsClient  # type: ignore

            secrets = UserSecretsClient()
            os.environ.setdefault("HF_TOKEN", secrets.get_secret("HF_TOKEN"))
            os.environ.setdefault("WANDB_API_KEY", secrets.get_secret("WANDB_API_KEY"))
            os.environ.setdefault("GITHUB_TOKEN", secrets.get_secret("GITHUB_TOKEN") or "")
            print("✅ Secrets loaded from Kaggle Secrets.")

        else:
            # Running locally or in CI
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
            print("✅ Secrets loaded from .env file.")
    except Exception as e:
        # Fallback to .env
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
            "HF_TOKEN not found. Set it in .env, Colab secrets, or Kaggle Secrets."
        )
    return token


def get_wandb_key() -> str:
    """Return the W&B API key, raising if missing."""
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise EnvironmentError(
            "WANDB_API_KEY not found. Set it in .env, Colab secrets, or Kaggle Secrets."
        )
    return key
