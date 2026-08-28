"""
config.py: Environment and secrets management.

Loads variables from a .env file (local), Colab userdata (Google Colab),
or Kaggle Secrets (Kaggle kernels).

Connection to the project WPs
-----------------------------
* WP1-T1/T4 & WP2-T1/T3/T4/T6: loads the GITHUB/HF/WANDB tokens per runtime so
  every Kaggle notebook can clone the repo, pull models and authenticate W&B.
"""

import os


def load_env(environment: str = "local") -> None:
    """Load environment variables from the appropriate secrets backend.

    Colab reads via `google.colab.userdata`, Kaggle via
    `kaggle_secrets.UserSecretsClient`, local via a `.env` file (python-dotenv).
    Tokens end up in `os.environ`.

    Parameters
    ----------
    environment : str
        One of `"colab"`, `"kaggle"`, or `"local"` (the default).
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
            # Local or CI
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
            print("✅ Secrets loaded from .env file.")
    except Exception as e:
        # Fall back to .env
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        print(f"⚠️ Warning: Error loading secrets: {e}")
        print("Trying to load from .env file as fallback...")
        print("Some secrets may not be loaded. Ensure python-dotenv is installed and .env file exists.")
