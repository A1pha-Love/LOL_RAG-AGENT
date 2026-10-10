"""Single place that resolves the DashScope base URLs.

Two host styles exist:

* **generic** — ``https://dashscope.aliyuncs.com/...`` authenticates purely by
  API key, so nothing but ``DASHSCOPE_API_KEY`` is required;
* **workspace-scoped** — ``https://{workspace}.{region}.maas.aliyuncs.com/...``
  additionally requires the key to belong to that business space, and returns
  ``access_denied: Workspace endpoint access denied`` when it does not.

Generic is the default because it keeps configuration to a single secret. Set
``DASHSCOPE_BASE_URL`` for a different host, or ``DASHSCOPE_WORKSPACE_ID`` to
opt back into the workspace-scoped host.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

GENERIC_HOST = "https://dashscope.aliyuncs.com"
DEFAULT_REGION = "cn-beijing"


def _base(service_path: str) -> str:
    configured = os.getenv("DASHSCOPE_BASE_URL", "").strip().rstrip("/")
    if configured:
        return f"{configured}{service_path}"

    workspace = os.getenv("DASHSCOPE_WORKSPACE_ID", "").strip()
    if workspace:
        region = os.getenv("DASHSCOPE_REGION", DEFAULT_REGION).strip() or DEFAULT_REGION
        return f"https://{workspace}.{region}.maas.aliyuncs.com{service_path}"

    return f"{GENERIC_HOST}{service_path}"


def chat_base_url() -> str:
    """Base URL for chat and embeddings (the OpenAI-compatible route)."""
    return _base("/compatible-mode/v1")


def rerank_url() -> str:
    """Full URL of the rerank route (a different service path from the above)."""
    return _base("/compatible-api/v1") + "/reranks"


def require_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is missing. Set it in the project .env file.")
    return api_key
