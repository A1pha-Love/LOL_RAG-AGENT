from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from openai import OpenAI

from .config import chat_base_url, require_api_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class DashScopeEmbeddings(Embeddings):
    """LangChain adapter that sends raw string arrays to DashScope."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=1024,
            encoding_format="float",
        )
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embeddings() -> DashScopeEmbeddings:
    """Create the configured Qwen embedding client; no request occurs here."""
    model = os.getenv("QWEN_EMBEDDING_MODEL", "qwen3.7-text-embedding").strip()
    if model != "qwen3.7-text-embedding":
        raise RuntimeError("QWEN_EMBEDDING_MODEL must be qwen3.7-text-embedding.")
    client = OpenAI(api_key=require_api_key(), base_url=chat_base_url())
    return DashScopeEmbeddings(client=client, model=model)
