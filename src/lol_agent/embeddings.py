from __future__ import annotations

# ============================================================================
# 中文说明
#
# 本模块把 DashScope 的 embedding 接口适配成 LangChain 的 `Embeddings` 接口，
# 供知识库建索引与检索时调用。适配的目的只有一个：让上层（knowledge.py）用
# LangChain 的统一接口拿向量，而不必关心底层是 OpenAI 兼容协议还是别的东西。
#
# 关键约定：**向量维度固定为 1024**（见 embed_documents 里的 dimensions 参数，
# 与 knowledge.EMBEDDING_DIMENSIONS 一致）。已建好的索引里存的就是 1024 维向量，
# 换模型或改维度会让全部存量向量失效、必须重跑 scripts/index_knowledge.py，因此
# QWEN_EMBEDDING_MODEL 在本项目里被写死校验（见 build_embeddings）。
# ============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from openai import OpenAI

from .config import chat_base_url, require_api_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 与 config.py 一样在导入时加载 .env；重复加载是幂等的，不会覆盖已设置的真实环境变量
# （load_dotenv 默认不 override）。
load_dotenv(PROJECT_ROOT / ".env")


class DashScopeEmbeddings(Embeddings):
    """LangChain adapter that sends raw string arrays to DashScope.

    把字符串数组直接交给 DashScope 的适配器（不做任何预处理，也不缓存向量）。
    """

    def __init__(self, client: OpenAI, model: str) -> None:
        """注入已配置好的客户端与模型名。

        参数:
            client: 已带 api_key 与 base_url 的 OpenAI 客户端。
            model: 模型名；本项目只允许 qwen3.7-text-embedding（见 build_embeddings）。
        """
        self.client = client
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量文本转向量。

        参数:
            texts: 待向量化的文本列表。

        返回:
            与输入顺序一一对应的向量列表。

        说明:
            * 空列表直接返回空：白跑一次网络请求没有意义，而且部分接口对空 input
              会直接报错。
            * dimensions=1024 与索引里已存储的维度必须一致（见模块说明）。
            * 按 `item.index` 排序后再返回：接口返回的 data 顺序不保证与输入一致，
              若不排序，向量与文本就可能错位，检索会返回毫不相关的文档。
        """
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
        """单条查询文本转向量（复用批量实现，保证与文档用同一套参数）。"""
        return self.embed_documents([text])[0]


def build_embeddings() -> DashScopeEmbeddings:
    """Create the configured Qwen embedding client; no request occurs here.

    创建 embedding 客户端；本函数不发起任何网络请求。

    返回:
        已配置好的 DashScopeEmbeddings 实例。

    异常:
        RuntimeError: 环境变量里的模型名不是 qwen3.7-text-embedding 时抛出。

    为什么把模型名写死校验:
        索引是"用某个模型算出的向量"这件事的产物。换 embedding 模型后，新旧向量
        不在同一语义空间里，相似度检索会静默变差（不会报错），所以宁可在这里明确
        拒绝，也不允许换模型后继续用旧索引。
    """
    model = os.getenv("QWEN_EMBEDDING_MODEL", "qwen3.7-text-embedding").strip()
    if model != "qwen3.7-text-embedding":
        raise RuntimeError("QWEN_EMBEDDING_MODEL must be qwen3.7-text-embedding.")
    # chat 与 embedding 走同一个 OpenAI 兼容 base URL。
    client = OpenAI(api_key=require_api_key(), base_url=chat_base_url())
    return DashScopeEmbeddings(client=client, model=model)
