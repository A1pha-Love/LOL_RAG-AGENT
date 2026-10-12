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

# ============================================================================
# 中文说明
#
# 本模块只有一个职责：把"用哪个 DashScope 域名"这件事集中在一个地方决定，
# 让 chat / embedding / rerank 三处调用不会各写一套拼接逻辑。
#
# 两种主机风格（这正是最容易踩的坑）：
#
#   1. 通用域名（默认）
#      https://dashscope.aliyuncs.com/compatible-mode/v1
#      只靠 API key 鉴权，配置里只需要一个 DASHSCOPE_API_KEY 就能跑起来，
#      所以它是默认值。
#
#   2. workspace 专用域名（业务空间专属）
#      https://{workspace}.{region}.maas.aliyuncs.com/compatible-mode/v1
#      除了 key 之外，还要求这把 key 属于该业务空间。若 key 不属于该空间，服务端会
#      返回 `access_denied: Workspace endpoint access denied`——403 的成因通常是
#      "key 与 workspace 不匹配"，而不是 key 本身失效。遇到这个报错时，正确做法是
#      清掉 DASHSCOPE_WORKSPACE_ID 回退到通用域名，而不是反复换 key。
#
# 优先级：显式配置的 DASHSCOPE_BASE_URL > DASHSCOPE_WORKSPACE_ID > 通用域名。
# 环境变量都在导入时由 load_dotenv 从项目根目录的 .env 读入，因此本模块被导入
# 一次就完成了配置加载。
# ============================================================================

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# parents[2]：src/lol_agent/config.py -> lol_agent -> src -> 项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 在模块导入时就加载 .env（而不是等到 require_api_key 被调用）：这样所有依赖本模块
# 的入口（CLI、后端、脚本）都自动获得同一份配置，不需要各自记得加载。
load_dotenv(PROJECT_ROOT / ".env")

GENERIC_HOST = "https://dashscope.aliyuncs.com"
# 默认地域：workspace 专用域名必须写成 {workspace}.{region}.maas.aliyuncs.com。
DEFAULT_REGION = "cn-beijing"


def _base(service_path: str) -> str:
    """按优先级拼出服务的基础 URL。

    参数:
        service_path: 服务路径（如 "/compatible-mode/v1"），由调用方给出，
            因为 chat/embedding 与 rerank 用的路径并不相同。

    返回:
        完整的基础 URL（末尾不带斜杠，服务路径自带前导斜杠）。

    说明:
        `.rstrip("/")` 是为了容忍用户在 .env 里把 base URL 写成
        "https://host/" 这种带尾斜杠的形式，否则会拼出 "//compatible-mode/v1"。
    """
    configured = os.getenv("DASHSCOPE_BASE_URL", "").strip().rstrip("/")
    if configured:
        return f"{configured}{service_path}"

    workspace = os.getenv("DASHSCOPE_WORKSPACE_ID", "").strip()
    if workspace:
        # region 允许被环境变量覆盖；写成空白（而不是不设）时也要回退到默认地域，
        # 否则会拼出带空 region 的非法域名。
        region = os.getenv("DASHSCOPE_REGION", DEFAULT_REGION).strip() or DEFAULT_REGION
        return f"https://{workspace}.{region}.maas.aliyuncs.com{service_path}"

    return f"{GENERIC_HOST}{service_path}"


def chat_base_url() -> str:
    """Base URL for chat and embeddings (the OpenAI-compatible route).

    chat 与 embedding 共用的基础 URL（OpenAI 兼容路由）。
    """
    return _base("/compatible-mode/v1")


def rerank_url() -> str:
    """Full URL of the rerank route (a different service path from the above).

    rerank 的完整 URL。注意服务路径不同（compatible-api 而不是 compatible-mode），
    且 rerank 接口本身要带上 /reranks 资源名，所以它是"完整 URL"而不是基础 URL。
    """
    return _base("/compatible-api/v1") + "/reranks"


def require_api_key() -> str:
    """取出 API key；缺失时立刻抛 RuntimeError。

    返回:
        去掉首尾空白的 API key。

    说明:
        这里刻意"早失败"：缺少 key 时构造客户端不会报错，只有真正发请求时才失败，
        而且错误信息往往离配置问题很远。提前抛错能让使用者立刻知道要改 .env。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is missing. Set it in the project .env file.")
    return api_key
