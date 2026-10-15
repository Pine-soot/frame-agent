"""LLM 抽象层。

- :mod:`frame_agent.llm.base`：协议与数据结构（无厂商绑定）
- :mod:`frame_agent.llm.router`：降级链 / 重试退避 / 预算熔断 / 记账
- :mod:`frame_agent.llm.mock`：确定性离线后端（保可复现）
- :mod:`frame_agent.llm.openai_compat`：任意 OpenAI 兼容端点

:func:`build_backends` 是唯一需要改动的装配点：想把 mock 换成真模型，
只改环境变量即可，流水线代码一行不动。
"""

from __future__ import annotations

from frame_agent.llm.base import (
    BudgetExceeded,
    LLMBackend,
    LLMError,
    LLMResponse,
    Message,
    PermanentLLMError,
    ToolCall,
    TransientLLMError,
    Usage,
)
from frame_agent.llm.mock import MockBackend
from frame_agent.llm.openai_compat import OpenAICompatBackend, from_env
from frame_agent.llm.router import Budget, CallRecord, ModelRouter

__all__ = [
    "LLMBackend",
    "LLMError",
    "LLMResponse",
    "Message",
    "ToolCall",
    "Usage",
    "TransientLLMError",
    "PermanentLLMError",
    "BudgetExceeded",
    "MockBackend",
    "OpenAICompatBackend",
    "from_env",
    "ModelRouter",
    "Budget",
    "CallRecord",
    "build_backends",
]


def build_backends(
    spec: str = "mock",
    *,
    budget: Budget | None = None,
) -> list:
    """把 ``--backend`` 字符串解析成后端列表（有序 = 降级优先级）。

    语法：``mock`` | ``openai`` | ``mock,openai`` | ``openai,mock``

    ``openai,mock`` 这个组合是实际部署里最实用的：优先真实模型，
    一旦限流/超时就自动降级到离线后端，保证流水线**永远能出一个可用产物**，
    而不是整批任务因为一次 429 全部失败。
    """
    backends: list = []
    for token in [t.strip().lower() for t in (spec or "mock").split(",") if t.strip()]:
        if token in {"mock", "offline"}:
            backends.append(MockBackend())
        elif token in {"openai", "openai-compat", "real"}:
            backends.append(from_env())
        else:
            raise ValueError(f"未知后端：{token!r}（可选 mock / openai）")
    if not backends:
        raise ValueError("后端列表为空")
    return backends
