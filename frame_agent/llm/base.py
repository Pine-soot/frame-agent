"""LLM 后端的统一抽象。

本模块只定义**协议与数据结构**，不含任何厂商绑定。三个实现分别位于
:mod:`frame_agent.llm.mock`（确定性离线）、:mod:`frame_agent.llm.openai_compat`
（任意 OpenAI 兼容端点），测试里还有一个用于注入故障的替身。

不引入官方 SDK：只用标准库 ``urllib``，让"换模型"这件事退化成改环境变量。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

__all__ = [
    "ToolCall",
    "Message",
    "Usage",
    "LLMResponse",
    "LLMBackend",
    "LLMError",
    "TransientLLMError",
    "PermanentLLMError",
    "BudgetExceeded",
]


# --------------------------------------------------------------------------
# 异常：区分"可重试"与"不可重试"是重试策略的前提
# --------------------------------------------------------------------------

class LLMError(RuntimeError):
    """所有 LLM 相关错误的基类。"""


class TransientLLMError(LLMError):
    """瞬时故障：限流、超时、5xx、连接重置。**应当**重试。"""


class PermanentLLMError(LLMError):
    """确定性故障：鉴权失败、参数非法、模型不存在。重试没有意义。"""


class BudgetExceeded(LLMError):
    """调用次数 / token / 墙钟时间任一预算耗尽。

    这是**熔断**而不是错误：在 Agent 流水线里，一个退化的循环可能无限调用模型。
    把它作为异常抛出，是为了让上层能明确决定"降级"还是"整体失败"，
    而不是悄悄产出半成品。
    """


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------

@dataclass
class ToolCall:
    """模型发起的一次工具调用。"""

    id: str
    name: str
    arguments: dict

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class Message:
    """对话消息。

    用 ``role`` 单字段区分 ``system`` / ``user`` / ``assistant`` / ``tool``，
    各后端自行把它翻译成自己的格式（这里是唯一需要适配的地方）。
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    name: str | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.name:
            data["name"] = self.name
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        return data

    def to_openai(self) -> dict:
        """转成 OpenAI Chat Completions 的消息格式。"""
        if self.role == "tool":
            return {
                "role": "tool",
                "content": self.content,
                "tool_call_id": self.tool_call_id or "",
            }
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]
        return data


@dataclass
class Usage:
    """token 计量。真实后端由 API 返回，离线后端按字符数估算。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass
class LLMResponse:
    """一次模型调用的结果。"""

    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    raw: dict = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class LLMBackend(Protocol):
    """后端协议。

    只要实现 ``name`` 与 ``complete`` 两个成员即可被 :class:`ModelRouter` 接纳。
    用 ``Protocol`` 而非 ABC，是为了让测试替身不必继承任何基类，
    Duck typing 在这里比继承更贴合"可替换"的设计意图。
    """

    name: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:  # pragma: no cover - 协议声明
        ...
