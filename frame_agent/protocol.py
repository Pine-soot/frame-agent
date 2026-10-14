"""Agent 与后端之间的**线协议**（wire protocol）。

单独抽一个模块是为了一个很实际的原因：确定性离线后端
(:mod:`frame_agent.llm.mock`) 需要读懂 Agent 到底在请求什么任务，
但 ``llm`` 层不应该反向依赖 ``agents`` 层（会造成循环依赖）。

于是把双方都要遵守的约定放在这里，谁都可以引用它，依赖图保持无环：

.. code-block:: text

    agents ──┐
             ├──> protocol  (任务标签 / 上下文封装 / 观测探测)
    llm   ───┘

三个约定：

``[TASK:XXX]``
    任务标签，写在 system prompt 里。真实模型会忽略它（它只是一段普通文本，
    语义上相当于"你现在执行的是 XXX 任务"，本来就是有用的提示）；
    离线后端则用它做意图分发。

``<ctx>...</ctx>``
    结构化上下文信封。Agent 把入参序列化成 JSON 塞进用户消息。
    对真实模型而言这是清晰的输入格式；对离线后端而言这是唯一的信息来源。
    **同一份上下文喂给两种后端**，是"离线跑通的结果能代表在线行为"的前提。

``has_tool_observation``
    判断对话里是否已经出现过工具返回结果。离线后端用它决定
    "该发起工具调用" 还是 "该输出最终答案"，从而让工具循环无需任何内部状态。
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

__all__ = [
    "TASK_MEDIA_ANALYZE",
    "TASK_SCRIPT_WRITE",
    "TASK_SCRIPT_CRITIQUE",
    "TASK_PROMPT_EXPAND",
    "TASK_TAGS",
    "inject_task",
    "detect_task",
    "encode_ctx",
    "decode_ctx",
    "has_tool_observation",
]

# 任务标签：与四个 Agent 一一对应
TASK_MEDIA_ANALYZE = "MEDIA_ANALYZE"
TASK_SCRIPT_WRITE = "SCRIPT_WRITE"
TASK_SCRIPT_CRITIQUE = "SCRIPT_CRITIQUE"
TASK_PROMPT_EXPAND = "PROMPT_EXPAND"

TASK_TAGS = (
    TASK_MEDIA_ANALYZE,
    TASK_SCRIPT_WRITE,
    TASK_SCRIPT_CRITIQUE,
    TASK_PROMPT_EXPAND,
)

_TASK_RE = re.compile(r"\[TASK:([A-Z_]+)\]")
_CTX_RE = re.compile(r"<ctx>\s*(.*?)\s*</ctx>", re.DOTALL)


def inject_task(system_prompt: str, task: str) -> str:
    """把任务标签追加到 system prompt 末尾。"""
    return f"{system_prompt.rstrip()}\n\n[TASK:{task}]"


def detect_task(system_prompt: str) -> str | None:
    """从 system prompt 中读出任务标签。"""
    match = _TASK_RE.search(system_prompt or "")
    return match.group(1) if match else None


def encode_ctx(payload: dict) -> str:
    """把结构化上下文序列化成信封。

    ``ensure_ascii=False`` 是刻意的：中文上下文保持可读，
    一方面方便人工审查落盘的 trace，另一方面真实模型对中文原文的
    理解优于 ``\\uXXXX`` 转义序列。
    """
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    return f"<ctx>\n{body}\n</ctx>"


def decode_ctx(messages: Sequence[Any]) -> dict:
    """从消息列表里取出最近一个上下文信封并反序列化。

    倒序扫描：修复轮会把上下文留在更早的消息里，最新的 user 消息
    可能是"你上次输出不对，请修正"。倒序 + 取最近一个信封才正确。
    """
    for message in reversed(list(messages)):
        content = getattr(message, "content", "") or ""
        match = _CTX_RE.search(content)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    return {}


def has_tool_observation(messages: Sequence[Any]) -> bool:
    """对话里是否已经出现过工具返回结果。

    这是让工具循环**无状态**的关键：离线后端不记录"我调用到第几轮"，
    只看历史里有没有 tool 角色消息，就足以判断该继续调用还是该收尾。
    好处是可重放、可并行、不依赖任何实例变量。
    """
    return any(getattr(m, "role", None) == "tool" for m in messages)
