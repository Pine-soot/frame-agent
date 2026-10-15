"""Agent 基类：两阶段执行 + 结构化产出保证。

为什么是"两阶段"而不是"一个 ReAct 循环跑到底"
============================================

常见的 ReAct 写法是把工具调用和最终答案混在同一个循环里：模型边想边调工具，
直到它自己决定收尾。这在**探索型**任务上很好，但在我们这个场景里有两个问题：

1. **上下文会被工具噪声污染。** 工具返回的原始 JSON 往往很长（比如上百条字幕），
   连续塞进对话后，模型对最开始那句任务目标的注意力会被稀释，
   输出质量随工具轮数增加而下降。
2. **结构化产出难以保证。** 边调工具边吐 JSON，模型很容易在"半途"就返回一个
   缺字段的对象，把校验失败和工具循环纠缠在一起，问题定位困难。

于是本基类把流程切成两个**职责单一**的阶段：

.. code-block:: text

    阶段一  证据采集 (evidence)
      模型只做一件事：决定要调哪些工具。执行完后立刻收手，把结果收进字典。
      只跑一轮，不循环。理由：多轮采集的边际收益低，而成本线性增长；
         如果真需要多轮，正确的做法是拆成两个 Agent（本项目正是这么做的：
         Analyst 负责理解素材，Critic 负责复核产出）。

    阶段二  上下文收敛 + 结构化产出 (synthesis)
      把阶段一的观测**收敛成一份干净的结构化上下文**，重新发起一次请求，
      并要求输出严格符合 schema 的 JSON，失败则带错误路径自动修复。
      到这里，"上下文纯净"和"结构可保证"两件事同时成立。

这个切分带来一个额外好处：**离线后端与真实模型走的是同一条路径**，
所以离线跑出来的控制流（调用次数、工具序列、修复轮数）是真实可信的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from frame_agent.llm.base import Message
from frame_agent.protocol import encode_ctx, inject_task
from frame_agent.schema import GuardResult, SchemaGuard

__all__ = ["AgentResult", "Agent"]


@dataclass
class AgentResult:
    """一个 Agent 的完整产出与执行痕迹。"""

    agent: str
    ok: bool
    data: dict | None = None
    text: str = ""
    model_calls: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    observations: dict = field(default_factory=dict)
    guard: GuardResult | None = None
    degradation: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "ok": self.ok,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "observation_keys": sorted(self.observations),
            "degradation": self.degradation,
            "guard": self.guard.summary() if self.guard else None,
        }


class Agent:
    """所有 Agent 的基类。

    子类需要提供：

    - ``task``：:mod:`frame_agent.protocol` 中的任务标签
    - ``system_prompt``：角色设定（不含任务标签，由 :meth:`system` 自动追加）
    - ``output_schema``：产出必须满足的 schema
    - :meth:`consolidate`：把原始 payload 与工具观测收敛成最终上下文
    """

    task: str = ""
    name: str = "agent"
    system_prompt: str = ""
    output_schema: dict | None = None
    output_name: str = "payload"
    use_tools: bool = False
    max_repair: int = 2
    max_tokens: int = 2600
    temperature: float = 0.2

    def __init__(
        self,
        router: Any,
        registry: Any = None,
        *,
        max_repair: int | None = None,
    ) -> None:
        self.router = router
        self.registry = registry
        self.guard = (
            SchemaGuard(
                self.output_schema,
                name=self.output_name,
                max_repair=self.max_repair if max_repair is None else max_repair,
            )
            if self.output_schema
            else None
        )

    # -- 提示词 -------------------------------------------------------------

    def system(self) -> str:
        return inject_task(self.system_prompt, self.task)

    def task_instruction(self) -> str:
        """阶段二的指令。子类可覆写以强调各自的产出重点。"""
        return "请基于以下上下文完成本次任务，并只输出符合约定结构的 JSON。"

    # -- 上下文收敛 ---------------------------------------------------------

    def consolidate(self, payload: dict, observations: dict) -> dict:
        """把原始 payload 与工具观测合并成最终上下文。

        默认行为是把观测挂到 ``observations`` 键下。子类覆写它可以做
        **字段级归一**（例如把 ``parse_transcript`` 的结果提升为顶层
        ``transcript``），让最终提示词里的上下文更贴近模型的直觉，
        而不是暴露内部工具名。
        """
        merged = dict(payload)
        if observations:
            merged["observations"] = observations
        return merged

    # -- 执行 ---------------------------------------------------------------

    def run(self, payload: dict) -> AgentResult:
        model_calls = 0
        tool_log: list[dict] = []
        observations: dict[str, Any] = {}

        # ---- 阶段一：证据采集 ----
        if self.use_tools and self.registry is not None:
            plan_messages = [
                Message(role="system", content=self.system()),
                Message(
                    role="user",
                    content=(
                        f"{self.task_instruction()}\n\n"
                        "如需更多证据，请先调用可用工具；不需要则直接回答。\n\n"
                        + encode_ctx(payload)
                    ),
                ),
            ]
            plan = self.router.call(
                plan_messages,
                tools=self.registry.schemas(),
                temperature=self.temperature,
                max_tokens=768,
                purpose=f"{self.name}:evidence",
            )
            model_calls += 1
            for call in plan.tool_calls:
                result = self.registry.invoke(call.name, call.arguments)
                observations[call.name] = (
                    result.data if result.ok else {"__error__": result.error}
                )
                tool_log.append(
                    {
                        "tool": call.name,
                        "ok": result.ok,
                        "reason": (call.arguments or {}).get("_reason", ""),
                        "arguments": {
                            k: v for k, v in (call.arguments or {}).items() if k != "_reason"
                        },
                        "error": result.error,
                    }
                )

        # ---- 阶段二：上下文收敛 + 结构化产出 ----
        enriched = self.consolidate(payload, observations)
        synthesis_messages = [
            Message(role="system", content=self.system()),
            Message(
                role="user",
                content=f"{self.task_instruction()}\n\n{encode_ctx(enriched)}",
            ),
        ]

        degradation: list[str] = []
        if self.guard is None:  # pragma: no cover - 目前所有 Agent 都有 schema
            response = self.router.call(
                synthesis_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                purpose=f"{self.name}:synthesis",
            )
            model_calls += 1
            return AgentResult(
                agent=self.name,
                ok=True,
                data={"text": response.text},
                text=response.text,
                model_calls=model_calls,
                tool_calls=tool_log,
                observations=observations,
            )

        guard_result = self.guard.generate(
            self.router,
            synthesis_messages,
            purpose=f"{self.name}:synthesis",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        # 修复轮每次都会真实调用模型，必须计入成本统计
        model_calls += 1 + guard_result.repair_rounds

        if not guard_result.ok:
            degradation.append(
                f"{self.name} 的结构化产出在 {guard_result.repair_rounds} 轮修复后仍未通过校验"
            )

        last_text = guard_result.attempts[-1].raw_text if guard_result.attempts else ""
        return AgentResult(
            agent=self.name,
            ok=guard_result.ok,
            data=guard_result.data,
            text=last_text,
            model_calls=model_calls,
            tool_calls=tool_log,
            observations=observations,
            guard=guard_result,
            degradation=degradation,
        )
