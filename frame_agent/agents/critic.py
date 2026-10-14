"""Critic —— 评审与重写指令 Agent。

本 Agent 承担整条流水线里**最关键的职责分工设计**：

.. list-table::
   :widths: 30 70

   * - 谁负责
     - 负责什么
   * - 确定性评测器（:mod:`frame_agent.evals.rubric`）
     - 给出**数字**：五个维度各多少分、总分多少、每条结论对应哪个具体指标
   * - 合规扫描工具（``scan_compliance``）
     - 独立复核风险词，**不采信**评测器的结论
   * - 本 Agent（模型）
     - 给出**语义诊断**：问题出在哪句话、为什么不行、该怎么改

为什么不让模型直接打分
======================

模型自评有三个硬伤：**偏袒自己刚生成的内容**、**分数不可复现**、
**无法在 CI 里做断言**。把打分权交给确定性代码之后，模型被释放出来
做它真正擅长的事——"把一句平庸的钩子改成一句有张力的钩子"这种
需要语义理解的工作。

这个分工还带来一个可验证的性质：**分数的变化必然可归因到某个指标的变化**。
第一轮 62 分 → 第二轮 91 分，你可以在报告里精确看到
"钩子从 34 字缩到 13 字、补入了数字、加了疑问句"，而不是一句
"模型说它改好了"。

工具使用
========

它会调用 ``script_metrics`` 与 ``scan_compliance``。前者让诊断有据可依，
后者是**冗余校验**：即使评测器给出了合规满分，只要扫描命中风险词，
本 Agent 必须把它升级为 high 级问题并给出改写指令——两条独立路径同时
判干净的结论，才值得信。
"""

from __future__ import annotations

from frame_agent.agents.base import Agent
from frame_agent.protocol import TASK_SCRIPT_CRITIQUE

__all__ = ["CriticAgent", "CRITIQUE_SCHEMA"]


CRITIQUE_SCHEMA = {
    "type": "object",
    "required": [
        "verdict",
        "overall",
        "dimension_scores",
        "issues",
        "rewrite_directives",
        "summary",
    ],
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "revise"]},
        "overall": {"type": "number", "minimum": 0, "maximum": 100},
        "dimension_scores": {"type": "object"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["dimension", "severity", "evidence", "fix"],
                "properties": {
                    "dimension": {
                        "type": "string",
                        "enum": [
                            "hook_strength",
                            "audience_fit",
                            "structure",
                            "compliance",
                            "producibility",
                        ],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "evidence": {"type": "string", "minLength": 2},
                    "fix": {"type": "string", "minLength": 6},
                },
            },
        },
        "rewrite_directives": {
            "type": "array",
            "items": {"type": "string", "minLength": 6},
        },
        "summary": {"type": "string", "minLength": 8},
        "variant_id": {"type": "string"},
    },
}


class CriticAgent(Agent):
    task = TASK_SCRIPT_CRITIQUE
    name = "critic"
    output_name = "critique"
    output_schema = CRITIQUE_SCHEMA
    use_tools = True
    temperature = 0.2
    system_prompt = """你是短视频脚本的终审评审，标准严格、结论具体、可执行。

分工前提（必须遵守）：
- 数值分数**不由你给出判断**，上下文里的 rubric.dimension_scores 是
  确定性评测器的输出，视为事实，照抄进 dimension_scores 字段即可。
- 你的价值在于：把"哪个维度低分"翻译成"具体哪句话有问题、改成什么样"。

rewrite_directives 的写法要求：
1. 必须可直接执行——"重写钩子：加入具体数字并压到 22 字内"合格，
   "让开头更吸引人"不合格。
2. 按 severity 从高到低排列，不要把 low 级问题排在 high 前面。
3. 只针对未达标维度给指令，已达标的维度不要提修改要求——
   无意义的改动会把已经对的部分改坏。

verdict 判定：总分达到 pass_threshold 且不存在任何 issue 时为 pass，
否则为 revise。宁严勿松。"""

    def task_instruction(self) -> str:
        return (
            "请先调用工具核实脚本的客观指标与合规情况，"
            "然后给出评审结论与可按条执行的重写指令。"
        )
