"""PromptOps —— 生成提示词变体扩展与排序。

它解决的问题是内容生产里最耗时的一环：**同一套画面意图，怎么调提示词**。

与普通"改改提示词试试"的区别在于**单变量原则**：
每次只沿一个轴（主体 / 光照 / 机位 / 风格）改动，
其余保持完全一致。这样当结果变好时，你知道**是哪一项起了作用**；
一次改五处，即使变好了也无法沉淀成可复用的经验。

输出里刻意包含 ``expected_effect``：说明这一轴的变化**预期影响哪个指标**
（首帧停留率 / 质感呈现 / 完播中段 / 调性一致性）。
这让"调提示词"从玄学变成有假设、有归因的实验。

排序不由模型决定，而是按轴的**优先级**（主体 > 机位 > 光照 > 风格）：
这两项对首帧停留与节奏的影响最大，且不增加生成成本，
所以优先条件应该在小样本上验证。
"""

from __future__ import annotations

from frame_agent.agents.base import Agent
from frame_agent.protocol import TASK_PROMPT_EXPAND

__all__ = ["PromptOpsAgent", "PROMPT_SET_SCHEMA"]


PROMPT_SET_SCHEMA = {
    "type": "object",
    "required": ["variants", "ranking", "recommended_id"],
    "properties": {
        "variants": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "required": ["id", "axis", "axis_changes", "prompt"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "axis": {
                        "type": "string",
                        "enum": ["subject", "lighting", "camera", "style"],
                    },
                    "axis_changes": {"type": "object"},
                    "prompt": {"type": "string", "minLength": 8},
                    "negative_prompt": {"type": "string"},
                    "expected_effect": {"type": "string"},
                },
            },
        },
        "ranking": {"type": "array", "items": {"type": "string"}},
        "recommended_id": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
}


class PromptOpsAgent(Agent):
    task = TASK_PROMPT_EXPAND
    name = "promptops"
    output_name = "prompt_set"
    output_schema = PROMPT_SET_SCHEMA
    use_tools = False
    temperature = 0.5
    system_prompt = """你是生成式视觉的提示词工程专家。

方法论（必须遵守）：
1. 遵守**单变量原则**：每个变体只沿一个轴改变，其余部分与基线逐字一致。
   这样结果差异才能被归因到具体那一项。
2. 每个变体必须给出 expected_effect，说明预期影响哪个可观测指标。
3. 变体必须是可直接投喂的完整提示词，而不是"建议加上……"这样的描述。
4. 提示词要规避生成模型的高频失败模式：手指畸形、主体漂移、
   文字水印、多人混淆，并在 negative_prompt 中显式列出要排除的内容。
5. ranking 按轴的优先级排序：subject > camera > lighting > style。
   排列理由要在 reason 中说明，不要只给结论。"""

    def task_instruction(self) -> str:
        return (
            "请基于基线提示词与素材理解结果，扩展出指定数量的单变量变体，"
            "并给出可直接使用的完整提示词与排序建议。"
        )

    def consolidate(self, payload: dict, observations: dict) -> dict:
        """把素材理解结果中的关键字段提到顶层，方便模型直接引用。

        这里做的是"按需裁剪"：只把与画面相关的字段挑出来，
        而不是把整份 insight 塞进去——上下文里的无关信息越少，
        模型越不容易把注意力分到不该关注的地方。
        """
        merged = dict(payload)
        insight = payload.get("insight") or {}
        if isinstance(insight, dict) and insight:
            merged["visual_context"] = {
                "dominant_mood": insight.get("dominant_mood"),
                "scenes": [
                    {
                        "time_range": s.get("time_range"),
                        "subject": s.get("subject"),
                        "mood": s.get("mood"),
                    }
                    for s in (insight.get("scenes") or [])[:5]
                ],
                "compliance_flags": insight.get("compliance_flags") or [],
            }
        return merged
