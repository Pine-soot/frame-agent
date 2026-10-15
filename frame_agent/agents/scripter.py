"""Scripter：脚本与分镜生成 Agent。

它在反思循环里被调用**两次以上**，这是本 Agent 最需要注意的设计点：

- **第 1 轮**：只拿 brief + 素材理解结果，生成 N 个不同"角度"的变体。
  要求多角度，是为了让后续评测有可比对象（单一方案无法比较优劣）。
- **第 2..R 轮**：额外拿到 Critic 的 ``rewrite_directives``，
  并被明确要求"逐条落实指令，并说明每条指令改在哪了"。

为什么让同一个 Agent 拿着指令重写，而不是重开一个新 Agent？
==========================================================

因为重写**必须保留上一轮的上下文**（原角度、原卖点分布、目标时长约束），
否则"重写"会退化成"重新生成"，导致已达标的部分被改坏，这是反思式
Agent 最常见的失败模式。所以 payload 里同时带上原变体、评测结论与指令。
"""

from __future__ import annotations

from frame_agent.agents.base import Agent
from frame_agent.protocol import TASK_SCRIPT_WRITE

__all__ = ["ScripterAgent", "SCRIPT_SET_SCHEMA"]


SCRIPT_SET_SCHEMA = {
    "type": "object",
    "required": ["variants"],
    "properties": {
        "variants": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "required": [
                    "id",
                    "angle",
                    "hook",
                    "beats",
                    "cta",
                    "target_duration_sec",
                ],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "angle": {"type": "string", "minLength": 2},
                    "hook": {"type": "string", "minLength": 4, "maxLength": 90},
                    "beats": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "required": ["t", "line", "shot", "on_screen"],
                            "properties": {
                                "t": {"type": "string", "minLength": 2},
                                "line": {"type": "string", "minLength": 1},
                                "shot": {"type": "string", "minLength": 2},
                                "on_screen": {"type": "string"},
                            },
                        },
                    },
                    "cta": {"type": "string", "minLength": 2},
                    "target_duration_sec": {"type": "number", "minimum": 5},
                    "rationale": {"type": "string"},
                },
            },
        }
    },
}


class ScripterAgent(Agent):
    task = TASK_SCRIPT_WRITE
    name = "scripter"
    output_name = "script_set"
    output_schema = SCRIPT_SET_SCHEMA
    use_tools = False  # 创作阶段不需要取证，证据已在 context 里
    temperature = 0.6  # 需要多样性；这也是本流水线里唯一调高温度的地方
    system_prompt = """你是一名短视频脚本与分镜编剧，服务对象是需要批量产出的内容团队。

产出要求：
1. **钩子**必须在 3 秒内建立张力，尽量控制在 22 字以内；
   优先使用具体数字、疑问句、冲突或反差，避免"大家好今天我们来聊聊"式开场。
2. **节拍**要给出明确时间区间、口播台词、镜头描述与屏幕字幕，
   确保拍摄团队可以照着直接执行，不需要二次追问。
3. **口播总字数**要与目标时长匹配：中文口播约每秒 4.5 字，
   30 秒片子的口播总量控制在 120-140 字之间。
4. **CTA** 要与 brief.goal 一致，并含明确动作词。
5. 严禁使用 brief.product.constraints 里禁止的表述；
   功效类描述只写成分与机制，不做效果承诺。
6. 每个变体必须是**不同的切入角度**，不是同一句话的换皮。

若上下文中给出了 rewrite_directives（上一轮评审意见），
必须逐条落实，并在 rationale 中说明每条指令具体改在了哪里。"""

    def task_instruction(self) -> str:
        return (
            "请基于 brief 与素材理解结果，生成指定数量的脚本变体。"
            "若存在 rewrite_directives，请优先执行这些指令。"
        )
