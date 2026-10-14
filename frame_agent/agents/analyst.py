"""Analyst —— 素材理解 Agent。

职责边界（很重要）
==================

它**只回答"素材里有什么"**，不回答"应该怎么写"。把这两件事分开是为了：

- 让"事实层"可以被独立验证（场景切分对不对、有没有合规风险）；
- 让后续的 Scripter 拿到一份**稳定、结构化**的事实输入，
  而不是让它一边理解素材一边创作——那是最容易产生幻觉的混合任务。

工具使用
========

它会通过 ``parse_transcript`` 与 ``extract_keyframes`` 两个工具主动取证，
再由 :meth:`Consolidate` 把两份观测归一成顶层 ``transcript`` / ``frames``。

注意 :meth:`consolidate` 里对 ``synthetic`` 帧的**强制标注**：
如果抽帧降级到了合成占位（无 ffmpeg 且无旁挂清单），
这里会显式写入 ``frames_source`` 与一条降级说明。这样报告里
"视觉结论的可信级别"是可见的，而不是被悄悄抹平。
"""

from __future__ import annotations

from frame_agent.agents.base import Agent
from frame_agent.protocol import TASK_MEDIA_ANALYZE

__all__ = ["AnalystAgent", "MEDIA_INSIGHT_SCHEMA"]


MEDIA_INSIGHT_SCHEMA = {
    "type": "object",
    "required": [
        "scenes",
        "dominant_mood",
        "audience_hint",
        "compliance_flags",
        "summary",
    ],
    "properties": {
        "scenes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {
                "type": "object",
                "required": [
                    "index",
                    "time_range",
                    "subject",
                    "mood",
                    "on_screen_text",
                    "compliance_risk",
                ],
                "properties": {
                    "index": {"type": "integer", "minimum": 1},
                    "time_range": {"type": "string", "minLength": 1},
                    "subject": {"type": "string", "minLength": 1},
                    "mood": {"type": "string", "minLength": 1},
                    "on_screen_text": {"type": "string"},
                    "compliance_risk": {"type": "array", "items": {"type": "string"}},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "dominant_mood": {"type": "string", "minLength": 1},
        "audience_hint": {"type": "string", "minLength": 1},
        "compliance_flags": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string", "minLength": 10},
        "frames_seen": {"type": "integer", "minimum": 0},
        "frames_source": {"type": "string", "enum": ["ffmpeg", "manifest", "synthetic"]},
    },
}


class AnalystAgent(Agent):
    task = TASK_MEDIA_ANALYZE
    name = "analyst"
    output_name = "media_insight"
    output_schema = MEDIA_INSIGHT_SCHEMA
    use_tools = True
    temperature = 0.1  # 事实抽取任务，越低越好
    system_prompt = """你是一名短视频素材分析专家，服务于内容生产流水线的最前端。

你的唯一职责是**如实描述素材里有什么**，不提供创作建议、不写文案。
输出必须让下游程序可以直接消费。

分析原则：
1. 场景切分以语义转折为准，而不是机械等分；每个场景给出时间区间。
2. 主体、屏幕文字、情绪调性都要基于观测到的证据，禁止推测未出现的内容。
3. 合规风险只报"文本/画面中实际出现的风险表述"，并原样引用命中词。
4. 若抽帧来源为 synthetic，说明画面细节未真实获取，此时画面相关结论必须保守。
5. 不要复述输入，不要输出解释性前言。"""

    def task_instruction(self) -> str:
        return (
            "请先判断是否需要调用工具补齐证据（字幕时间轴、关键帧清单），"
            "然后输出素材理解结果。"
        )

    def consolidate(self, payload: dict, observations: dict) -> dict:
        """把工具观测提升为顶层字段，并对降级情况做显式标注。"""
        merged = dict(payload)

        transcript = observations.get("parse_transcript")
        if isinstance(transcript, dict) and not transcript.get("__error__"):
            merged["transcript"] = {
                "path": transcript.get("path"),
                "stats": transcript.get("stats"),
                "cues": transcript.get("cues"),
            }
        elif transcript:
            merged.setdefault("degradation", []).append(
                f"字幕解析失败：{transcript.get('__error__')}"
            )

        frames = observations.get("extract_keyframes")
        if isinstance(frames, dict) and not frames.get("__error__"):
            merged["frames"] = frames.get("frames")
            merged["frames_source"] = frames.get("source")
            merged["frames_note"] = frames.get("reason") or ""
            if frames.get("source") == "synthetic":
                merged.setdefault("degradation", []).append(
                    "关键帧为合成占位（无 ffmpeg 且无旁挂清单），"
                    "画面层面的结论不可作为事实采信"
                )
        elif frames:
            merged.setdefault("degradation", []).append(
                f"关键帧抽取失败：{frames.get('__error__')}"
            )

        # 工具观测已经在上面归一成具体字段，无需再冗余挂一份 observations
        return merged
