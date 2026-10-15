"""工具层装配。

:func:`build_registry` 在每个 run 开始时按当次上下文（brief / 素材路径）
构造一份工具注册表。**用闭包绑定上下文**，而不是让模型在参数里传路径，
这既避免了模型编造路径，也让工具 schema 保持极简（参数越少，模型选错越少）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from frame_agent.tools.base import ToolRegistry, ToolSpec
from frame_agent.tools.media import extract_keyframes as _extract_keyframes
from frame_agent.tools.media import probe_video
from frame_agent.tools.metrics import analyze_script
from frame_agent.tools.transcript import load_transcript, transcript_stats

__all__ = [
    "ToolRegistry",
    "ToolSpec",
    "build_registry",
    "analyze_script",
    "load_transcript",
    "transcript_stats",
    "probe_video",
]


def build_registry(context: dict[str, Any]) -> ToolRegistry:
    """构造本次运行的工具集。

    ``context`` 是**可变的**：流水线在后半程会把生成好的脚本写回
    ``context["scripts"]``，于是 ``script_metrics`` 工具在后半程才可用。
    工具是"活的"，能反映当前运行状态，而不是一份静态快照。
    """
    registry = ToolRegistry()

    media = context.get("media") or {}
    transcript_path = media.get("transcript")
    video_path = media.get("video")
    brief = context.get("brief") or {}
    platform = brief.get("platform") or {}
    duration_hint = float(platform.get("target_duration_sec") or 0) * 3

    # ---- 工具 1：字幕解析 --------------------------------------------------
    def parse_transcript(path: str | None = None) -> dict:
        target = path or transcript_path
        if not target:
            raise FileNotFoundError(
                "本次运行未提供字幕文件；请在 brief.media.transcript 中指定"
            )
        cues = load_transcript(target, default_duration=duration_hint)
        return {
            "path": str(target),
            "stats": transcript_stats(cues),
            "cues": [c.to_dict() for c in cues],
        }

    registry.register(
        ToolSpec(
            name="parse_transcript",
            description=(
                "解析本次运行关联的字幕/转写文件，返回逐条时间轴以及统计量"
                "（总时长、语速字/秒、最长静默间隔、平均句长）。"
                "在需要判断素材节奏、或需要按时间轴切分场景时调用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "字幕文件路径；省略则使用 brief 中配置的文件",
                    }
                },
                "additionalProperties": False,
            },
            handler=parse_transcript,
        )
    )

    # ---- 工具 2：关键帧抽取 -----------------------------------------------
    def extract_keyframes(count: int = 6, video_path_arg: str | None = None) -> dict:
        target = video_path_arg or video_path or transcript_path
        if not target:
            raise FileNotFoundError("本次运行既无视频也无字幕文件，无法抽帧")
        # 旁挂清单（<name>.frames.json）由 media.load_frames_manifest 自动发现，
        # 无需在这里手动指定路径。
        return _extract_keyframes(
            Path(target),
            out_dir=context.get("frames_dir"),
            count=int(count),
            duration_hint=duration_hint,
        )

    registry.register(
        ToolSpec(
            name="extract_keyframes",
            description=(
                "抽取视频关键帧，返回帧的时间点与引用。"
                "注意返回值中的 source 字段标明来源级别：ffmpeg=真实抽帧、"
                "manifest=旁挂清单、synthetic=占位合成。"
                "若为 synthetic，则不能据此对画面内容下任何结论。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "期望抽取的帧数，默认 6",
                        "minimum": 1,
                        "maximum": 24,
                    },
                    "video_path_arg": {
                        "type": "string",
                        "description": "视频路径；省略则使用 brief 中配置的素材",
                    },
                },
                "additionalProperties": False,
            },
            handler=extract_keyframes,
        )
    )

    # ---- 工具 3：合规扫描 --------------------------------------------------
    def scan_compliance(text: str) -> dict:
        from frame_agent.llm.mock import _RISK_TERMS  # 复用同一份词表，避免两处漂移

        hits = [term for term in _RISK_TERMS if term in (text or "")]
        constraints = (brief.get("product") or {}).get("constraints") or []
        return {
            "text_length": len(text or ""),
            "risk_terms_hit": hits,
            "is_clean": not hits,
            "brief_constraints": constraints,
            "advice": (
                "命中风险词，建议改写为成分/机制层面的客观描述"
                if hits
                else "未命中内置风险词；仍需按 brief.constraints 人工复核"
            ),
        }

    registry.register(
        ToolSpec(
            name="scan_compliance",
            description=(
                "对一段文案做广告法风险词扫描，返回命中的风险词与 brief 中约定的约束。"
                "在输出最终文案前调用，用于自查。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "待扫描的文案全文"}
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=scan_compliance,
        )
    )

    # ---- 工具 4：脚本指标 --------------------------------------------------
    def script_metrics(variant_index: int = 0) -> dict:
        scripts = context.get("scripts") or []
        if not scripts:
            raise ValueError("当前运行尚未生成脚本，请先生成脚本再调用本工具")
        index = max(0, min(int(variant_index), len(scripts) - 1))
        variant = scripts[index]
        metrics = analyze_script(variant, brief)
        return {
            "variant_id": variant.get("id"),
            "variant_index": index,
            "metrics": metrics,
            "available_variants": len(scripts),
        }

    registry.register(
        ToolSpec(
            name="script_metrics",
            description=(
                "计算指定脚本变体的确定性量化指标：钩子长度与加分项、"
                "节拍数、每拍时长、预估成片时长与目标时长的偏差、"
                "CTA 是否具备、平均句长。用于客观核查脚本是否达标。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "variant_index": {
                        "type": "integer",
                        "description": "脚本在本次运行结果中的序号，从 0 开始",
                        "minimum": 0,
                    }
                },
                "additionalProperties": False,
            },
            handler=script_metrics,
        )
    )

    # ---- 工具 5：素材探测 --------------------------------------------------
    def video_probe() -> dict:
        target = video_path or transcript_path
        if not target:
            raise FileNotFoundError("本次运行未提供任何媒体文件")
        result = probe_video(Path(target))
        result["path"] = str(target)
        return result

    registry.register(
        ToolSpec(
            name="video_probe",
            description="探测媒体文件的分辨率、时长、帧率等元信息。",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=video_probe,
        )
    )

    return registry
