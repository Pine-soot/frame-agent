"""确定性离线后端 —— 让整条流水线**不依赖 API Key 也能完整跑通**。

为什么值得写这一个"假"后端
==========================

1. **可复现**：评审者 clone 仓库后直接运行，不需要任何 Key、不消耗任何额度，
   就能拿到与 README 中完全一致的指标。这是把"作品"变成"可验证证据"的前提。
2. **可测试**：CI 里没有网络也能覆盖多 Agent 编排、工具循环、反思收敛、
   预算熔断等最难的逻辑路径。
3. **可对比**：换真实模型时，输入上下文与输出 schema 完全一致，
   于是"离线基线 vs 在线效果"是一个受控实验，能直接量化模型增益。

它**不是**随机文本生成器
========================

必须说清楚它的诚实边界：这里的输出是由**同一份结构化上下文**渲染出来的
模板化内容，不是模型推理。它可以用来验证"管线是否正确地串联、校验、评测、
收敛"，但**不能**代表真实模型的创作水平。因此：

- 所有产出的报告里都会标注 ``backend=mock``；
- 数值类结论（评分、收敛轮数、成本）在离线态只证明**流程正确**；
- 真实创作质量必须接 :mod:`frame_agent.llm.openai_compat` 后重跑同一套评测。

即便如此，它仍严格实现了真实后端的两条关键行为，否则无法承担测试职责：

- **会发起工具调用**：首轮返回 ``tool_calls``，等 Agent 回灌观测后再输出最终 JSON；
- **会被"重写指令"真正改变**：第二轮拿到 critic 的 directives 后，
  钩子从"描述型"改为"数字+悬念型"，于是确定性评测器给出的分数**真的会上升**，
  反思循环的收敛是可观测的，而不是假装收敛。

另有一个刻意设计：**所有内容都由上下文推导，不使用随机数**。
同一个 brief 跑一万次，输出逐字节相同。这让测试可以做精确断言。
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from frame_agent.llm.base import (
    LLMResponse,
    Message,
    PermanentLLMError,
    ToolCall,
    Usage,
)
from frame_agent.protocol import (
    TASK_MEDIA_ANALYZE,
    TASK_PROMPT_EXPAND,
    TASK_SCRIPT_CRITIQUE,
    TASK_SCRIPT_WRITE,
    decode_ctx,
    detect_task,
    has_tool_observation,
)

__all__ = ["MockBackend"]

# --------------------------------------------------------------------------
# 固定的题材词表：内容由 brief 推导，但取值域是封闭的，便于断言
# --------------------------------------------------------------------------

_MOODS = ["克制专业", "轻松日常", "痛点共鸣", "理性对比", "温暖治愈"]

_ANGLES = [
    ("痛点切入", "从目标受众最难受的瞬间进入，先共情再给方案"),
    ("成分科普", "拆解核心成分/机制，用可信度换取停留时长"),
    ("对比测评", "同价位横向对比，用具体差异建立决策依据"),
    ("场景代入", "把产品放进一个具体生活场景，弱化广告感"),
]

_SHOT_POOL = [
    "中景口播，人物居左、右侧留白给字幕",
    "产品特写，浅景深，缓慢横移",
    "手部演示，俯拍 45 度，节奏加快",
    "分屏对比，左侧竞品右侧本品，同步播放",
    "生活空镜，手持微晃，营造真实感",
    "纯字幕卡，大字报式，强节奏卡点",
]

_RISK_TERMS = [
    "美白", "祛斑", "根治", "彻底", "最好", "第一", "永久", "百分百",
    "包治", "国家级", "全网最低",
]

_TENSION_WORDS = ["别再", "还在", "竟然", "其实", "为什么", "？", "?", "错了", "真相", "别买", "别再花"]


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """粗略估算 token：中文约 1 字 ≈ 0.6 token，这里取 len//2 作为保守近似。

    仅用于离线记账与测试，真实后端一律以 API 返回的 usage 为准。
    """
    return max(1, len(text) // 2)


def _first_number(candidates: Sequence[str], fallback: str = "3") -> str:
    """取出候选文本里第一个数字（只保留数字与小数点）。

    刻意**剥掉 ``%`` 等单位符号**：钩子模板是领域无关的，
    如果直接把 "5%" 塞进"5处关键点"这类句式会产出病句，
    而带上单位又没法在别的领域复用（"3个系统" → "3个关键点"）。
    统一只取数值是最稳妥的做法。
    """
    for item in candidates:
        match = re.search(r"\d+(?:\.\d+)?", item or "")
        if match:
            return match.group(0)
    return fallback


def _extract_days(candidates: Sequence[str], fallback: int = 30) -> int:
    for item in candidates:
        match = re.search(r"(\d+)\s*天", item or "")
        if match:
            return int(match.group(1))
    return fallback


def _split_timeline(total_sec: int, parts: int) -> list[str]:
    """把一个总时长均匀切成 ``parts`` 段，返回 ``"0-3s"`` 形式的区间。"""
    parts = max(1, parts)
    total = max(parts, int(total_sec))
    bounds = [round(i * total / parts) for i in range(parts + 1)]
    return [f"{bounds[i]}-{bounds[i + 1]}s" for i in range(parts)]


def _brief_facets(ctx: dict) -> dict:
    """从 brief 里抽出生成内容所需的全部素材，缺失时给安全兜底。"""
    brief = ctx.get("brief", {}) or {}
    product = brief.get("product", {}) or {}
    audience = brief.get("audience", {}) or {}
    platform = brief.get("platform", {}) or {}

    points = list(product.get("selling_points", []) or [])
    keywords = list(audience.get("keywords", []) or [])
    pains = list(audience.get("pain_points", []) or [])
    constraints = list(product.get("constraints", []) or [])

    return {
        "name": product.get("name") or "本产品",
        "category": product.get("category") or "同类产品",
        "points": points,
        "price": product.get("price_cny"),
        "constraints": constraints,
        "segment": audience.get("segment") or "目标受众",
        "keywords": keywords,
        "pains": pains,
        "platform": platform.get("name") or "短视频平台",
        "duration": int(platform.get("target_duration_sec") or 30),
        "goal": brief.get("goal") or "引导兴趣与转化",
        "number": _first_number(points),
        "days": _extract_days(points + constraints),
    }


# --------------------------------------------------------------------------
# 任务一：素材理解
# --------------------------------------------------------------------------

def _media_analyze(ctx: dict) -> dict:
    facets = _brief_facets(ctx)
    transcript = ctx.get("transcript", {}) or {}
    frames = ctx.get("frames", []) or []
    cues = transcript.get("cues", []) or []

    # 依据真实字幕内容切分场景：有字幕就按字幕切，没字幕就按固定段数切
    if cues:
        chunk = max(1, len(cues) // 4)
        groups = [cues[i : i + chunk] for i in range(0, len(cues), chunk)][:5]
    else:
        groups = [[]]

    scenes = []
    for index, group in enumerate(groups):
        text = " ".join(c.get("text", "") for c in group)
        scene_no = index + 1
        on_screen = ""
        if group:
            on_screen = (group[0].get("text") or "")[:14]
        scenes.append(
            {
                "index": scene_no,
                "time_range": (
                    f"{group[0].get('start', 0)}-{group[-1].get('end', 0)}s"
                    if group
                    else f"{index * 8}-{(index + 1) * 8}s"
                ),
                "subject": _pick_subject(text, facets),
                "mood": _MOODS[index % len(_MOODS)],
                "on_screen_text": on_screen,
                "keywords": _keywords_in(text, facets)[:4],
                "compliance_risk": _scan_risk(text + " " + on_screen),
            }
        )

    risk_flags = sorted({t for s in scenes for t in s["compliance_risk"]})
    dominant_mood = scenes[0]["mood"] if scenes else "克制专业"
    return {
        "scenes": scenes,
        "dominant_mood": dominant_mood,
        "audience_hint": facets["segment"],
        "compliance_flags": risk_flags,
        "summary": (
            f"素材共 {len(scenes)} 个语义段落，主体为{facets['category']}相关演示，"
            f"整体调性{dominant_mood}，"
            f"{'存在风险表述需要改写：' + '、'.join(risk_flags) if risk_flags else '未发现明显合规风险'}。"
        ),
        "frames_seen": len(frames),
    }


def _pick_subject(text: str, facets: dict) -> str:
    for point in facets["points"]:
        head = point.split()[0] if point else ""
        if head and head in text:
            return head
    return facets["category"]


def _keywords_in(text: str, facets: dict) -> list[str]:
    return [kw for kw in facets["keywords"] if kw and kw in text]


def _scan_risk(text: str) -> list[str]:
    return [term for term in _RISK_TERMS if term in text]


# --------------------------------------------------------------------------
# 任务二：脚本 + 分镜
# --------------------------------------------------------------------------

def _hook_descriptive(facets: dict, angle: str, index: int) -> str:
    """第一轮钩子：**刻意写得"安全但平庸"**——长、无数字、无悬念标记。

    这不是偷懒，而是为了演示反思循环真的在起作用：
    第一轮必然低于阈值，critic 给出指令，第二轮才有对比可言。

    注意这里**刻意不出现产品名与卖点**：卖点里带 "5%"、"28 天" 这类数字，
    一旦混进钩子就会命中 ``hook_has_number`` 而虚高。要演示"缺数字"，
    就必须保证钩子文本里真的没有数字。
    """
    pain = facets["pains"][index % len(facets["pains"])] if facets["pains"] else "日常困扰"
    category = facets["category"]
    if angle == "痛点切入":
        return f"很多人在面对{pain}的时候一直没有找到合适的方法，今天我们把这件事从头讲清楚"
    if angle == "成分科普":
        return f"关于{category}里的各种说法市场上差异很大，我们把它一步一步拆开讲明白"
    if angle == "对比测评":
        return f"我把同价位的几款{category}放在一起用了一段时间，结果和预期有一些差别"
    return f"想象一下这样的场景，你每天都会遇到{pain}，换一种方式之后会有什么不一样"


def _hook_optimized(facets: dict, angle: str, index: int) -> str:
    """第二轮钩子：**数字 + 悬念 + 受众词**，长度压到 22 字内。

    这些正是确定性评测器中 ``hook_strength`` 的加分明细项，
    所以分数上升是"设计出来的"，不是"碰巧的"。
    """
    kw = facets["keywords"][index % len(facets["keywords"])] if facets["keywords"] else "这个"
    kw2 = facets["keywords"][(index + 1) % len(facets["keywords"])] if facets["keywords"] else "问题"
    price = facets["price"]
    price_text = f"{price}元" if price else ""

    if angle == "痛点切入":
        return f"{kw}？{facets['days']}天自己看变化"
    if angle == "成分科普":
        return f"{kw2}？{facets['number']}处关键点先讲清"
    if angle == "对比测评":
        return f"别再乱买了，{price_text}差在这{facets['number']}点"
    return f"{kw}还在忍？{facets['days']}天见结果"


# 第一轮口播：偏长、偏铺垫，预估时长明显短于目标（时长维度也会失分）
_BEATS_V1 = (
    "今天我们来聊聊{point0}这件事",
    "很多人会担心{pain0}，这个问题值得展开",
    "接下来看一下{point1}",
    "如果感兴趣的话可以继续往下看",
)
_BEATS_V1_ONSCREEN = ("开场铺垫", "问题陈述", "卖点说明", "继续观看")
_BEATS_V1_CTA = "如果感兴趣可以继续往下看"

# 第二轮口播：字数贴近目标时长，且信息密度更高
_BEATS_V2 = (
    "先给结论：{point0}，重点是稳定和耐受度",
    "你现在的困扰是{pain0}，主要原因是角质层代谢节奏变慢",
    "具体怎么用：早晚各一次，先建立耐受再逐步上量，{point1}",
    "想要的话直接点下方购物车，现在下单还会附赠小样",
)
_BEATS_V2_ONSCREEN = ("结论前置", "原因拆解", "用法演示", "点击购物车")
_BEATS_V2_CTA = "点击下方购物车，{price}元直接带走"


def _beats(facets: dict, angle: str, optimized: bool, count: int = 4) -> list[dict]:
    timeline = _split_timeline(facets["duration"], count)
    point_cycle = facets["points"] or ["核心卖点"]
    pain_cycle = facets["pains"] or ["日常困扰"]
    # 不同角度从镜头池的不同位置起取，保证四个角度的分镜不雷同
    offset = next((i for i, (name, _) in enumerate(_ANGLES) if name == angle), 0)
    shots = [_SHOT_POOL[(i + offset) % len(_SHOT_POOL)] for i in range(count)]

    templates = _BEATS_V2 if optimized else _BEATS_V1
    on_screen = _BEATS_V2_ONSCREEN if optimized else _BEATS_V1_ONSCREEN

    lines = [
        text.format(
            point0=point_cycle[0],
            point1=point_cycle[min(1, len(point_cycle) - 1)],
            pain0=pain_cycle[0],
            price=facets["price"] or "",
        )
        for text in templates
    ]

    return [
        {
            "t": timeline[i],
            "line": lines[i % len(lines)],
            "shot": shots[i % len(shots)],
            "on_screen": on_screen[i % len(on_screen)],
        }
        for i in range(count)
    ]


_ANGLE_DESC = dict(_ANGLES)


def _script_write(ctx: dict) -> dict:
    facets = _brief_facets(ctx)
    count = int(ctx.get("variant_count") or 3)
    round_index = int(ctx.get("round") or 1)
    directives = ctx.get("rewrite_directives") or []
    optimized = bool(directives) or round_index > 1

    variants = []
    for index in range(count):
        angle, angle_desc = _ANGLES[index % len(_ANGLES)]
        hook = (
            _hook_optimized(facets, angle, index)
            if optimized
            else _hook_descriptive(facets, angle, index)
        )
        beats = _beats(facets, angle, optimized)
        if optimized:
            cta = (
                _BEATS_V2_CTA.format(price=facets["price"])
                if facets["price"]
                else "点击下方购物车立即查看"
            )
        else:
            # 第一轮刻意给一个**没有明确动作词**的收尾：
            # 这会让评测器的 cta_present 判定为 False，转化维度真实失分。
            cta = _BEATS_V1_CTA
        variants.append(
            {
                "id": f"v{index + 1}",
                "angle": angle,
                "hook": hook,
                "beats": beats,
                "cta": cta,
                "target_duration_sec": facets["duration"],
                "rationale": f"{angle_desc}；面向{facets['segment']}，投放在{facets['platform']}。",
                "round": round_index,
            }
        )
    return {"variants": variants, "angle_count": len(_ANGLES)}


# --------------------------------------------------------------------------
# 任务三：评审与重写指令
# --------------------------------------------------------------------------

_DIMENSION_MAX = {
    "hook_strength": 100,
    "audience_fit": 100,
    "structure": 100,
    "compliance": 100,
    "producibility": 100,
}

_DIRECTIVE_BY_DIMENSION = {
    "hook_strength": "重写钩子：加入具体数字、长度压到 22 字以内、用疑问句或冲突句制造张力",
    "audience_fit": "让钩子与首句命中目标受众的关键词与痛点原话，不要用泛化描述",
    "structure": "节拍数量与时长必须与目标时长对齐，每拍给出明确的时间区间",
    "compliance": "改写含风险表述的句子，改为成分/机制层面的客观描述",
    "producibility": "补齐每个节拍的镜头描述与屏幕字幕，确保可直接照拍",
}


def _script_critique(ctx: dict) -> dict:
    variant = ctx.get("variant", {}) or {}
    rubric = ctx.get("rubric", {}) or {}
    scores = rubric.get("dimension_scores", {}) or {}
    threshold = float(ctx.get("pass_threshold") or 78)
    observations = ctx.get("observations") or {}

    issues: dict[str, dict] = {}

    # 来源一：确定性评测器中未达标的维度
    for dimension, score in scores.items():
        maximum = _DIMENSION_MAX.get(dimension, 100)
        ratio = (score / maximum) if maximum else 0.0
        if ratio >= 0.78:
            continue
        severity = "high" if ratio < 0.6 else "medium"
        notes = (rubric.get("notes", {}) or {}).get(dimension, []) or []
        issues[dimension] = {
            "dimension": dimension,
            "severity": severity,
            "evidence": "；".join(notes) if notes else f"该维度得分 {score}/{maximum}",
            "fix": _DIRECTIVE_BY_DIMENSION[dimension],
        }

    # 来源二：**独立的合规扫描**。这一条刻意不依赖打分器：
    # 只要扫描命中风险词，无论评测器给多少分，都必须升级为 high 并给出改写指令。
    compliance = observations.get("scan_compliance") or {}
    risk_hits = compliance.get("risk_terms_hit") or []
    if risk_hits:
        issues["compliance"] = {
            "dimension": "compliance",
            "severity": "high",
            "evidence": f"合规扫描工具独立命中风险词：{'、'.join(risk_hits)}",
            "fix": _DIRECTIVE_BY_DIMENSION["compliance"],
        }
    elif compliance.get("is_clean") and "compliance" in issues:
        # 打分器判低但独立扫描认为干净 —— 说明是约束条款层面的问题，
        # 降级为 medium，避免对脚本做无谓的大改。
        issues["compliance"]["severity"] = "medium"
        issues["compliance"]["evidence"] = (
            f"未命中内置风险词，但需核对 brief 约束：{compliance.get('brief_constraints')}"
        )

    ordered = sorted(
        issues.values(), key=lambda i: {"high": 0, "medium": 1, "low": 2}[i["severity"]]
    )

    overall = float(rubric.get("overall") or 0.0)
    verdict = "pass" if overall >= threshold and not ordered else "revise"

    return {
        "verdict": verdict,
        "overall": overall,
        "dimension_scores": scores,
        "issues": ordered,
        "rewrite_directives": [i["fix"] for i in ordered],
        "summary": (
            f"总分 {overall:.1f}（阈值 {threshold:.0f}），"
            f"{'通过，可用于投放测试' if verdict == 'pass' else f'不达标，需针对 {len(ordered)} 个维度重写'}。"
        ),
        "variant_id": variant.get("id"),
    }


# --------------------------------------------------------------------------
# 任务四：Prompt 变体扩展
# --------------------------------------------------------------------------

_PROMPT_AXES = {
    "subject": [
        "主体置于画面中心偏下，占据约 1/3 画幅",
        "主体偏侧构图，为产品特写留出右侧空间",
        "改为手部特写，弱化人脸信息，降低肖像风险",
    ],
    "lighting": [
        "柔和侧逆光，色温约 5200K，保留皮肤质感",
        "顶部硬光制造阴影，强化产品轮廓",
        "漫反射柔光箱，消除高光油皮反光",
    ],
    "camera": [
        "固定机位中景，缓慢推近，节奏平稳",
        "手持轻微晃动跟拍，增强真实感与临场感",
        "滑轨横向移动，制造节奏变化与空间感",
    ],
    "style": [
        "清透日系，低饱和高明亮度，干净通透",
        "高对比冷调，强化科技感与专业感",
        "暖调生活流，叠加轻微颗粒，弱化广告感",
    ],
}

_AXIS_EFFECT = {
    "subject": "降低人物肖像依赖，提高素材复用率与合规安全度",
    "lighting": "改善产品质感呈现，直接影响首帧停留率",
    "camera": "改变观看节奏，影响完播曲线中段的保持率",
    "style": "决定内容与账号既有调性的一致性，影响关注转化",
}


def _prompt_expand(ctx: dict) -> dict:
    base = ctx.get("base_prompt") or ""
    axes = list(ctx.get("axes") or list(_PROMPT_AXES.keys()))
    count = int(ctx.get("count") or 4)
    negative = ctx.get("negative_prompt") or "低清晰度, 变形, 多余手指, 文字水印"

    variants = []
    for index in range(count):
        axis = axes[index % len(axes)]
        options = _PROMPT_AXES.get(axis, ["保持原设定不变"])
        chosen = options[index % len(options)]
        prompt = f"{base}；{chosen}"
        variants.append(
            {
                "id": f"p{index + 1}",
                "axis": axis,
                "axis_changes": {axis: chosen},
                "prompt": prompt,
                "negative_prompt": negative,
                "expected_effect": _AXIS_EFFECT.get(axis, "保持基线效果"),
            }
        )

    # 排序按轴优先级（subject > camera > lighting > style），确定性且可解释
    priority = {"subject": 0, "camera": 1, "lighting": 2, "style": 3}
    ranking = sorted(variants, key=lambda v: (priority.get(v["axis"], 9), v["id"]))

    return {
        "variants": variants,
        "ranking": [v["id"] for v in ranking],
        "recommended_id": ranking[0]["id"] if ranking else None,
        "reason": "优先改动主体与机位：这两项对首帧停留与节奏的影响最大，且不增加生成成本。",
    }


# --------------------------------------------------------------------------
# 后端实现
# --------------------------------------------------------------------------

class MockBackend:
    """确定性离线后端。

    ``latency_ms`` 会真实 sleep，用于演示延迟画像；测试里传 0 保持零耗时。
    """

    name = "mock"

    def __init__(self, *, latency_ms: float = 0.0) -> None:
        self.latency_ms = latency_ms
        self.calls = 0

    # -- 工具循环：首轮发起调用，拿到观测后输出最终答案 ----------------------

    def _tool_calls_for(self, task: str, ctx: dict) -> list[ToolCall]:
        """任务到工具调用的映射。

        注意 ``arguments`` 里带 ``_reason`` 字段：真实模型常会附带一段
        调用理由，把它记录下来对调试极有价值（"模型为什么调这个工具"）。
        """
        if task == TASK_MEDIA_ANALYZE:
            return [
                ToolCall(
                    id="call_transcript",
                    name="parse_transcript",
                    arguments={"_reason": "需要字幕时间轴来切分场景"},
                ),
                ToolCall(
                    id="call_frames",
                    name="extract_keyframes",
                    arguments={"_reason": "需要关键帧清单来判定画面主体", "count": 6},
                ),
            ]

        if task == TASK_SCRIPT_CRITIQUE:
            variant = ctx.get("variant") or {}
            # 合规复核必须基于**脚本全文**，而不是依赖已有的评分结论
            full_text = " ".join(
                [str(variant.get("hook") or "")]
                + [str(b.get("line") or "") for b in (variant.get("beats") or [])]
                + [str(variant.get("cta") or "")]
            )
            return [
                ToolCall(
                    id="call_metrics",
                    name="script_metrics",
                    arguments={
                        "variant_index": int(ctx.get("variant_index") or 0),
                        "_reason": "用客观指标支撑评审结论，避免主观打分",
                    },
                ),
                ToolCall(
                    id="call_scan",
                    name="scan_compliance",
                    arguments={
                        "text": full_text,
                        "_reason": "合规需独立扫描复核，不能只信打分器",
                    },
                ),
            ]

        return []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        import time as _time

        self.calls += 1
        if self.latency_ms:
            _time.sleep(self.latency_ms / 1000.0)

        system = "\n".join(m.content for m in messages if m.role == "system")
        task = detect_task(system)
        if task is None:
            raise PermanentLLMError(
                "离线后端只能处理带 [TASK:...] 标签的请求；"
                f"收到 system prompt 前 120 字符：{system[:120]!r}"
            )

        prompt_text = "".join(m.content or "" for m in messages)
        usage = Usage(prompt_tokens=_estimate_tokens(prompt_text))

        # 先解出上下文信封：工具调用的参数需要从中推导
        ctx = decode_ctx(messages)

        # 工具循环第一步：宣告要调用的工具，此时不产出最终答案
        pending = self._tool_calls_for(task, ctx)
        if tools and pending and not has_tool_observation(messages):
            text = json.dumps(
                {"status": "need_tools", "planned": [tc.name for tc in pending]},
                ensure_ascii=False,
            )
            return LLMResponse(
                text=text,
                model=self.name,
                usage=usage + Usage(completion_tokens=_estimate_tokens(text)),
                tool_calls=pending,
                finish_reason="tool_calls",
            )

        if not ctx:
            raise PermanentLLMError(
                f"任务 {task} 缺少 <ctx> 上下文信封，离线后端无法生成确定性内容"
            )

        handlers = {
            TASK_MEDIA_ANALYZE: _media_analyze,
            TASK_SCRIPT_WRITE: _script_write,
            TASK_SCRIPT_CRITIQUE: _script_critique,
            TASK_PROMPT_EXPAND: _prompt_expand,
        }
        handler = handlers.get(task)
        if handler is None:  # pragma: no cover - TASK_TAGS 已穷尽
            raise PermanentLLMError(f"离线后端不支持任务 {task}")

        data = handler(ctx)
        text = json.dumps(data, ensure_ascii=False, indent=2)
        return LLMResponse(
            text=text,
            model=self.name,
            usage=usage + Usage(completion_tokens=_estimate_tokens(text)),
            tool_calls=[],
            finish_reason="stop",
        )
