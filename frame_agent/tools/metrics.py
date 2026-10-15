"""脚本量化指标：把"文案好不好"里**可量化**的那部分抽出来。

为什么要做这件事
================

让模型自己给脚本打分，是 Agent 评测里最容易自欺的做法：模型倾向于
给自己刚生成的内容打高分（self-preference bias），而且分数无法复现。

本模块的思路是：**先把脚本压成一组确定性、可复现、可断言的数字**，
再让这些数字进评测器。这样带来三个好处：

1. 评分可复现，同样的脚本永远得到同样的分数，CI 里能做精确断言；
2. 评分可解释，每个分数都能追到某个具体数字（"钩子 22 字以内 +25"）；
3. 模型只做它擅长的事，语义诊断、创意改写，而不是充当计分器。

评测器（:mod:`frame_agent.evals.rubric`）就建立在这些指标之上。
"""

from __future__ import annotations

import re
import statistics

__all__ = ["analyze_script", "estimate_duration", "BEATS_PER_SECOND_ZH"]

# 中文口播舒适语速：约 4.5 字/秒（对应 270 字/分钟）
BEATS_PER_SECOND_ZH = 4.5

_NUMBER_RE = re.compile(r"\d")

# 制造悬念/张力的标记：疑问、否定、反差、时间承诺
_TENSION_MARKERS = (
    "？", "?", "别再", "还在", "竟然", "其实", "为什么", "错了",
    "真相", "别买", "千万别", "注意", "警告", "居然", "没想到",
)

# 明确的行动号召标记
_CTA_MARKERS = (
    "点击", "购物车", "主页", "关注", "下单", "领取", "私信", "评论",
    "链接", "小黄车", "橱窗", "收藏", "转发", "了解更多",
)

# 分镜时间区间："0-8s" / "0~8秒" / "0 - 8 s"
_TIME_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-~—–至]\s*(\d+(?:\.\d+)?)")

_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;，,、\n]+")


def estimate_duration(text: str, chars_per_second: float = BEATS_PER_SECOND_ZH) -> float:
    """由中文文案反推成片时长（秒）。

    只统计实际要念出来的字，剔除标点与空白。用于校验
    "脚本能不能塞进目标时长"，这是投放侧最实际的约束之一。
    """
    if not text:
        return 0.0
    spoken = re.sub(r"[\s\u3000]", "", text)
    spoken = re.sub(r"[，。！？、；：\"'（）()【】\[\]，,\.!\?;:]", "", spoken)
    return round(len(spoken) / max(0.1, chars_per_second), 2)


def _beat_durations(beats: list[dict]) -> list[float]:
    """从分镜的 ``t`` 字段解析每拍时长。"""
    durations: list[float] = []
    for beat in beats or []:
        raw = str(beat.get("t") or beat.get("time") or "")
        match = _TIME_RANGE_RE.search(raw)
        if match:
            start, end = float(match.group(1)), float(match.group(2))
            if end >= start:
                durations.append(round(end - start, 2))
    return durations


def _sentence_lengths(text: str) -> list[int]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text or "")]
    return [len(p) for p in parts if p]


def analyze_script(variant: dict, brief: dict | None = None) -> dict:
    """把一个脚本变体压成指标字典。

    ``brief`` 可选。一旦提供，就能额外算出"受众契合度"相关的量：
    钩子是否命中了目标受众的关键词与痛点原话。缺少 brief 时这些字段为 0，
    评测器会自动跳过对应维度而不是给 0 分，避免因为缺输入而误判脚本。
    """
    variant = variant or {}
    brief = brief or {}
    audience = brief.get("audience", {}) or {}

    hook = str(variant.get("hook") or "")
    beats = list(variant.get("beats") or [])
    cta = str(variant.get("cta") or "")
    target_duration = float(variant.get("target_duration_sec") or 0) or float(
        (brief.get("platform") or {}).get("target_duration_sec") or 30
    )

    body_text = " ".join(str(b.get("line") or "") for b in beats)
    full_text = f"{hook} {body_text} {cta}".strip()

    keywords = [str(k) for k in (audience.get("keywords") or []) if str(k).strip()]
    pains = [str(p) for p in (audience.get("pain_points") or []) if str(p).strip()]

    # 关键词命中：钩子权重最高（前 3 秒决定去留），所以单独统计
    hook_keyword_hits = [k for k in keywords if k in hook]
    body_keyword_hits = [k for k in keywords if k in body_text or k in cta]
    hook_pain_hits = [p for p in pains if p in hook or any(tok in hook for tok in _tokens(p))]

    durations = _beat_durations(beats)
    sentence_lengths = _sentence_lengths(full_text)
    est = estimate_duration(full_text)

    return {
        # -- 钩子 --
        "hook": hook,
        "hook_char_count": len(hook),
        "hook_has_number": bool(_NUMBER_RE.search(hook)),
        "hook_is_concise": 0 < len(hook) <= 22,
        "hook_has_tension": any(marker in hook for marker in _TENSION_MARKERS),
        "hook_keyword_hits": hook_keyword_hits,
        "hook_pain_hits": hook_pain_hits,
        "hook_question": ("？" in hook) or ("?" in hook),
        # -- 结构 --
        "beat_count": len(beats),
        "beats_with_shot": sum(1 for b in beats if str(b.get("shot") or "").strip()),
        "beats_with_on_screen": sum(
            1 for b in beats if str(b.get("on_screen") or "").strip()
        ),
        "beat_durations": durations,
        "pacing_stddev": round(statistics.pstdev(durations), 2) if len(durations) > 1 else 0.0,
        # -- 转化 --
        "cta": cta,
        "cta_present": any(marker in cta for marker in _CTA_MARKERS),
        # -- 篇幅与时长 --
        "char_count": len(full_text),
        "body_char_count": len(body_text) + len(cta),
        "est_duration_sec": est,
        "target_duration_sec": target_duration,
        "duration_delta_ratio": (
            round(abs(est - target_duration) / target_duration, 3)
            if target_duration > 0
            else 0.0
        ),
        # -- 语言 --
        "sentence_lengths": sentence_lengths,
        "mean_sentence_len": (
            round(sum(sentence_lengths) / len(sentence_lengths), 2)
            if sentence_lengths
            else 0.0
        ),
        "max_sentence_len": max(sentence_lengths) if sentence_lengths else 0,
        # -- 受众素材（供评测器判断是否具备评测前提）--
        "audience_keyword_pool": keywords,
        "audience_pain_pool": pains,
        "body_keyword_hits": body_keyword_hits,
    }


def _tokens(text: str) -> list[str]:
    """把一句中文痛点切成 2 字滑窗，用于宽松匹配（"熬夜后肤色暗沉" ↔ "暗沉"）。

    中文没有空格，精确子串匹配会大面积漏判。2-gram 滑窗是个便宜且够用的近似。
    """
    clean = re.sub(r"[\s，。！？、；：]", "", text or "")
    if len(clean) <= 2:
        return [clean] if clean else []
    return [clean[i : i + 2] for i in range(len(clean) - 1)]
