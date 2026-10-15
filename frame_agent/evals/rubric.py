"""确定性评测器：整条流水线的"裁判"。

核心立场
========

**评分权不能交给模型。** 理由与模型能力无关，问题出在"自己评自己"
这件事在方法论上成立：

- 模型倾向于给刚生成的内容打高分（self-preference bias），这是已知现象；
- 模型给出的分数不可复现，换个温度、换次调用就变，CI 里无法断言；
- 模型给不出"为什么是 73 分而不是 78 分"这种可追溯到指标的解释。

所以本模块只做一件事：**把 :mod:`frame_agent.tools.metrics` 算出的客观指标
映射成五个维度的分数**，每个分数都由若干条明确规则加权得出，
每条规则命中与否都记录在 ``notes`` 里。

五个维度与权重
==============

===============  ======  ============================================
维度              权重     为什么是这个权重
===============  ======  ============================================
hook_strength     0.35    前 3 秒决定去留，是短视频唯一的"生死线"
audience_fit      0.20    受众不匹配，钩子再强也是无效流量
structure         0.20    结构决定完播曲线中段能否守住
compliance        0.15    一票否决性质，但它是"门槛"而非"加分项"
producibility     0.10    拍不出来等于零，但它是必要条件而非充分条件
===============  ======  ============================================

权重之和为 1.0，由 :func:`_check_weights` 在导入时断言，
避免有人改了权重忘记同步，这类错误在评测系统里非常隐蔽。
"""

from __future__ import annotations

from typing import Callable

__all__ = [
    "DIMENSIONS",
    "DIMENSION_WEIGHTS",
    "evaluate",
    "score_hook",
    "score_audience",
    "score_structure",
    "score_compliance",
    "score_producibility",
]

# (维度名, 权重, 一句话说明)
DIMENSIONS: tuple[tuple[str, float, str], ...] = (
    ("hook_strength", 0.35, "前 3 秒抓取力"),
    ("audience_fit", 0.20, "与目标受众的贴合度"),
    ("structure", 0.20, "节拍结构与时长匹配"),
    ("compliance", 0.15, "合规安全度"),
    ("producibility", 0.10, "可拍摄性"),
)

DIMENSION_WEIGHTS: dict[str, float] = {name: weight for name, weight, _ in DIMENSIONS}

# 无效开场白：出现在钩子里说明前 3 秒被浪费掉了
_WEAK_OPENINGS = ("大家好", "今天我们来", "今天我们来聊", "我们先来说", "本期视频")


def _check_weights() -> None:
    total = round(sum(DIMENSION_WEIGHTS.values()), 6)
    if abs(total - 1.0) > 1e-9:
        raise AssertionError(f"评测维度权重之和必须为 1.0，当前为 {total}")


_check_weights()


# --------------------------------------------------------------------------
# 各维度打分：返回 (分数, 命中说明)
# --------------------------------------------------------------------------

def score_hook(metrics: dict) -> tuple[int, list[str]]:
    """钩子强度，权重最高，也是反思循环里主要"重写"的维度。

    加分项全部对应可观测的字面特征，不做语义判断（语义判断交给 Critic）。
    这正是分工的意义：字面特征机器判定，语义质量模型判定。
    """
    score = 0
    notes: list[str] = []

    hook = metrics.get("hook", "")
    length = int(metrics.get("hook_char_count") or 0)

    if metrics.get("hook_has_number"):
        score += 25
        notes.append("钩子含具体数字（可信度锚点）+25")
    else:
        notes.append("钩子缺少具体数字，说服力弱（未得分）")

    if metrics.get("hook_is_concise"):
        score += 25
        notes.append(f"钩子长度 {length} 字，在 22 字以内 +25")
    elif length <= 45:
        score += 8
        notes.append(f"钩子长度 {length} 字，偏长但尚可容忍 +8")
    else:
        notes.append(f"钩子长度 {length} 字，过长，前 3 秒念不完（未得分）")

    if metrics.get("hook_has_tension"):
        score += 20
        notes.append("钩子含悬念/冲突/反差标记 +20")

    if metrics.get("hook_question"):
        score += 10
        notes.append("钩子为疑问句，天然制造信息缺口 +10")

    if 8 <= length <= 22:
        score += 20
        notes.append("钩子长度落在最佳区间 8-22 字 +20")
    elif length > 22:
        notes.append("钩子长度超出最佳区间 8-22 字（未得分）")

    if hook and not any(opening in hook for opening in _WEAK_OPENINGS):
        score += 15
        notes.append("钩子未使用无效开场白 +15")

    return min(100, score), notes


def score_audience(metrics: dict) -> tuple[int, list[str]]:
    """受众贴合度，检查目标受众的关键词与痛点是否真的进入了文案。"""
    score = 0
    notes: list[str] = []

    hook_hits = metrics.get("hook_keyword_hits") or []
    hook_pain = metrics.get("hook_pain_hits") or []
    body_hits = metrics.get("body_keyword_hits") or []

    if hook_hits:
        score += 25
        notes.append(f"钩子命中受众关键词：{'、'.join(hook_hits)} +25")
    else:
        notes.append("钩子未命中任何受众关键词（未得分）")

    if hook_pain:
        score += 20
        notes.append(f"钩子直指受众痛点：{'、'.join(hook_pain)} +20")
    else:
        notes.append("钩子未直指受众痛点原话（未得分）")

    if len(body_hits) >= 2:
        score += 25
        notes.append(f"正文覆盖 {len(body_hits)} 个受众关键词 +25")
    elif len(body_hits) == 1:
        score += 15
        notes.append("正文仅覆盖 1 个受众关键词 +15")
    else:
        notes.append("正文未覆盖受众关键词（未得分）")

    # 是否出现具体的使用场景/人群指代，而不是泛化的"大家"
    full_text = (metrics.get("hook", "") + metrics.get("cta", "") + " " + " ".join(
        []  # 正文已折算进 body_keyword_hits，这里只看钩子与 CTA 的具体性
    ))
    if any(token in full_text for token in ("你", "你的", "通勤", "上班", "每天")):
        score += 15
        notes.append("文案使用了第二人称或具体场景，代入感强 +15")
    else:
        notes.append("文案缺少第二人称或具体场景指代（未得分）")

    distinct = set(hook_hits) | set(body_hits)
    # 阈值设为 2 而不是 3：中文受众词往往高度相关（"暗沉""熬夜"常同时出现），
    # 要求 3 个不同词会让大量合格文案因为"用词集中"而被误判为覆盖面不足。
    if len(distinct) >= 2:
        score += 15
        notes.append(f"全文覆盖 {len(distinct)} 个不同受众词，覆盖面达标 +15")
    else:
        notes.append(f"全文仅覆盖 {len(distinct)} 个不同受众词，覆盖面偏窄（未得分）")

    return min(100, score), notes


def score_structure(metrics: dict) -> tuple[int, list[str]]:
    """结构，节拍数量、镜头完整性、以及最实际的"塞不塞得进时长"。"""
    score = 0
    notes: list[str] = []

    beat_count = int(metrics.get("beat_count") or 0)
    if 3 <= beat_count <= 5:
        score += 30
        notes.append(f"{beat_count} 个节拍，符合短视频结构 +30")
    elif beat_count > 0:
        score += 10
        notes.append(f"{beat_count} 个节拍，偏离最佳区间 3-5 +10")
    else:
        notes.append("没有任何节拍（未得分）")

    if beat_count and metrics.get("beats_with_shot") == beat_count:
        score += 25
        notes.append("每个节拍都有镜头描述，可直接照拍 +25")
    else:
        notes.append("存在缺少镜头描述的节拍（未得分）")

    if beat_count and metrics.get("beats_with_on_screen") == beat_count:
        score += 20
        notes.append("每个节拍都有屏幕字幕设计 +20")
    else:
        notes.append("存在缺少屏幕字幕的节拍（未得分）")

    delta = float(metrics.get("duration_delta_ratio") or 0.0)
    est = metrics.get("est_duration_sec")
    target = metrics.get("target_duration_sec")
    if delta <= 0.15:
        score += 25
        notes.append(f"预估成片 {est}s vs 目标 {target}s，偏差 {delta:.0%} +25")
    elif delta <= 0.35:
        score += 12
        notes.append(f"预估成片 {est}s vs 目标 {target}s，偏差 {delta:.0%}，偏大但可剪辑修正 +12")
    else:
        notes.append(f"预估成片 {est}s vs 目标 {target}s，偏差 {delta:.0%}，超出可接受范围（未得分）")

    return min(100, score), notes


def score_compliance(metrics: dict, brief: dict | None = None) -> tuple[int, list[str]]:
    """合规，一票否决式维度：命中风险词直接腰斩，命中多个直接压到地板。"""
    from frame_agent.llm.mock import _RISK_TERMS

    text = " ".join(
        [
            str(metrics.get("hook") or ""),
            str(metrics.get("cta") or ""),
        ]
    )
    hits = [term for term in _RISK_TERMS if term in text]
    notes: list[str] = []

    if not hits:
        notes.append("未命中广告法风险词 +100")
        return 100, notes

    if len(hits) == 1:
        notes.append(f"命中 1 个风险词：{hits[0]}，必须改写 +30")
        return 30, notes

    notes.append(f"命中 {len(hits)} 个风险词：{'、'.join(hits)}，不可投放 +0")
    return 0, notes


def score_producibility(metrics: dict) -> tuple[int, list[str]]:
    """可拍摄性，只判断"能不能照着拍出来"，不判断拍得好不好。"""
    score = 0
    notes: list[str] = []

    beat_count = int(metrics.get("beat_count") or 0)
    if beat_count and metrics.get("beats_with_shot") == beat_count:
        score += 40
        notes.append("镜头描述完备 +40")
    if beat_count and metrics.get("beats_with_on_screen") == beat_count:
        score += 30
        notes.append("屏幕字幕完备 +30")
    if beat_count >= 3:
        score += 15
        notes.append("节拍数量足够支撑一条完整成片 +15")
    if metrics.get("cta_present"):
        score += 15
        notes.append("存在明确的行动号召 +15")
    else:
        notes.append("缺少明确行动词（点击/购物车/关注等），转化路径断（未得分）")

    return min(100, score), notes


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------

_SCORERS: dict[str, Callable[..., tuple[int, list[str]]]] = {
    "hook_strength": score_hook,
    "audience_fit": score_audience,
    "structure": score_structure,
    "compliance": score_compliance,
    "producibility": score_producibility,
}


def evaluate(
    variant: dict,
    brief: dict | None = None,
    *,
    pass_threshold: float = 82.0,
    metrics: dict | None = None,
) -> dict:
    """对一个脚本变体给出完整评测结果。

    返回结构里 ``notes`` 是**逐条加分明细**，这一点很重要：
    它为 Critic 提供了"为什么低分"的具体证据，也为报告提供了可审计的评分链。
    没有明细的分数在复盘时毫无价值。
    """
    from frame_agent.tools.metrics import analyze_script

    metrics = metrics or analyze_script(variant, brief)

    dimension_scores: dict[str, int] = {}
    notes: dict[str, list[str]] = {}
    for name, _weight, _desc in DIMENSIONS:
        scorer = _SCORERS[name]
        if name == "compliance":
            score, detail = scorer(metrics, brief)
        else:
            score, detail = scorer(metrics)
        dimension_scores[name] = score
        notes[name] = detail

    overall = round(
        sum(dimension_scores[name] * weight for name, weight, _ in DIMENSIONS), 2
    )
    weak = [
        name
        for name, _w, _d in DIMENSIONS
        if dimension_scores[name] < 0.78 * 100
    ]

    return {
        "variant_id": variant.get("id"),
        "angle": variant.get("angle"),
        "dimension_scores": dimension_scores,
        "weights": dict(DIMENSION_WEIGHTS),
        "notes": notes,
        "overall": overall,
        "pass_threshold": pass_threshold,
        "verdict": "pass" if overall >= pass_threshold else "revise",
        "weak_dimensions": weak,
        "metrics": metrics,
    }
