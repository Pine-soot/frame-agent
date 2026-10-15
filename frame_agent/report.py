"""交付报告渲染：把一堆 JSON 变成人能在 2 分钟内读完的东西。

报告按"先结论、再证据、最后过程"的顺序组织：

1. 顶部一屏放采纳方案、总分、是否达标、是否有降级，
   不往下翻就能知道这次能不能用；
2. 中部放逐维度评分明细（含每条加分规则），"为什么是 73 分"
   比"73 分"本身更值得看；
3. 底部放成本画像与完整叙事链（第几轮改了什么、分数怎么变的），
   用于复盘和方法论沉淀。

报告里会显式标注后端类型与降级项。一份用离线后端跑出来的高分报告，
如果被误当成真实模型效果，就会误导决策，所以在标题区直接写明 ``backend``。
"""

from __future__ import annotations

from typing import Any

__all__ = ["render_markdown"]

_BACKEND_LABEL = {
    "mock": "离线确定性后端（用于验证流程与可复现性，不代表真实创作水平）",
    "openai-compat": "真实模型后端（OpenAI 兼容端点）",
}


def _bullet_list(items: list[str], empty: str = "无") -> str:
    if not items:
        return f"- {empty}"
    return "\n".join(f"- {item}" for item in items)


def _dimension_table(scores: dict[str, int], weights: dict[str, float], notes: dict) -> str:
    from frame_agent.evals.rubric import DIMENSIONS

    rows = [
        "| 维度 | 得分 | 权重 | 说明 |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, weight, desc in DIMENSIONS:
        rows.append(
            f"| {name} | {scores.get(name, 0)} | {weight:.2f} | {desc} |"
        )
    return "\n".join(rows)


def _scoring_detail(notes: dict) -> str:
    blocks: list[str] = []
    for dimension, items in notes.items():
        blocks.append(f"**{dimension}**")
        for item in items:
            if "未得分" in item:
                blocks.append(f"- [ ] {item}")
            else:
                blocks.append(f"- [x] {item}")
        blocks.append("")
    return "\n".join(blocks)


def _storyboard_table(variant: dict) -> str:
    rows = [
        "| 时间 | 口播 | 镜头 | 屏幕字幕 |",
        "| --- | --- | --- | --- |",
    ]
    for beat in variant.get("beats") or []:
        rows.append(
            f"| {beat.get('t','')} | {beat.get('line','')} | "
            f"{beat.get('shot','')} | {beat.get('on_screen','')} |"
        )
    return "\n".join(rows)


def render_markdown(
    *,
    brief: dict,
    insight: dict,
    scripts: list[dict],
    scores: list[dict],
    critique: dict,
    prompt_set: dict,
    cost: dict,
    rounds: int,
    degradation_notes: list[str],
    rubric_result: dict,
    run_id: str,
    backend: str,
    round_history: list[dict] | None = None,
) -> str:
    """渲染完整报告。所有输入都来自落盘产物，因此可以事后重渲染。"""
    product = (brief.get("product") or {}).get("name") or "（未命名产品）"
    platform = (brief.get("platform") or {}).get("name") or "短视频平台"
    threshold = rubric_result.get("pass_threshold")
    overall = rubric_result.get("overall")
    verdict = rubric_result.get("verdict")
    best_id = rubric_result.get("variant_id")

    best = next((s for s in scripts if s.get("id") == best_id), scripts[0] if scripts else {})

    label = _BACKEND_LABEL.get(backend, backend)
    if backend not in _BACKEND_LABEL and backend.startswith("openai-compat"):
        label = _BACKEND_LABEL["openai-compat"]

    score_by_id = {s.get("variant_id"): s for s in scores}

    lines: list[str] = []
    lines.append(f"# 内容生产报告 · {product}")
    lines.append("")
    lines.append(f"- **运行 ID**：`{run_id}`")
    lines.append(f"- **投放平台**：{platform}")
    lines.append(f"- **后端**：`{backend}`，{label}")
    lines.append(
        f"- **结论**：采纳变体 `{best_id}`，总分 {overall} / 阈值 {threshold}，"
        f"{'达标' if verdict == 'pass' else '未达标'}"
    )
    lines.append(f"- **反思轮数**：{rounds}")
    lines.append(f"- **模型调用 / token**：{cost.get('total_calls')} 次 / {cost.get('total_tokens')}")
    if degradation_notes:
        lines.append(f"- **降级项**：{len(degradation_notes)} 条（见文末）")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- 1. 采纳方案 ----
    lines.append("## 1. 采纳方案")
    lines.append("")
    if best:
        lines.append(f"**角度**：{best.get('angle','')}")
        lines.append("")
        lines.append(f"**钩子**：{best.get('hook','')}")
        lines.append("")
        lines.append(f"**CTA**：{best.get('cta','')}")
        lines.append("")
        if best.get("rationale"):
            lines.append(f"**创作说明**：{best.get('rationale')}")
            lines.append("")
        lines.append("### 分镜表")
        lines.append("")
        lines.append(_storyboard_table(best))
        lines.append("")

    # ---- 2. 评分明细 ----
    lines.append("## 2. 评分明细（确定性评测器）")
    lines.append("")
    lines.append(
        _dimension_table(
            rubric_result.get("dimension_scores", {}),
            rubric_result.get("weights", {}),
            rubric_result.get("notes", {}),
        )
    )
    lines.append("")
    lines.append(f"**加权总分：{overall}**")
    lines.append("")
    lines.append("### 逐条加分/失分明细")
    lines.append("")
    lines.append(_scoring_detail(rubric_result.get("notes", {})))
    lines.append("")

    # ---- 3. 反思循环轨迹 ----
    lines.append("## 3. 反思循环轨迹（收敛过程）")
    lines.append("")
    if round_history:
        lines.append("| 轮次 | 采纳变体 | 本轮钩子 | 各变体总分 | 最高分 | 评审 |")
        lines.append("| ---: | --- | --- | --- | ---: | --- |")
        for entry in round_history:
            score_text = " / ".join(
                f"{k}:{v}" for k, v in (entry.get("scores") or {}).items()
            )
            lines.append(
                f"| {entry.get('round')} | {entry.get('best_id')} | {entry.get('hook')} | "
                f"{score_text} | {entry.get('best_score')} | `{entry.get('verdict')}` |"
            )
        lines.append("")
        for entry in round_history:
            directives = entry.get("directives") or []
            if directives:
                lines.append(
                    f"**第 {entry['round']} 轮的评审意见（直接作为下一轮的重写指令）：**"
                )
                lines.append("")
                lines.append(_bullet_list(directives))
                lines.append("")
        if len(round_history) > 1:
            first = round_history[0].get("best_score")
            last = round_history[-1].get("best_score")
            if first is not None and last is not None:
                lines.append(
                    f"最高分变化：{first} → {last}（{round(last - first, 2):+}）。"
                    "每一分都对应上表某个具体指标的变动。"
                )
                lines.append("")
    else:
        lines.append("本次未执行反思循环（max_rounds=1 或首轮即达标）。")
        lines.append("")

    # ---- 4. 全部候选择优对比 ----
    lines.append("## 4. 候选变体对比")
    lines.append("")
    lines.append("| 变体 | 角度 | 钩子 | 总分 | 判定 |")
    lines.append("| --- | --- | --- | ---: | --- |")
    for script in scripts:
        entry = score_by_id.get(script.get("id"), {})
        lines.append(
            f"| {script.get('id')} | {script.get('angle')} | {script.get('hook')} | "
            f"{entry.get('overall','-')} | {entry.get('verdict','-')} |"
        )
    lines.append("")

    # ---- 4. 评审结论 ----
    lines.append("## 5. 评审结论与重写指令")
    lines.append("")
    lines.append(f"**verdict**：`{critique.get('verdict')}`" + ("（兜底结论）" if critique.get("fallback") else ""))
    lines.append("")
    if critique.get("summary"):
        lines.append(critique["summary"])
        lines.append("")
    issues = critique.get("issues") or []
    if issues:
        lines.append("| 维度 | 严重度 | 证据 | 修改要求 |")
        lines.append("| --- | --- | --- | --- |")
        for issue in issues:
            lines.append(
                f"| {issue.get('dimension')} | {issue.get('severity')} | "
                f"{issue.get('evidence')} | {issue.get('fix')} |"
            )
        lines.append("")
    else:
        lines.append("无待修问题。")
        lines.append("")

    # ---- 6. 提示词优化 ----
    lines.append("## 6. 提示词单变量变体")
    lines.append("")
    if prompt_set.get("variants"):
        lines.append(f"**推荐**：`{prompt_set.get('recommended_id')}`：{prompt_set.get('reason','')}")
        lines.append("")
        lines.append("| 变体 | 轴 | 改动 | 预期影响 |")
        lines.append("| --- | --- | --- | --- |")
        for item in prompt_set["variants"]:
            changes = "；".join(f"{k}: {v}" for k, v in (item.get("axis_changes") or {}).items())
            lines.append(
                f"| {item.get('id')} | {item.get('axis')} | {changes} | "
                f"{item.get('expected_effect','')} |"
            )
        lines.append("")
        lines.append("### 完整提示词")
        lines.append("")
        for item in prompt_set["variants"]:
            lines.append(f"- **{item.get('id')}**（{item.get('axis')}）")
            lines.append(f"  ```")
            lines.append(f"  {item.get('prompt','')}")
            lines.append(f"  ```")
            if item.get("negative_prompt"):
                lines.append(f"  negative: `{item['negative_prompt']}`")
        lines.append("")
    else:
        lines.append("本次未产出提示词变体。")
        lines.append("")

    # ---- 6. 素材理解 ----
    lines.append("## 7. 素材理解结果")
    lines.append("")
    if insight:
        lines.append(f"- **主导情绪**：{insight.get('dominant_mood','')}")
        lines.append(f"- **受众提示**：{insight.get('audience_hint','')}")
        lines.append(f"- **合规风险**：{insight.get('compliance_flags') or '无'}")
        lines.append("")
        lines.append(f"> {insight.get('summary','')}")
        lines.append("")
        scenes = insight.get("scenes") or []
        if scenes:
            lines.append("| # | 时间 | 主体 | 情绪 | 屏幕文字 |")
            lines.append("| ---: | --- | --- | --- | --- |")
            for scene in scenes:
                lines.append(
                    f"| {scene.get('index')} | {scene.get('time_range')} | "
                    f"{scene.get('subject')} | {scene.get('mood')} | "
                    f"{scene.get('on_screen_text')} |"
                )
            lines.append("")
    else:
        lines.append("本次未获得素材上下文（已降级）。")
        lines.append("")

    # ---- 7. 成本画像 ----
    lines.append("## 8. 成本与延迟画像")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 总调用 | {cost.get('total_calls')} |")
    lines.append(f"| 成功 / 失败 | {cost.get('successful_calls')} / {cost.get('failed_calls')} |")
    lines.append(f"| 重试调用 | {cost.get('retry_calls')} |")
    lines.append(f"| 总 token | {cost.get('total_tokens')} |")
    latency = cost.get("latency_ms") or {}
    lines.append(f"| 延迟 p50 / p95 | {latency.get('p50')} ms / {latency.get('p95')} ms |")
    budget = cost.get("budget") or {}
    lines.append(
        f"| 预算使用 | 调用 {budget.get('calls')}/{budget.get('calls_limit')}，"
        f"token {budget.get('tokens')}/{budget.get('tokens_limit')} |"
    )
    lines.append("")

    # ---- 9. 降级说明 ----
    lines.append("## 9. 降级与风险说明")
    lines.append("")
    if degradation_notes:
        lines.append(_bullet_list(degradation_notes))
    else:
        lines.append("本次运行未触发任何降级。")
    lines.append("")
    lines.append(
        "> 说明：本报告由确定性评测器打分，分数可复现、可追溯至具体指标；"
        "语义诊断部分由模型给出。评分只用于初筛与回归对比，"
        "最终投放决策仍需人工终审。"
    )
    lines.append("")
    return "\n".join(lines)
