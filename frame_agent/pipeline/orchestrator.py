"""编排层 —— 把四个 Agent 串成一条可交付、可观测、可续跑的流水线。

反思循环（reflection loop）的收敛条件
=====================================

::

    round = 1
    scripts = Scripter(brief, insight)                  # 多角度生成
    best, rubric = 用确定性评测器挑最优                  # 谁最好由数字决定
    while round < max_rounds:
        critique = Critic(best, rubric)                 # 语义诊断 + 重写指令
        if critique.verdict == "pass": break            # 达标即停，不做无意义改写
        scripts = Scripter(brief, insight,              # 带着指令重写
                           previous=best,
                           directives=critique.rewrite_directives,
                           round=round + 1)
        best, rubric = 重新评测并挑最优
        round += 1

三个刻意的设计决定：

1. **达标即停**，不是固定跑满轮数。第 1 轮就过阈值的 case 只花 1 轮成本，
   这对批量生产是决定性的成本差异。
2. **重写时把上一版一起带上**，否则模型会"重新发明"而不是"针对性修改"，
   容易把已经达标的部分改坏（详见 :mod:`frame_agent.agents.scripter`）。
3. **预算熔断优先于收敛**。如果 token/调用次数快耗尽，宁可在未收敛时停下
   并如实标注，也不要为了"跑满"而超支——真实业务里超支比不完美更不可接受。

降级而非崩溃
============

任何一个 Agent 失败（结构校验过不去、模型全挂了），本层都不会让整个 run 崩掉，
而是把失败**记进 degradation_notes** 并继续能走的部分。
理由：内容生产的产出是"给人初筛用的候选集"，
交出"3 个方案 + 1 条明确的失败说明"远比什么都不交有价值。
但如果最终一个可用变体都没有，就如实返回失败——不伪造产物。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from frame_agent.agents import AnalystAgent, CriticAgent, PromptOpsAgent, ScripterAgent
from frame_agent.evals.rubric import evaluate
from frame_agent.llm.base import BudgetExceeded
from frame_agent.pipeline.state import RunState
from frame_agent.tools import build_registry

__all__ = ["ContentPipeline"]


class ContentPipeline:
    """内容生产流水线。

    :param router: :class:`~frame_agent.llm.router.ModelRouter`
    :param out_dir: 产物目录（每次 run 一个目录，便于对比）
    :param max_rounds: 反思循环的最大轮数（1 表示不做重写）
    :param variant_count: 每轮生成的脚本变体数
    :param pass_threshold: 达标分数线
    """

    def __init__(
        self,
        router: Any,
        *,
        out_dir: str | Path,
        max_rounds: int = 2,
        variant_count: int = 3,
        pass_threshold: float = 82.0,
        frames_dir: str | Path | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        self.router = router
        self.out_dir = Path(out_dir)
        self.max_rounds = max(1, int(max_rounds))
        self.variant_count = max(1, int(variant_count))
        self.pass_threshold = float(pass_threshold)
        self.frames_dir = frames_dir
        # brief 里的素材路径若为相对路径，按此目录解析（默认当前工作目录）。
        # 显式引入这个参数是为了让内置示例能在任意 cwd 下跑通，
        # 而不必要求使用者先 cd 到仓库根目录。
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.state: RunState | None = None
        # 每轮生成+评审的分数轨迹，用于报告里的"收敛过程"一节
        self.round_history: list[dict] = []

    # -- 内部工具 -----------------------------------------------------------

    def _resolve_media(self, media: dict) -> dict:
        """把相对素材路径解析成实际存在的绝对路径。

        只在"原路径不存在、但 base_dir 下存在"时才替换——
        这样显式传入的正确相对路径（相对 cwd）不会被误改。
        """
        resolved = dict(media)
        for key in ("transcript", "video", "frames_manifest"):
            value = resolved.get(key)
            if not value or Path(str(value)).is_absolute():
                continue
            if Path(str(value)).exists():
                continue
            candidate = self.base_dir / str(value)
            if candidate.exists():
                resolved[key] = str(candidate)
        return resolved

    @property
    def _calls_so_far(self) -> int:
        return len(self.router.records)

    def _tokens_so_far(self) -> int:
        return sum(
            r.prompt_tokens + r.completion_tokens for r in self.router.records
        )

    def _write_json(self, name: str, payload: Any, kind: str, description: str) -> str:
        import json

        assert self.state is not None
        path = self.out_dir / name
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self.state.add_artifact(name, path, kind, description)
        return str(path)

    def _budget_exhausted(self) -> bool:
        try:
            self.router.budget.check()
        except BudgetExceeded:
            return True
        return False

    # -- 主流程 -------------------------------------------------------------

    def run(
        self,
        brief: dict,
        *,
        brief_path: str = "",
        backend: str = "mock",
    ) -> dict:
        """执行一次完整流水线，返回结构化结果并落盘全部产物。"""
        state = RunState.create(
            self.out_dir, brief, brief_path=brief_path, backend=backend
        )
        self.state = state

        media = self._resolve_media(dict(brief.get("media") or {}))
        context: dict[str, Any] = {
            "brief": brief,
            "media": media,
            "frames_dir": str(self.frames_dir) if self.frames_dir else None,
            "scripts": [],       # 由 Scripter 填充，供 script_metrics 工具读取
            "variant_index": 0,  # 由 Critic 阶段填充
        }
        registry = build_registry(context)

        state.log_step("ingest", "ok", detail=f"素材配置：{sorted(media) or '无'}")
        state.append_trace({"event": "run_start", "brief_hash": state.brief_hash})

        # ---- 1. 素材理解 ----
        insight = self._step_analyze(brief, context, registry)

        # ---- 2. 生成 + 反思循环 ----
        scripts, rubric_result, critique, rounds = self._step_generate(
            brief, insight, context, registry
        )

        # ---- 3. Prompt 优化 ----
        prompt_set = self._step_promptops(brief, insight, scripts, rubric_result, registry)

        # ---- 4. 落盘 ----
        return self._finalize(brief, insight, scripts, rubric_result, critique, prompt_set, rounds)

    # -- 各阶段 -------------------------------------------------------------

    def _step_analyze(self, brief: dict, context: dict, registry: Any) -> dict:
        assert self.state is not None
        agent = AnalystAgent(self.router, registry)
        before = self._calls_so_far
        result = agent.run({"brief": brief, "media": context["media"]})
        calls = self._calls_so_far - before

        detail = f"{calls} 次模型调用，{len(result.tool_calls)} 次工具调用"
        if result.ok and result.data:
            insight = result.data
            state_status = "ok"
            detail += f"，识别 {len(insight.get('scenes') or [])} 个场景"
            if insight.get("compliance_flags"):
                detail += f"，风险项 {insight['compliance_flags']}"
        else:
            insight = {}
            state_status = "degraded"
            note = f"素材理解未产出可用结果：{result.degradation or '结构校验未通过'}"
            self.state.note_degradation(note)
            detail += "，已降级为无素材上下文"

        # Agent 自己上报的降级原因（例如关键帧只有合成占位）要一路带到最终报告
        for line in result.degradation:
            self.state.note_degradation(f"素材层：{line}")

        self.state.log_step(
            "analyze", state_status, detail=detail, metrics={"model_calls": calls}
        )
        self.state.append_trace({"event": "agent", **result.to_dict()})
        return insight

    def _score_all(self, scripts: list[dict], brief: dict) -> list[dict]:
        return [evaluate(s, brief, pass_threshold=self.pass_threshold) for s in scripts]

    def _step_generate(
        self,
        brief: dict,
        insight: dict,
        context: dict,
        registry: Any,
    ) -> tuple[list[dict], dict, dict, int]:
        assert self.state is not None
        scripter = ScripterAgent(self.router)
        critic = CriticAgent(self.router, registry)

        scripts: list[dict] = []
        scripts_text = ""
        rubric_result: dict = {}
        critique: dict = {}
        rounds = 0
        round_history: list[dict] = []

        for round_index in range(1, self.max_rounds + 1):
            # 预算见底就不要开始新的一轮——宁可少一轮，也不要半途熔断
            if self._budget_exhausted():
                note = (
                    f"预算在第 {round_index} 轮开始前耗尽，已停止反思循环；"
                    "结果可能未收敛"
                )
                self.state.note_degradation(note)
                self.state.log_step("generate", "degraded", detail=note)
                break

            payload: dict[str, Any] = {
                "brief": brief,
                "insight": insight,
                "variant_count": self.variant_count,
                "round": round_index,
            }
            if rounds >= 1 and scripts_text:
                payload["previous_scripts"] = scripts_text
                payload["rewrite_directives"] = critique.get("rewrite_directives") or []
                payload["previous_scores"] = rubric_result.get("dimension_scores")

            before = self._calls_so_far
            result = scripter.run(payload)
            calls = self._calls_so_far - before
            rounds = round_index

            if not result.ok or not (result.data or {}).get("variants"):
                note = f"第 {round_index} 轮脚本生成失败，已停止反思循环"
                self.state.note_degradation(note)
                self.state.log_step(
                    "generate", "degraded", detail=note,
                    metrics={"round": round_index, "model_calls": calls},
                )
                break

            scripts = list(result.data["variants"])
            context["scripts"] = scripts
            scored = self._score_all(scripts, brief)
            best_index = max(
                range(len(scored)), key=lambda i: scored[i]["overall"]
            )
            rubric_result = scored[best_index]
            scripts_text = _render_scripts_for_critic(scripts)

            self.state.log_step(
                f"generate:round{round_index}",
                "ok",
                detail=(
                    f"生成 {len(scripts)} 个变体，最高分 "
                    f"{rubric_result['overall']}（{rubric_result['variant_id']}），"
                    f"下限 {min(s['overall'] for s in scored)}"
                ),
                metrics={
                    "round": round_index,
                    "model_calls": calls,
                    "best_score": rubric_result["overall"],
                },
            )
            self.state.append_trace(
                {
                    "event": "generate",
                    "round": round_index,
                    "scores": [s["overall"] for s in scored],
                    "variant_ids": [s["variant_id"] for s in scored],
                }
            )

            # ---- 评审 ----
            context["variant_index"] = best_index
            critic_before = self._calls_so_far
            critique_result = critic.run(
                {
                    "brief": brief,
                    "variant": scripts[best_index],
                    "variant_index": best_index,
                    "rubric": {
                        "dimension_scores": rubric_result["dimension_scores"],
                        "notes": rubric_result["notes"],
                        "overall": rubric_result["overall"],
                    },
                    "pass_threshold": self.pass_threshold,
                    "round": round_index,
                }
            )
            critic_calls = self._calls_so_far - critic_before

            if not critique_result.ok or not critique_result.data:
                note = f"第 {round_index} 轮评审失败，采用评测器结论作为兜底"
                self.state.note_degradation(note)
                critique = _synthetic_critique(rubric_result, self.pass_threshold)
            else:
                critique = critique_result.data

            self.state.log_step(
                f"critique:round{round_index}",
                "ok",
                detail=(
                    f"verdict={critique.get('verdict')}，"
                    f"{len(critique.get('issues') or [])} 个问题，"
                    f"{len(critique.get('rewrite_directives') or [])} 条重写指令"
                ),
                metrics={"round": round_index, "model_calls": critic_calls},
            )
            self.state.append_trace(
                {
                    "event": "critique",
                    "round": round_index,
                    "verdict": critique.get("verdict"),
                    "issues": [i.get("dimension") for i in (critique.get("issues") or [])],
                    "directives": critique.get("rewrite_directives") or [],
                    "agent_trace": critique_result.to_dict(),
                }
            )

            if critique.get("verdict") == "pass":
                round_history.append(
                    {
                        "round": round_index,
                        "best_id": rubric_result.get("variant_id"),
                        "best_score": rubric_result.get("overall"),
                        "scores": {s["variant_id"]: s["overall"] for s in scored},
                        "verdict": critique.get("verdict"),
                        "directives": [],
                        "hook": (scripts[best_index] or {}).get("hook"),
                        "dimension_scores": rubric_result.get("dimension_scores"),
                    }
                )
                break

            round_history.append(
                {
                    "round": round_index,
                    "best_id": rubric_result.get("variant_id"),
                    "best_score": rubric_result.get("overall"),
                    "scores": {s["variant_id"]: s["overall"] for s in scored},
                    "verdict": critique.get("verdict"),
                    "directives": critique.get("rewrite_directives") or [],
                    "hook": (scripts[best_index] or {}).get("hook"),
                    "dimension_scores": rubric_result.get("dimension_scores"),
                }
            )

        self.round_history = round_history
        return scripts, rubric_result, critique, rounds

    def _step_promptops(
        self,
        brief: dict,
        insight: dict,
        scripts: list[dict],
        rubric_result: dict,
        registry: Any,
    ) -> dict:
        assert self.state is not None
        if not scripts:
            self.state.log_step("promptops", "skipped", detail="无可用脚本，跳过")
            return {}

        best = _pick_best(scripts, rubric_result)
        agent = PromptOpsAgent(self.router)
        before = self._calls_so_far
        result = agent.run(
            {
                "brief": brief,
                "insight": insight,
                "base_prompt": _build_base_prompt(brief, insight, best),
                "axes": ["subject", "camera", "lighting", "style"],
                "count": 4,
                "negative_prompt": "低清晰度, 变形, 多余手指, 文字水印, 比例失真",
            }
        )
        calls = self._calls_so_far - before

        if result.ok and result.data:
            self.state.log_step(
                "promptops",
                "ok",
                detail=f"生成 {len(result.data.get('variants') or [])} 个提示词变体，"
                f"推荐 {result.data.get('recommended_id')}",
                metrics={"model_calls": calls},
            )
            return result.data

        self.state.note_degradation("提示词扩展未产出可用结果")
        self.state.log_step("promptops", "degraded", detail="已跳过")
        return {}

    # -- 收尾 ---------------------------------------------------------------

    def _finalize(
        self,
        brief: dict,
        insight: dict,
        scripts: list[dict],
        rubric_result: dict,
        critique: dict,
        prompt_set: dict,
        rounds: int,
    ) -> dict:
        assert self.state is not None
        from frame_agent.report import render_markdown

        if not scripts or not rubric_result:
            self.state.log_step("finalize", "failed", detail="没有任何可用脚本变体")
            self.state.save()
            return {
                "run_id": self.state.run_id,
                "status": "failed",
                "reason": "流水线未能产出任何可用的脚本变体",
                "degradation_notes": self.state.degradation_notes,
                "artifacts": [],
            }

        best = _pick_best(scripts, rubric_result)
        scored = self._score_all(scripts, brief)
        cost = self.router.report()

        self._write_json("media_insight.json", insight, "insight", "素材理解结果")
        self._write_json("scripts.json", {"variants": scripts}, "scripts", "脚本与分镜变体")
        self._write_json(
            "scores.json", {"variants": scored}, "scores", "各变体的确定性评测结果"
        )
        self._write_json("critique.json", critique, "critique", "评审结论与重写指令")
        self._write_json("prompt_set.json", prompt_set, "prompts", "提示词单变量变体")
        self._write_json("cost.json", cost, "cost", "模型调用成本与延迟画像")

        report_path = self.out_dir / "report.md"
        report_path.write_text(
            render_markdown(
                brief=brief,
                insight=insight,
                scripts=scripts,
                scores=scored,
                critique=critique,
                prompt_set=prompt_set,
                cost=cost,
                rounds=rounds,
                degradation_notes=self.state.degradation_notes,
                rubric_result=rubric_result,
                run_id=self.state.run_id,
                backend=self.state.backend,
                round_history=self.round_history,
            ),
            encoding="utf-8",
        )
        self.state.add_artifact("report.md", report_path, "report", "人读版交付报告")

        self.state.log_step(
            "finalize",
            "ok",
            detail=f"采纳 {best.get('id')}，总分 {rubric_result['overall']}，共 {rounds} 轮",
            metrics={
                "rounds": rounds,
                "total_model_calls": cost.get("total_calls"),
                "total_tokens": cost.get("total_tokens"),
            },
        )
        self.state.save()

        return {
            "run_id": self.state.run_id,
            "status": "ok",
            "backend": self.state.backend,
            "rounds": rounds,
            "best_variant": best,
            "eval": rubric_result,
            "scores": scored,
            "critique": critique,
            "insight": insight,
            "prompt_set": prompt_set,
            "cost": cost,
            "model_calls": cost.get("total_calls"),
            "degradation_notes": self.state.degradation_notes,
            "round_history": self.round_history,
            "artifacts": [a["path"] for a in self.state.artifacts],
        }


# --------------------------------------------------------------------------
# 辅助函数（模块级，便于单测）
# --------------------------------------------------------------------------

def _pick_best(scripts: list[dict], rubric_result: dict) -> dict:
    """按评测结果挑出最佳变体。以 ``variant_id`` 匹配，而不是下标——
    下标在过滤/重排之后极易错位，这是最容易埋的坑之一。"""
    target = rubric_result.get("variant_id")
    for script in scripts:
        if script.get("id") == target:
            return script
    return scripts[0] if scripts else {}


def _render_scripts_for_critic(scripts: list[dict]) -> str:
    """把脚本渲染成给 Critic 看的紧凑文本。

    只保留诊断需要的字段（钩子、台词、CTA），丢掉镜头描述等无关信息，
    避免上下文被噪音占满。
    """
    import json

    compact = [
        {
            "id": s.get("id"),
            "angle": s.get("angle"),
            "hook": s.get("hook"),
            "lines": [b.get("line") for b in (s.get("beats") or [])],
            "cta": s.get("cta"),
        }
        for s in scripts
    ]
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _build_base_prompt(brief: dict, insight: dict, best: dict) -> str:
    """由 brief、素材理解结果与最佳脚本推导出基线画面提示词。"""
    product = brief.get("product") or {}
    audience = brief.get("audience") or {}
    platform = brief.get("platform") or {}
    return (
        f"{platform.get('name') or '短视频'} 竖屏画面（{platform.get('aspect_ratio') or '9:16'}）："
        f"主体为{product.get('name') or '产品'}，"
        f"画面基调{insight.get('dominant_mood') or '干净克制'}，"
        f"开场需承载钩子意图「{best.get('hook') or ''}」，"
        f"面向{audience.get('segment') or '目标受众'}，"
        f"整体质感真实、主体清晰、无夸张特效"
    )


def _synthetic_critique(rubric_result: dict, threshold: float) -> dict:
    """评审 Agent 失败时的兜底：用评测器结论直接构造一份最小可用评审。

    这样即使模型侧完全不可用，流水线仍能产出"分数 + 明确的不达标说明"，
    下游（人或程序）依然有决策依据。
    """
    overall = float(rubric_result.get("overall") or 0.0)
    weak = rubric_result.get("weak_dimensions") or []
    directives = [
        f"针对未达标维度 {name} 重写：{rubric_result.get('notes', {}).get(name, [''])[-1]}"
        for name in weak
    ]
    return {
        "verdict": "pass" if overall >= threshold and not weak else "revise",
        "overall": overall,
        "dimension_scores": rubric_result.get("dimension_scores") or {},
        "issues": [
            {
                "dimension": name,
                "severity": "high" if rubric_result["dimension_scores"].get(name, 0) < 60 else "medium",
                "evidence": "由确定性评测器给出（评审 Agent 不可用）",
                "fix": directive,
            }
            for name, directive in zip(weak, directives)
        ],
        "rewrite_directives": directives,
        "summary": f"评审 Agent 不可用，使用评测器结论兜底：总分 {overall}（阈值 {threshold}）。",
        "variant_id": rubric_result.get("variant_id"),
        "fallback": True,
    }
