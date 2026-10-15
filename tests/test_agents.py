"""Agent 层测试：两阶段执行、工具循环、结构化产出保证。

重点验证的是**控制流**而不是输出文本本身，
因为离线后端的文本是模板化的，需要保证的是：
工具被正确调用、观测被正确收敛、schema 不通过时是否会触发修复。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from frame_agent.agents import AnalystAgent, CriticAgent, PromptOpsAgent, ScripterAgent
from frame_agent.llm.base import LLMResponse, Message, Usage
from frame_agent.llm.mock import MockBackend
from frame_agent.llm.router import Budget, ModelRouter
from frame_agent.protocol import detect_task, encode_ctx, has_tool_observation
from frame_agent.tools import build_registry

BRIEF = {
    "brief_id": "t",
    "product": {
        "name": "测试精华",
        "category": "护肤品",
        "selling_points": ["5% 烟酰胺高浓度配方", "28 天改善暗沉"],
        "price_cny": 129,
        "constraints": ["不得宣称美白"],
    },
    "audience": {
        "segment": "上班族",
        "pain_points": ["熬夜后肤色暗沉"],
        "keywords": ["暗沉", "熬夜"],
    },
    "platform": {"name": "抖音", "target_duration_sec": 30},
}

SRT = "\n\n".join(
    f"{i}\n00:00:{i * 3:02d},000 --> 00:00:{(i + 1) * 3:02d},000\n第{i}句字幕内容\n"
    for i in range(1, 6)
)


def _router(**kwargs) -> ModelRouter:
    kwargs.setdefault("sleep", lambda _s: None)
    return ModelRouter([MockBackend()], budget=Budget(max_calls=32), **kwargs)


class TestProtocol(unittest.TestCase):
    def test_task_tag_roundtrip(self):
        text = "角色设定\n\n[TASK:SCRIPT_WRITE]"
        self.assertEqual(detect_task(text), "SCRIPT_WRITE")
        self.assertIsNone(detect_task("没有标签"))

    def test_ctx_roundtrip_including_repair_round(self):
        """修复轮里最新的 user 消息不是上下文信封，倒序扫描必须仍然能找到它。"""
        messages = [
            Message(role="system", content="s"),
            Message(role="user", content=encode_ctx({"a": 1})),
            Message(role="assistant", content="坏输出"),
            Message(role="user", content="请修正"),
        ]
        from frame_agent.protocol import decode_ctx

        self.assertEqual(decode_ctx(messages), {"a": 1})

    def test_has_tool_observation(self):
        self.assertFalse(has_tool_observation([Message(role="user", content="x")]))
        self.assertTrue(
            has_tool_observation([Message(role="tool", content="x", tool_call_id="1")])
        )


class TestAnalyst(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.srt = Path(self.tmp.name) / "clip.srt"
        self.srt.write_text(SRT, encoding="utf-8")
        self.context = {
            "brief": BRIEF,
            "media": {"transcript": str(self.srt)},
            "scripts": [],
            "variant_index": 0,
        }
        self.registry = build_registry(self.context)

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_phase_execution_and_tool_use(self):
        agent = AnalystAgent(_router(), self.registry)
        result = agent.run({"brief": BRIEF, "media": self.context["media"]})

        self.assertTrue(result.ok, result.guard.summary() if result.guard else "")
        # 阶段一：1 次（决定调用哪些工具）；阶段二：1 次（产出结构化结论）
        self.assertEqual(result.model_calls, 2)
        tool_names = {c["tool"] for c in result.tool_calls}
        self.assertEqual(tool_names, {"parse_transcript", "extract_keyframes"})
        self.assertTrue(all(c["ok"] for c in result.tool_calls))
        # 工具自带的调用理由必须被记录下来（便于调试"模型为什么调它"）
        self.assertTrue(all(c["reason"] for c in result.tool_calls))

    def test_compliance_flag_surfaces_from_transcript_scan(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".srt", encoding="utf-8", delete=False
        ) as handle:
            handle.write("1\n00:00:00,000 --> 00:00:03,000\n这支精华可以美白\n")
            risky = Path(handle.name)
        try:
            context = dict(self.context)
            context["media"] = {"transcript": str(risky)}
            agent = AnalystAgent(_router(), build_registry(context))
            result = agent.run({"brief": BRIEF, "media": context["media"]})
            self.assertIn("美白", result.data.get("compliance_flags") or [])
        finally:
            risky.unlink(missing_ok=True)

    def test_no_media_still_produces_result_or_degrades_cleanly(self):
        """没有素材时工具会失败，但 Agent 必须把失败当观测处理，而不是崩掉。"""
        context = {"brief": BRIEF, "media": {}, "scripts": [], "variant_index": 0}
        agent = AnalystAgent(_router(), build_registry(context))
        result = agent.run({"brief": BRIEF, "media": {}})
        self.assertTrue(result.ok)
        self.assertTrue(
            all(not c["ok"] for c in result.tool_calls) or not result.tool_calls
        )


class TestScripter(unittest.TestCase):
    def test_generates_requested_variant_count_with_distinct_angles(self):
        agent = ScripterAgent(_router())
        result = agent.run({"brief": BRIEF, "insight": {}, "variant_count": 3, "round": 1})
        self.assertTrue(result.ok)
        variants = result.data["variants"]
        self.assertEqual(len(variants), 3)
        self.assertEqual(len({v["angle"] for v in variants}), 3)
        self.assertTrue(all(v["beats"] for v in variants))

    def test_rewrite_directives_change_the_output(self):
        """带重写指令的第二轮必须与第一轮不同，否则反思循环是假收敛。"""
        agent = ScripterAgent(_router())
        round1 = agent.run({"brief": BRIEF, "insight": {}, "variant_count": 1, "round": 1})
        round2 = agent.run(
            {
                "brief": BRIEF,
                "insight": {},
                "variant_count": 1,
                "round": 2,
                "rewrite_directives": ["重写钩子：加入具体数字并压到 22 字内"],
            }
        )
        hook1 = round1.data["variants"][0]["hook"]
        hook2 = round2.data["variants"][0]["hook"]
        self.assertNotEqual(hook1, hook2)
        self.assertLess(len(hook2), len(hook1))


class TestCritic(unittest.TestCase):
    def test_uses_independent_compliance_scan(self):
        """合规结论必须来自独立扫描工具，而不是复述评测器的分数。"""
        context = {"brief": BRIEF, "media": {}, "scripts": [], "variant_index": 0}
        registry = build_registry(context)
        variant = {
            "id": "v1",
            "angle": "痛点切入",
            "hook": "可以美白？",
            "beats": [{"t": "0-5s", "line": "这支精华可以美白", "shot": "中景", "on_screen": "美白"}],
            "cta": "点击购买",
            "target_duration_sec": 30,
        }
        context["scripts"] = [variant]
        agent = CriticAgent(_router(), registry)
        result = agent.run(
            {
                "brief": BRIEF,
                "variant": variant,
                "variant_index": 0,
                "rubric": {"dimension_scores": {"compliance": 100}, "notes": {}, "overall": 100.0},
                "pass_threshold": 82,
                "round": 1,
            }
        )
        self.assertTrue(result.ok)
        issues = {i["dimension"]: i["severity"] for i in result.data["issues"]}
        self.assertEqual(issues.get("compliance"), "high")
        self.assertIn("美白", str(result.data["issues"]))
        self.assertEqual(result.data["verdict"], "revise")


class TestPromptOps(unittest.TestCase):
    def test_single_variable_axis_and_ranking(self):
        agent = PromptOpsAgent(_router())
        result = agent.run(
            {
                "brief": BRIEF,
                "insight": {"dominant_mood": "克制专业", "scenes": [{"time_range": "0-3s", "subject": "精华", "mood": "克制"}], "compliance_flags": []},
                "base_prompt": "基线画面",
                "axes": ["subject", "camera", "lighting", "style"],
                "count": 4,
            }
        )
        self.assertTrue(result.ok)
        variants = result.data["variants"]
        self.assertEqual(len(variants), 4)
        # 单变量原则：每个变体只改一个轴
        self.assertTrue(all(len(v["axis_changes"]) == 1 for v in variants))
        # 排序按轴优先级 subject > camera > lighting > style
        self.assertEqual(result.data["ranking"][0], "p1")
        self.assertEqual(result.data["recommended_id"], "p1")


class TestAgentRobustness(unittest.TestCase):
    def test_schema_failure_triggers_repair_and_is_recorded(self):
        """后端连续给出坏 JSON 时，Agent 必须报 ok=False 并留下修复痕迹，
        而不是静默返回半成品。"""

        class BrokenBackend:
            name = "broken"

            def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
                return LLMResponse(text="完全不是 JSON", model="broken", usage=Usage(5, 5))

        router = ModelRouter([BrokenBackend()], budget=Budget(max_calls=16), sleep=lambda _s: None)
        agent = ScripterAgent(router, max_repair=1)
        result = agent.run({"brief": BRIEF, "insight": {}, "variant_count": 1, "round": 1})

        self.assertFalse(result.ok)
        self.assertIsNone(result.data)
        self.assertTrue(result.degradation)
        self.assertEqual(result.guard.repair_rounds, 1)
        self.assertEqual(len(result.guard.attempts), 2)

    def test_router_raises_budget_exceeded_is_propagated(self):
        router = ModelRouter(
            [MockBackend()], budget=Budget(max_calls=0), sleep=lambda _s: None
        )
        agent = ScripterAgent(router)
        from frame_agent.llm.base import BudgetExceeded

        with self.assertRaises(BudgetExceeded):
            agent.run({"brief": BRIEF, "insight": {}, "variant_count": 1, "round": 1})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
