"""端到端测试 —— 本项目最重要的一组用例。

它验证的是"整条流水线作为一个系统"的行为，而不是单个模块：

1. **反思循环真的收敛**（第一轮不达标、第二轮通过，且分数可复现地上升）；
2. **同一输入两次运行结果完全一致**（离线后端的确定性是这套测试的前提）；
3. **产物完整落盘**，报告包含关键章节；
4. **批量回归套件**能跑通并给出正确汇总；
5. **max_rounds=1 时跳过重写**，成本随之下调。

这些性质一旦被破坏，说明流水线的"可靠性承诺"出了问题——
比某个函数的返回值错更严重。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from frame_agent.cli import examples_dir
from frame_agent.evals.runner import run_suite, summarize_cases
from frame_agent.llm.mock import MockBackend
from frame_agent.llm.router import Budget, ModelRouter
from frame_agent.pipeline import ContentPipeline, RunState, brief_fingerprint

REPO_ROOT = examples_dir().parent
BRIEF_PATH = examples_dir() / "briefs" / "skincare_launch.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pipeline(out_dir: Path, **kwargs) -> ContentPipeline:
    router = ModelRouter(
        [MockBackend()], budget=Budget(max_calls=64, max_tokens=500_000), sleep=lambda _s: None
    )
    kwargs.setdefault("max_rounds", 2)
    kwargs.setdefault("variant_count", 3)
    return ContentPipeline(
        router, out_dir=out_dir, base_dir=REPO_ROOT, **kwargs
    )


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "run"

    def tearDown(self):
        self.tmp.cleanup()

    # -- 基本形态 -----------------------------------------------------------

    def test_full_run_shape(self):
        brief = _load(BRIEF_PATH)
        result = _pipeline(self.out).run(brief, brief_path=str(BRIEF_PATH), backend="mock")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["backend"], "mock")
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(len(result["scores"]), 3)
        self.assertIn("hook", result["best_variant"])
        self.assertGreater(result["model_calls"], 0)
        self.assertEqual(result["degradation_notes"], [])

    def test_all_expected_artifacts_written(self):
        brief = _load(BRIEF_PATH)
        result = _pipeline(self.out).run(brief, brief_path=str(BRIEF_PATH))

        for name in (
            "media_insight.json",
            "scripts.json",
            "scores.json",
            "critique.json",
            "prompt_set.json",
            "cost.json",
            "report.md",
            "run_state.json",
            "trace.jsonl",
        ):
            self.assertTrue((self.out / name).exists(), f"缺少产物 {name}")

        report = (self.out / "report.md").read_text(encoding="utf-8")
        for section in ("反思循环轨迹", "评分明细", "候选变体对比", "成本与延迟画像"):
            self.assertIn(section, report)

    # -- 反思循环收敛 -------------------------------------------------------

    def test_reflection_loop_converges_and_is_evidence_backed(self):
        brief = _load(BRIEF_PATH)
        result = _pipeline(self.out).run(brief)

        history = result["round_history"]
        self.assertEqual(len(history), 2, "应当恰好跑两轮：第一轮不达标，第二轮达标")

        first, second = history
        self.assertEqual(first["verdict"], "revise")
        self.assertEqual(second["verdict"], "pass")
        self.assertLess(first["best_score"], second["best_score"])
        self.assertGreaterEqual(second["best_score"], 82.0)

        # 第一轮必须给出可执行的重写指令，否则第二轮无从改进
        self.assertTrue(first["directives"])
        self.assertTrue(all(len(d) > 6 for d in first["directives"]))

        # 分数上升必须可归因：钩子强度提升，且钩子确实变短了
        self.assertGreater(
            second["dimension_scores"]["hook_strength"],
            first["dimension_scores"]["hook_strength"],
        )
        self.assertLess(len(second["hook"]), len(first["hook"]))

    def test_max_rounds_one_skips_rewrite_and_costs_less(self):
        brief = _load(BRIEF_PATH)
        one_round = _pipeline(self.out / "one", max_rounds=1).run(brief)
        self.assertEqual(one_round["rounds"], 1)
        self.assertEqual(len(one_round["round_history"]), 1)
        # 第一轮的评审意见会被记录下来（有诊断价值），但没有下一轮去执行它
        self.assertEqual(one_round["round_history"][0]["verdict"], "revise")

        two_round = _pipeline(self.out / "two", max_rounds=2).run(brief)
        # 不做重写的一轮必然更省：少一次 Scripter + 少一次 Critic
        self.assertLess(one_round["model_calls"], two_round["model_calls"])
        self.assertLess(one_round["cost"]["total_tokens"], two_round["cost"]["total_tokens"])

    # -- 确定性 -------------------------------------------------------------

    def test_two_runs_are_byte_identical_in_outcome(self):
        """离线后端无随机数，两次运行的结果必须完全一致——
        这是把"报告里的数字"当作可验证证据的前提。"""
        brief = _load(BRIEF_PATH)
        a = _pipeline(self.out / "a").run(brief)
        b = _pipeline(self.out / "b").run(brief)

        self.assertEqual(a["eval"]["overall"], b["eval"]["overall"])
        self.assertEqual(a["best_variant"]["hook"], b["best_variant"]["hook"])
        self.assertEqual(
            [s["overall"] for s in a["scores"]], [s["overall"] for s in b["scores"]]
        )
        self.assertEqual(a["cost"]["total_calls"], b["cost"]["total_calls"])

    def test_second_example_brief_also_runs(self):
        """第二个 brief（SaaS 类目）也必须跑通——
        否则说明提示词或评测器被单一类目过拟合了。"""
        saas = examples_dir() / "briefs" / "saas_demo.json"
        result = _pipeline(self.out / "saas").run(_load(saas), brief_path=str(saas))
        self.assertEqual(result["status"], "ok")
        self.assertGreater(len(result["best_variant"]["hook"]), 0)

    # -- 状态与断点续跑 -----------------------------------------------------

    def test_run_state_is_resumable_only_for_same_brief(self):
        brief = _load(BRIEF_PATH)
        _pipeline(self.out).run(brief, brief_path=str(BRIEF_PATH))

        state = RunState.load(self.out)
        self.assertIsNotNone(state)
        self.assertEqual(state.brief_hash, brief_fingerprint(brief))
        self.assertTrue(state.steps)

        # 指纹一致 → 可复用
        self.assertIsNotNone(RunState.resumable(self.out, brief))

        # brief 改了 → 拒绝复用（防止用旧状态产出与 brief 不符的报告）
        mutated = dict(brief)
        mutated["goal"] = "完全不同的目标"
        self.assertIsNone(RunState.resumable(self.out, mutated))

    def test_trace_is_append_only_and_parseable(self):
        brief = _load(BRIEF_PATH)
        _pipeline(self.out).run(brief)
        state = RunState.load(self.out)
        records = state.read_trace()
        self.assertGreaterEqual(len(records), 3)
        events = {r.get("event") for r in records}
        self.assertIn("run_start", events)
        self.assertIn("generate", events)
        self.assertIn("critique", events)


class TestRegressionSuite(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "suite"

    def tearDown(self):
        self.tmp.cleanup()

    def test_suite_runs_both_example_briefs(self):
        brief_paths = sorted((examples_dir() / "briefs").glob("*.json"))

        def run_case(brief_path: str, out_dir: str) -> dict:
            return _pipeline(Path(out_dir)).run(
                _load(Path(brief_path)), brief_path=brief_path, backend="mock"
            )

        suite = run_suite(run_case, brief_paths, self.out)
        summary = suite["summary"]

        self.assertEqual(summary["case_count"], 2)
        self.assertTrue(all(c["status"] == "ok" for c in suite["cases"]))
        self.assertEqual(summary["pass_count"], 2)
        self.assertEqual(summary["pass_rate"], 1.0)
        self.assertEqual(summary["cases_with_degradation"], [])
        self.assertTrue((self.out / "suite.json").exists())
        self.assertTrue((self.out / "skincare_launch").is_dir())

    def test_suite_isolates_failing_case(self):
        """单个 case 抛异常不能中断整批——批量回归的价值就在于总能给出全景。"""

        def run_case(brief_path: str, out_dir: str) -> dict:
            if "saas" in brief_path:
                raise RuntimeError("模拟这一类目挂了")
            return {"eval": {"verdict": "pass", "overall": 90.0, "dimension_scores": {}}, "rounds": 1}

        suite = run_suite(run_case, ["a.json", "b_saas.json", "c.json"], self.out)
        statuses = [c["status"] for c in suite["cases"]]
        self.assertEqual(statuses, ["ok", "error", "ok"])
        summary = suite["summary"]
        self.assertEqual(summary["case_count"], 3)
        self.assertEqual(summary["pass_count"], 2)

    def test_summarize_handles_empty_input(self):
        summary = summarize_cases([])
        self.assertEqual(summary["case_count"], 0)
        self.assertEqual(summary["pass_rate"], 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
