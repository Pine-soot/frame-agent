"""路由层测试：降级链、重试策略、预算熔断、记账。

路由是"模型不听话时系统还能不能活"的保障层。这里的每个用例都对应一个
真实会发生的故障：限流、鉴权失败、超时、循环失控导致成本爆炸。
"""

from __future__ import annotations

import unittest

from frame_agent.llm.base import (
    BudgetExceeded,
    LLMError,
    LLMResponse,
    Message,
    PermanentLLMError,
    TransientLLMError,
    Usage,
)
from frame_agent.llm.router import Budget, ModelRouter


class StubBackend:
    """可控后端：按预设的异常/响应序列行为。"""

    def __init__(self, name: str, script: list) -> None:
        self.name = name
        self.script = list(script)
        self.calls = 0

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        item = self.script.pop(0) if self.script else LLMResponse(
            text="ok", model=self.name, usage=Usage(10, 5)
        )
        if isinstance(item, Exception):
            raise item
        return item


def _ok(name: str, text: str = "ok", pt: int = 10, ct: int = 5) -> LLMResponse:
    return LLMResponse(text=text, model=name, usage=Usage(pt, ct))


def _router(*backends, **kwargs) -> ModelRouter:
    # sleep 注入空函数：退避逻辑要测，但不该真的让测试变慢
    kwargs.setdefault("sleep", lambda _s: None)
    return ModelRouter(list(backends), **kwargs)


class TestFallbackChain(unittest.TestCase):
    def test_falls_back_to_second_backend(self):
        primary = StubBackend("primary", [TransientLLMError("429")] * 3)
        backup = StubBackend("backup", [_ok("backup")])
        router = _router(primary, backup, max_retries=0)
        response = router.call([Message(role="user", content="hi")])
        self.assertEqual(response.text, "ok")
        self.assertEqual(backup.calls, 1)

    def test_transient_error_is_retried_within_same_backend(self):
        backend = StubBackend("a", [TransientLLMError("timeout"), _ok("a", "recovered")])
        router = _router(backend, max_retries=2)
        response = router.call([Message(role="user", content="hi")])
        self.assertEqual(response.text, "recovered")
        self.assertEqual(backend.calls, 2)

    def test_permanent_error_switches_backend_without_retry(self):
        """鉴权失败重试 100 次也没用，必须立刻换后端。"""
        bad = StubBackend("bad", [PermanentLLMError("401")] * 5)
        good = StubBackend("good", [_ok("good")])
        router = _router(bad, good, max_retries=3)
        router.call([Message(role="user", content="hi")])
        self.assertEqual(bad.calls, 1)
        self.assertEqual(good.calls, 1)

    def test_all_backends_fail_raises_with_last_error(self):
        a = StubBackend("a", [PermanentLLMError("nope")])
        b = StubBackend("b", [PermanentLLMError("nope2")])
        router = _router(a, b, max_retries=0)
        with self.assertRaises(LLMError) as ctx:
            router.call([Message(role="user", content="hi")])
        self.assertIn("全部 2 个后端", str(ctx.exception))

    def test_unexpected_exception_is_contained(self):
        """某个后端抛了非预期异常，不应该炸掉整条链。"""
        weird = StubBackend("weird", [ValueError("boom")])
        good = StubBackend("good", [_ok("good")])
        router = _router(weird, good, max_retries=0)
        self.assertEqual(router.call([Message(role="user", content="hi")]).text, "ok")


class TestBudget(unittest.TestCase):
    def test_call_budget_exhausts(self):
        backend = StubBackend("a", [])
        router = _router(backend, budget=Budget(max_calls=2, max_tokens=10**9))
        router.call([Message(role="user", content="1")])
        router.call([Message(role="user", content="2")])
        with self.assertRaises(BudgetExceeded):
            router.call([Message(role="user", content="3")])

    def test_token_budget_exhausts(self):
        backend = StubBackend("a", [])
        # 单次调用计 15 tokens（10 prompt + 5 completion）
        router = _router(backend, budget=Budget(max_calls=100, max_tokens=10))
        router.call([Message(role="user", content="1")])  # 首次：检查 0 < 10，放行
        with self.assertRaises(BudgetExceeded):
            router.call([Message(role="user", content="2")])  # 已用 15 >= 10，拦截

    def test_time_budget_uses_injected_clock(self):
        now = {"t": 0.0}
        backend = StubBackend("a", [])
        router = _router(
            backend,
            budget=Budget(max_calls=100, max_tokens=10**9, max_seconds=5),
            clock=lambda: now["t"],
        )
        router.call([Message(role="user", content="1")])
        now["t"] = 10.0  # 穿越时间，无需真的 sleep
        with self.assertRaises(BudgetExceeded):
            router.call([Message(role="user", content="2")])

    def test_budget_checked_before_spending(self):
        """超支必须在调用**之前**拦住，否则"熔断"就变成"事后追认"。"""
        backend = StubBackend("a", [])
        router = _router(backend, budget=Budget(max_calls=1, max_tokens=10**9))
        router.call([Message(role="user", content="1")])
        with self.assertRaises(BudgetExceeded):
            router.call([Message(role="user", content="2")])
        self.assertEqual(backend.calls, 1)


class TestAccounting(unittest.TestCase):
    def test_failed_calls_are_recorded(self):
        """失败的调用同样消耗了延迟与配额，漏记会让成本画像严重失真。"""
        bad = StubBackend("bad", [TransientLLMError("x")] * 3)
        good = StubBackend("good", [_ok("good")])
        router = _router(bad, good, max_retries=1)
        router.call([Message(role="user", content="hi")], purpose="test")

        report = router.report()
        self.assertEqual(report["total_calls"], 3)  # bad×2 + good×1
        self.assertEqual(report["failed_calls"], 2)
        self.assertEqual(report["successful_calls"], 1)
        self.assertEqual(report["retry_calls"], 1)

    def test_report_groups_by_backend(self):
        a = StubBackend("a", [_ok("a")])
        b = StubBackend("b", [_ok("b")])
        router = _router(a, b)
        router.call([Message(role="user", content="1")])
        report = router.report()
        self.assertIn("a", report["by_backend"])
        self.assertEqual(report["by_backend"]["a"]["calls"], 1)

    def test_usage_is_accumulated_into_budget(self):
        backend = StubBackend("a", [_ok("a", pt=100, ct=50), _ok("a", pt=10, ct=5)])
        router = _router(backend)
        router.call([Message(role="user", content="1")])
        router.call([Message(role="user", content="2")])
        self.assertEqual(router.budget.tokens, 165)
        self.assertEqual(router.budget.calls, 2)

    def test_percentile_handles_small_samples(self):
        backend = StubBackend("a", [_ok("a")])
        router = _router(backend)
        router.call([Message(role="user", content="1")])
        latency = router.report()["latency_ms"]
        self.assertIn("p50", latency)
        self.assertIn("p95", latency)


class TestConstruction(unittest.TestCase):
    def test_empty_backend_list_rejected(self):
        with self.assertRaises(ValueError):
            ModelRouter([])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
