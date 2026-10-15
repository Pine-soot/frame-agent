"""多模型路由：降级链、重试退避、预算熔断、调用记账。

这三件事在生产环境里**必须**放在同一层，因为它们互相耦合：

- 没有降级链，单点限流就打挂整条流水线；
- 没有预算熔断，一个退化的反思循环能把账单刷爆；
- 没有记账，你根本不知道成本花在哪个 Agent 上，也就无法优化。

本模块把它们收在一个 :class:`ModelRouter` 里，并暴露 :meth:`ModelRouter.report`
产出可直接进报告的成本/延迟画像。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from frame_agent.llm.base import (
    BudgetExceeded,
    LLMError,
    LLMResponse,
    Message,
    PermanentLLMError,
    TransientLLMError,
    Usage,
)

__all__ = ["Budget", "CallRecord", "ModelRouter"]


# --------------------------------------------------------------------------
# 预算
# --------------------------------------------------------------------------

@dataclass
class Budget:
    """一次流水线运行的资源上限。

    三个维度任一耗尽即熔断。``max_seconds`` 用可注入的时钟读取，
    这样单测里可以瞬间"穿越时间"而不用真的 sleep。
    """

    max_calls: int = 64
    max_tokens: int = 200_000
    max_seconds: float = 600.0
    calls: int = 0
    tokens: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def check(self, clock: Callable[[], float] = time.monotonic) -> None:
        """在执行**之前**检查，避免"先超支后报错"。"""
        if self.calls >= self.max_calls:
            raise BudgetExceeded(
                f"调用次数预算耗尽（{self.calls}/{self.max_calls}）"
            )
        if self.tokens >= self.max_tokens:
            raise BudgetExceeded(f"token 预算耗尽（{self.tokens}/{self.max_tokens}）")
        elapsed = clock() - self.started_at
        if elapsed > self.max_seconds:
            raise BudgetExceeded(
                f"时间预算耗尽（{elapsed:.1f}s/{self.max_seconds:.1f}s）"
            )

    def charge(self, usage: Usage) -> None:
        self.calls += 1
        self.tokens += usage.total_tokens

    def snapshot(self, clock: Callable[[], float] = time.monotonic) -> dict:
        return {
            "calls": self.calls,
            "calls_limit": self.max_calls,
            "tokens": self.tokens,
            "tokens_limit": self.max_tokens,
            "elapsed_seconds": round(clock() - self.started_at, 3),
            "seconds_limit": self.max_seconds,
        }


# --------------------------------------------------------------------------
# 记账
# --------------------------------------------------------------------------

@dataclass
class CallRecord:
    """单次调用的审计记录。"""

    seq: int
    backend: str
    purpose: str
    ok: bool
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    attempt: int = 1

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "backend": self.backend,
            "purpose": self.purpose,
            "ok": self.ok,
            "attempt": self.attempt,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "error": self.error,
        }


def _percentile(values: list[float], pct: float) -> float:
    """线性插值分位数。数据量小的时候比 numpy 省一个依赖。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


# --------------------------------------------------------------------------
# 路由
# --------------------------------------------------------------------------

class ModelRouter:
    """带降级链的模型调用入口。

    调用语义
    --------
    对一次 :meth:`call`：

    1. **先查预算**：超支立刻抛 :class:`BudgetExceeded`，绝不"先花再报"；
    2. **按顺序试后端**：首选失败则降级到次选（这就是 fallback chain）；
    3. **同一后端内做退避重试**：仅对 :class:`TransientLLMError` 重试，
       遇到 :class:`PermanentLLMError` 立刻换后端（重试一个鉴权错误纯属浪费）；
    4. **无论成功失败都记账**：失败的调用同样消耗了延迟与配额，不能漏记。

    ``sleep`` 可注入，测试里传一个空函数即可零延迟跑完退避逻辑。
    """

    def __init__(
        self,
        backends: Sequence[Any],
        *,
        budget: Budget | None = None,
        max_retries: int = 2,
        backoff_base: float = 0.4,
        backoff_cap: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not backends:
            raise ValueError("ModelRouter 至少需要一个后端")
        self.backends = list(backends)
        self.budget = budget or Budget()
        self.max_retries = max(0, max_retries)
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self._clock = clock
        self._sleep = sleep
        # 预算的计时起点必须与路由器使用同一个时钟，否则注入的时钟
        # （比如测试用的假时钟）会让"已用时间"算出无意义的负数。
        self.budget.started_at = self._clock()
        self.records: list[CallRecord] = []
        self._seq = 0

    # -- 内部 ---------------------------------------------------------------

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _record(self, **kwargs: Any) -> None:
        self.records.append(CallRecord(**kwargs))

    # -- 对外 ---------------------------------------------------------------

    def call(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        purpose: str = "",
    ) -> LLMResponse:
        """按降级链调用，返回第一个成功的响应。"""
        self.budget.check(self._clock)
        last_error: Exception | None = None

        for backend_index, backend in enumerate(self.backends):
            attempts_for_this_backend = self.max_retries + 1
            for attempt in range(1, attempts_for_this_backend + 1):
                seq = self._next_seq()
                started = self._clock()
                try:
                    response = backend.complete(
                        messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except PermanentLLMError as exc:
                    latency = (self._clock() - started) * 1000
                    self._record(
                        seq=seq, backend=backend.name, purpose=purpose, ok=False,
                        latency_ms=latency, error=f"permanent: {exc}", attempt=attempt,
                    )
                    last_error = exc
                    break  # 确定性错误：不重试，直接降级到下一个后端
                except (TransientLLMError, TimeoutError, ConnectionError) as exc:
                    latency = (self._clock() - started) * 1000
                    self._record(
                        seq=seq, backend=backend.name, purpose=purpose, ok=False,
                        latency_ms=latency, error=f"transient: {exc}", attempt=attempt,
                    )
                    last_error = exc
                    if attempt < attempts_for_this_backend:
                        delay = min(
                            self.backoff_base * (2 ** (attempt - 1)), self.backoff_cap
                        )
                        self._sleep(delay)
                    continue  # 换一次重试
                except Exception as exc:  # noqa: BLE001 - 兜底，避免单后端炸掉整链
                    latency = (self._clock() - started) * 1000
                    self._record(
                        seq=seq, backend=backend.name, purpose=purpose, ok=False,
                        latency_ms=latency, error=f"unexpected: {exc!r}", attempt=attempt,
                    )
                    last_error = LLMError(str(exc))
                    break
                else:
                    latency = (self._clock() - started) * 1000
                    usage = response.usage or Usage()
                    self._record(
                        seq=seq, backend=backend.name, purpose=purpose, ok=True,
                        latency_ms=latency,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        attempt=attempt,
                    )
                    self.budget.charge(usage)
                    return response

        raise LLMError(
            f"全部 {len(self.backends)} 个后端均调用失败，最后错误：{last_error!r}"
        )

    def report(self) -> dict:
        """成本与延迟画像，可直接进产物报告。"""
        ok = [r for r in self.records if r.ok]
        failed = [r for r in self.records if not r.ok]
        by_backend: dict[str, dict] = {}
        for record in self.records:
            bucket = by_backend.setdefault(
                record.backend, {"calls": 0, "failures": 0, "tokens": 0, "latencies": []}
            )
            bucket["calls"] += 1
            bucket["failures"] += 0 if record.ok else 1
            bucket["tokens"] += record.prompt_tokens + record.completion_tokens
            bucket["latencies"].append(record.latency_ms)

        return {
            "total_calls": len(self.records),
            "successful_calls": len(ok),
            "failed_calls": len(failed),
            "retry_calls": sum(1 for r in self.records if r.attempt > 1),
            "total_tokens": sum(r.prompt_tokens + r.completion_tokens for r in self.records),
            "latency_ms": {
                "p50": round(_percentile([r.latency_ms for r in ok], 0.50), 2),
                "p95": round(_percentile([r.latency_ms for r in ok], 0.95), 2),
            },
            "by_backend": {
                name: {
                    "calls": b["calls"],
                    "failures": b["failures"],
                    "tokens": b["tokens"],
                    "p50_ms": round(_percentile(b["latencies"], 0.50), 2),
                }
                for name, b in by_backend.items()
            },
            "budget": self.budget.snapshot(self._clock),
        }
