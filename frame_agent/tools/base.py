"""工具层：Agent 能调用的"手和眼"。

设计要点
========

**工具错误不是异常，而是观测。**
:class:`ToolRegistry.invoke` 永远返回 :class:`ToolResult`，绝不向上抛。
理由是 Agent 的核心优势恰恰在于"失败后能看到失败并调整"——
如果工具一报错就把整条链炸掉，Agent 就退化成了一个普通函数调用序列。
把 ``{"ok": false, "error": "..."}`` 作为观测回灌给模型，
它才有机会换成别的参数、换别的工具，或者如实汇报做不到。

**工具的 schema 就是它的文档。**
``parameters`` 里每个字段都带 ``description``，因为这些 schema
会原样进入模型的 function-calling 上下文——描述写得清楚，
模型选错工具的概率就低，这比在 system prompt 里写十行"请正确选择工具"有效得多。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["ToolSpec", "ToolResult", "ToolRegistry"]


@dataclass
class ToolResult:
    """工具调用结果。``ok=False`` 时要保证 ``error`` 是可读的字符串。"""

    ok: bool
    data: Any = None
    error: str | None = None
    tool: str = ""

    def to_observation(self) -> str:
        """序列化成回灌给模型的观测文本。"""
        import json

        if self.ok:
            return json.dumps(
                {"tool": self.tool, "ok": True, "data": self.data},
                ensure_ascii=False,
                default=str,
            )
        return json.dumps(
            {"tool": self.tool, "ok": False, "error": self.error},
            ensure_ascii=False,
        )


@dataclass
class ToolSpec:
    """工具定义：名称、用途说明、参数 schema、处理函数。"""

    name: str
    description: str
    parameters: dict = field(default_factory=dict)
    handler: Callable[..., Any] | None = None

    def to_openai_schema(self) -> dict:
        """转成 OpenAI tools 字段的格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
                or {"type": "object", "properties": {}, "additionalProperties": True},
            },
        }


class ToolRegistry:
    """工具注册表：统一注册、暴露 schema、安全分发。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.calls: list[dict] = []

    # -- 注册 ---------------------------------------------------------------

    def register(self, spec: ToolSpec) -> ToolSpec:
        if not spec.name:
            raise ValueError("工具必须有 name")
        if spec.handler is None:
            raise ValueError(f"工具 {spec.name} 缺少 handler")
        self._tools[spec.name] = spec
        return spec

    def tool(
        self,
        name: str,
        description: str,
        parameters: dict | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """装饰器形式的注册，让工具定义就地可读。"""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                ToolSpec(
                    name=name,
                    description=description,
                    parameters=parameters or {},
                    handler=func,
                )
            )
            return func

        return decorator

    # -- 查询 ---------------------------------------------------------------

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict]:
        return [spec.to_openai_schema() for spec in self._tools.values()]

    # -- 调用 ---------------------------------------------------------------

    def invoke(self, name: str, arguments: dict | None = None) -> ToolResult:
        """安全调用。

        三类失败都被收敛成 :class:`ToolResult`：

        1. **工具不存在** —— 模型幻觉出一个不存在的工具名（很常见）；
        2. **参数非法** —— 模型给的 JSON 与函数签名不匹配；
        3. **执行期异常** —— ffmpeg 没装、文件不存在、权限不足。

        三者对模型都应该表现为"可读的错误 + 建议"，而不是进程崩溃。
        """
        args = dict(arguments or {})
        # 剥掉模型自带的调用理由字段，它不属于工具签名
        args.pop("_reason", None)

        spec = self._tools.get(name)
        if spec is None:
            result = ToolResult(
                ok=False,
                error=f"工具 {name!r} 不存在。可用工具：{self.names()}",
                tool=name,
            )
            self.calls.append({"tool": name, "ok": False, "args": args})
            return result

        try:
            data = spec.handler(**args)  # type: ignore[misc]
        except TypeError as exc:
            result = ToolResult(
                ok=False,
                error=f"参数不匹配：{exc}。该工具的参数 schema 为 {spec.parameters}",
                tool=name,
            )
        except Exception as exc:  # noqa: BLE001 - 工具层兜底，刻意不向上传播
            result = ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}", tool=name)
        else:
            result = ToolResult(ok=True, data=data, tool=name)

        self.calls.append({"tool": name, "ok": result.ok, "args": args})
        return result
