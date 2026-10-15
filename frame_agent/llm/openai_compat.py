"""OpenAI 兼容后端：用标准库直连任意兼容端点。

**不引入 openai / dashscope / requests 等 SDK。** 原因：

- 兼容面反而更大：只要实现 ``POST {base_url}/chat/completions`` 的服务都能用，
  包括 OpenAI、DeepSeek、通义千问（兼容模式）、Moonshot、vLLM、Ollama、LM Studio、
  以及本地自建网关。装一个 SDK 反而把自己锁在某一家的兼容层上。
- 依赖为零：``urllib.request`` + ``json`` 就够了，评审者不必 pip install 任何东西。
- 关键逻辑可见：错误分类（哪些该重试、哪些该换后端）写在这里，而不是藏在
  SDK 的重试装饰器里，这段逻辑正好是 Agent 稳定性最需要被 review 的部分。

配置全部来自环境变量，便于在 CI / 本地 / 服务器之间切换而不改代码：

===========================  ==========================================
``FRAME_AGENT_BASE_URL``     例如 ``https://api.deepseek.com/v1``
``FRAME_AGENT_API_KEY``      API Key
``FRAME_AGENT_MODEL``        模型名，例如 ``deepseek-chat``
``FRAME_AGENT_TIMEOUT``      请求超时秒数，默认 60
===========================  ==========================================
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Sequence

from frame_agent.llm.base import (
    LLMResponse,
    Message,
    PermanentLLMError,
    ToolCall,
    TransientLLMError,
    Usage,
)

__all__ = ["OpenAICompatBackend", "from_env"]

# 这些 HTTP 状态码代表"等一会儿再试可能就好了"
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class OpenAICompatBackend:
    """最小可用的 OpenAI 兼容客户端。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        name: str | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url 不能为空")
        if not api_key:
            raise ValueError("api_key 不能为空")
        if not model:
            raise ValueError("model 不能为空")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        # 后端名带模型名，便于在记账报告里区分同一网关下的不同模型
        self.name = name or f"openai-compat:{model}"

    # -- 请求构造 -----------------------------------------------------------

    def _payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        return payload

    # -- 响应解析 -----------------------------------------------------------

    @staticmethod
    def _parse_usage(data: dict) -> Usage:
        raw = data.get("usage") or {}
        return Usage(
            prompt_tokens=int(raw.get("prompt_tokens") or 0),
            completion_tokens=int(raw.get("completion_tokens") or 0),
        )

    @staticmethod
    def _parse_tool_calls(message: dict) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            raw_args = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                # 模型偶尔会输出非法 JSON 参数。不在这里抛错，
                # 把它降级成 ``{"__raw__": ...}`` 交给工具层报可读错误，
                # 让 Agent 有机会"看到错误并自己修正"，而不是整轮崩掉。
                arguments = {"__raw__": raw_args}
            calls.append(
                ToolCall(
                    id=item.get("id") or f"call_{len(calls)}",
                    name=function.get("name") or "",
                    arguments=arguments if isinstance(arguments, dict) else {"value": arguments},
                )
            )
        return calls

    # -- 主入口 -------------------------------------------------------------

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(
            self._payload(messages, tools, temperature, max_tokens), ensure_ascii=False
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # noqa: BLE001 - 读错误体失败不应掩盖原始错误
                detail = "<无法读取错误响应体>"
            if exc.code in _TRANSIENT_STATUS:
                raise TransientLLMError(f"HTTP {exc.code}：{detail}") from exc
            raise PermanentLLMError(f"HTTP {exc.code}：{detail}") from exc
        except urllib.error.URLError as exc:
            # DNS / 连接被拒 / TLS 失败：属于瞬时问题，交给路由器重试与降级
            raise TransientLLMError(f"网络错误：{exc.reason!r}") from exc
        except TimeoutError as exc:
            raise TransientLLMError(f"请求超时（{self.timeout}s）") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TransientLLMError(
                f"响应不是合法 JSON，前 200 字符：{raw[:200]!r}"
            ) from exc

        if "error" in data and not data.get("choices"):
            raise PermanentLLMError(f"接口返回错误：{data['error']}")

        choices = data.get("choices") or []
        if not choices:
            raise TransientLLMError(f"响应缺少 choices 字段：{str(data)[:200]}")

        choice = choices[0]
        message = choice.get("message") or {}
        return LLMResponse(
            text=message.get("content") or "",
            model=data.get("model") or self.model,
            usage=self._parse_usage(data),
            tool_calls=self._parse_tool_calls(message),
            finish_reason=choice.get("finish_reason") or "stop",
            raw=data,
        )


def from_env(prefix: str = "FRAME_AGENT") -> OpenAICompatBackend:
    """按约定从环境变量构建后端，缺任一项即报错（fail fast，别静默降级）。"""
    base_url = os.environ.get(f"{prefix}_BASE_URL", "")
    api_key = os.environ.get(f"{prefix}_API_KEY", "")
    model = os.environ.get(f"{prefix}_MODEL", "")
    timeout = float(os.environ.get(f"{prefix}_TIMEOUT", "60") or 60)

    missing = [
        name
        for name, value in (
            (f"{prefix}_BASE_URL", base_url),
            (f"{prefix}_API_KEY", api_key),
            (f"{prefix}_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "缺少环境变量：" + ", ".join(missing) + "。"
            "如只想离线体验，请使用 --backend mock。"
        )
    return OpenAICompatBackend(
        base_url=base_url, api_key=api_key, model=model, timeout=timeout
    )
