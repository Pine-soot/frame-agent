"""零依赖的结构化输出校验 + 自动修复闭环。

为什么需要这个模块
==================

Agent 流水线里，LLM 的自由文本是**不可信输入**。只要一个字段缺失或类型漂移，
下游的评分、渲染、落盘就会连环失败——而且失败点离根因很远，极难排查。

常规做法是引入 pydantic 之类的重型依赖。本项目把它压到一个约 200 行的
JSON Schema 子集校验器上，理由有三：

1. **运行期零依赖** —— 评审者 clone 下来直接就能跑，不需要装任何东西；
2. **错误信息带 JSON 路径** —— 校验失败时产出 ``$.variants[0].hook: 缺少必需字段``
   这类可读错误，可以直接原文回灌给模型做自我修复（这是关键：模型能看懂路径）；
3. **模型无关** —— 换任何后端（OpenAI / Qwen / vLLM / 本地 7B）都不用改校验层。

三个能力
========

- :func:`extract_json` —— 从"夹带解释性文字 / 带 markdown 围栏 / 前缀闲聊"的
  原始输出里稳健地抠出 JSON 对象。真实模型几乎从不老老实实只输出 JSON。
- :func:`validate` —— JSON Schema 子集校验，返回带路径的错误列表。
- :class:`SchemaGuard` —— 把二者包成 *生成 → 校验 → 带错误重试* 的闭环，
  把结构化输出的可靠性从"祈祷模型听话"变成"工程上可保证"。

支持的 Schema 关键字
====================

``type`` ``required`` ``properties`` ``items`` ``enum`` ``minimum`` ``maximum``
``minItems`` ``maxItems`` ``minLength`` ``maxLength`` ``pattern``
``additionalProperties``
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "validate",
    "extract_json",
    "SchemaGuard",
    "GuardResult",
    "JSONDecodeFailure",
]


class JSONDecodeFailure(ValueError):
    """无法从原始文本中抽取出合法 JSON。"""


# --------------------------------------------------------------------------
# 类型判定
# --------------------------------------------------------------------------

def _type_ok(value: Any, expected: str) -> bool:
    """按 JSON Schema 语义判定类型。

    注意 ``bool`` 在 Python 里是 ``int`` 的子类，必须显式排除，
    否则 ``True`` 会被判定为合法的 integer——这是最容易踩的坑。
    """
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "null":
        return value is None
    return True  # 未知类型关键字不阻塞


# --------------------------------------------------------------------------
# 校验器
# --------------------------------------------------------------------------

def validate(instance: Any, schema: dict, path: str = "$") -> list[str]:
    """校验 ``instance`` 是否符合 ``schema``，返回**错误信息列表**。

    返回空列表表示通过。之所以返回列表而不是抛异常，是因为
    :class:`SchemaGuard` 需要把**全部**错误一次性喂回给模型——
    一次只报一个错误会导致修复轮次爆炸。

    错误信息形如 ``$.variants[0].hook: 缺少必需字段（required）``，
    路径语法直接对齐 JSONPath，模型对它有很强的先验。
    """
    errors: list[str] = []
    if not isinstance(schema, dict):
        return errors

    expected = schema.get("type")
    if expected and not _type_ok(instance, expected):
        actual = "null" if instance is None else type(instance).__name__
        errors.append(f"{path}: 期望类型 {expected}，实际为 {actual}")
        return errors  # 类型都不对，往下校验没有意义

    # ---- 枚举 ----
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: 取值 {instance!r} 不在允许集合 {schema['enum']} 中")

    # ---- 字符串 ----
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(
                f"{path}: 长度 {len(instance)} 小于最小长度 {schema['minLength']}"
            )
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(
                f"{path}: 长度 {len(instance)} 超过最大长度 {schema['maxLength']}"
            )
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append(f"{path}: 不匹配模式 {schema['pattern']!r}")

    # ---- 数值（显式排除 bool） ----
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} 小于下界 {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} 超过上界 {schema['maximum']}")

    # ---- 数组 ----
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(
                f"{path}: 元素数 {len(instance)} 少于 minItems={schema['minItems']}"
            )
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(
                f"{path}: 元素数 {len(instance)} 超过 maxItems={schema['maxItems']}"
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(instance):
                errors.extend(validate(item, item_schema, f"{path}[{i}]"))

    # ---- 对象 ----
    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                # 错误定位到**缺失字段本身**（``$.variants[0].hook``）而不是父对象。
                # 路径越具体，模型在修复轮里越容易一次改对。
                errors.append(f"{path}.{key}: 缺少必需字段（required）")
        for key, sub_schema in props.items():
            if key in instance:
                errors.extend(validate(instance[key], sub_schema, f"{path}.{key}"))
        if schema.get("additionalProperties") is False:
            extra = sorted(set(instance) - set(props))
            if extra:
                errors.append(f"{path}: 出现未声明字段 {extra}")

    return errors


# --------------------------------------------------------------------------
# 从脏文本里抠 JSON
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """从模型原始输出里提取 JSON。

    按"宽容度递增"的顺序尝试四层策略，任何一层成功即返回：

    1. 直接 ``json.loads``（模型完全听话的情况，最快路径）
    2. 剥离 markdown 围栏后再解析（最常见的情况）
    3. 从头扫描括号配对，截出第一个完整 JSON 值（应对"我先解释一下……{...} 完毕"）
    4. 从最后一个 ``{`` 回溯匹配（应对被截断的多余尾注）

    四层都失败才抛 :class:`JSONDecodeFailure`。
    """
    if text is None:
        raise JSONDecodeFailure("输入为 None")

    raw = text.strip()
    if not raw:
        raise JSONDecodeFailure("输入为空字符串")

    # 策略 1：直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 策略 2：剥围栏
    fence = _FENCE_RE.search(raw)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            raw = candidate

    # 策略 3 / 4：括号配对扫描
    for opener, closer in (("{", "}"), ("[", "]")):
        start = raw.find(opener)
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            for i in range(start, len(raw)):
                ch = raw[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        snippet = raw[start : i + 1]
                        try:
                            return json.loads(snippet)
                        except json.JSONDecodeError:
                            break
            start = raw.find(opener, start + 1)

    raise JSONDecodeFailure(
        f"四层提取策略均失败，原始输出前 200 字符：{raw[:200]!r}"
    )


# --------------------------------------------------------------------------
# 生成 → 校验 → 修复
# --------------------------------------------------------------------------

@dataclass
class Attempt:
    """一次生成尝试的完整记录，用于事后归因（模型第几次才做对）。"""

    index: int
    raw_text: str
    errors: list[str]
    ok: bool

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "ok": self.ok,
            "errors": self.errors,
            "raw_text_head": self.raw_text[:300],
        }


@dataclass
class GuardResult:
    """:class:`SchemaGuard` 的产出。

    ``ok`` 为 False 时 ``data`` 为 None，调用方**必须**显式处理降级路径——
    这是刻意的设计：静默返回半成品数据比直接失败危险得多。
    """

    ok: bool
    data: Any | None
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def repair_rounds(self) -> int:
        """实际发生的修复轮数（0 表示一次通过）。"""
        return max(0, len(self.attempts) - 1)

    def summary(self) -> dict:
        return {
            "ok": self.ok,
            "repair_rounds": self.repair_rounds,
            "attempts": [a.to_dict() for a in self.attempts],
        }


_REPAIR_TEMPLATE = """你上一次的输出未通过结构校验，无法被下游程序消费。

【上一次的原始输出】
{raw}

【校验错误】
{errors}

【修正要求】
- 逐条修复上述错误，特别是路径中指示的缺失字段与类型不匹配；
- 只输出**一个 JSON 对象**，不要输出任何解释文字、道歉或 markdown 围栏；
- 保持内容质量，不要为了通过校验而删减信息量。"""


class SchemaGuard:
    """把"模型生成结构化数据"变成一条可靠路径。

    工作方式是一个**有界修复循环**：

    .. code-block:: text

        生成 -> extract_json -> validate
                  ^                  |
                  |  错误列表未空     |
                  +---- 回灌修复 ------+   最多 max_repair 轮
                                     |
                                  通过 -> 返回 data

    与"重试整轮对话"的朴素做法相比，差别在于**把校验错误作为观测回灌**：
    模型看到的是 ``$.variants[0].hook: 缺少必需字段（required）``，
    而不是一句无信息的"再试一次"。实践上这能把一次修复成功率拉到很高，
    从而显著减少整体重试次数与 token 消耗。

    ``max_repair`` 默认为 2：轮数再多，边际收益已经很低，
    而且往往说明 schema 或提示词本身有问题，应该让人来看——
    所以这里的策略是"有限修复 + 明确失败"，而不是无限重试。
    """

    def __init__(
        self,
        schema: dict | None = None,
        *,
        name: str = "payload",
        max_repair: int = 2,
    ) -> None:
        self.schema = schema
        self.name = name
        self.max_repair = max(0, int(max_repair))

    def validate_payload(self, data: Any) -> list[str]:
        if self.schema is None:
            return []
        return validate(data, self.schema, path="$")

    def coerce(self, raw_text: str) -> tuple[Any | None, list[str]]:
        """只做解析 + 校验，不调用模型（便于单测与离线复用）。

        未提供 schema 时直接返回原始文本：调用方既然没声明结构要求，
        就不应该因为"输出不是 JSON"而失败——那是自由文本场景。
        """
        if self.schema is None:
            return raw_text, []
        try:
            data = extract_json(raw_text)
        except JSONDecodeFailure as exc:
            return None, [f"$: JSON 解析失败 —— {exc}"]
        return data, self.validate_payload(data)

    def generate(
        self,
        router: Any,
        messages: Sequence[Any],
        *,
        purpose: str = "",
        tools: Any = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> GuardResult:
        """执行"生成 → 校验 → 修复"闭环。

        ``router`` 需要提供 ``call(messages, ...) -> LLMResponse`` 接口。
        这里用鸭子类型而不是硬引用 :class:`~frame_agent.llm.router.ModelRouter`，
        是为了让本模块可以被任何实现了该协议的对象复用（含测试替身）。
        """
        attempts: list[Attempt] = []
        convo = list(messages)
        label = purpose or self.name

        for index in range(self.max_repair + 1):
            # 修复轮不再透传工具：此时模型的唯一任务是产出合法 JSON，
            # 继续给它工具反而会诱发新的工具调用，偏离目标。
            active_tools = tools if index == 0 else None
            response = router.call(
                convo,
                tools=active_tools,
                temperature=temperature,
                max_tokens=max_tokens,
                purpose=f"{label}#attempt{index + 1}",
            )
            raw_text = response.text or ""
            data, errors = self.coerce(raw_text)
            ok = data is not None and not errors
            attempts.append(
                Attempt(index=index + 1, raw_text=raw_text, errors=errors, ok=ok)
            )
            if ok:
                return GuardResult(ok=True, data=data, attempts=attempts)

            if index == self.max_repair:
                break

            # 构造修复对话：把模型的原始输出和错误路径一起回灌
            from frame_agent.llm.base import Message  # 局部导入，避免循环依赖

            convo = convo + [
                Message(role="assistant", content=raw_text),
                Message(
                    role="user",
                    content=_REPAIR_TEMPLATE.format(
                        raw=raw_text.strip()[:2000],
                        errors="\n".join(f"- {e}" for e in errors) or "- 无法解析出 JSON",
                    ),
                ),
            ]

        return GuardResult(ok=False, data=None, attempts=attempts)
