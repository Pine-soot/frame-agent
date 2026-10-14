"""结构化输出层的测试：校验器、JSON 提取、修复闭环。

这三块是整条流水线的"地基"，它们出问题的表现是**下游莫名其妙地崩**，
排查成本极高。所以这里的用例刻意写得很细，包含各种脏输入。
"""

from __future__ import annotations

import unittest

from frame_agent.llm.base import LLMResponse, Message, Usage
from frame_agent.schema import (
    JSONDecodeFailure,
    SchemaGuard,
    extract_json,
    validate,
)


class FakeRouter:
    """按预设脚本依次返回响应，用于测试修复闭环。

    只实现 ``call``，因为 SchemaGuard 只依赖这一个方法（鸭子类型的好处：
    测试替身不需要继承任何东西，也不需要真的网络）。
    """

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def call(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        text = self.responses.pop(0) if self.responses else ""
        return LLMResponse(text=text, model="fake", usage=Usage(1, 1))


class TestValidate(unittest.TestCase):
    SCHEMA = {
        "type": "object",
        "required": ["name", "count"],
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string", "minLength": 2, "maxLength": 5},
            "count": {"type": "integer", "minimum": 1, "maximum": 10},
            "tags": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "mode": {"type": "string", "enum": ["a", "b"]},
            "ratio": {"type": "number"},
        },
    }

    def test_valid_payload_passes(self):
        data = {"name": "abc", "count": 3, "tags": ["x"], "mode": "a", "ratio": 1.5}
        self.assertEqual(validate(data, self.SCHEMA), [])

    def test_missing_required_field_reports_path(self):
        errors = validate({"name": "abc"}, self.SCHEMA)
        self.assertEqual(len(errors), 1)
        self.assertIn("$.count", errors[0])
        self.assertIn("缺少必需字段", errors[0])

    def test_type_mismatch_reports_actual_type(self):
        errors = validate({"name": "abc", "count": "3"}, self.SCHEMA)
        self.assertTrue(any("$.count" in e and "期望类型 integer" in e for e in errors))

    def test_bool_is_not_integer(self):
        """bool 是 int 的子类，这是 JSON Schema 实现里最经典的坑，必须挡住。"""
        errors = validate({"name": "abc", "count": True}, self.SCHEMA)
        self.assertTrue(any("$.count" in e for e in errors))

    def test_number_accepts_int_and_float(self):
        self.assertEqual(validate({"name": "abc", "count": 1, "ratio": 2}, self.SCHEMA), [])

    def test_string_length_bounds(self):
        too_short = validate({"name": "a", "count": 1}, self.SCHEMA)
        too_long = validate({"name": "abcdef", "count": 1}, self.SCHEMA)
        self.assertTrue(any("最小长度" in e for e in too_short))
        self.assertTrue(any("最大长度" in e for e in too_long))

    def test_numeric_bounds(self):
        errors = validate({"name": "abc", "count": 99}, self.SCHEMA)
        self.assertTrue(any("上界" in e for e in errors))

    def test_enum_violation(self):
        errors = validate({"name": "abc", "count": 1, "mode": "c"}, self.SCHEMA)
        self.assertTrue(any("允许集合" in e for e in errors))

    def test_array_items_and_min_items(self):
        empty = validate({"name": "abc", "count": 1, "tags": []}, self.SCHEMA)
        wrong_item = validate({"name": "abc", "count": 1, "tags": [1]}, self.SCHEMA)
        self.assertTrue(any("minItems" in e for e in empty))
        self.assertTrue(any("$.tags[0]" in e for e in wrong_item))

    def test_additional_properties_false(self):
        errors = validate({"name": "abc", "count": 1, "extra": 1}, self.SCHEMA)
        self.assertTrue(any("未声明字段" in e for e in errors))

    def test_error_paths_are_jsonpath_like(self):
        """错误路径必须能让模型看懂，所以形状是 $.a.b[0].c 而不是 Python repr。"""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "object", "required": ["k"]},
                }
            },
        }
        errors = validate({"items": [{"k": 1}, {}]}, schema)
        self.assertTrue(any(e.startswith("$.items[1]") for e in errors), errors)

    def test_type_error_short_circuits(self):
        """类型都不对时不应继续下沉校验，否则会报出一堆无意义的从属错误。"""
        errors = validate("not-an-object", self.SCHEMA)
        self.assertEqual(len(errors), 1)


class TestExtractJson(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_json_fenced(self):
        text = '这是结果：\n```json\n{"a": 1}\n```\n以上。'
        self.assertEqual(extract_json(text), {"a": 1})

    def test_fence_without_language(self):
        self.assertEqual(extract_json("```\n[1, 2]\n```"), [1, 2])

    def test_prose_before_and_after(self):
        text = '好的，我分析如下 {"a": {"b": [1, 2]}} 希望有帮助'
        self.assertEqual(extract_json(text), {"a": {"b": [1, 2]}})

    def test_string_containing_braces_is_not_confused(self):
        """字符串字面量里的花括号不能干扰括号配对。"""
        text = '{"note": "这里有个 } 花括号", "n": 1}'
        self.assertEqual(extract_json(text)["n"], 1)

    def test_escaped_quote_inside_string(self):
        text = r'{"note": "他说 \"你好\"", "n": 2}'
        self.assertEqual(extract_json(text)["n"], 2)

    def test_array_payload(self):
        self.assertEqual(extract_json("前言 [1, 2, 3] 后记"), [1, 2, 3])

    def test_invalid_raises(self):
        with self.assertRaises(JSONDecodeFailure):
            extract_json("完全不是 JSON")

    def test_empty_raises(self):
        with self.assertRaises(JSONDecodeFailure):
            extract_json("   ")


class TestSchemaGuard(unittest.TestCase):
    SCHEMA = {
        "type": "object",
        "required": ["hook"],
        "properties": {"hook": {"type": "string", "minLength": 4}},
    }

    def test_first_attempt_success_needs_one_call(self):
        router = FakeRouter(['{"hook": "暗沉？28天看变化"}'])
        guard = SchemaGuard(self.SCHEMA)
        result = guard.generate(router, [Message(role="user", content="x")])
        self.assertTrue(result.ok)
        self.assertEqual(result.repair_rounds, 0)
        self.assertEqual(router.calls, 1)

    def test_repair_loop_recovers_on_second_attempt(self):
        router = FakeRouter(["这不是 JSON", '{"hook": "补救后的钩子"}'])
        guard = SchemaGuard(self.SCHEMA)
        result = guard.generate(router, [Message(role="user", content="x")])
        self.assertTrue(result.ok)
        self.assertEqual(result.repair_rounds, 1)
        self.assertEqual(router.calls, 2)
        self.assertFalse(result.attempts[0].ok)
        self.assertTrue(result.attempts[1].ok)

    def test_gives_up_after_max_repair(self):
        router = FakeRouter(["坏", "还是坏", "依然坏"])
        guard = SchemaGuard(self.SCHEMA, max_repair=2)
        result = guard.generate(router, [Message(role="user", content="x")])
        self.assertFalse(result.ok)
        self.assertIsNone(result.data)
        self.assertEqual(router.calls, 3)  # 首次 + 2 次修复

    def test_repair_prompt_carries_error_paths(self):
        """修复请求里必须带上带路径的错误，这是修复能生效的前提。"""
        captured: list[list] = []

        class CapturingRouter(FakeRouter):
            def call(self, messages, **kwargs):  # noqa: ANN001, ANN003
                captured.append(list(messages))
                return super().call(messages, **kwargs)

        router = CapturingRouter(["{}", '{"hook": "第二次对了"}'])
        guard = SchemaGuard(self.SCHEMA, max_repair=1)
        guard.generate(router, [Message(role="user", content="x")])

        repair_message = captured[1][-1].content
        self.assertIn("$.hook", repair_message)
        self.assertIn("缺少必需字段", repair_message)

    def test_no_schema_means_no_validation(self):
        guard = SchemaGuard(None)
        router = FakeRouter(["随便什么文本"])
        result = guard.generate(router, [Message(role="user", content="x")])
        self.assertTrue(result.ok)
        self.assertEqual(result.data, "随便什么文本")

    def test_summary_is_serializable(self):
        router = FakeRouter(['{"hook": "暗沉？28天看变化"}'])
        guard = SchemaGuard(self.SCHEMA)
        result = guard.generate(router, [Message(role="user", content="x")])
        summary = result.summary()
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["repair_rounds"], 0)
        self.assertIn("raw_text_head", summary["attempts"][0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
