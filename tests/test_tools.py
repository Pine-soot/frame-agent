"""工具层测试：字幕解析、指标计算、注册表容错、媒体降级。

工具层的失败必须表现为"可读的错误观测"而不是异常，
所以这里重点覆盖三类失败：工具不存在、参数不匹配、执行期异常。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from frame_agent.tools.base import ToolRegistry, ToolSpec
from frame_agent.tools.media import extract_keyframes
from frame_agent.tools.metrics import analyze_script, estimate_duration
from frame_agent.tools.transcript import load_transcript, parse_subtitles, transcript_stats

BRIEF = {
    "product": {"name": "精华", "category": "护肤品", "selling_points": ["5% 烟酰胺"], "price_cny": 129},
    "audience": {
        "segment": "上班族",
        "pain_points": ["熬夜后肤色暗沉"],
        "keywords": ["暗沉", "熬夜"],
    },
    "platform": {"name": "抖音", "target_duration_sec": 30},
}


class TestSubtitleParsing(unittest.TestCase):
    SRT = (
        "1\n00:00:00,000 --> 00:00:03,200\n第一句字幕\n\n"
        "2\n00:00:03,200 --> 00:00:06,400\n第二句字幕\n"
    )

    def test_basic_srt(self):
        cues = parse_subtitles(self.SRT)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].text, "第一句字幕")
        self.assertAlmostEqual(cues[0].start, 0.0)
        self.assertAlmostEqual(cues[1].end, 6.4)

    def test_vtt_header_and_settings_are_ignored(self):
        vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000 align:start\n内容\n"
        cues = parse_subtitles(vtt)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "内容")

    def test_millisecond_padding(self):
        """``00:00:00,5`` 应理解为 500ms，而不是 5ms。"""
        cues = parse_subtitles("1\n00:00:00,5 --> 00:00:01,50\nx\n")
        self.assertAlmostEqual(cues[0].start, 0.5)
        self.assertAlmostEqual(cues[0].end, 1.5)

    def test_webvtt_tags_stripped(self):
        cues = parse_subtitles("1\n00:00:00,000 --> 00:00:01,000\n<v 张三>你好</v>\n")
        self.assertEqual(cues[0].text, "你好")

    def test_empty_input(self):
        self.assertEqual(parse_subtitles(""), [])

    def test_stats(self):
        cues = parse_subtitles(self.SRT)
        stats = transcript_stats(cues)
        self.assertEqual(stats["cue_count"], 2)
        self.assertAlmostEqual(stats["duration_sec"], 6.4)
        self.assertEqual(stats["char_count"], 10)
        self.assertGreater(stats["chars_per_second"], 0)

    def test_stats_on_empty_is_safe(self):
        stats = transcript_stats([])
        self.assertEqual(stats["cue_count"], 0)
        self.assertEqual(stats["chars_per_second"], 0.0)

    def test_txt_fallback_generates_timeline(self):
        """只有纯文本、没有时间轴时，要均分出一个可用的时间轴而不是返回空。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.txt"
            path.write_text("第一行\n第二行\n第三行\n", encoding="utf-8")
            cues = load_transcript(path, default_duration=30)
        self.assertEqual(len(cues), 3)
        self.assertAlmostEqual(cues[-1].end, 30.0)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_transcript("不存在的文件.srt")


class TestMetrics(unittest.TestCase):
    def test_estimate_duration_ignores_punctuation(self):
        # 9 个有效字 / 4.5 = 2.0 秒
        self.assertAlmostEqual(estimate_duration("一二三四，五六七八九。"), 2.0)

    def test_hook_flags_detected(self):
        variant = {
            "hook": "暗沉？28天自己看变化",
            "beats": [
                {"t": "0-8s", "line": "熬夜后暗沉", "shot": "中景", "on_screen": "钩子"},
                {"t": "8-15s", "line": "用法演示", "shot": "特写", "on_screen": "用法"},
            ],
            "cta": "点击下方购物车",
            "target_duration_sec": 30,
        }
        metrics = analyze_script(variant, BRIEF)
        self.assertTrue(metrics["hook_has_number"])
        self.assertTrue(metrics["hook_is_concise"])
        self.assertTrue(metrics["hook_has_tension"])
        self.assertTrue(metrics["hook_question"])
        self.assertTrue(metrics["cta_present"])
        self.assertEqual(metrics["hook_keyword_hits"], ["暗沉"])
        self.assertEqual(metrics["beat_count"], 2)
        # 两拍时长 8s / 7s，总体标准差为 0.5，用来验证节拍时长确实被解析出来
        self.assertAlmostEqual(metrics["pacing_stddev"], 0.5)

    def test_long_hook_without_number_fails_flags(self):
        variant = {
            "hook": "很多人在面对这个问题的时候一直没有找到合适的方法，今天我们把这件事讲清楚",
            "beats": [{"t": "0-10s", "line": "x", "shot": "中景", "on_screen": "y"}],
            "cta": "继续往下看",
            "target_duration_sec": 30,
        }
        metrics = analyze_script(variant, BRIEF)
        self.assertFalse(metrics["hook_has_number"])
        self.assertFalse(metrics["hook_is_concise"])
        self.assertFalse(metrics["hook_question"])
        self.assertFalse(metrics["cta_present"])

    def test_pain_point_fuzzy_match(self):
        """中文无分词，用 2-gram 滑窗做宽松匹配（"熬夜后肤色暗沉" ↔ "暗沉"）。"""
        variant = {
            "hook": "暗沉怎么办",
            "beats": [{"t": "0-5s", "line": "x", "shot": "s", "on_screen": "o"}],
            "cta": "点击",
            "target_duration_sec": 30,
        }
        metrics = analyze_script(variant, BRIEF)
        self.assertIn("熬夜后肤色暗沉", metrics["hook_pain_hits"])

    def test_missing_brief_does_not_crash(self):
        variant = {"hook": "暗沉？", "beats": [], "cta": "", "target_duration_sec": 30}
        metrics = analyze_script(variant, None)
        self.assertEqual(metrics["audience_keyword_pool"], [])


class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()

        @self.registry.tool("echo", "回显", {"type": "object", "properties": {"x": {"type": "string"}}})
        def echo(x: str = "") -> dict:
            return {"x": x}

        @self.registry.tool("boom", "总是失败")
        def boom() -> dict:
            raise RuntimeError("内部炸了")

    def test_successful_invocation(self):
        result = self.registry.invoke("echo", {"x": "hi"})
        self.assertTrue(result.ok)
        self.assertEqual(result.data, {"x": "hi"})

    def test_unknown_tool_returns_readable_error(self):
        """模型幻觉出不存在的工具名是很常见的，必须给出可用工具清单。"""
        result = self.registry.invoke("nope", {})
        self.assertFalse(result.ok)
        self.assertIn("不存在", result.error)
        self.assertIn("echo", result.error)

    def test_handler_exception_is_captured(self):
        result = self.registry.invoke("boom", {})
        self.assertFalse(result.ok)
        self.assertIn("RuntimeError", result.error)

    def test_argument_mismatch_is_captured_with_schema(self):
        result = self.registry.invoke("echo", {"unexpected": 1})
        self.assertFalse(result.ok)
        self.assertIn("参数不匹配", result.error)
        self.assertIn("schema", result.error)

    def test_reason_field_is_stripped(self):
        """模型常带一个 _reason 说明调用理由，它不属于工具签名，必须剥离。"""
        result = self.registry.invoke("echo", {"x": "hi", "_reason": "因为我想"})
        self.assertTrue(result.ok)

    def test_schemas_are_openai_shaped(self):
        schemas = self.registry.schemas()
        self.assertTrue(all(s["type"] == "function" for s in schemas))
        self.assertTrue(all("parameters" in s["function"] for s in schemas))

    def test_observation_serialization(self):
        ok_obs = json.loads(self.registry.invoke("echo", {"x": "1"}).to_observation())
        bad_obs = json.loads(self.registry.invoke("nope", {}).to_observation())
        self.assertTrue(ok_obs["ok"])
        self.assertFalse(bad_obs["ok"])

    def test_registering_without_handler_rejected(self):
        with self.assertRaises(ValueError):
            self.registry.register(ToolSpec(name="x", description="d"))


class TestMediaDegradation(unittest.TestCase):
    def test_manifest_branch_is_used_when_no_ffmpeg(self):
        """无 ffmpeg 时走旁挂清单（第二级降级），并且**如实标注来源**。"""
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "clip.srt"
            media.write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
            (Path(tmp) / "clip.frames.json").write_text(
                json.dumps({"frames": [{"time_sec": 1.0, "caption": "测试帧"}]}),
                encoding="utf-8",
            )
            result = extract_keyframes(media, count=3)

        self.assertEqual(result["source"], "manifest")
        self.assertTrue(result["available"])
        self.assertEqual(result["frames"][0]["caption"], "测试帧")

    def test_synthetic_branch_is_labelled_and_deterministic(self):
        """既无 ffmpeg 也无清单时合成占位，但必须标注 synthetic 且结果可复现。

        这里用一个 .bin 假文件：即使机器上装了 ffmpeg，ffprobe 也会失败，
        最终仍会落到第三级降级，因此断言可以无条件成立。
        """
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "no_manifest.bin"
            media.write_bytes(b"fake video bytes")
            first = extract_keyframes(media, count=4)
            second = extract_keyframes(media, count=4)

        self.assertEqual(first["source"], "synthetic")
        self.assertFalse(first["available"])
        self.assertIn("不可用", first["reason"])
        self.assertEqual(
            [f["ref"] for f in first["frames"]], [f["ref"] for f in second["frames"]]
        )

    def test_missing_file_still_returns_a_result(self):
        """文件不存在也不能抛异常，工具失败必须以观测形式返回。"""
        result = extract_keyframes("D:/definitely/not/here.mp4", count=2)
        self.assertEqual(result["source"], "synthetic")
        self.assertEqual(len(result["frames"]), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
