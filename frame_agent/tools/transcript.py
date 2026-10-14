"""字幕/转写文本解析与统计。

为什么要自己写 SRT 解析
=======================

- ``pysrt`` 之类的库要引入依赖；而 SRT 的格式简单到不值得。
- 真正有价值的部分不是"解析"，而是**从时间轴里算出可被评测的统计量**：
  语速（字/秒）、最长静默间隔、句子长度分布。这些量直接决定了
  "这条素材的节奏是否适合剪成 30 秒短视频"，是脚本生成的重要输入。

支持格式：``.srt`` / ``.vtt`` / 纯文本 ``.txt``（无时间轴时按时长均分兜底）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

__all__ = ["Cue", "parse_subtitles", "load_transcript", "transcript_stats"]

_TIME_RE = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})[.,](?P<ms>\d{1,3})"
)
_ARROW_RE = re.compile(r"\s*-->\s*")


@dataclass
class Cue:
    """一条字幕。"""

    index: int
    start: float  # 秒
    end: float
    text: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "text": self.text,
        }


def _to_seconds(match: re.Match) -> float:
    h, m, s = int(match.group("h")), int(match.group("m")), int(match.group("s"))
    ms_raw = match.group("ms")
    # "5" 应理解为 500ms，"50" 是 500ms，"500" 是 500ms —— 统一左对齐补零
    ms = int(ms_raw.ljust(3, "0")[:3])
    return h * 3600 + m * 60 + s + ms / 1000.0


def parse_subtitles(text: str) -> list[Cue]:
    """解析 SRT/VTT 文本为字幕列表。

    对格式噪声相当宽容：空行、序号缺失、多余空白、VTT 的 ``WEBVTT`` 头与
    cue 设置（``align:start``）都能正确处理——真实素材几乎不会规整。
    """
    cues: list[Cue] = []
    # 先按时间行切块，再取该行之后的文本作为内容
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))

    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        if lines[0].upper().startswith("WEBVTT"):
            lines = lines[1:]
            if not lines:
                continue

        timeline_index = None
        for i, line in enumerate(lines):
            if "-->" in line:
                timeline_index = i
                break
        if timeline_index is None:
            continue

        parts = _ARROW_RE.split(lines[timeline_index])
        if len(parts) < 2:
            continue
        start_match = _TIME_RE.search(parts[0])
        end_match = _TIME_RE.search(parts[1])
        if not start_match or not end_match:
            continue

        body = " ".join(lines[timeline_index + 1 :]).strip()
        # VTT 的 karaoke/位置标签要清掉，否则会污染文本
        body = re.sub(r"<[^>]+>", "", body).strip()

        cues.append(
            Cue(
                index=len(cues) + 1,
                start=_to_seconds(start_match),
                end=_to_seconds(end_match),
                text=body,
            )
        )

    return cues


def load_transcript(path: str | Path, *, default_duration: float = 0.0) -> list[Cue]:
    """读取字幕文件。

    扩展名为 ``.txt``（或任何解析不出时间轴的文件）时，
    按 ``default_duration`` 均分生成等长时间轴，而不是返回空列表——
    这样"只有文案、没有字幕时间轴"的素材依然能进入后续流程。
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"字幕文件不存在：{file_path}")

    raw = file_path.read_text(encoding="utf-8", errors="replace")

    if file_path.suffix.lower() in {".srt", ".vtt"}:
        cues = parse_subtitles(raw)
        if cues:
            return cues

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return []
    total = max(default_duration, float(len(lines) * 3))
    step = total / len(lines)
    return [
        Cue(index=i + 1, start=round(i * step, 2), end=round((i + 1) * step, 2), text=ln)
        for i, ln in enumerate(lines)
    ]


def transcript_stats(cues: list[Cue]) -> dict:
    """从时间轴推导可用于评测与决策的统计量。

    ``chars_per_second`` 是最有用的一个：短视频口播的舒适区间大约
    4.0–6.0 字/秒，偏低说明节奏拖，偏高说明赶。它同时也用于
    由脚本文本反推成片时长，从而校验"脚本能不能塞进 30 秒"。
    """
    if not cues:
        return {
            "cue_count": 0,
            "duration_sec": 0.0,
            "char_count": 0,
            "chars_per_second": 0.0,
            "max_gap_sec": 0.0,
            "gaps_over_1s": 0,
            "sentence_count": 0,
            "mean_sentence_len": 0.0,
        }

    duration = max(cue.end for cue in cues) - min(cue.start for cue in cues)
    char_count = sum(len(cue.text) for cue in cues)

    gaps = [
        max(0.0, cues[i + 1].start - cues[i].end) for i in range(len(cues) - 1)
    ]

    sentences = [s for cue in cues for s in re.split(r"[。！？!?；;]", cue.text) if s.strip()]
    lengths = [len(s) for s in sentences] or [0]

    return {
        "cue_count": len(cues),
        "duration_sec": round(duration, 2),
        "char_count": char_count,
        "chars_per_second": round(char_count / duration, 2) if duration > 0 else 0.0,
        "max_gap_sec": round(max(gaps), 2) if gaps else 0.0,
        "gaps_over_1s": sum(1 for g in gaps if g > 1.0),
        "sentence_count": len(sentences),
        "mean_sentence_len": round(sum(lengths) / len(lengths), 2),
    }
