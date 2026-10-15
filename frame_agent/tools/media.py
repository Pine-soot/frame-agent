"""媒体处理工具：关键帧抽取与探测，**带三级降级**。

这里体现的是生产环境里最常见的现实：你的流水线要跑在很多台机器上，
其中相当一部分**没有装 ffmpeg**（CI 容器、评审者的笔记本、同事的 Windows）。

如果"没装 ffmpeg"就等于"整条流水线不可用"，这个项目就失去了可复现性。
所以本模块定义了三级降级，每一级都**如实标注**自己是什么级别：

======  ==================  ===========================================
级别    来源 ``source`` 值   何时触发
======  ==================  ===========================================
1      ``ffmpeg``           检测到 ffmpeg，真实抽帧
2      ``manifest``         同目录存在 ``<name>.frames.json`` 旁挂清单
3      ``synthetic``        都没有，按确定性规则合成帧引用（明确标记）
======  ==================  ===========================================

关键点：**降级不是静默的**。``source`` 字段会一路带进最终报告，
让任何读到结果的人都知道"这次的关键帧是真实的还是合成的"。
静默降级比直接报错危险得多，它会让假的结论看起来像真的。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

__all__ = [
    "ffmpeg_available",
    "ffprobe_available",
    "probe_video",
    "extract_keyframes",
    "load_frames_manifest",
]


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def _run(cmd: list[str], timeout: float = 30.0) -> tuple[int, str, str]:
    """执行外部命令，任何异常都收敛为 (returncode, stdout, stderr)。"""
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return completed.returncode, completed.stdout or "", completed.stderr or ""
    except (subprocess.TimeoutExpired, OSError) as exc:
        return -1, "", f"{type(exc).__name__}: {exc}"


def probe_video(path: str | Path) -> dict:
    """探测视频元信息。无 ffprobe 时返回 ``available=False`` 而非抛错。"""
    file_path = Path(path)
    if not file_path.exists():
        return {"available": False, "reason": f"文件不存在：{file_path}"}

    if not ffprobe_available():
        return {
            "available": False,
            "reason": "未检测到 ffprobe，无法读取容器元信息",
            "size_bytes": file_path.stat().st_size,
        }

    code, stdout, stderr = _run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration",
            "-show_entries", "format=duration",
            "-of", "json",
            str(file_path),
        ]
    )
    if code != 0:
        return {"available": False, "reason": f"ffprobe 失败：{stderr.strip()[:200]}"}

    try:
        data = json.loads(stdout)
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format") or {}
        duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
        return {
            "available": True,
            "width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0),
            "frame_rate": stream.get("r_frame_rate") or "",
            "duration_sec": round(duration, 2),
            "size_bytes": file_path.stat().st_size,
        }
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return {"available": False, "reason": f"解析 ffprobe 输出失败：{exc}"}


def load_frames_manifest(path: str | Path) -> list[dict] | None:
    """读取旁挂关键帧清单 ``<name>.frames.json``。

    允许人工/上游系统预先产出关键帧描述，用于"素材已经过了视觉理解管线、
    现在只需要跑文本侧"的场景。返回 None 表示没有清单。
    """
    file_path = Path(path)
    candidates = [
        file_path.with_suffix(".frames.json"),
        file_path.with_name(file_path.stem + ".frames.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
            if isinstance(data, dict):
                data = data.get("frames", [])
            if isinstance(data, list):
                return data
    return None


def _synthetic_frames(path: Path, count: int, duration: float) -> list[dict]:
    """确定性合成帧引用。

    不生成任何图片，只产出"帧的位置与标识"。基于文件内容哈希，
    所以同一素材每次得到完全一致的清单，测试可精确断言。
    """
    try:
        digest = hashlib.sha256(path.read_bytes()[:65536]).hexdigest()[:12]
    except OSError:
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]

    step = (duration or count * 2) / max(1, count)
    return [
        {
            "index": i + 1,
            "time_sec": round(i * step, 2),
            "ref": f"synthetic:{digest}:{i + 1}",
            "source": "synthetic",
        }
        for i in range(count)
    ]


def extract_keyframes(
    video_path: str | Path,
    out_dir: str | Path | None = None,
    count: int = 6,
    duration_hint: float = 0.0,
) -> dict:
    """抽取关键帧，返回 ``{"frames": [...], "source": "...", "available": bool}``。

    返回结构里同时给出 ``source`` 与 ``reason``，让调用方（以及最终报告）
    能明确知道这批帧的可信级别。
    """
    file_path = Path(video_path)
    count = max(1, int(count))

    # ---- 级别 1：真实抽帧 ----
    if file_path.exists() and ffmpeg_available():
        meta = probe_video(file_path)
        duration = float(meta.get("duration_sec") or duration_hint or 0.0)
        target_dir = Path(out_dir) if out_dir else file_path.parent / f"{file_path.stem}_frames"
        target_dir.mkdir(parents=True, exist_ok=True)

        if duration > 0:
            # 均匀采样，避开首尾各 5% 的黑场/片尾
            step = duration / (count + 1)
            timestamps = [round(step * (i + 1), 2) for i in range(count)]
        else:
            timestamps = [round(i * 2.0, 2) for i in range(count)]

        frames: list[dict] = []
        for i, ts in enumerate(timestamps):
            out_file = target_dir / f"frame_{i + 1:03d}_{ts}s.jpg"
            code, _, stderr = _run(
                [
                    "ffmpeg", "-y", "-ss", str(ts), "-i", str(file_path),
                    "-frames:v", "1", "-q:v", "3", str(out_file),
                ]
            )
            if code == 0 and out_file.exists():
                frames.append(
                    {
                        "index": i + 1,
                        "time_sec": ts,
                        "path": str(out_file),
                        "ref": str(out_file),
                        "source": "ffmpeg",
                    }
                )
            else:
                frames.append(
                    {
                        "index": i + 1,
                        "time_sec": ts,
                        "ref": f"failed:{file_path.name}:{ts}",
                        "source": "ffmpeg",
                        "error": stderr.strip()[:160],
                    }
                )
        if any(f.get("path") for f in frames):
            return {
                "frames": frames,
                "source": "ffmpeg",
                "available": True,
                "duration_sec": duration,
                "reason": "",
            }

    # ---- 级别 2：旁挂清单 ----
    manifest = load_frames_manifest(file_path) if file_path.exists() else None
    if manifest:
        normalized = [
            {
                "index": i + 1,
                "time_sec": float(item.get("time_sec", item.get("t", i * 2.0)) or i * 2.0),
                "ref": item.get("ref") or item.get("path") or f"manifest:{i + 1}",
                "caption": item.get("caption") or item.get("description") or "",
                "source": "manifest",
            }
            for i, item in enumerate(manifest[:count])
        ]
        return {
            "frames": normalized,
            "source": "manifest",
            "available": True,
            "duration_sec": duration_hint,
            "reason": "使用旁挂关键帧清单（未调用 ffmpeg）",
        }

    # ---- 级别 3：确定性合成 ----
    return {
        "frames": _synthetic_frames(file_path, count, duration_hint),
        "source": "synthetic",
        "available": False,
        "duration_sec": duration_hint,
        "reason": (
            "既无 ffmpeg 也无旁挂清单，已合成占位帧引用。"
            "下游若依赖真实画面内容，结果须视为不可用。"
        ),
    }
