"""运行状态：检查点、轨迹、断点续跑。

为什么 Agent 流水线必须有检查点
==============================

一次 run 可能包含近十次模型调用与多次工具执行，耗时从几秒到几分钟不等。
如果只在最后写一次结果，那么：

- 中途限流/超时 → 前面所有工作白做，重跑还要再花一遍钱；
- 出问题时没有任何中间态可查，"模型到底在哪一步走偏了"只能靠猜。

所以本模块维护两类持久化：

``run_state.json``
    结构化检查点：步骤列表、产物清单、预算快照、brief 指纹。
    用于**续跑**与**程序化消费**（报告、CI 断言）。

``trace.jsonl``
    追加式轨迹：每一次调用、每一次评分、每一次重写都写一行。
    用 JSONL 而不是 JSON 数组，是为了**崩溃安全**——已经落盘的完整行永远可读，
    不会因为最后一行写坏而丢掉整个文件。

续跑的正确性依赖 ``brief_hash``：只有 brief 内容完全一致时才允许复用检查点。
只看 run_id 是不够的——同一个目录里改完 brief 再跑，
如果直接复用旧状态，产出的报告会与 brief 不符，这种错最难查。
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["RunState", "brief_fingerprint"]


def brief_fingerprint(brief: dict) -> str:
    """对 brief 做规范化指纹。键排序 + 紧凑分隔符，保证同一语义得到同一哈希。"""
    canonical = json.dumps(brief, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class RunState:
    """一次流水线运行的全部可持久化状态。"""

    run_id: str
    out_dir: Path
    brief_hash: str
    brief_path: str = ""
    backend: str = "mock"
    started_at: str = ""
    steps: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    degradation_notes: list[str] = field(default_factory=list)

    # -- 路径 ---------------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.out_dir / "run_state.json"

    @property
    def trace_path(self) -> Path:
        return self.out_dir / "trace.jsonl"

    # -- 构造 ---------------------------------------------------------------

    @classmethod
    def create(
        cls,
        out_dir: str | Path,
        brief: dict,
        *,
        brief_path: str = "",
        backend: str = "mock",
        run_id: str | None = None,
    ) -> "RunState":
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return cls(
            run_id=run_id or uuid.uuid4().hex[:12],
            out_dir=directory,
            brief_hash=brief_fingerprint(brief),
            brief_path=brief_path,
            backend=backend,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    # -- 记录 ---------------------------------------------------------------

    def log_step(
        self,
        name: str,
        status: str,
        *,
        detail: str = "",
        metrics: dict | None = None,
    ) -> None:
        self.steps.append(
            {
                "index": len(self.steps) + 1,
                "name": name,
                "status": status,
                "detail": detail,
                "metrics": metrics or {},
                "at": time.strftime("%H:%M:%S"),
            }
        )

    def add_artifact(
        self, name: str, path: str | Path, kind: str, description: str = ""
    ) -> None:
        self.artifacts.append(
            {
                "name": name,
                "path": str(path),
                "kind": kind,
                "description": description,
            }
        )

    def note_degradation(self, message: str) -> None:
        if message not in self.degradation_notes:
            self.degradation_notes.append(message)

    def append_trace(self, record: dict) -> None:
        """追加一行轨迹。**立即 flush**，崩溃时也不丢已完成的记录。"""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {"ts": time.time(), "run_id": self.run_id, **record},
            ensure_ascii=False,
            default=str,
        )
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    # -- 持久化 -------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "brief_hash": self.brief_hash,
            "brief_path": self.brief_path,
            "backend": self.backend,
            "started_at": self.started_at,
            "steps": self.steps,
            "artifacts": self.artifacts,
            "degradation_notes": self.degradation_notes,
        }

    def save(self) -> Path:
        """原子写入：先写临时文件再替换，避免中断留下半个 JSON。

        这一步在长跑任务里非常关键——直接覆盖写时如果进程被杀，
        留下的破损 run_state.json 会让"续跑"这个功能本身失效。
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        temp = self.state_path.with_suffix(".json.tmp")
        temp.write_text(payload, encoding="utf-8")
        temp.replace(self.state_path)
        return self.state_path

    @classmethod
    def load(cls, out_dir: str | Path) -> "RunState | None":
        directory = Path(out_dir)
        path = directory / "run_state.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return cls(
            run_id=data.get("run_id") or "unknown",
            out_dir=directory,
            brief_hash=data.get("brief_hash") or "",
            brief_path=data.get("brief_path") or "",
            backend=data.get("backend") or "mock",
            started_at=data.get("started_at") or "",
            steps=list(data.get("steps") or []),
            artifacts=list(data.get("artifacts") or []),
            degradation_notes=list(data.get("degradation_notes") or []),
        )

    @classmethod
    def resumable(
        cls, out_dir: str | Path, brief: dict
    ) -> "RunState | None":
        """仅当 brief 指纹一致时返回可复用的状态，否则返回 None。"""
        state = cls.load(out_dir)
        if state is None:
            return None
        if state.brief_hash != brief_fingerprint(brief):
            return None
        return state

    def read_trace(self) -> list[dict]:
        if not self.trace_path.exists():
            return []
        records: list[dict] = []
        for line in self.trace_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 容忍崩溃时留下的半行
        return records
