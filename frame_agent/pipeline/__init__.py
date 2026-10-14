"""编排层。

:class:`ContentPipeline` 是唯一的对外入口：
它负责把 Agent、工具、评测器、预算与持久化粘在一起，
并对**每一步都留下可审计的痕迹**（``run_state.json`` + ``trace.jsonl``）。
"""

from frame_agent.pipeline.orchestrator import ContentPipeline
from frame_agent.pipeline.state import RunState, brief_fingerprint

__all__ = ["ContentPipeline", "RunState", "brief_fingerprint"]
