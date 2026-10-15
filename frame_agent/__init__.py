"""frame-agent：垂直内容生产 Agent 流水线。

把"把一堆素材变成能发的短视频脚本"这件事拆成一条可观测、可评测、
可复现的流水线：

    ingest -> analyst -> scripter -> critic(反思循环) -> promptops -> eval -> report

设计上保持 **运行期零第三方依赖**，并提供确定性的离线后端，
使任何评审者 clone 之后无需 API Key 即可跑通全链路并复现全部指标。
"""

from frame_agent.schema import SchemaGuard, extract_json, validate

__version__ = "1.0.0"

__all__ = ["__version__", "SchemaGuard", "validate", "extract_json"]
