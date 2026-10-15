"""Agent 层。

四个 Agent 按流水线顺序各司其职，**职责不重叠**是有意的设计约束：

==================  ====================  ================================
Agent               输入                   输出
==================  ====================  ================================
:class:`Analyst`    brief + 素材            结构化的素材事实（场景/情绪/风险）
:class:`Scripter`   brief + 素材事实        N 个不同角度的脚本变体（含分镜）
:class:`Critic`     变体 + 确定性评测      语义诊断 + 可执行的重写指令
:class:`PromptOps`  基线提示词 + 视觉上下文 单变量提示词变体与排序
==================  ====================  ================================

共同基类保证三件事：结构必须过 schema、工具调用有据可查、
失败必须显式降级而不是静默返回半成品。
"""

from frame_agent.agents.analyst import AnalystAgent
from frame_agent.agents.base import Agent, AgentResult
from frame_agent.agents.critic import CriticAgent
from frame_agent.agents.promptops import PromptOpsAgent
from frame_agent.agents.scripter import ScripterAgent

__all__ = [
    "Agent",
    "AgentResult",
    "AnalystAgent",
    "ScripterAgent",
    "CriticAgent",
    "PromptOpsAgent",
]
