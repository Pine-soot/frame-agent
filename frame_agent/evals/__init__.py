"""评测层。

- :mod:`frame_agent.evals.rubric`：确定性评分（谁给分）
- :mod:`frame_agent.evals.runner`：批量回归（跑一批 brief 看是否退化）

拆开的原因：打分逻辑要被流水线**在线调用**（每生成一版脚本就评一次），
而批量回归是**离线动作**（改提示词后跑一遍看有没有把别的 case 改坏）。
两者生命周期不同，职责也不同，混在一个模块里迟早会互相牵扯。
"""

from frame_agent.evals.rubric import (
    DIMENSIONS,
    DIMENSION_WEIGHTS,
    evaluate,
)
from frame_agent.evals.runner import run_suite

__all__ = ["DIMENSIONS", "DIMENSION_WEIGHTS", "evaluate", "run_suite"]
