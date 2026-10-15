"""批量回归评测：改提示词之后，怎么知道没有把别的 case 改坏？

这就是本模块存在的意义。内容生产流水线最典型的退化方式是：
为了让 A 类目（美妆）的钩子更强，把提示词改成"必须用数字开头"，
结果 B 类目（SaaS）的脚本被改得不像人话。**单看一个 case 是发现不了的。**

所以 :func:`run_suite` 跑一批 brief，输出一张可直接贴进复盘文档的矩阵：

- 逐 case：最终采纳的变体、五维分数、总评、收敛轮数、模型调用数、token
- 汇总：通过率、平均分、平均收敛轮数、总成本、降级项清单

``degradation_notes`` 是保留的一列：它汇总所有"被降级"的地方
（结构校验失败、关键帧只有合成占位、达到最大反思轮数仍未达标）。
一份没有任何降级说明的报告，通常意味着你没有认真看降级说明。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

from frame_agent.evals.rubric import evaluate

__all__ = ["run_suite", "summarize_cases"]


def summarize_cases(cases: Sequence[dict]) -> dict:
    """把逐 case 结果汇成总体画像。"""
    if not cases:
        return {
            "case_count": 0,
            "pass_rate": 0.0,
            "mean_score": 0.0,
            "mean_rounds": 0.0,
            "total_model_calls": 0,
            "total_tokens": 0,
        }

    passed = [c for c in cases if c.get("verdict") == "pass"]
    scores = [float(c.get("overall") or 0.0) for c in cases]

    # 逐维度平均分，用来看"整体是哪个维度在拖后腿"
    dimension_totals: dict[str, list[float]] = {}
    for case in cases:
        for name, value in (case.get("dimension_scores") or {}).items():
            dimension_totals.setdefault(name, []).append(float(value))

    return {
        "case_count": len(cases),
        "pass_count": len(passed),
        "pass_rate": round(len(passed) / len(cases), 3),
        "mean_score": round(sum(scores) / len(scores), 2),
        "min_score": round(min(scores), 2),
        "max_score": round(max(scores), 2),
        "mean_rounds": round(
            sum(float(c.get("rounds") or 0) for c in cases) / len(cases), 2
        ),
        "total_model_calls": sum(int(c.get("model_calls") or 0) for c in cases),
        "total_tokens": sum(int(c.get("tokens") or 0) for c in cases),
        "mean_dimension_scores": {
            name: round(sum(values) / len(values), 2)
            for name, values in sorted(dimension_totals.items())
        },
        "cases_with_degradation": [
            c["brief_id"] for c in cases if c.get("degradation_notes")
        ],
    }


def run_suite(
    run_case: Callable[[str, str], dict],
    brief_paths: Sequence[str | Path],
    out_root: str | Path,
    *,
    verbose: bool = False,
) -> dict:
    """跑一批 brief。

    ``run_case`` 由调用方注入（签名 ``(brief_path, out_dir) -> result``），
    这样评测套件**不需要知道管线是怎么实现的**。好处是可以用一个
    "已知答案的假管线"来单测评测套件本身，测试不该依赖被它测试的对象。

    单个 case 抛异常不会中断整批：会被记成 ``status="error"`` 并继续，
    因为批量回归最重要的性质是"总是跑完并给出全景"，而不是"第一个错就停"。
    """
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []
    for brief_path in brief_paths:
        brief_path = Path(brief_path)
        started = time.monotonic()
        try:
            result = run_case(str(brief_path), str(root / brief_path.stem))
            case = result.get("eval", {}) if isinstance(result, dict) else {}
            cases.append(
                {
                    "brief_id": brief_path.stem,
                    "brief_path": str(brief_path),
                    "status": "ok",
                    "verdict": case.get("verdict"),
                    "overall": case.get("overall"),
                    "dimension_scores": case.get("dimension_scores"),
                    "rounds": result.get("rounds"),
                    "model_calls": result.get("model_calls"),
                    "tokens": (result.get("cost") or {}).get("total_tokens"),
                    "best_variant_id": (result.get("best_variant") or {}).get("id"),
                    "best_hook": (result.get("best_variant") or {}).get("hook"),
                    "degradation_notes": result.get("degradation_notes") or [],
                    "elapsed_sec": round(time.monotonic() - started, 3),
                }
            )
        except Exception as exc:  # noqa: BLE001 - 批量回归必须跑完
            cases.append(
                {
                    "brief_id": brief_path.stem,
                    "brief_path": str(brief_path),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_sec": round(time.monotonic() - started, 3),
                }
            )
        if verbose:  # pragma: no cover - 仅交互式使用
            last = cases[-1]
            print(
                f"[{last['brief_id']}] {last.get('status')} "
                f"score={last.get('overall')} verdict={last.get('verdict')}"
            )

    result = {"cases": cases, "summary": summarize_cases(cases)}

    # 汇总结果由 run_suite 自己落盘：它是这次批量回归的产物，
    # 交给调用方写会让"有没有写"变成调用方的责任，容易漏。
    import json

    (root / "suite.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return result


def rescore_variant(variant: dict, brief: dict, threshold: float = 82.0) -> dict:
    """离线重算某个已落盘变体的分数。

    用途：调整 rubric 权重后，对历史产物做**回溯重评**，
    不需要重新调用模型。这是把评测与生成解耦之后才可能有的能力。
    """
    return evaluate(variant, brief, pass_threshold=threshold)
