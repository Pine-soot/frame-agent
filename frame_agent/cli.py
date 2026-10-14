"""命令行入口。

四条子命令覆盖了三个使用场景：

``demo``
    零参数跑通内置示例。评审者第一次接触这个项目时唯一该做的事——
    不该让人先读文档再猜参数。

``run``
    对单个 brief 跑一次完整流水线。日常生产用。

``eval``
    对一批 brief 跑回归，输出分数矩阵。改提示词/改权重之后必跑，
    用来确认没有把别的 case 改坏。

``inspect``
    查看某次运行的检查点与轨迹摘要。出问题时第一步该做的排查动作。

设计上刻意把"参数解析"和"业务逻辑"分开：``main`` 只负责把 argv 翻译成
调用，所有实际工作都在 :mod:`frame_agent.pipeline` 里，便于被程序化复用
（比如嵌到别的系统里当函数调，而不是 subprocess）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from frame_agent import __version__
from frame_agent.evals.runner import run_suite
from frame_agent.llm import Budget, ModelRouter, build_backends
from frame_agent.pipeline import ContentPipeline, RunState

__all__ = ["main", "build_parser", "examples_dir"]


def examples_dir() -> Path:
    """内置示例目录（随包一起分发）。"""
    return Path(__file__).resolve().parent.parent / "examples"


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frame-agent",
        description="垂直内容生产 Agent 流水线：素材理解 → 脚本分镜 → 确定性评测 → 反思重写 → 提示词优化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"frame-agent {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--backend",
            default="mock",
            help="后端与降级链，逗号分隔。mock=离线确定性；openai=真实端点。"
            "推荐生产用 'openai,mock' 保证永远有产物。默认 mock。",
        )
        target.add_argument("--max-calls", type=int, default=64, help="模型调用次数上限（熔断阈值）")
        target.add_argument("--max-tokens", type=int, default=200_000, help="token 上限（熔断阈值）")
        target.add_argument("--max-seconds", type=float, default=600.0, help="墙钟时间上限（秒）")

    # ---- run ----
    run = sub.add_parser("run", help="对单个 brief 跑一次完整流水线")
    run.add_argument("--brief", required=True, help="brief JSON 文件路径")
    run.add_argument("--out", required=True, help="产物输出目录")
    run.add_argument("--rounds", type=int, default=2, help="反思循环最大轮数，1 表示不重写")
    run.add_argument("--variants", type=int, default=3, help="每轮生成的脚本变体数")
    run.add_argument("--threshold", type=float, default=82.0, help="达标分数线")
    run.add_argument("--frames-dir", default=None, help="关键帧输出目录")
    run.add_argument("--json", action="store_true", help="以 JSON 输出完整结果（默认输出摘要）")
    add_common(run)

    # ---- demo ----
    demo = sub.add_parser("demo", help="用内置示例零参数跑通全链路")
    demo.add_argument("--out", default="runs/demo", help="产物输出目录")
    demo.add_argument("--case", default="skincare_launch", help="示例 brief 名称")
    demo.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    add_common(demo)

    # ---- eval ----
    ev = sub.add_parser("eval", help="对一批 brief 跑回归评测")
    ev.add_argument("--briefs", default=None, help="brief 目录（默认用内置示例）")
    ev.add_argument("--out", default="runs/suite", help="回归产物输出目录")
    ev.add_argument("--rounds", type=int, default=2, help="反思循环最大轮数")
    ev.add_argument("--variants", type=int, default=3, help="每轮生成的脚本变体数")
    ev.add_argument("--threshold", type=float, default=82.0, help="达标分数线")
    add_common(ev)

    # ---- inspect ----
    ins = sub.add_parser("inspect", help="查看某次运行的检查点与轨迹")
    ins.add_argument("--state", default="runs/demo", help="运行目录（内含 run_state.json）")
    ins.add_argument("--trace", action="store_true", help="同时打印最近若干条轨迹")

    return parser


def _build_router(args: argparse.Namespace) -> ModelRouter:
    budget = Budget(
        max_calls=args.max_calls,
        max_tokens=args.max_tokens,
        max_seconds=args.max_seconds,
    )
    backends = build_backends(args.backend, budget=budget)
    return ModelRouter(backends, budget=budget)


def _resolve_brief_paths(briefs: str | None) -> list[Path]:
    if briefs:
        target = Path(briefs)
        if target.is_dir():
            return sorted(target.glob("*.json"))
        return [target]
    return sorted((examples_dir() / "briefs").glob("*.json"))


def _base_dir_for(brief_path: Path) -> Path:
    """决定本次运行的素材路径解析基准目录。

    规则：brief 位于仓库 ``examples/briefs/`` 下时，基准取仓库根——
    因为示例 brief 里的素材路径是相对仓库根书写的；其余情况用 cwd，
    符合"相对路径相对当前工作目录"的一般直觉。
    """
    resolved = brief_path.resolve()
    if resolved.parent.name == "briefs" and resolved.parent.parent.name == "examples":
        return resolved.parent.parent.parent
    return Path.cwd()


def _print_summary(result: dict) -> None:
    """人读摘要。刻意短——详细内容都在 report.md 里。"""
    if result.get("status") != "ok":
        print(f"❌ 运行失败：{result.get('reason')}")
        for note in result.get("degradation_notes") or []:
            print(f"   - {note}")
        return

    ev = result.get("eval") or {}
    best = result.get("best_variant") or {}
    cost = result.get("cost") or {}

    print("=" * 68)
    print(f"运行 ID   : {result.get('run_id')}")
    print(f"后端      : {result.get('backend')}")
    print(f"采纳变体  : {best.get('id')}（{best.get('angle')}）")
    print(f"钩子      : {best.get('hook')}")
    print(f"总分      : {ev.get('overall')} / 阈值 {ev.get('pass_threshold')} "
          f"→ {'✅ 达标' if ev.get('verdict') == 'pass' else '⚠️ 未达标'}")
    print(f"反思轮数  : {result.get('rounds')}")
    print(f"模型调用  : {cost.get('total_calls')} 次，token {cost.get('total_tokens')}")
    print("-" * 68)
    print("维度得分：")
    for name, score in (ev.get("dimension_scores") or {}).items():
        bar = "█" * int(score / 5)
        print(f"  {name:<16} {score:>3}  {bar}")
    if result.get("degradation_notes"):
        print("-" * 68)
        print("降级项：")
        for note in result["degradation_notes"]:
            print(f"  ⚠️ {note}")
    print("=" * 68)
    print("产物：")
    for path in result.get("artifacts") or []:
        print(f"  - {path}")


def cmd_run(args: argparse.Namespace) -> int:
    brief = _load_json(args.brief)
    pipeline = ContentPipeline(
        _build_router(args),
        out_dir=args.out,
        max_rounds=args.rounds,
        variant_count=args.variants,
        pass_threshold=args.threshold,
        frames_dir=args.frames_dir,
        base_dir=_base_dir_for(Path(args.brief)),
    )
    result = pipeline.run(brief, brief_path=str(args.brief), backend=args.backend)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        _print_summary(result)
    return 0 if result.get("status") == "ok" else 1


def cmd_demo(args: argparse.Namespace) -> int:
    brief_path = examples_dir() / "briefs" / f"{args.case}.json"
    if not brief_path.exists():
        print(f"❌ 找不到示例 brief：{brief_path}", file=sys.stderr)
        return 2
    brief = _load_json(brief_path)
    repo_root = examples_dir().parent
    pipeline = ContentPipeline(
        _build_router(args),
        out_dir=args.out,
        max_rounds=int(brief.get("max_rounds") or 2),
        variant_count=int(brief.get("variants") or 3),
        pass_threshold=float(brief.get("pass_threshold") or 82.0),
        # 内置示例的素材路径相对仓库根目录书写，显式把 base_dir 指过去，
        # 这样在任何 cwd 下执行都能跑通。
        base_dir=repo_root,
    )
    result = pipeline.run(brief, brief_path=str(brief_path), backend=args.backend)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        _print_summary(result)
    return 0 if result.get("status") == "ok" else 1


def cmd_eval(args: argparse.Namespace) -> int:
    brief_paths = _resolve_brief_paths(args.briefs)
    if not brief_paths:
        print("❌ 没有找到任何 brief", file=sys.stderr)
        return 2

    def run_case(brief_path: str, out_dir: str) -> dict:
        brief = _load_json(brief_path)
        pipeline = ContentPipeline(
            _build_router(args),
            out_dir=out_dir,
            max_rounds=args.rounds,
            variant_count=args.variants,
            pass_threshold=args.threshold,
            base_dir=_base_dir_for(Path(brief_path)),
        )
        return pipeline.run(brief, brief_path=brief_path, backend=args.backend)

    suite = run_suite(run_case, brief_paths, args.out)
    # suite.json 由 run_suite 负责写入，这里只负责呈现
    out_path = Path(args.out) / "suite.json"

    summary = suite["summary"]
    print("=" * 78)
    print("回归评测结果")
    print("=" * 78)
    print(f"| {'brief':<22} | {'状态':<6} | {'总分':>6} | {'判定':<7} | {'轮数':>4} | 采纳钩子")
    print("-" * 78)
    for case in suite["cases"]:
        print(
            f"| {case.get('brief_id',''):<22} | {case.get('status',''):<6} | "
            f"{str(case.get('overall','-')):>6} | {str(case.get('verdict','-')):<7} | "
            f"{str(case.get('rounds','-')):>4} | {case.get('best_hook','') or case.get('error','')}"
        )
    print("-" * 78)
    print(
        f"通过率 {summary['pass_rate']:.0%}（{summary.get('pass_count',0)}/{summary['case_count']}）"
        f"｜均分 {summary['mean_score']}｜平均轮数 {summary['mean_rounds']}"
        f"｜总 token {summary['total_tokens']}"
    )
    if summary.get("mean_dimension_scores"):
        print("维度均分：" + "，".join(
            f"{k}={v}" for k, v in summary["mean_dimension_scores"].items()
        ))
    if summary.get("cases_with_degradation"):
        print(f"含降级的 case：{', '.join(summary['cases_with_degradation'])}")
    print(f"\n完整结果：{out_path}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    state = RunState.load(args.state)
    if state is None:
        print(f"❌ 在 {args.state} 下找不到 run_state.json", file=sys.stderr)
        return 2

    print("=" * 68)
    print(f"run_id     : {state.run_id}")
    print(f"brief_hash : {state.brief_hash}")
    print(f"brief_path : {state.brief_path}")
    print(f"backend    : {state.backend}")
    print(f"started_at : {state.started_at}")
    print("-" * 68)
    print("步骤：")
    for step in state.steps:
        metrics = step.get("metrics") or {}
        extra = f" {metrics}" if metrics else ""
        print(f"  {step['index']:>2}. [{step['status']:<8}] {step['name']}: {step['detail']}{extra}")
    print("-" * 68)
    print("产物：")
    for artifact in state.artifacts:
        print(f"  - [{artifact['kind']:<8}] {artifact['path']}")
    if state.degradation_notes:
        print("-" * 68)
        print("降级项：")
        for note in state.degradation_notes:
            print(f"  ⚠️ {note}")

    if args.trace:
        records = state.read_trace()
        print("-" * 68)
        print(f"轨迹（共 {len(records)} 条，显示最近 12 条）：")
        for record in records[-12:]:
            event = record.get("event", "?")
            detail = {
                k: v
                for k, v in record.items()
                if k not in {"ts", "run_id", "event"}
            }
            print(f"  [{event}] {json.dumps(detail, ensure_ascii=False, default=str)[:150]}")
    print("=" * 68)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "run": cmd_run,
        "demo": cmd_demo,
        "eval": cmd_eval,
        "inspect": cmd_inspect,
    }
    handler = handlers[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:  # pragma: no cover
        print("\n已中断", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI 顶层兜底，给出可读错误而非堆栈
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
