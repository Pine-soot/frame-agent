# 架构与数据流

## 1. 分层

```text
┌──────────────────────────────────────────────────────────────┐
│  cli / __main__           run · demo · eval · inspect        │
├──────────────────────────────────────────────────────────────┤
│  pipeline                 RunState(检查点/轨迹) · Orchestrator │
│                           反思循环 · 降级汇总 · 产物落盘       │
├──────────────────────────────────────────────────────────────┤
│  agents                   Analyst · Scripter · Critic · PromptOps
│                           两阶段执行：证据采集 → 结构化产出     │
├──────────────────────────────────────────────────────────────┤
│  evals                    rubric(确定性评分) · runner(批量回归) │
├──────────────────────────────────────────────────────────────┤
│  tools                    registry · media · transcript · metrics
├──────────────────────────────────────────────────────────────┤
│  llm                      base(协议) · router(降级/预算/记账)  │
│                           · mock(离线) · openai_compat(在线)   │
├──────────────────────────────────────────────────────────────┤
│  schema / protocol        结构校验与修复 · 线协议约定           │
└──────────────────────────────────────────────────────────────┘
```

依赖方向严格单向向下。`llm` 不依赖 `agents`，二者共同依赖 `protocol` 与 `schema`，所以依赖图无环。

这条约束不只是洁癖。`mock` 后端需要理解 Agent 在请求什么任务，
如果它反向 import `agents`，就会出现"改一个 prompt 让 mock 也崩"的耦合。
把这些约定抽到 `protocol.py` 之后，离线后端只依赖一份稳定的标签表。

## 2. 一次运行的完整数据流

```text
brief.json
    │
    ├─▶ _resolve_media()          相对路径 → 绝对路径
    │
    ▼
build_registry(context)          5 个工具，闭包绑定当次上下文
    │
    ▼
┌── AnalystAgent ──────────────────────────────────────────────┐
│  阶段一  model(tools=5) → tool_calls[parse_transcript,        │
│                          extract_keyframes]                   │
│          → registry.invoke ×2 → observations{}                │
│  阶段二  consolidate(payload, observations)                   │
│          → 顶层 transcript / frames / frames_source           │
│          → SchemaGuard.generate() → media_insight             │
└──────────────────────────────────────────────────────────────┘
    │  insight
    ▼
┌── 反思循环  round = 1..max_rounds ────────────────────────────┐
│                                                              │
│   ScripterAgent ──▶ scripts[N]                                │
│        │  (round>1 时附带 previous_scripts + rewrite_directives)
│        ▼                                                      │
│   rubric.evaluate() ×N  ──▶ 挑最高分（按 variant_id 匹配）     │
│        │                                                      │
│        ▼                                                      │
│   CriticAgent ──▶ model(tools=2) → script_metrics,           │
│                   scan_compliance → 诊断 + 重写指令            │
│        │                                                      │
│        ├─ verdict == "pass" ──▶ break（达标即停）             │
│        └─ verdict == "revise" ──▶ 下一轮                      │
└──────────────────────────────────────────────────────────────┘
    │  best_scripts, rubric_result, critique, round_history
    ▼
┌── PromptOpsAgent ──▶ prompt_set（单变量变体 + 排序）──────────┐
└──────────────────────────────────────────────────────────────┘
    │
    ▼
report.render_markdown()  +  9 个 JSON 产物  +  run_state.json
```

## 3. 五个评测维度与权重

| 维度 | 权重 | 由什么构成 | 数据来源 |
| --- | ---: | --- | --- |
| `hook_strength` | 0.35 | 数字锚点、长度、悬念标记、疑问句、无效开场 | `metrics.analyze_script` |
| `audience_fit` | 0.20 | 关键词命中（钩子/正文）、痛点原话、场景具体性 | `metrics` + brief |
| `structure` | 0.20 | 节拍数、镜头完备度、字幕完备度、时长偏差 | `metrics` |
| `compliance` | 0.15 | 风险词命中数（一票否决式） | 内置词表 |
| `producibility` | 0.10 | 镜头/字幕完备、节拍充足、CTA 存在 | `metrics` |

权重之和为 1.0，在模块导入时断言（见 `rubric._check_weights`）。

hook 为什么占 0.35：前 3 秒决定去留，是短视频最关键的那道门槛。
compliance 为什么只有 0.15：它是门槛而不是加分项，但命中风险词时会直接腰斩到 30 分或 0 分，
实际效果仍然是一票否决。
producibility 为什么只有 0.10：拍得出来是必要条件，不是充分条件。

## 4. 错误处理策略

三层，各司其职：

| 层 | 策略 | 理由 |
| --- | --- | --- |
| 工具层 | 永不抛异常，返回 `ToolResult(ok=False, error=...)` | Agent 的价值就在于"失败后能看到失败并调整" |
| Agent 层 | schema 不通过 → 有界修复（默认 2 轮）→ 仍失败则 `ok=False` + `degradation` | 有限修复加明确失败，胜过无限重试 |
| 编排层 | 任何 Agent 失败都记入 `degradation_notes` 并继续能走的部分 | 交出"3 个方案 + 1 条失败说明"，比什么都不交有价值 |

例外只有预算熔断：`BudgetExceeded` 会向上抛出并被编排层捕获，
用来在"未收敛"和"超支"之间明确选择前者。

## 5. 状态与产物

```text
runs/<name>/
├── run_state.json    结构化检查点：步骤、产物、降级项、brief 指纹
├── trace.jsonl       追加式轨迹：每次调用/评分/重写一行，崩溃安全
├── report.md         人读版报告
├── scripts.json      scripts.json / scores.json / critique.json
├── prompt_set.json   media_insight.json / cost.json
```

`trace.jsonl` 用 JSONL 而不是 JSON 数组，是为了崩溃安全：
已落盘的完整行永远可读，不会因最后一行写坏而丢掉整个文件。
`read_trace()` 也容忍半行。
