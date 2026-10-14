# frame-agent

> 垂直内容生产 Agent 流水线：**素材理解 → 脚本分镜生成 → 确定性评测 → 自反思重写 → 提示词优化**
>
> 零运行期依赖 · 离线可复现 · 指标可追溯 · 支持任意 OpenAI 兼容模型

`frame-agent` 不是"一个会写文案的提示词"。它把"把一批素材变成可投放的短视频脚本"
这件事拆成一条**可观测、可评测、可复现、可续跑**的流水线，并且对每一步都留下证据。

```text
$ python -m frame_agent demo
====================================================================
运行 ID   : f6c2fe493c9d
后端      : mock
采纳变体  : v1（痛点切入）
钩子      : 暗沉？28天自己看变化
总分      : 97.0 / 阈值 82.0 → ✅ 达标
反思轮数  : 2
模型调用  : 9 次，token 20756
--------------------------------------------------------------------
维度得分：
  hook_strength    100  ████████████████████
  audience_fit      85  █████████████████
  structure        100  ████████████████████
  compliance       100  ████████████████████
  producibility    100  ████████████████████
====================================================================
```

---

## 1. 这个项目解决什么问题

内容生产团队的真实痛点不是"写不出文案"，而是三个更难的问题：

| 问题 | 具体表现 | 本项目的做法 |
| --- | --- | --- |
| **质量不可复现** | 同一个提示词今天出好文案，明天出废文案，无法归因 | 用**确定性评测器**打分，每一分都能追到某个具体指标 |
| **批量生产时质量漂移** | 改了提示词让 A 类目变好，B 类目却悄悄变差 | **批量回归套件** + 逐维度均分，退化一眼可见 |
| **成本不可控** | 反思循环或工具循环退化时无上限烧 token | 调用次数/token/时间**三重预算熔断**，超支即停并如实降级 |

设计上的一条主线贯穿全部代码：**凡是能被确定的，就不要交给模型。**

- 打分 → 确定性评测器（`evals/rubric.py`）
- 合规扫描 → 独立工具，不复述模型结论（`tools/__init__.py`）
- 素材解析、时长推算、节奏统计 → 纯代码（`tools/`）
- 模型只负责它真正擅长的：**语义诊断、创意改写、归因说明**

---

## 2. 核心功能

### 2.1 四个职责单一的 Agent

| Agent | 输入 | 输出 | 关键设计 |
| --- | --- | --- | --- |
| `Analyst` | brief + 素材 | 场景切分、情绪、屏幕文字、合规风险 | 主动调工具取证；**只描述事实，不做创作建议** |
| `Scripter` | brief + 素材事实 (+ 重写指令) | N 个不同角度的脚本 + 分镜表 | 多角度是为了**可比较**；带上一版一起重写，避免"重新发明" |
| `Critic` | 变体 + 确定性评分 | 语义诊断 + 逐条可执行的重写指令 | **数值不由模型给**；合规结论来自独立扫描工具 |
| `PromptOps` | 基线提示词 + 视觉上下文 | **单变量**提示词变体 + 排序 | 遵守单变量原则，结果差异才能归因到具体那一项 |

### 2.2 真正会收敛的反思循环

```text
round=1  Scripter → 评测器挑最优 → Critic
         68.55 分（未达阈值）→ 给出「重写钩子：加入具体数字、压到 22 字内、制造张力」
round=2  Scripter 带着指令重写（保留上一版上下文）
         97.0 分 → Critic 判定 pass，停止
```

这个 `68.55 → 97.0` 是**真实的、可复现的数字**，不是示意。
每次运行的产物里都带完整的收敛轨迹表，每一分都能对应到上表某个指标的变动。

达标即停 —— 首轮就过的 case 只花 1 轮成本。这对批量生产的成本差异是决定性的。

### 2.3 结构可保证（不是"祈祷模型输出合法 JSON"）

``SchemaGuard`` 实现 *生成 → 提取 → 校验 → 带错误路径重试* 的闭环：

```text
模型输出 → extract_json（四层宽容策略） → validate
              ↑                              │ 有错误
              └──── 把 $.variants[0].hook: 缺少必需字段（required）回灌 ────┘
                                             │ 通过
                                             ▼
                                        返回结构化数据
```

关键差别在于**把校验错误作为观测回灌**：模型看到的是带路径的具体错误，
而不是一句无信息的"再试一次"。

### 2.4 三级降级（不依赖 ffmpeg 也能跑通）

| 级别 | `source` | 触发条件 |
| --- | --- | --- |
| 1 | `ffmpeg` | 检测到 ffmpeg，真实抽帧 |
| 2 | `manifest` | 存在旁挂清单 `<name>.frames.json` |
| 3 | `synthetic` | 都没有，合成占位并**明确标注不可采信** |

降级永远不静默 —— `source` 会一路带进最终报告的"降级与风险说明"章节。
静默降级比直接报错危险得多：它会让假的结论看起来像真的。

### 2.5 多模型降级 + 预算熔断 + 调用记账

```bash
--backend openai,mock   # 优先真实模型；限流/超时自动降级到离线后端
```

一次调用失败会记录一条带延迟与错误分类的审计记录；
`report.py` 输出的成本画像包含按后端的 p50/p95 延迟与 token 消耗。

---

## 3. 目录结构

```text
frame-agent/
├── frame_agent/
│   ├── protocol.py          # Agent 与后端之间的线协议（任务标签 / 上下文信封）
│   ├── schema.py            # 零依赖 JSON Schema 子集校验器 + 修复闭环
│   ├── report.py            # 人读版报告渲染（9 个章节）
│   ├── cli.py               # 命令行入口：run / demo / eval / inspect
│   ├── llm/
│   │   ├── base.py          # 协议与数据结构（无厂商绑定）
│   │   ├── router.py        # 降级链 / 退避重试 / 预算熔断 / 记账
│   │   ├── mock.py          # 确定性离线后端（保可复现）
│   │   └── openai_compat.py # 任意 OpenAI 兼容端点（仅标准库 urllib）
│   ├── tools/
│   │   ├── base.py          # 工具注册表：错误一律收敛为可读观测
│   │   ├── media.py         # 关键帧抽取（三级降级）
│   │   ├── transcript.py    # SRT/VTT/TXT 解析与节奏统计
│   │   ├── metrics.py       # 脚本量化指标（钩子/结构/时长/语速）
│   │   └── __init__.py      # 按运行上下文装配工具集
│   ├── agents/
│   │   ├── base.py          # 两阶段执行：证据采集 → 上下文收敛 → 结构化产出
│   │   ├── analyst.py       # 素材理解
│   │   ├── scripter.py      # 脚本与分镜生成
│   │   ├── critic.py        # 评审与重写指令
│   │   └── promptops.py     # 提示词单变量扩展
│   ├── pipeline/
│   │   ├── state.py         # 检查点 + JSONL 轨迹 + 断点续跑
│   │   └── orchestrator.py  # 编排：反思循环 / 降级处理 / 产物落盘
│   └── evals/
│       ├── rubric.py        # 五维加权确定性评分（权重和为 1.0，导入时断言）
│       └── runner.py        # 批量回归 + 逐维度均分
├── examples/
│   ├── briefs/              # two 个不同类目的 brief
│   └── media/               # 示例字幕 + 旁挂关键帧清单
├── tests/                   # 89 个用例（unittest，零第三方依赖）
└── docs/
    ├── architecture.md      # 架构与数据流
    ├── design-decisions.md  # 八个关键设计取舍
    ├── case-study.md        # 项目案例说明
    └── example-run/         # 一次真实运行的完整产物（可核对）
```

---

## 4. 运行与验证

### 4.1 环境要求

**Python ≥ 3.10，无任何第三方依赖。** 不需要 API Key，不需要 ffmpeg。

```bash
git clone https://github.com/Pine-soot/frame-agent.git
cd frame-agent
```

### 4.2 零参数跑通全链路

```bash
python -m frame_agent demo
```

产物落在 `runs/demo/`：

| 文件 | 内容 |
| --- | --- |
| `report.md` | 人读版报告（结论 / 评分明细 / 收敛轨迹 / 分镜表 / 成本 / 降级说明） |
| `scripts.json` | 全部脚本变体与分镜 |
| `scores.json` | 每个变体的逐维度评分与逐条加分明细 |
| `critique.json` | 评审结论与重写指令 |
| `prompt_set.json` | 提示词单变量变体 |
| `media_insight.json` | 素材理解结果 |
| `cost.json` | 调用/token/延迟画像 |
| `run_state.json` | 检查点（支持续跑） |
| `trace.jsonl` | 追加式全过程轨迹 |

### 4.3 跑测试（89 个用例）

```bash
python -m unittest discover -s tests -t .
```

```text
Ran 89 tests in 3.3s

OK
```

覆盖内容包括：Schema 校验器的边界（含 `bool` 是 `int` 子类这个经典坑）、
JSON 提取的四层策略、修复闭环、降级链与预算熔断、工具容错、
**反思循环是否真的收敛**、**两次运行结果是否逐字节一致**、断点续跑的指纹校验。

### 4.4 批量回归

```bash
python -m frame_agent eval --out runs/suite
```

```text
通过率 100%（2/2）｜均分 95.7｜平均轮数 2.0｜总 token 41743
维度均分：audience_fit=85.0，compliance=100.0，hook_strength=100.0，
         producibility=100.0，structure=93.5
```

### 4.5 接真实模型

任何实现了 `POST {base_url}/chat/completions` 的服务都可以：

```bash
export FRAME_AGENT_BASE_URL=https://api.deepseek.com/v1
export FRAME_AGENT_API_KEY=sk-xxxx
export FRAME_AGENT_MODEL=deepseek-chat

python -m frame_agent run \
  --brief examples/briefs/skincare_launch.json \
  --out runs/real \
  --backend openai,mock      # 真实模型优先，失败自动降级
```

**换成真实模型时，输入上下文与输出 schema 完全不变**，于是
"离线基线 vs 在线效果"是一个受控实验，可以直接量化模型增益。

### 4.6 排查

```bash
python -m frame_agent inspect --state runs/demo --trace
```

输出每一步的状态、耗时、模型调用数、产物清单、降级项，以及最近的轨迹记录。

---

## 5. 关键实现细节

### 5.1 为什么 Agent 是"两阶段"而不是一个大 ReAct 循环

结构化产出任务上，把工具调用和最终答案混在一个循环里有两个硬伤：
工具返回的原始 JSON（可能上百条字幕）会持续稀释模型对任务目标的注意力；
而且容易在"半途"返回缺字段的对象，让校验失败与工具循环纠缠在一起。

所以基类切成两段：**证据采集**（一轮，只决定调哪些工具）→
**上下文收敛 + 结构化产出**（把观测归一成干净上下文，重新发起一次请求）。
副产品是离线后端与真实模型走**同一条控制流**，离线跑出来的调用次数
与工具序列因此是真实可信的。

### 5.2 参数为什么用闭包绑定，而不是让模型传

```python
def build_registry(context):            # 每个 run 构造一次
    def parse_transcript(path=None):    # 路径由闭包提供
        ...
```

模型的参数 schema 越简单，选错的概率越低。让模型传文件路径 =
给它一个编造路径的机会。同时 `context` 是**可变的**：
`script_metrics` 工具在后半程才真正可用，因为那时 `context["scripts"]` 才被填充。

### 5.3 工具错误是观测，不是异常

`ToolRegistry.invoke` 永远返回 `ToolResult`，从不向上抛。三类失败
（工具不存在 / 参数不匹配 / 执行期异常）都被收敛成可读错误 + 建议。
**工具一报错就炸掉整条链，Agent 就退化成了一个普通函数调用序列。**

### 5.4 检查点是原子写入的

```python
temp.write_text(payload)
temp.replace(self.state_path)   # 原子替换
```

直接覆盖写时如果进程被杀，留下的破损 `run_state.json`
会让"续跑"这个功能本身失效。续跑还要求 `brief` 指纹一致 ——
只看 run_id 不够，改完 brief 复用旧状态会产出与 brief 不符的报告，这种错最难查。

### 5.5 评测器的权重和为 1.0 是断言出来的

```python
def _check_weights():
    if abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) > 1e-9:
        raise AssertionError(...)
```

有人改了权重忘记同步是极隐蔽的错误 —— 分数会系统性偏移，
但所有单点测试都还是绿的。

---

## 6. 已验证的性质

以下是**可复现的**断言，不是声称：

| 性质 | 证据 |
| --- | --- |
| 反思循环真的收敛 | 首轮 68.55（`revise`）→ 次轮 97.0（`pass`），见 `docs/example-run/report.md` |
| 分数上升可归因 | `hook_strength` 23 → 100，钩子从 42 字缩到 11 字并补入数字与疑问句 |
| 运行确定性 | 同一 brief 两次运行的 `overall` / 钩子 / 各变体分数完全一致 |
| 批量不退化 | 两个不同类目（美妆 / SaaS）brief 均 2/2 通过，无降级项 |
| 结构可保证 | 89 个单测覆盖修复闭环、四层 JSON 提取、`bool` vs `int` 等边界 |
| 降级如实标注 | 无 ffmpeg 环境下关键帧 `source=manifest`，合成占位会标记不可采信 |

---

## 7. 已知边界（诚实说明）

- **离线后端的输出是模板化的，不代表真实创作水平。**
  它的作用是"让流水线可复现地跑通 + 让 CI 覆盖控制流"。
  所有报告都会标注 `backend=mock`，真实质量结论必须接真实模型后重跑同一套评测。
- **评测器只覆盖可量化的部分。** 钩子的"字面强度"（数字、长度、悬念标记）能算，
  但"这句话是否真的有洞察"不能算 —— 那部分交给 Critic，并最终仍需人工终审。
- **合规词表是内置的静态表。** 生产环境应替换为平台/法务维护的规则源。
- **未实现并发调度。** 当前是单 brief 串行；批量场景由 `eval` 顺序执行。

---

## 8. License

MIT
