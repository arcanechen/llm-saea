# LLM-SAEA：可审计的算法生成与代理辅助进化优化

已新增 DeepSeek 官方接口适配，接入与在线实验说明见 `README_DEEPSEEK.md`。本文件中的 OpenAI 专属说明仅适用于 `--provider openai`；DeepSeek 使用 `--provider deepseek`。

Python >= 3.10；本次实际环境、依赖版本和运行参数见实验 manifest 与 requirements-lock.txt。

## 安装与本地运行

在本目录打开终端。Windows PowerShell 示例：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m llm_saea run --method saea --problem sphere --dimension 5 --budget 100 --seed 0 --output my_runs/sphere_saea
.\.venv\Scripts\python.exe examples/custom_problem.py
```

Linux/macOS 将 `.\.venv\Scripts\python.exe` 替换为 `.venv/bin/python`。固定依赖重现可使用 `requirements-lock.txt`；跨系统安装仍需相应版本的可用 wheel。

## 方法名称与结果含义

| CLI 方法 | 实现 | 是否调用在线 LLM |
|---|---|---|
| random | 相同初始 LHS 后均匀随机采样 | 否 |
| de | DE/rand/1/bin，F=0.7、CR=0.9 | 否 |
| saea | 固定配置，验证选择代理模型，代理上做 DE 搜索 | 否 |
| adaptive | 公开的停滞响应规则调整 SAEA 配置 | 否 |
| llm | OpenAI / DeepSeek 根据运行状态生成并更新配置 | 是，失败则停止 |
| replay | 按保存的事件序列重放策略 | 否，不能标成新在线实验 |

五个主比较方法使用相同随机种子下完全相同的初始 LHS。随机种子只控制本地算法随机性；在线模型输出不保证可重复，应保留日志并使用 replay。

## OpenAI 接口（已实现，尚未通过真实服务验证）

使用官方 SDK 的 `client.responses.create`，严格 JSON Schema 输出，以及额外数值范围验证。模型只选择白名单内的代理、算子及超参数；随后可信 Python 模板生成可执行算法。**这属于受限算法语法内的组合与配置生成，不是任意源代码的自由生成。**

先通过安全配置流程保存密钥。运行器只从 `OPENAI_API_KEY` 环境变量，或显式指定的本地私有 env 文件读取密钥；不在命令行参数、日志或提交包中保存密钥。以下命令需在安全配置完成后执行，示例模型需以账号实际可用模型为准：

```powershell
# 单次生成：最多 1 次模型请求，导出 Python 算法与来源日志。
python -m llm_saea --env-file .env.local generate --model gpt-6.1-sol --dimension 5 --budget 100 --output generated/openai_algorithm.py

# 动态生成：默认在真实评价次数 24、44、64、84 后更新，最多 4 次请求。
python -m llm_saea --env-file .env.local run --method llm --model gpt-6.1-sol --problem sphere --dimension 5 --budget 100 --seed 0 --output my_runs/sphere_llm

# 精确重放策略：问题、维度、种子、预算、初始点数须与原运行一致。
python -m llm_saea run --method replay --problem sphere --dimension 5 --budget 100 --seed 0 --replay my_runs/sphere_llm/result.json --output my_runs/sphere_replay
```

每次模型请求输出上限 1600 token、超时 60 秒、SDK 自动重试为 0；输入 token 与实际费用取决于模型和状态文本。`--max-designs` 限制请求次数，不是金额限额。拒绝、超时、无效 JSON 或超范围数值都会明确停止，不悄悄替换成离线策略。

结果目录包含 `history.csv`、`result.json`、最终冻结配置生成的 `generated_final_algorithm.py`；在线运行还包含 `llm_calls/call_*.json`。CLI 在每次真实评价后追加 `evaluation_journal.jsonl`，即使后续 API 失败也保留已完成样本；该日志不提供自动续跑。最终冻结算法与动态运行不同，动态轨迹应通过 `result.json` 的 events 重放。

## 自定义昂贵问题

```python
import numpy as np
from llm_saea import Problem, optimize

def expensive_objective(x):
    # 替换为真实仿真；返回有限的标量目标值。
    return float(np.sum((x - 0.25) ** 2))

problem = Problem('my_problem', np.zeros(4), np.ones(4), expensive_objective)
result = optimize(problem, budget=100, seed=0, method='saea')
print(result.best_x, result.best_y)
```

`Problem.description` 可描述问题，但会在在线模式传给模型。目标函数代码、名称、已知最优值、变量坐标和 API 密钥不进入模型提示。目标必须是确定性的；失败或非有限返回会中断运行，该次尝试仍可能消耗外部仿真资源。当前运行中断后不自动恢复已花费的外部预算。

一般不等式约束应自行定义尺度明确的惩罚目标或可行性变换，并分别验证约束满足情况；仅传入惩罚函数不会得到可行性保证。整数舍入会引入重复设计点，应自行建立缓存，本核心不直接支持离散变量。

## 复现实验

```powershell
python experiments/run_suite.py --output new_results/offline
python experiments/analyze.py new_results/offline
```

默认：6 函数 × 2 维度 × 5 种子 × 4 方法 = 240 次独立运行，每次 100 次真实评价，共 24000 次。未人为加入 sleep；廉价解析函数用于测试样本效率，不能据此声称真实昂贵仿真的总时间加速。输出目录必须为空，防止覆盖原始证据。

在线小样本验证（另建目录，不与离线结果混用）：

```powershell
python experiments/run_suite.py --output new_results/online_pilot --methods random de saea adaptive llm --functions sphere rosenbrock --dimensions 5 --seeds 0 1 2 --model gpt-6.1-sol --env-file .env.local
python experiments/analyze.py new_results/online_pilot
```

上述在线小样本最多 24 次模型请求；其结论仅能视为试验性观察。完整在线对比应使用相同 12 个函数-维度组合及 5 个种子，并报告实际请求数、token、模型版本、失败率和墙钟开销。最终提交包已使用 DeepSeek 完成同一完整协议；命令和结果见下文。

## 文件导航

- `src/llm_saea/problem.py`：统一问题接口、6 个解析基准与 LHS。
- `policy.py` / `llm.py`：配置语法、合法性校验、导出器与 OpenAI 接口。
- `surrogate.py`：RBF / 二次回归与留出验证选择。
- `optimizer.py`：代理内进化搜索和真实评价循环。
- `experiments`：运行、审计、汇总、统计与制图。
- `tools/build_documents.py`：从代码协议和已审核数据重新生成文档。

核心依赖 NumPy；分析依赖 SciPy / Matplotlib；文档依赖 ReportLab。在线接口的单元测试使用假客户端，只验证请求结构、错误处理和日志行为，不构成真实 API 连通性测试。

## 最终在线实验与无 API 审计

最终主分析在 `../03_experiments/online`：300 个运行，其中 240 个对照从原 offline 目录逐字节复用，新增 60 个在线运行，最多/实际 240 次请求。模型输出没有用于修改基准、筛选种子或回头调参。

以下从本目录执行；私有 env 文件路径替换成自己的实际路径。第一条会产生新的 API 费用，后两条仅本地分析与解析函数计算。

```powershell
python experiments/run_online.py --baseline ../03_experiments/offline --output new_results/online --env-file ../../.env.local
python experiments/analyze.py new_results/online
python experiments/audit_online.py new_results/online
```

`run_online.py` 每次运行落盘、失败立即停止，保留逐次评价与调用日志。`--resume` 仅跳过已完整保存的运行，不会重复部分运行中的付费调用；它不是任意位置的自动断点恢复。

`audit_online.py` 验证原始对照哈希、边界与目标值、所有调用用量/配置，并精确重放 60 次动态运行。首次执行还生成 60 次“冻结首个 LLM 配置”消融与可执行 Python 算法（`online/frozen_initial`）；这些是本地运行，不标成新在线采样。重放和目标重算是审计，不计入独立实验样本数。仅对随包解析基准运行此审计脚本。

图表/统计：`online/summary.csv`、`paired_tests.csv`、`frozen_comparison.csv`、`api_calls.csv`、`online_audit.json`。主比较 36 项检验统一 Holm 校正，冻结消融 12 项单独校正，n=5 的结论为探索性。

文档再生成：`python tools/build_documents.py`；最终打包：`python tools/package_submission.py --output ../../新的提交包.zip`。文档生成需要中文字体与完整在线分析文件。
