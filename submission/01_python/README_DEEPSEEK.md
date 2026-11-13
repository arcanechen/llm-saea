# DeepSeek V4.1-Flash-1017 接入

官方接口：`https://api.deepseek.com`；当前 API 模型标识：`deepseek-flash`。
依据：https://api-docs.deepseek.com/quick_start/agent_integrations/ （2026-10-23 核对）。

项目通过官方兼容的 Chat Completions 接口调用，启用 JSON Output，随后在本地验证 Plan 字段与数值范围。JSON Output 本身不保证满足 JSON Schema，不能等同于 OpenAI 的严格结构化输出。

私有配置保存在工作区根目录 `.env.local`，未纳入提交包：

- `LLM_PROVIDER=deepseek`
- `DEEPSEEK_BASE_URL=https://api.deepseek.com`
- `DEEPSEEK_MODEL=deepseek-flash`
- `DEEPSEEK_API_KEY`：已授权保存的私有密钥，不在此处显示。

从工作区根目录执行（配置文件路径也可显式更改）：

```powershell
# 一次算法配置生成；最多一个请求，不自动重试。
.\.venv\Scripts\python.exe -m llm_saea --env-file .env.local generate --provider deepseek --dimension 2 --budget 25 --output my_runs/deepseek_generated.py

# 动态优化示例；会产生模型请求费用。
.\.venv\Scripts\python.exe -m llm_saea --env-file .env.local run --method llm --provider deepseek --problem sphere --dimension 5 --budget 100 --max-designs 4 --output my_runs/deepseek_dynamic
```

DeepSeek 请求关闭 thinking，输出上限 768 token、超时 60 秒、自动重试次数 0。错误时保留经清理的错误类型和 HTTP 状态，不输出密钥或完整异常正文。来源标签、实际返回模型名和 token 用量保存在 `llm_calls` 或 `*.llm_logs` 中。

当前 `deepseek-flash` 是可更新的服务别名，并非固定快照名称。当前映射依据官方文档；日志记录请求与实际返回的模型标识，不推测服务端权重版本。

最小接入验证已通过：1 次 API 请求，共 866 token；配置校验成功，导出的程序在 2 维 Sphere 上完成 25 次评价。原始证据见 `03_experiments/deepseek_smoke`。后续用户已授权完整在线实验：新增 60 次动态优化、240 次请求，全部结果见 `03_experiments/online`。此前 240 次离线对照保持不变，五方法主分析合计 300 次运行；另附 60 次冻结初始配置消融和 60 次无 API 精确重放审计。具体性能与用量以最终实验报告及 `online/online_audit.json` 为准。
