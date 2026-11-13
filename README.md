# LLM-SAEA

基于 LLM 的自动代理模型辅助的进化算法生成。面向有限边界、连续变量、确定性单目标黑盒最小化，通过代理模型、差分进化搜索与 LLM 配置生成协作，在固定真实评价预算下搜索更优解。

## 项目内容

- [Python 代码与安装说明](submission/01_python/README.md)：源码、wheel、测试、自定义目标函数示例、实验与审计脚本。
- [算法说明](submission/02_documents/算法说明文档.pdf)与[流程图](submission/02_documents/algorithm_flow.pdf)。
- [最终实验报告](submission/03_experiments/实验结果报告.pdf)、[完整在线实验数据](submission/03_experiments/online)与原始调用日志。
- [完整提交压缩包](https://github.com/arcanechen/llm-saea/releases)：三部分材料与 SHA256 校验。

## 安装与运行

Python >= 3.10。在仓库根目录执行：

```bash
python -m pip install -r submission/01_python/requirements.txt
python -m pip install -e submission/01_python
python -m unittest discover -s submission/01_python/tests -v
python -m llm_saea run --method saea --problem sphere --dimension 5 --budget 100 --seed 0 --output my_runs/sphere
```

离线运行不需要 API 密钥。DeepSeek 在线配置见 [README_DEEPSEEK.md](submission/01_python/README_DEEPSEEK.md)，密钥须保存在自己的私有环境中。在线运行会产生 API 费用。仓库不包含密钥、私有 env 或虚拟环境。

## 实验结果

实验日期为 **2026-10-23**；本次 GitHub 发布日期为 **2026-11-13**。

- 6 个函数 × 2 个维度 × 5 个种子 × 5 个方法 = **300 次主实验**，每次 100 次真实目标评价。
- 其中新增 **60 次 DeepSeek 在线运行，240 次请求全部成功，共 239,318 tokens**。
- 另有 **60 次冻结首次配置的本地消融**；60 次动态事件精确重放通过，13 项测试通过。
- 平均名次（越低越好）：规则 SAEA 1.58、固定 SAEA 1.83、LLM-SAEA 2.58、Random 4.50、DE 4.50。

这些小样本解析基准未证明 LLM 普遍优于固定或规则策略，也不等同于真实工业昂贵仿真的验证。完整结果、统计检验和限制均保留在报告中。`offline` 与 `online` 目录中复用的 240 个对照是同一批运行，不应重复计数；重放属于审计，不是新的独立实验。

## 可复现与算法边界

模型输出限制在预定义配置语法中，经过字段和范围校验后，由可信模板导出 Python 算法；不会执行模型提供的任意代码。`replay` 可从保存事件复现本地搜索轨迹，无需重新请求 API。

DeepSeek 的 `deepseek-flash` 是可更新模型别名，日志记录执行时的实际返回信息。仓库保存的是上述日期的实验，不保证未来 API 输出一致。OpenAI 接口代码保留，真实在线实验使用 DeepSeek。

归档内文件校验见 [SHA256SUMS.txt](submission/SHA256SUMS.txt)。完整使用方法和适用范围见[提交说明](submission/README_请先阅读.md)。
