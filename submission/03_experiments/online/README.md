# 完整在线实验：2026-10-23

本目录是最终五方法主分析。300 次运行包含 240 次原有离线对照（复制且哈希未变）和新增 60 次 DeepSeek 动态优化。100 次评价/运行，共 30000 次；原始 offline 目录中的重复对照不应再次计数。

新增请求 240 次，全部成功，总用量 239,318 token。此前最小测试 866 token 另列在 ../deepseek_smoke，不在该统计中。

在线 LLM 在 2/12 个实例上的最终中位数优于固定 SAEA，在 3/12 个实例上优于规则 SAEA。五方法平均名次为 random=4.50；de=4.50；saea=1.83；adaptive=1.58；llm=2.58。在线 LLM 对两个 SAEA 对照的 Holm 校正显著项分别为 0 和 0。这是一组小样本基准证据，不支持声称 LLM 对任意问题普遍更优。

## 文件

- manifest.json：执行前协议、来源文件哈希、代码哈希和运行时间。
- runs：五方法真实评价记录；llm 子目录含 4 次请求日志、完整配置事件、逐次 journal。
- per_run.csv / summary.csv：每次结果与汇总。
- paired_tests.csv：3 组比较 × 12 实例，共 36 个双侧 Wilcoxon 检验，统一 Holm 校正。
- figures：五方法收敛中位数/IQR、平均名次。
- api_calls.csv / online_audit.json：请求、用量、耗时及校验记录。
- frozen_initial：60 次冻结首次模型配置的本地消融；每个运行有真实历史和生成的可执行 Python 算法。
- frozen_per_run.csv / frozen_comparison.csv：动态与冻结比较，12 项单独 Holm 校正。
- replay_audit.csv：60 次完整坐标/目标精确重放检查；不是新独立在线样本。

冻结消融为补充探索性分析，在主在线实验期间加入；未用于修改模型提示、算法参数或筛选运行。冻结组共享每次运行的第一个 LLM 配置和种子，前 44 个评价点与动态组完全相同；之后差异包括配置改变以及立即重拟合时点的改变。

真实在线模型输出可能变化，种子只控制本地优化。离线 replay 可以重现已保存配置对应的轨迹。run_online.py 会消费模型额度，analyze.py 和 audit_online.py 不发送模型请求。

用量估价仅作成本记录，不是账号账单；官方定价来源及估价假设见 online_audit.json。主分析之外的审计重算/重放不混入独立样本数。
