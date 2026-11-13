"""Final report derived only from the completed, audited online benchmark."""
import csv
import json
from reportlab.platypus import Spacer, PageBreak, Image


def create_report(root, p, table, banner, build):
    exp = root / "03_experiments"
    online = exp / "online"
    analysis = json.loads((online / "analysis.json").read_text(encoding="utf-8"))
    audit = json.loads((online / "online_audit.json").read_text(encoding="utf-8"))
    def csvrows(name):
        with (online / name).open(encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    summary, paired, frozen, per_run = [csvrows(n) for n in ["summary.csv", "paired_tests.csv", "frozen_comparison.csv", "per_run.csv"]]
    methods = ["random", "de", "saea", "adaptive", "llm"]
    names = ["sphere", "ellipsoid", "rosenbrock", "rastrigin", "ackley", "griewank"]
    ranks = analysis["average_ranks"]
    rank_text = "；".join(f"{m}={ranks[m]:.2f}" for m in methods)
    stats = {}
    for comparison in ["adaptive vs saea", "llm vs saea", "llm vs adaptive"]:
        rows = [r for r in paired if r["comparison"] == comparison]
        stats[comparison] = {k:sum(int(r[k]) for r in rows) for k in ["wins", "ties", "losses"]}
        stats[comparison]["significant"] = sum(float(r["p_holm"]) < .05 for r in rows)
    median_wins = {}
    for control in ["saea", "adaptive"]:
        median_wins[control] = sum(
            float(next(r["median"] for r in summary if r["function"]==name and int(r["dimension"])==dim and r["method"]=="llm")) <
            float(next(r["median"] for r in summary if r["function"]==name and int(r["dimension"])==dim and r["method"]==control))
            for name in names for dim in [5,10])
    conclusion = (f"在线 LLM 在 {median_wins['saea']}/12 个实例上的最终中位数优于固定 SAEA，在 {median_wins['adaptive']}/12 个实例上优于规则 SAEA。"
                  f"五方法平均名次为 {rank_text}。在线 LLM 对两个 SAEA 对照的 Holm 校正显著项分别为 "
                  f"{stats['llm vs saea']['significant']} 和 {stats['llm vs adaptive']['significant']}。这是一组小样本基准证据，不支持声称 LLM 对任意问题普遍更优。")
    protocol = ("6 个函数 × 2 个维度（5D、10D）× 5 个种子（0-4）× 5 个方法，共 300 次独立优化运行、30000 次真实目标评价。"
                "其中 240 次既有离线对照逐字节保留，新增 60 次在线 DeepSeek 动态优化。每次预算 100，初始 LHS 为 24 点；"
                "同一函数/维度/种子下，各方法共享初始样本。没有依据在线结果修改参数或删除不利运行。")
    md = ["# 实验结果与可复现性报告\n\n2026-10-23；最终提交版\n", "## 实验协议\n\n"+protocol, "## 核心结果\n\n"+conclusion]
    story = [p("实验结果与可复现性报告", "title"), p("DeepSeek 在线对比完成 / 最终提交版 / 2026-10-23", "h2"),
             banner(conclusion), Spacer(1,12), p("1  实验协议", "h1"), p(protocol),
             table([["项目", "设置"], ["方法", "Random、DE、固定 SAEA、规则 SAEA、在线 LLM-SAEA"],
                    ["在线控制", "deepseek-flash；评价次数 24、44、64、84 后更新，最多 4 次/运行"],
                    ["输出与错误", "JSON 模式 + 本地字段/范围校验；768 输出 token；thinking 关闭；重试 0"],
                    ["统计指标", "最终实测最优目标均值/样本标准差/中位数；收敛轨迹；名次；配对比较"],
                    ["原始证据", "online/runs：history.csv、result.json、LLM 日志与逐次评价 journal"],
                    ["补充验证", "60 次冻结初始配置消融；60 次动态事件精确重放，均无新增 API 调用"]], [92,401]),
             Spacer(1,10), p("基准是廉价解析函数，用于检查有限评价预算下的样本效率；没有人为 sleep，不等同于真实工业昂贵仿真。OpenAI 路径仍未通过真实服务测试。", "small"), PageBreak()]
    for dim in [5,10]:
        story += [p(f"2  最终目标结果：{dim} 维", "h1"), p("均值 ± 样本标准差，5 个种子；越小越好。全部是实际 objective 调用结果，未使用代理预测值替代。")]
        rows = [["函数", "Random", "DE", "固定 SAEA", "规则 SAEA", "LLM SAEA"]]
        mrows = ["| 函数 | Random | DE | 固定 SAEA | 规则 SAEA | LLM SAEA |", "|---|---:|---:|---:|---:|---:|"]
        for name in names:
            cells = []
            for method in methods:
                r = next(r for r in summary if r["function"]==name and int(r["dimension"])==dim and r["method"]==method)
                cells.append(f"{float(r['mean']):.3e}\n± {float(r['std']):.2e}")
            rows.append([name,*cells]); mrows.append("| "+" | ".join([name,*[c.replace("\n"," ") for c in cells]])+" |")
        story += [table(rows,[73,84,84,84,84,84]), Spacer(1,20), p("如何解释", "h2"),
                  p("各函数量纲不同，应在相同函数、维度和预算下比较。平均名次以每个实例的最终中位数排序；均值与中位数可能给出不同排序。全部种子均保留，完整中位数、最小值、最大值见 online/summary.csv。"),
                  p("六个函数的理论最优值均为 0。Rosenbrock 最优点是全 1，其余为全 0；定义及有限搜索边界均在 problem.py 中。LLM 只接收泛化的问题描述和运行状态，不接收函数名称、公式、坐标或已知最优值。"), PageBreak()]
        md.append(f"## {dim}D 最终目标\n\n"+"\n".join(mrows))
    for dim in [5,10]:
        story += [p(f"3  收敛轨迹：{dim} 维", "h1"),
                  p("实线为 5 个种子的中位数，阴影是第 25-75 百分位区间（IQR），不是置信区间。横轴为实际目标评价次数；虚线为 24 个初始样本结束。"),
                  Image(str(online/"figures"/f"convergence_{dim}d.png"),width=493,height=271), Spacer(1,16),
                  p("读图边界", "h2"), p("纵轴使用对数尺度；只在绘图时将不大于 1e-12 的值设为显示下限，原始 CSV 保留原值。候选在代理上的搜索不消耗真实评价预算，所有轨迹变化都来源于真实函数评价。"),
                  p("在线运行除了评价预算，还支付模型请求时间与 token 成本；图中评价效率不代表墙钟时间优势。原始曲线及可编辑 SVG 位于 online/figures。"), PageBreak()]
        md.append(f"## {dim}D 收敛\n\n![中位数与 IQR](online/figures/convergence_{dim}d.png)")
    srows = [["比较（左者胜/平/负）", "胜", "平", "负", "显著实例/12"]]
    for comparison, s in stats.items():
        srows.append([comparison, s["wins"], s["ties"], s["losses"], s["significant"]])
    story += [p("4  排名与配对统计", "h1"), p(conclusion),
              Image(str(online/"figures"/"average_rank.png"),width=430,height=231),
              table(srows,[213,55,55,55,115]), Spacer(1,12),
              p("胜/平/负汇总来自 12 个实例 × 5 对同种子运行的数值比较，不将它们视为 60 个同分布独立问题。每个实例做双侧 Wilcoxon 配对检验，主分析三个比较的全部 36 个 p 值共同作 Holm 校正。"),
              p("每项只有 5 对样本，检验能力很低；无零差时精确双侧 p 值的最小值为 2/32=0.0625。因此不能用不显著证明等效，也不能仅凭名次宣称统计显著优势。", "small"), PageBreak()]
    md.append("## 统计\n\n"+conclusion+"\n\n完整 36 项检验及统一 Holm 校正见 online/paired_tests.csv；n=5，仅作探索性分析。")
    f = audit["dynamic_vs_frozen"]
    frozen_text = (f"冻结每次在线运行的第一次有效配置，以相同种子重新优化 100 次评价，得到 60 次本地消融运行。"
                   f"动态对冻结的逐种子胜/平/负为 {f['wins']}/{f['ties']}/{f['losses']}；"
                   f"12 项配对检验在独立的 Holm 校正族中显著项为 {audit['frozen_holm_significant']}。"
                   "它检验后续配置更新的作用，不是新在线模型采样。")
    frows = [["函数 / 维度", "动态中位数", "冻结中位数", "胜/平/负", "Holm p"]]
    for r in frozen:
        frows.append([f"{r['function']} / {r['dimension']}D", f"{float(r['dynamic_median']):.3e}",
                      f"{float(r['frozen_median']):.3e}", f"{r['wins']}/{r['ties']}/{r['losses']}", f"{float(r['p_holm']):.3f}"])
    story += [p("5  冻结初始配置消融", "h1"), p(frozen_text), table(frows,[133,105,105,80,70]),
              Spacer(1,10), p("冻结组沿用首次 LLM 输出；动态组在 44/64/84 次评价处更新并立即重拟合。消融同时包含配置变化与更新时重拟合的影响，不能单独归因于某个参数。头 44 个评价点已逐项核对一致。补充实验未用于反向调参。", "small"), PageBreak()]
    md.append("## 冻结初始配置消融\n\n"+frozen_text+"\n\n数据见 online/frozen_comparison.csv、frozen_per_run.csv 和 frozen_initial/。")
    live_seconds = sum(float(r["elapsed_seconds"]) for r in per_run if r["method"] == "llm")
    cost_text = (f"新增在线实验成功请求 {audit['successful_requests']} 次、失败 {audit['failed_requests']} 次；"
                 f"输入 {audit['prompt_tokens']:,} token，输出 {audit['completion_tokens']:,} token，总计 {audit['total_tokens']:,} token。"
                 f"API 耗时合计 {audit['api_seconds_sum']:.2f} 秒；60 次在线优化的运行耗时合计 {live_seconds:.2f} 秒。")
    story += [p("6  在线开销与证据审计", "h1"), p(cost_text),
              table([["指标", "实测 / 说明"], ["API 延迟", f"中位数 {audit['api_seconds_median']:.3f}s；最大值 {audit['api_seconds_max']:.3f}s"],
                     ["不同配置数", f"{audit['unique_numeric_plans']} / 240（去掉 reason 后，对其余字段去重）"],
                     ["代理选择", str(audit['surrogate_choices'])], ["算子选择", str(audit['operator_choices'])],
                     ["保守费用估计", f"USD {audit['usd_peak_uncached_upper_estimate']:.4f}，使用全部输入未命中缓存的高峰价"],
                     ["主分析审计", "300 次运行 / 30000 次评价；边界、预算、累计最优、初始样本一致；按坐标重算目标一致"],
                     ["策略精确重放", "60/60 运行的全部坐标与目标逐位一致；无 API 请求"],
                     ["本地消融", "60 次运行 / 6000 次评价，另存且不混入五方法主分析"]], [112,381]),
              Spacer(1,12), p("费用是估算，不是账号账单。2026-10-23 官方价格页的高峰未缓存输入价为 $0.30/百万 token，输出为 $1.20/百万；缓存和非高峰折扣可能降低实际扣款。https://api-docs.deepseek.com/quick_start/pricing/", "small"),
              p("此前最小接入测试的 1 次请求 / 866 token 单独保留，未计入本页新增在线实验统计。原始 API 返回模型、response_id、用量、状态、耗时和有效配置均可追溯。", "small"), PageBreak()]
    md.append("## 调用与成本\n\n"+cost_text+f"\n\n按高峰未缓存价格保守估计 USD {audit['usd_peak_uncached_upper_estimate']:.4f}，不是账单。价格来源：https://api-docs.deepseek.com/quick_start/pricing/ 。审计明细见 online/online_audit.json。")
    limitations = [
        ("研究结论", conclusion),
        ("适用范围", "当前实现面向有限边界、连续变量、确定性单目标最小化。六个解析函数只验证框架与样本效率；尚无真实昂贵工业仿真，不能声称对任意昂贵优化问题都有增益。"),
        ("统计与时间", "只有 5 个种子，结论为探索性。离线对照来自早先同机运行，在线组于本次顺序执行；目标评价预算和初始样本相同，但运行时间不构成完全同期受控硬件对比。未评估多次模型采样或提示敏感性。"),
        ("模型可复现性", "deepseek-flash 是可更新别名。日志记录请求与返回名称，官方文档在执行日期映射到 V4.1-Flash-1017；不能证明服务端权重固定。后续重跑 API 不保证相同输出，精确复现依赖已保存事件的离线重放。"),
        ("提交与复现", "压缩包三部分齐全，含源码、wheel、流程图、说明、完整日志、汇总表和报告。SHA256SUMS.txt 校验归档内容。run_online.py 可在新目录复做在线实验（产生 API 费用）；analyze.py 与 audit_online.py 只做本地统计和验证。"),
        ("数据计数", "主分析是 300 个独立运行（含复用的 240 个对照）；另有 60 个冻结消融。60 次重放及按坐标重算目标是审计开销，不是新增独立实验样本。offline/ 与 online/ 下对照文件重复保存是为了来源审计，不得重复计数。"),
    ]
    story += [p("7  结论、局限与提交说明", "h1")]
    for head, body in limitations:
        story += [p(head,"h2"),p(body)]
        md.append("## "+head+"\n\n"+body)
    build(exp / "实验结果报告.pdf", story)
    (exp / "实验结果报告.md").write_text("\n\n".join(md),encoding="utf-8")
    (online / "report_findings.json").write_text(json.dumps(dict(conclusion=conclusion, median_wins=median_wins,
        comparisons=stats, live_run_seconds=live_seconds),ensure_ascii=False,indent=2),encoding="utf-8")
