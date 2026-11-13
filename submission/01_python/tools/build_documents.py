"""Build Chinese documentation and plots from the verified experiment bundle."""
import argparse
import csv
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Polygon
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
from reportlab.lib.pagesizes import A4

BLUE="#173d57"; TEAL="#087c79"; GRAY="#526577"; AMBER="#fff0cf"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--font",default=os.environ.get("CJK_FONT",r"C:\Windows\Fonts\msyh.ttc"))
    args=parser.parse_args()
    root=Path(args.root); docs=root/"02_documents"; exp=root/"03_experiments"
    docs.mkdir(exist_ok=True)
    if not Path(args.font).exists():
        parser.error("Supply --font with a Chinese TrueType font path")
    pdfmetrics.registerFont(TTFont("CJK",args.font,subfontIndex=0))
    font=FontProperties(fname=args.font)
    styles={
        "body":ParagraphStyle("body",fontName="CJK",fontSize=10.2,leading=17,textColor=colors.HexColor(BLUE),spaceAfter=9,wordWrap="CJK"),
        "title":ParagraphStyle("title",fontName="CJK",fontSize=26,leading=38,textColor=colors.HexColor(BLUE),spaceAfter=20,wordWrap="CJK"),
        "h1":ParagraphStyle("h1",fontName="CJK",fontSize=19,leading=27,textColor=colors.HexColor(BLUE),spaceAfter=17,wordWrap="CJK"),
        "h2":ParagraphStyle("h2",fontName="CJK",fontSize=12.5,leading=20,textColor=colors.HexColor(TEAL),spaceBefore=9,spaceAfter=7,wordWrap="CJK"),
        "small":ParagraphStyle("small",fontName="CJK",fontSize=8.3,leading=13,textColor=colors.HexColor(GRAY),spaceAfter=7,wordWrap="CJK"),
        "cell":ParagraphStyle("cell",fontName="CJK",fontSize=8.4,leading=13,textColor=colors.HexColor(BLUE),wordWrap="CJK"),
        "code":ParagraphStyle("code",fontName="Courier",fontSize=8,leading=12,textColor=colors.HexColor(BLUE),spaceAfter=9),
    }
    def p(text,style="body"):
        return Paragraph(escape(text).replace("\n","<br/>"),styles[style])
    def table(rows,widths):
        cells=[[p(str(c),"cell") for c in row] for row in rows]
        t=Table(cells,colWidths=widths,hAlign="LEFT",repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dfecef")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.whitesmoke,colors.white]),
            ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),8),
            ("RIGHTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),8),
            ("BOTTOMPADDING",(0,0),(-1,-1),8),("LINEBELOW",(0,0),(-1,0),.6,colors.HexColor(TEAL))]))
        return t
    def banner(text):
        t=Table([[p(text,"body")]],colWidths=[493])
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor(AMBER)),("BOX",(0,0),(-1,-1),.5,colors.HexColor("#d9b765")),("LEFTPADDING",(0,0),(-1,-1),13),("RIGHTPADDING",(0,0),(-1,-1),13),("TOPPADDING",(0,0),(-1,-1),12),("BOTTOMPADDING",(0,0),(-1,-1),8)]))
        return t
    def footer(canvas,doc):
        canvas.setFont("CJK",8)
        canvas.setFillColor(colors.HexColor(GRAY))
        canvas.drawString(50,812,"LLM-SAEA  /  自动算法生成与代理辅助优化")
        canvas.setStrokeColor(colors.HexColor("#d7e1e7"));canvas.line(50,801,545,801)
        canvas.drawString(50,28,"2026-10-23  |  代码、文档与可复现实验")
        canvas.drawRightString(545,28,str(doc.page))
    def build(path,story):
        SimpleDocTemplate(str(path),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=61,bottomMargin=52,
                          title=path.stem,author="LLM-SAEA project").build(story,onFirstPage=footer,onLaterPages=footer)

    # Editable flowchart source plus native vector/PDF and raster outputs.
    mermaid='''flowchart TD
    A[输入目标函数、变量边界和评价预算] --> B[LHS 初始采样；真实评价；建立档案]
    B --> C{真实评价预算耗尽?}
    C -- 是 --> Z[输出最优实测解、日志与生成算法]
    C -- 否 --> D{到达配置更新时间?}
    D -- 是 --> E[状态摘要：预算、停滞、改进、验证误差]
    E --> F[LLM 生成结构化算法配置]
    F --> G{Schema 与数值范围合法?}
    G -- 否 --> X[明确报错并停止；保留调用日志]
    G -- 是 --> H[应用已验证算法配置]
    H --> I[留出验证选择并拟合代理模型]
    D -- 否 --> I
    I --> J[在代理采集函数上执行 DE 搜索]
    J --> K[选择未评价候选；真实函数评价一次]
    K --> L[更新数据档案与最优实测值]
    L --> C
'''
    (docs/"algorithm_flow.mmd").write_text(mermaid,encoding="utf-8")
    fig,ax=plt.subplots(figsize=(9,10.3))
    fig.patch.set_facecolor("white");ax.set(xlim=(0,10.8),ylim=(0,10.7));ax.axis("off")
    def box(x,y,w,h,text,color="#e8f3f3"):
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.06,rounding_size=.10",facecolor=color,edgecolor=TEAL,lw=1.2))
        ax.text(x,y,text,ha="center",va="center",fontproperties=font,fontsize=10,color=BLUE)
    def diamond(x,y,text):
        ax.add_patch(Polygon([(x,y+.48),(x+1.4,y),(x,y-.48),(x-1.4,y)],facecolor="#fff2d7",edgecolor="#b88824",lw=1.2))
        ax.text(x,y,text,ha="center",va="center",fontproperties=font,fontsize=9,color=BLUE)
    def arrow(x1,y1,x2,y2,label=None):
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops={"arrowstyle":"->","color":GRAY,"lw":1.2})
        if label:ax.text((x1+x2)/2+.12,(y1+y2)/2+.08,label,fontproperties=font,fontsize=9,color=GRAY)
    ax.text(5.4,10.35,"LLM 驱动的代理辅助进化算法",ha="center",fontproperties=font,fontsize=17,color=BLUE)
    box(4.5,9.5,4.0,.65,"输入：黑盒函数 / 边界 / 评价预算")
    box(4.5,8.45,4.0,.65,"LHS 初始采样 → 真实评价 → 数据档案")
    diamond(4.5,7.35,"预算耗尽？")
    box(8.9,7.35,2.8,.85,"输出最优实测解\n保存数据与生成算法")
    diamond(4.5,6.05,"到达配置更新时间？")
    box(8.9,6.05,2.8,.85,"提取状态摘要\nLLM 生成配置")
    box(8.9,4.7,2.8,.85,"校验 Schema 与范围\n应用已验证算法配置")
    box(4.5,4.7,4.0,.7,"留出验证选择代理；拟合已评价数据")
    box(4.5,3.5,4.0,.7,"DE 搜索：预测目标 - 探索权重 × 距离")
    box(8.9,3.45,2.8,.7,"校验 / API 失败\n明确停止并留日志",color="#fbe8e8")
    box(4.5,2.3,4.0,.7,"筛选未评价候选；真实函数评价 1 次")
    box(4.5,1.1,4.0,.7,"更新档案、最优值与预算计数")
    arrow(4.5,9.16,4.5,8.8);arrow(4.5,8.1,4.5,7.85)
    arrow(5.9,7.35,7.42,7.35,"是");arrow(4.5,6.86,4.5,6.54,"否")
    arrow(5.9,6.05,7.42,6.05,"是");arrow(4.5,5.56,4.5,5.08,"否")
    arrow(8.9,5.60,8.9,5.16);arrow(7.43,4.7,6.56,4.7,"合法")
    arrow(8.9,4.25,8.9,3.84,"失败")
    for top,bottom in [(4.32,3.88),(3.12,2.68),(1.92,1.48)]:arrow(4.5,top,4.5,bottom)
    ax.plot([2.43,1.2,1.2],[1.1,1.1,7.35],color=GRAY,lw=1.2);arrow(1.2,7.35,3.08,7.35)
    ax.text(.65,4.3,"循\n环",ha="center",fontproperties=font,fontsize=10,color=GRAY)
    ax.text(5.4,.3,"离线对照在设计器处显式选用固定配置或规则策略；不是 API 失败后的自动替代。",ha="center",fontproperties=font,fontsize=9,color=GRAY)
    fig.tight_layout()
    fig.savefig(docs/"algorithm_flow.png",dpi=180)
    fig.savefig(docs/"algorithm_flow.pdf")
    fig.savefig(docs/"algorithm_flow.svg")
    plt.close(fig)

    sections=[
      ("1  问题定义与交付边界",[
        ("目标", "在有限真实函数评价预算 B 下，最小化确定性黑盒目标 f(x)，其中每个连续变量都有有限上下界。只要求提供 Python callable，不依赖解析梯度。真实昂贵性可以来自仿真、实验或模型训练；本次验证采用解析函数和严格评价次数限制。"),
        ("数学形式", "min f(x),  subject to l_j <= x_j <= u_j, j=1,...,d.\n归一化变量 z_j=(x_j-l_j)/(u_j-l_j)，在 [0,1]^d 上搜索。历史档案 D_t={(z_i,y_i)}，所有 y_i 都来自真实 objective 调用。输出 argmin_i y_i，不把代理预测值当成实测结果。"),
        ("“任意”的操作性解释", "任意满足接口约定的有限边界连续单目标问题可以接入；并不意味着任意维度、任意结构或任意预算均有效。一般约束、多目标、离散变量、随机噪声、非平稳目标需要独立适配与验证。当前未实现这些问题的通用求解器。"),
        ("算法生成的边界", "LLM 在预定义算法语法内选择代理类型、DE 算子和控制参数，系统校验后通过可信模板导出 Python 程序。结构变化包含模型族与搜索算子，数值变化包含探索和局部搜索强度。它不是无限制生成并执行任意 Python 代码，也没有复现 EoH 的完整算法种群演化。"),
      ]),
      ("2  代理模型与候选筛选",[
        ("初始设计与数据来源", "所有方法先使用同种子、同点数的拉丁超立方采样 LHS。默认 24 个点，预算不足 24 时只评价预算允许的点。代理只使用已真实评价的历史数据，每次选中候选才调用一次真实函数。"),
        ("代理组合与验证", "模型候选为高斯核 RBF 回归和含交叉项的二次岭回归。RBF 使用训练集成对距离中位数的 0.5、1、2 倍作为长度尺度候选，正则项 1e-6；二次模型使用增广最小二乘实现 1e-6 岭正则。固定索引 i mod 5=0 为验证集，其余为训练集，选择验证 NRMSE 最小的模型，再用全档案重拟合。"),
        ("采集函数", "a(z) = (m(z)-mean(y))/max(std(y),1e-12) - beta * min_i ||z-z_i|| / sqrt(d).\n第一项偏向预测目标更低的点，第二项奖励远离已评价数据的覆盖范围。距离是探索启发式，不是经过校准的预测标准差，图中也不把它解释成置信区间。"),
        ("模型管理", "默认每新增 3 个真实评价点重新拟合一次；配置改变时立即重拟合。中间使用缓存模型，减少训练开销。验证集来自适应性采样档案，不能视为独立测试集；它只服务于在线模型选择。"),
        ("代理内进化", "候选种群由均匀点与当前优秀解周围的高斯扰动组成。DE 在代理采集函数上迭代，候选搜索不消耗真实评价预算。边界用反射修复，交叉至少更新一个维度。最后剔除与真实档案距离不超过 1e-7 的候选，并选择采集值最小者真实评价。"),
      ]),
      ("3  LLM 设计器与程序生成",[
        ("输入与职责", "输入为维度、预算使用量、停滞评价次数、近期相对改进、上一轮验证误差及当前配置；可附加用户问题描述。基准函数名称、已知最优点、目标公式和密钥不会传给模型。LLM 负责低频算法设计，不负责伪造目标函数评价。"),
        ("结构化输出", "支持 OpenAI Responses 严格 JSON Schema 输出，以及 DeepSeek Chat Completions 的 JSON Output。DeepSeek 的 JSON 模式不保证 Schema 合法性，两条路径都执行本地 Plan 校验：字段必须齐全，拒绝多余字段、未知算子、非有限数值和越界参数。reason 保存简短决策说明。"),
        ("控制频率与开销", "动态模式默认在评价次数 24、44、64、84 时设计，最多 4 次请求。OpenAI 输出上限 1600 token，DeepSeek 为 768 token 且关闭 thinking；均超时 60 秒、不隐式重试。记录实际模型名、response_id、token、耗时和结果。本次完成 60 次动态在线运行、240 次模型请求；全部原始调用、配置事件与真实评价轨迹均随包提交。"),
        ("生成与执行", "export_program 将已验证配置以 Python 字面量写入可信模板，生成 solve(problem, budget, seed) 入口。程序从 llm_saea 导入优化内核。模型输出不会传给 eval/exec。动态运行保存完整配置事件；最终导出的冻结配置算法只能复现该配置的行为，不能替代动态事件重放。"),
        ("失败与复现", "API 失败、拒绝、响应不完整、JSON 无效或参数越界会明确终止，并保存不含凭据的诊断日志。系统不会悄悄退回离线规则。replay 使用保存的配置事件和原始问题/种子/预算复现本地随机过程，不再次请求 API；重放应单独标注来源。"),
      ]),
      ("4  接口、配置与复杂度",[
        ("调用方式", "安装方法、PowerShell 命令、单次算法生成、动态运行、策略重放及自定义昂贵仿真示例均见 01_python/README.md。普通 saea / de / random / adaptive 运行不需要密钥。在线运行必须显式提供实际可用模型，并安全配置本地密钥。"),
        ("配置默认值", "surrogate=auto；operator=current_to_best；mutation=0.65；crossover=0.85；exploration=0.2；local_fraction=0.3；radius=0.12；population=40；generations=6。默认初始样本数 24，真实评价预算 100，重拟合间隔 3。规则自适应方法在连续 12 次无改进时增加探索。"),
        ("成本拆分", "总墙钟时间 = 真实函数时间 + 代理训练/搜索时间 + LLM 调用时间 + 程序管理时间。RBF 稠密求解约 O(n^3)，预测约 O(Mnd)；二次特征数 p=1+d+d(d+1)/2，最小二乘代价随 n 和 p 增长。每轮搜索约 population*(generations+1) 个候选。当前无增量矩阵更新或档案压缩，不适合直接推断超高维性能。"),
        ("扩展建议", "昂贵确定性仿真可直接替换目标函数。一般约束应明确定义可行性处理与惩罚尺度；整数变量需去重与缓存；噪声函数需重复测量和噪声建模；多目标任务需非支配排序或明确标量化。上述扩展均需独立预算协议，当前交付不声称已验证。"),
      ]),
      ("5  验证证据与参考文献",[
        ("已执行验证", "13 项单元测试覆盖预算、边界、复现、重放、代理、导出、日志和两个服务接口契约。主实验新增 60 次 DeepSeek 在线优化、240 次成功请求，连同 240 次原始离线对照共 300 次运行；另有 60 次冻结消融及 60 次精确事件重放。OpenAI 服务仍未实测。"),
        ("实验结论边界", "在线 LLM 已按同预算与初始样本完成五方法对照；结论以实验报告中的真实数据为准。每个实例只有 5 个随机种子，统计能力有限，不能把局部胜出外推为普遍有效。六个解析函数不是实际工业昂贵仿真，当前未验证一般约束、噪声、多目标或离散问题。"),
        ("参考文献 [1]", "Jin, Y. (2011). Surrogate-assisted evolutionary computation: Recent advances and future challenges. Swarm and Evolutionary Computation, 1(2), 61-70. DOI: 10.1016/j.swevo.2011.05.001. 本项目借鉴历史评价数据训练代理、用真实函数校正候选的基本范式。"),
        ("参考文献 [2]", "Liu, F., et al. (2024). Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model. ICML 2024; arXiv:2401.02051. https://arxiv.org/abs/2401.02051 。该工作研究 LLM 与演化式启发式设计；本项目采用受限配置生成，范围与原论文不同。"),
        ("参考文献 [3]", "OpenAI. Structured model outputs. https://developers.openai.com/api/docs/guides/structured-outputs 。访问日期：2026-10-23。接口按 Responses API 严格结构化输出设计；模型可用性以账号实际权限为准。"),
        ("参考文献 [4]", "DeepSeek API Docs. Your First API Call; JSON Output. https://api-docs.deepseek.com/quick_start/agent_integrations/ ; https://api-docs.deepseek.com/guides/json_mode/ 。2026-10-23 文档将 deepseek-flash 映射到 V4.1-Flash-1017；服务别名可能更新。"),
      ]),
    ]
    story=[Spacer(1,64),p("基于 LLM 的自动代理模型\n辅助的进化算法生成","title"),p("算法说明文档  /  LLM-SAEA","h1"),p("面向有限边界连续黑盒最小化问题","h2"),Spacer(1,24),
           p("交付：Python 代码、流程图与说明、五方法在线对比及冻结配置消融。\n版本：0.1.1    日期：2026-10-23"),Spacer(1,25),
           banner("最终状态：60 次 DeepSeek 在线优化全部完成，240 次模型请求成功；连同原始对照共 300 次主实验。另附 60 次冻结配置消融、完整日志与重放审计。统计结果和适用边界见实验报告。"),Spacer(1,24),
           p("阅读顺序：问题定义 → 算法流程 → 代理与搜索 → LLM 设计 → 接口与复现 → 实验报告。","small"),PageBreak()]
    markdown=["# 基于 LLM 的自动代理模型辅助的进化算法生成\n\n日期：2026-10-23；版本 0.1.1\n\n**完整在线对照已完成：300 次主实验（60 次在线）、240 次新增模型请求；含消融与复现审计。性能结论以实验报告为准。**\n"]
    for index,(title,items) in enumerate(sections):
        if index==1:
            story += [p("算法流程图","h1"),Image(str(docs/"algorithm_flow.png"),width=480,height=549),p("流程源文件 algorithm_flow.mmd 可编辑；PDF / SVG 为矢量版本。默认每 20 次真实评价更新设计器，每新增 3 个样本重拟合代理。","small"),PageBreak()]
        story.append(p(title,"h1"));markdown.append("## "+title)
        for heading,body in items:
            story += [p(heading,"h2"),p(body)]
            markdown.append("### "+heading+"\n\n"+body+"\n")
        if index<len(sections)-1:story.append(PageBreak())
    build(docs/"算法说明文档.pdf",story)
    (docs/"算法说明文档.md").write_text("\n\n".join(markdown),encoding="utf-8")

    from final_report import create_report
    create_report(root, p, table, banner, build)
    print("Created three PDFs with completed online evidence.")


if __name__=="__main__":main()
