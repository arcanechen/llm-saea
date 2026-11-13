"""Audit raw histories and derive tables and figures without cherry-picking."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS={"random":"#8793a3","de":"#d58b22","saea":"#2574ac","adaptive":"#168574","llm":"#8f4ba5"}
LABELS={"random":"Random search","de":"DE/rand/1/bin","saea":"Fixed SAEA","adaptive":"Rule-adaptive SAEA","llm":"LLM SAEA"}


def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory")
    args=parser.parse_args()
    root=Path(args.directory)
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    parameters=manifest["parameters"]
    groups=defaultdict(list); individual=[]
    for file in sorted((root/"runs").rglob("result.json")):
        row=json.loads(file.read_text(encoding="utf-8"))
        data=np.genfromtxt(file.with_name("history.csv"),delimiter=",",skip_header=1)
        assert len(data)==parameters["budget"]==row["evaluations"]
        assert np.all(np.isfinite(data))
        np.testing.assert_allclose(data[:,0],np.arange(1,len(data)+1))
        np.testing.assert_allclose(data[:,2],np.minimum.accumulate(data[:,1]),atol=1e-12)
        assert np.isclose(row["best_y"],min(data[:,1]))
        groups[(row["problem"],row["metadata"]["dimension"],row["method"])].append((row,data[:,2],data[:,3:]))
        individual.append({"function":row["problem"],"dimension":row["metadata"]["dimension"],"method":row["method"],"seed":row["seed"],"best_y":row["best_y"],"elapsed_seconds":row["elapsed_seconds"],"objective_seconds":row["objective_seconds"]})
    expected=len(parameters["functions"])*len(parameters["dimensions"])*len(parameters["seeds"])*len(parameters["methods"])
    assert len(individual)==expected==manifest["completed_runs"],"Incomplete suite"
    for name in parameters["functions"]:
        for dim in parameters["dimensions"]:
            reference=sorted(groups[(name,dim,parameters["methods"][0])],key=lambda r:r[0]["seed"])
            for method in parameters["methods"]:
                runs=sorted(groups[(name,dim,method)],key=lambda r:r[0]["seed"])
                assert [r[0]["seed"] for r in runs]==sorted(parameters["seeds"])
                for a,b in zip(reference,runs):
                    np.testing.assert_array_equal(a[2][:min(24,parameters["budget"])],b[2][:min(24,parameters["budget"])])
    summary=[]
    for (name,dim,method),runs in sorted(groups.items()):
        values=np.array([r[0]["best_y"] for r in runs])
        summary.append({"function":name,"dimension":dim,"method":method,"n":len(values),
                        "mean":float(values.mean()),"std":float(values.std(ddof=1)) if len(values)>1 else 0.,
                        "median":float(np.median(values)),"min":float(values.min()),"max":float(values.max()),
                        "mean_seconds":float(np.mean([r[0]["elapsed_seconds"] for r in runs]))})
    write_csv(root/"per_run.csv",individual); write_csv(root/"summary.csv",summary)
    methods=parameters["methods"]
    rank_rows=[]
    for name in parameters["functions"]:
        for dim in parameters["dimensions"]:
            medians=[next(r["median"] for r in summary if (r["function"],r["dimension"],r["method"])==(name,dim,m)) for m in methods]
            ranks=rankdata(medians)
            rank_rows.append({"function":name,"dimension":dim,**dict(zip(methods,map(float,ranks)))})
    write_csv(root/"ranks.csv",rank_rows)
    comparisons=[]
    pairs = [("adaptive", "saea"), ("llm", "saea"), ("llm", "adaptive")]
    for left, right in pairs:
        if left not in methods or right not in methods:
            continue
        for name in parameters["functions"]:
            for dim in parameters["dimensions"]:
                av=[r[0]["best_y"] for r in sorted(groups[(name,dim,left)],key=lambda r:r[0]["seed"])]
                bv=[r[0]["best_y"] for r in sorted(groups[(name,dim,right)],key=lambda r:r[0]["seed"])]
                difference=np.array(av)-np.array(bv)
                p=1. if np.all(difference==0) else float(wilcoxon(difference,method="auto").pvalue)
                comparisons.append({"function":name,"dimension":dim,"comparison":f"{left} vs {right}", "paired_seeds":len(av),"wins":int(np.sum(difference<0)),"ties":int(np.sum(difference==0)),"losses":int(np.sum(difference>0)),"p_raw_two_sided":p})
    if comparisons:
        order=np.argsort([r["p_raw_two_sided"] for r in comparisons]); previous=0.
        for rank,index in enumerate(order):
            previous=max(previous,min(1.,(len(order)-rank)*comparisons[index]["p_raw_two_sided"]))
            comparisons[index]["p_holm"]=previous
        write_csv(root/"paired_tests.csv",comparisons)
    figures=root/"figures"; figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":170})
    for dim in parameters["dimensions"]:
        fig,axes=plt.subplots(2,3,figsize=(13.5,7.4),constrained_layout=True)
        for ax,name in zip(axes.flat,parameters["functions"]):
            for method in methods:
                history=np.array([r[1] for r in groups[(name,dim,method)]])
                q1,median,q3=np.quantile(history,[.25,.5,.75],axis=0)
                x=np.arange(1,history.shape[1]+1)
                ax.plot(x,np.maximum(median,1e-12),label=LABELS[method],color=COLORS[method],lw=1.6)
                ax.fill_between(x,np.maximum(q1,1e-12),np.maximum(q3,1e-12),color=COLORS[method],alpha=.10)
            ax.set(yscale="log",xlabel="True objective evaluations",ylabel="Best observed f(x), lower is better",title=f"{name.capitalize()} ({dim}D)")
            ax.axvline(24,ls=":",lw=.8,color="#777777")
            ax.grid(alpha=.16)
        for ax in list(axes.flat)[len(parameters["functions"]):]:
            ax.set_visible(False)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="outside lower center",ncol=len(methods),frameon=False)
        fig.suptitle(f"Equal-budget analytic benchmarks | median and IQR over {len(parameters['seeds'])} seeds\nOffline comparators; online LLM not evaluated" if "llm" not in methods else "Equal-budget benchmark comparisons",fontsize=14)
        fig.savefig(figures/f"convergence_{dim}d.png"); fig.savefig(figures/f"convergence_{dim}d.svg"); plt.close(fig)
    avg_rank={m:float(np.mean([r[m] for r in rank_rows])) for m in methods}
    fig,ax=plt.subplots(figsize=(8,4.3),constrained_layout=True)
    ax.barh([LABELS[m] for m in methods],[avg_rank[m] for m in methods],color=[COLORS[m] for m in methods])
    ax.set(xlim=(0,len(methods)+.4),xlabel="Average rank across function-dimension cases (lower is better)",title="Descriptive ranking by median final objective")
    for i,m in enumerate(methods): ax.text(avg_rank[m]+.05,i,f"{avg_rank[m]:.2f}",va="center")
    fig.savefig(figures/"average_rank.png"); plt.close(fig)
    (root/"analysis.json").write_text(json.dumps({"verified_runs":len(individual),"true_evaluations":sum(r[0]["evaluations"] for g in groups.values() for r in g),"average_ranks":avg_rank,"online_llm_evaluated":any(r[0]["metadata"]["online_llm"] for g in groups.values() for r in g),"any_holm_significant":any(r["p_holm"]<.05 for r in comparisons),"sum_runtime_seconds":sum(r["elapsed_seconds"] for r in individual)},indent=2),encoding="utf-8")
    print(json.dumps({"verified_runs":len(individual),"average_ranks":avg_rank},indent=2))


if __name__=="__main__":main()
