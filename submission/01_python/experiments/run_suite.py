"""Run a predeclared equal-budget benchmark. Never mixes live/offline results."""
import os
for variable in ["OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"]:
    os.environ.setdefault(variable,"1")
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from llm_saea import benchmark,optimize
from llm_saea.llm import create_designer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--functions",nargs="+",default=["sphere","ellipsoid","rosenbrock","rastrigin","ackley","griewank"])
    parser.add_argument("--dimensions",nargs="+",type=int,default=[5,10])
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2,3,4])
    parser.add_argument("--methods",nargs="+",choices=["random","de","saea","adaptive","llm"],default=["random","de","saea","adaptive"])
    parser.add_argument("--budget",type=int,default=100)
    parser.add_argument("--model")
    parser.add_argument("--provider",choices=["openai","deepseek"])
    parser.add_argument("--env-file")
    args=parser.parse_args()
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):
        parser.error("Choose a new or empty output directory; raw results are immutable")
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file,override=False)
    out.mkdir(parents=True,exist_ok=True)
    metadata={"started_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,
              "platform":platform.platform(),"processor":platform.processor(),
              "parameters":vars(args),"objective_delay_seconds":0,
              "protocol":"Six analytic functions, equal true-evaluation budget, common initial LHS per seed.",
              "llm_status":"requested" if "llm" in args.methods else "NOT_RUN",
              "packages":{m:importlib.metadata.version(m) for m in ["numpy","scipy","matplotlib","openai"]}}
    (out/"manifest.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    count=0
    for name in args.functions:
        for dimension in args.dimensions:
            for seed in args.seeds:
                for method in args.methods:
                    directory=out/"runs"/f"{name}_d{dimension}"/method/f"seed_{seed}"
                    designer=create_designer(args.model,directory/"llm_calls",max_calls=4,provider=args.provider) if method=="llm" else None
                    result=optimize(benchmark(name,dimension),budget=args.budget,seed=seed,method=method,designer=designer)
                    result.save(directory)
                    count+=1
                    print(f"{count:03d} {name:12s} d={dimension:2d} seed={seed} {method:8s} best={result.best_y:.6g} time={result.elapsed_seconds:.2f}s",flush=True)
    metadata["completed_utc"]=datetime.now(timezone.utc).isoformat()
    metadata["completed_runs"]=count
    (out/"manifest.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
