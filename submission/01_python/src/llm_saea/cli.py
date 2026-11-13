import argparse
import json
import os
from pathlib import Path
from .problem import benchmark
from .optimizer import optimize
from .policy import Plan, export_program
from .llm import create_designer


def main():
    parser = argparse.ArgumentParser(description="Auditable LLM-configured surrogate-assisted evolution")
    parser.add_argument("--env-file",help="Explicit local env file; never included in outputs")
    subs = parser.add_subparsers(dest="command",required=True)
    run = subs.add_parser("run")
    run.add_argument("--method",choices=["random","de","saea","adaptive","llm","replay"],default="saea")
    run.add_argument("--problem",default="sphere")
    run.add_argument("--dimension",type=int,default=5)
    run.add_argument("--budget",type=int,default=100)
    run.add_argument("--seed",type=int,default=0)
    run.add_argument("--initial",type=int,default=24)
    run.add_argument("--design-interval",type=int,default=20)
    run.add_argument("--max-designs",type=int,default=4)
    run.add_argument("--model",default=None)
    run.add_argument("--provider",choices=["openai","deepseek"],default=None)
    run.add_argument("--replay",help="Previous result.json with a saved plan schedule")
    run.add_argument("--output",required=True)
    gen = subs.add_parser("generate")
    gen.add_argument("--dimension",type=int,default=5)
    gen.add_argument("--budget",type=int,default=100)
    gen.add_argument("--model",default=None)
    gen.add_argument("--provider",choices=["openai","deepseek"],default=None)
    gen.add_argument("--output",required=True,help="New generated .py path")
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        if not Path(args.env_file).is_file():
            parser.error("Env file does not exist")
        load_dotenv(args.env_file,override=False)
    model = args.model
    try:
        if args.command == "generate":
            if args.dimension < 1 or args.budget < 1:
                parser.error("dimension and budget must be positive")
            if Path(args.output).exists():
                raise FileExistsError(args.output)
            designer = create_designer(model,Path(args.output).with_suffix(".llm_logs"),max_calls=1,provider=args.provider)
            state = {"dimension":args.dimension,"evaluations":0,"budget":args.budget,
                     "stagnation":0,"relative_recent_improvement":0.,"cv_normalized_rmse":None,
                     "current_plan":Plan().to_dict(),"description":"Bounded continuous black-box minimization."}
            plan = designer.design(state)
            export_program(args.output,plan,{"source":designer.provider,"model":designer.model,"log_directory":str(designer.directory)})
            print(f"Generated {args.output}")
            return
        out = Path(args.output)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Output directory must be empty or new")
        designer = create_designer(model,out/"llm_calls",max_calls=args.max_designs,provider=args.provider) if args.method=="llm" else None
        events = None
        if args.method == "replay":
            if not args.replay:
                parser.error("--replay result.json is required")
            old = json.loads(Path(args.replay).read_text(encoding="utf-8"))
            if (old["problem"],old["seed"],old["metadata"]["dimension"],old["evaluations"],old["metadata"]["n_initial"]) != (args.problem,args.seed,args.dimension,args.budget,min(args.initial,args.budget)):
                raise ValueError("Replay problem, seed, dimension, budget and initial count must match")
            events = old["events"]
        result = optimize(benchmark(args.problem,args.dimension),budget=args.budget,seed=args.seed,
                          method=args.method,designer=designer,n_initial=args.initial,
                          design_interval=args.design_interval,max_designs=args.max_designs,replay_events=events,
                          journal_path=out/"evaluation_journal.jsonl")
        result.save(out)
        if result.events:
            export_program(out/"generated_final_algorithm.py",Plan.from_dict(result.events[-1]["plan"]),
                           {"source":args.method,"note":"Final frozen plan; dynamic schedule remains in result.json."})
        print(json.dumps({"best_y":result.best_y,"evaluations":len(result.y),"output":str(out)},ensure_ascii=False))
    except (ValueError,RuntimeError,FileExistsError) as error:
        parser.exit(2,f"Error: {error}\n")
