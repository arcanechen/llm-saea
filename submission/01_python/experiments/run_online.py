"""Extend immutable offline controls with a predeclared live DeepSeek arm.

No automatic retries or offline fallback. A failed run preserves its journal.
--resume skips completed runs but refuses to repeat a partial paid run.
"""
import os
for variable in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from llm_saea import benchmark, optimize
from llm_saea.llm import create_designer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    base, out = Path(args.baseline), Path(args.output)
    original = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    params = dict(original["parameters"])
    assert params["methods"] == ["random", "de", "saea", "adaptive"]
    assert params["budget"] == 100 and original["completed_runs"] == 240
    params.update(methods=[*params["methods"], "llm"], provider="deepseek", model="deepseek-flash")
    params.pop("env_file", None)
    params.pop("output", None)
    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=False)
    if out.exists() and any(out.iterdir()):
        if not args.resume:
            parser.error("Nonempty output; use --resume only for a completed-run boundary")
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["parameters"] == params
    else:
        out.mkdir(parents=True, exist_ok=True)
        sources = {p.relative_to(base).as_posix(): digest(p) for p in (base / "runs").rglob("*") if p.is_file()}
        shutil.copytree(base / "runs", out / "runs")
        manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                        parameters=params, python=sys.version, platform=platform.platform(),
                        packages=original["packages"], objective_delay_seconds=0,
                        protocol="Predeclared 6 functions x 2 dimensions x 5 seeds; 100 evaluations, LHS24; LLM at 24/44/64/84. No tuning, retries, exclusions, or fallback.",
                        baseline_reuse="240 historical controls copied byte-for-byte; not new runs",
                        baseline_manifest_sha256=digest(base / "manifest.json"),
                        baseline_file_sha256=sources, completed_runs=240, planned_live_runs=60,
                        max_new_api_requests=240, state="RUNNING",
                        source_sha256={p.name:digest(p) for p in (Path(__file__).resolve().parents[1] / "src" / "llm_saea").glob("*.py")})
        save(out / "manifest.json", manifest)
    for rel, sha in manifest["baseline_file_sha256"].items():
        assert digest(out / rel) == sha and digest(base / rel) == sha
    completed = 0
    for name in params["functions"]:
        for dimension in params["dimensions"]:
            for seed in params["seeds"]:
                directory = out / "runs" / f"{name}_d{dimension}" / "llm" / f"seed_{seed}"
                if (directory / "result.json").exists():
                    completed += 1
                    continue
                if directory.exists() and any(directory.iterdir()):
                    raise RuntimeError("Partial live run preserved; not repeating paid requests. Inspect logs before recovery.")
                try:
                    designer = create_designer("deepseek-flash", directory / "llm_calls", max_calls=4, provider="deepseek")
                    result = optimize(benchmark(name, dimension), budget=100, seed=seed,
                                      method="llm", designer=designer,
                                      journal_path=directory / "evaluation_journal.jsonl")
                    result.save(directory)
                except Exception as error:
                    manifest.update(state="PAUSED_ON_ERROR", failed_run=f"{name}_d{dimension}/llm/seed_{seed}",
                                    error_type=type(error).__name__, completed_runs=240+completed,
                                    stopped_utc=datetime.now(timezone.utc).isoformat())
                    save(out / "manifest.json", manifest)
                    print("PAUSED: live run failed; sanitized call log and evaluation journal preserved.", flush=True)
                    return 2
                completed += 1
                manifest.update(completed_runs=240+completed)
                save(out / "manifest.json", manifest)
                print(f"LIVE {completed:02}/60 {name} d={dimension} seed={seed} best={result.best_y:.7g} calls={designer.calls} seconds={result.elapsed_seconds:.2f}", flush=True)
    manifest.update(state="COMPLETED", completed_utc=datetime.now(timezone.utc).isoformat())
    save(out / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
