"""Audit live evidence, replay all runs and run a frozen-initial-plan ablation.

This script makes NO network/model calls. Only the included analytic benchmarks
are reevaluated. Replays are validation and never counted as independent trials.
"""
import os
for variable in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from llm_saea import benchmark, optimize, Plan
from llm_saea.policy import export_program
from analyze import write_csv


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    root = Path(args.directory)
    manifest = read(root / "manifest.json")
    assert manifest["state"] == "COMPLETED" and manifest["completed_runs"] == 300
    for rel, sha in manifest["baseline_file_sha256"].items():
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == sha
    prompt = completion = total = cached = 0
    latency = []
    identities = set()
    call_rows, replay_rows, frozen_rows, plan_rows = [], [], [], []
    for path in sorted((root / "runs").rglob("result.json")):
        result = read(path)
        data = np.genfromtxt(path.with_name("history.csv"), delimiter=",", skip_header=1)
        problem = benchmark(result["problem"], result["metadata"]["dimension"])
        assert np.all(data[:, 3:] >= problem.lower) and np.all(data[:, 3:] <= problem.upper)
        np.testing.assert_allclose([problem.objective(x) for x in data[:, 3:]], data[:, 1], rtol=1e-12, atol=1e-12)
        if result["method"] != "llm":
            continue
        assert result["metadata"]["online_llm"] and result["metadata"]["llm_provider"] == "deepseek"
        events = result["events"]
        assert [e["at_evaluation"] for e in events] == [24, 44, 64, 84]
        logs = sorted((path.parent / "llm_calls").glob("call_*.json"))
        assert len(logs) == 4
        for log_path, event in zip(logs, events):
            log = read(log_path)
            assert log["status"] == "completed" and "error_type" not in log
            assert log["source"] == "deepseek" and log["validated_plan"] == event["plan"]
            assert log["state"]["evaluations"] == event["at_evaluation"]
            assert log["requested_model"] == log["returned_model"] == "deepseek-flash"
            assert log["response_id"] not in identities
            identities.add(log["response_id"])
            usage = log["usage"]
            assert usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]
            prompt += usage["prompt_tokens"]; completion += usage["completion_tokens"]; total += usage["total_tokens"]
            cached += usage.get("prompt_cache_hit_tokens", 0)
            latency.append(log["elapsed_seconds"])
            call_rows.append(dict(function=result["problem"], dimension=problem.dimension, seed=result["seed"],
                                  evaluation=event["at_evaluation"], prompt_tokens=usage["prompt_tokens"],
                                  completion_tokens=usage["completion_tokens"], total_tokens=usage["total_tokens"],
                                  seconds=log["elapsed_seconds"], log=log_path.relative_to(root).as_posix()))
            plan_rows.append({k:v for k,v in event["plan"].items() if k != "reason"})
        replay = optimize(problem, budget=100, seed=result["seed"], method="replay", replay_events=events)
        np.testing.assert_array_equal(replay.x, data[:, 3:])
        np.testing.assert_array_equal(replay.y, data[:, 1])
        replay_rows.append(dict(function=result["problem"], dimension=problem.dimension, seed=result["seed"],
                                exact_match=True, evaluations=100, seconds=replay.elapsed_seconds))
        frozen_dir = root / "frozen_initial" / f"{result['problem']}_d{problem.dimension}" / f"seed_{result['seed']}"
        if not (frozen_dir / "result.json").exists():
            frozen = optimize(problem, budget=100, seed=result["seed"], method="saea", plan=Plan.from_dict(events[0]["plan"]))
            np.testing.assert_array_equal(frozen.x[:44], data[:44, 3:])
            frozen.metadata.update(ablation="freeze first live plan; no further model calls", source_live_run=path.relative_to(root).as_posix())
            frozen.save(frozen_dir)
            export_program(frozen_dir / "generated_algorithm.py", Plan.from_dict(events[0]["plan"]),
                           dict(source="deepseek_first_plan_frozen", live_result=path.relative_to(root).as_posix(), independent_online_trial=False))
        frozen_result = read(frozen_dir / "result.json")
        frozen_rows.append(dict(function=result["problem"], dimension=problem.dimension, seed=result["seed"],
                                dynamic_best=result["best_y"], frozen_best=frozen_result["best_y"],
                                difference=result["best_y"] - frozen_result["best_y"]))
    assert len(call_rows) == 240 and len(replay_rows) == len(frozen_rows) == 60
    comparisons = []
    for name in manifest["parameters"]["functions"]:
        for dim in manifest["parameters"]["dimensions"]:
            rows = [r for r in frozen_rows if r["function"] == name and r["dimension"] == dim]
            diff = np.array([r["difference"] for r in rows])
            comparisons.append(dict(function=name, dimension=dim, n=len(rows),
                                    dynamic_median=float(np.median([r["dynamic_best"] for r in rows])),
                                    frozen_median=float(np.median([r["frozen_best"] for r in rows])),
                                    wins=int(sum(diff<0)), ties=int(sum(diff==0)), losses=int(sum(diff>0)),
                                    p_raw_two_sided=1. if np.all(diff==0) else float(wilcoxon(diff,method="auto").pvalue)))
    previous = 0.
    for rank, index in enumerate(np.argsort([r["p_raw_two_sided"] for r in comparisons])):
        previous = max(previous, min(1., (len(comparisons)-rank)*comparisons[index]["p_raw_two_sided"]))
        comparisons[index]["p_holm"] = previous
    write_csv(root / "api_calls.csv", call_rows)
    write_csv(root / "replay_audit.csv", replay_rows)
    write_csv(root / "frozen_per_run.csv", frozen_rows)
    write_csv(root / "frozen_comparison.csv", comparisons)
    costs = dict(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total, cache_hit_tokens=cached,
                 successful_requests=len(call_rows), failed_requests=0, replay_verified_runs=60,
                 frozen_ablation_runs=60, api_seconds_sum=float(sum(latency)),
                 api_seconds_median=float(np.median(latency)), api_seconds_max=float(max(latency)),
                 usd_peak_uncached_upper_estimate=(prompt*.3+completion*1.2)/1e6,
                 pricing_source="https://api-docs.deepseek.com/quick_start/pricing/", pricing_checked="2026-10-23",
                 cost_note="Conservative estimate using peak uncached rates; not an account invoice. Earlier one-call smoke test excluded.",
                 unique_numeric_plans=len({json.dumps(p,sort_keys=True) for p in plan_rows}),
                 surrogate_choices=dict(Counter(p["surrogate"] for p in plan_rows)),
                 operator_choices=dict(Counter(p["operator"] for p in plan_rows)),
                 dynamic_vs_frozen={key:sum(r[key] for r in comparisons) for key in ["wins","ties","losses"]},
                 frozen_holm_significant=sum(r["p_holm"]<.05 for r in comparisons),
                 measured_objectives_recomputed=True, baseline_hashes_unchanged=True)
    (root / "online_audit.json").write_text(json.dumps(costs,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(costs,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
