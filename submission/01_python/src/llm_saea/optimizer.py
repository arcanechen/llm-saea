from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import time
import numpy as np
from .problem import latin_hypercube
from .policy import Plan, offline_policy
from .surrogate import Portfolio, distances


def de_trials(pop, scores, rng, plan):
    n, d = pop.shape
    best = pop[int(np.argmin(scores))]
    trials = []
    for i in range(n):
        pool = np.delete(np.arange(n), i)
        a, b, c = rng.choice(pool, 3, replace=False)
        if plan.operator == "rand1":
            mutant = pop[a]+plan.mutation*(pop[b]-pop[c])
        elif plan.operator == "best1":
            mutant = best+plan.mutation*(pop[b]-pop[c])
        else:
            mutant = pop[i]+plan.mutation*(best-pop[i])+plan.mutation*(pop[a]-pop[b])
        # Reflection at bounds retains more variation than clipping.
        mutant = 1-np.abs(np.mod(mutant, 2)-1)
        mask = rng.random(d) < plan.crossover
        mask[rng.integers(d)] = True
        trials.append(np.where(mask, mutant, pop[i]))
    return np.asarray(trials)


def propose(model, x, y, rng, plan):
    n, d = plan.population, x.shape[1]
    pop = rng.random((n, d))
    local = int(n*plan.local_fraction)
    if local:
        elite = x[np.argsort(y)[:min(5,len(y))]]
        pop[:local] = np.clip(elite[rng.integers(len(elite),size=local)]+
                              rng.normal(0,plan.radius,(local,d)),0,1)
    score = model.acquisition(pop, plan.exploration)
    seen = [pop.copy()]
    for _ in range(plan.generations):
        trial = de_trials(pop,score,rng,plan)
        ts = model.acquisition(trial,plan.exploration)
        improve = ts < score
        pop[improve], score[improve] = trial[improve], ts[improve]
        seen.append(trial)
    pool = np.vstack(seen)
    fresh = np.min(distances(pool,x),axis=1)>1e-7
    if not np.any(fresh):
        return rng.random(d)
    pool = pool[fresh]
    return pool[int(np.argmin(model.acquisition(pool,plan.exploration)))]


@dataclass
class Result:
    problem: str
    method: str
    seed: int
    x: np.ndarray
    y: np.ndarray
    elapsed_seconds: float
    objective_seconds: float
    events: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def best_x(self):
        return self.x[int(np.argmin(self.y))].copy()

    @property
    def best_y(self):
        return float(np.min(self.y))

    def save(self, directory):
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        if (out/"result.json").exists():
            raise FileExistsError(f"Refusing to overwrite run at {out}")
        with (out/"history.csv").open("w",newline="",encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["evaluation","objective","best_so_far"]+[f"x{i}" for i in range(self.x.shape[1])])
            for i,(x,y,best) in enumerate(zip(self.x,self.y,np.minimum.accumulate(self.y)),1):
                writer.writerow([i,y,best,*x])
        summary = dict(problem=self.problem,method=self.method,seed=self.seed,
                       best_y=self.best_y,best_x=self.best_x.tolist(),evaluations=len(self.y),
                       elapsed_seconds=self.elapsed_seconds,objective_seconds=self.objective_seconds,
                       metadata=self.metadata,events=self.events)
        (out/"result.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def optimize(problem, budget=100, seed=0, method="saea", plan=None,
             designer=None, n_initial=24, design_interval=20, max_designs=4,
             replay_events=None, refit_interval=3, journal_path=None):
    """Minimize within exactly `budget` true objective evaluations.

    llm requires a designer and fails visibly on any model/validation error.
    Replay reproduces saved policy decisions without making API requests.
    """
    methods = {"random","de","saea","adaptive","llm","replay"}
    if method not in methods:
        raise ValueError(f"method must be one of {sorted(methods)}")
    for name,value in [("budget",budget),("n_initial",n_initial),("design_interval",design_interval),
                       ("max_designs",max_designs),("refit_interval",refit_interval)]:
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if n_initial < 4:
        raise ValueError("n_initial must be >= 4")
    if method == "llm" and designer is None:
        raise ValueError("LLM mode requires an explicit LLM designer")
    if method == "replay" and not replay_events:
        raise ValueError("Replay requires saved plan events")
    replay_map = {}
    for e in replay_events or []:
        if e["at_evaluation"] in replay_map:
            raise ValueError("Duplicate replay event")
        if type(e["at_evaluation"]) is not int or not min(budget,n_initial) <= e["at_evaluation"] < budget:
            raise ValueError("Replay event outside this run's evaluation range")
        replay_map[e["at_evaluation"]] = Plan.from_dict(e["plan"])
    journal = None
    if journal_path is not None:
        journal = Path(journal_path)
        journal.parent.mkdir(parents=True,exist_ok=True)
        with journal.open("x",encoding="utf-8"):
            pass
    def append_journal(record):
        if journal is not None:
            with journal.open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(record,allow_nan=False)+"\n")
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    x, y, events, timings = [], [], [], []
    best, last_improvement = float("inf"), 0

    def evaluate(z):
        nonlocal best, last_improvement
        if len(y) >= budget:
            raise RuntimeError("Evaluation budget exhausted")
        t = time.perf_counter()
        point = problem.decode(z).copy()
        try:
            value = float(problem.objective(point.copy()))
            if not np.isfinite(value):
                raise ValueError("Objective returned non-finite value; run aborted (no silent replacement).")
        except Exception as error:
            append_journal({"attempt":len(y)+1,"x":point.tolist(),"status":"failed", "error_type":type(error).__name__})
            raise
        timings.append(time.perf_counter()-t)
        x.append(z.copy()); y.append(value)
        if value < best:
            best, last_improvement = value, len(y)
        append_journal({"attempt":len(y),"x":point.tolist(),"status":"ok","objective":value,"best_so_far":best})
        return value

    for z in latin_hypercube(min(budget,n_initial),problem.dimension,rng):
        evaluate(z)
    n_init_actual = len(y)
    current = plan or Plan()
    population, fitness = np.asarray(x).copy(), np.asarray(y).copy()
    model, fitted_at, calls = None, -1000, 0
    replay_used = set()
    while len(y) < budget:
        xa, ya = np.asarray(x), np.asarray(y)
        if method == "random":
            evaluate(rng.random(problem.dimension)); continue
        if method == "de":
            trial = de_trials(population,fitness,rng,Plan(operator="rand1",mutation=.7,crossover=.9))
            for i, z in enumerate(trial):
                if len(y) >= budget:
                    break
                value = evaluate(z)
                if value <= fitness[i]:
                    population[i], fitness[i] = z, value
            continue
        state = {"dimension":problem.dimension,"evaluations":len(y),"budget":budget,
                 "stagnation":len(y)-last_improvement,
                 "relative_recent_improvement":float((min(y[:max(n_init_actual,len(y)-10)])-best)/max(float(np.std(ya)),1e-12)),
                 "cv_normalized_rmse":None if model is None else model.cv_error,
                 "current_plan":current.to_dict(),"description":problem.description}
        scheduled = (len(y)-n_init_actual)%design_interval == 0
        changed = False
        if method == "replay" and len(y) in replay_map:
            current = replay_map[len(y)]
            replay_used.add(len(y)); changed = True
        elif method in {"llm","adaptive"} and scheduled and calls < max_designs:
            current = designer.design(state) if method == "llm" else offline_policy(state)
            calls += 1; changed = True
        if changed or not events:
            events.append({"at_evaluation":len(y),"plan":current.to_dict(),
                           "source":designer.provider if method=="llm" else method})
        if model is None or changed or len(y)-fitted_at >= refit_interval:
            model = Portfolio().fit(xa,ya,current.surrogate)
            fitted_at = len(y)
        z = propose(model,xa,ya,rng,current)
        evaluate(z)
    if method == "replay" and replay_used != set(replay_map):
        raise ValueError("Replay event evaluations do not match this run's budget/initialization")
    elapsed = time.perf_counter()-start
    actual_x = np.asarray([problem.decode(z) for z in x])
    return Result(problem.name,method,seed,actual_x,np.asarray(y),elapsed,sum(timings),events,
                  {"budget":budget,"dimension":problem.dimension,"n_initial":n_init_actual,
                   "design_interval":design_interval,"max_designs":max_designs,
                   "refit_interval":refit_interval,"policy_calls":calls,
                   "llm_provider":designer.provider if method=="llm" else None,
                   "llm_model":designer.model if method=="llm" else None,
                   "online_llm":method=="llm" and calls>0, "objective_delay_injected":False})
