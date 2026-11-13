"""Replace simulation() with your real expensive solver; no model key required."""
import numpy as np
from llm_saea import Problem, optimize


def simulation(x):
    # Demonstration scalar cost. Put finite element/CFD/simulation calls here.
    # For known constraints, explicitly add a documented penalty or transform.
    return float((x[0]-1.2)**2+2*(x[1]+.4)**2+.2*(x[0]*x[1])**2)


if __name__ == "__main__":
    problem=Problem("custom_simulation",np.array([-3.,-3.]),np.array([3.,3.]),simulation,
                    "Two continuous design variables; minimize simulated cost.")
    result=optimize(problem,budget=60,method="saea",seed=42)
    print("best x:",result.best_x,"best measured objective:",result.best_y)
