from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class Problem:
    """A finite box, a deterministic scalar objective, and optional description.

    Only this callable can spend the expensive evaluation budget. A caller may
    wrap a simulation, penalized objective, or scalarization explicitly.
    """
    name: str
    lower: np.ndarray
    upper: np.ndarray
    objective: Callable[[np.ndarray], float]
    description: str = "User-supplied bounded black-box minimization."
    optimum: float | None = None

    def __post_init__(self):
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        if (self.lower.ndim != 1 or self.lower.size == 0 or
                self.upper.shape != self.lower.shape or
                not np.all(np.isfinite(self.lower)) or
                not np.all(np.isfinite(self.upper)) or
                np.any(self.upper <= self.lower)):
            raise ValueError("Bounds must be finite 1-D arrays with lower < upper.")

    @property
    def dimension(self):
        return len(self.lower)

    def decode(self, z):
        z = np.asarray(z, dtype=float)
        if z.shape != self.lower.shape or not np.all(np.isfinite(z)):
            raise ValueError("Invalid candidate shape or non-finite candidate.")
        if np.any(z < 0) or np.any(z > 1):
            raise ValueError("Normalized candidate outside [0, 1].")
        return self.lower + z * (self.upper - self.lower)


def benchmark(name: str, dimension: int = 5) -> Problem:
    if dimension < 2:
        raise ValueError("Benchmark dimension must be >= 2.")
    functions = {
        "sphere": (lambda x: np.sum(x*x), -5.0, 5.0),
        "rosenbrock": (lambda x: np.sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2), -2.0, 2.0),
        "rastrigin": (lambda x: 10*len(x)+np.sum(x*x-10*np.cos(2*np.pi*x)), -5.12, 5.12),
        "ackley": (lambda x: -20*np.exp(-0.2*np.sqrt(np.mean(x*x)))-np.exp(np.mean(np.cos(2*np.pi*x)))+20+np.e, -5.0, 5.0),
        "griewank": (lambda x: np.sum(x*x)/4000-np.prod(np.cos(x/np.sqrt(np.arange(1,len(x)+1))))+1, -50.0, 50.0),
        "ellipsoid": (lambda x: np.sum(np.logspace(0, 6, len(x))*x*x), -5.0, 5.0),
    }
    if name not in functions:
        raise ValueError(f"Unknown benchmark: {name}")
    f, lo, hi = functions[name]
    # No function identity, optimum or analytic formula is passed to the LLM.
    return Problem(name, np.full(dimension, lo), np.full(dimension, hi),
                   f, "Deterministic bounded continuous black-box minimization.", 0.0)


def latin_hypercube(n, d, rng):
    return np.column_stack([(rng.permutation(n)+rng.random(n))/n for _ in range(d)])
