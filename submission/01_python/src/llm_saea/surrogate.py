"""Small-data surrogate portfolio, trained only on truly evaluated points."""
import numpy as np


def distances(a, b):
    return np.sqrt(np.maximum(np.sum(a*a, axis=1)[:, None] +
                              np.sum(b*b, axis=1)[None, :] - 2*a@b.T, 0.))


def quadratic_features(x):
    d = x.shape[1]
    return np.column_stack([np.ones(len(x)), x] +
                           [x[:, i]*x[:, j] for i in range(d) for j in range(i, d)])


class Regressor:
    def __init__(self, kind, length=1.):
        self.kind, self.length = kind, length

    def fit(self, x, y):
        self.x = x
        self.center, self.scale = float(np.mean(y)), max(float(np.std(y)), 1e-12)
        t = (y-self.center)/self.scale
        if self.kind == "rbf":
            k = np.exp(-.5*(distances(x,x)/self.length)**2)
            self.weights = np.linalg.solve(k+1e-6*np.eye(len(x)), t)
        else:
            q = quadratic_features(x)
            # Ridge via augmented least squares is stable for underdetermined data.
            self.weights = np.linalg.lstsq(
                np.vstack([q, 1e-3*np.eye(q.shape[1])]),
                np.r_[t, np.zeros(q.shape[1])], rcond=None)[0]
        return self

    def predict(self, x):
        q = np.exp(-.5*(distances(x,self.x)/self.length)**2) if self.kind == "rbf" else quadratic_features(x)
        return (q@self.weights)*self.scale+self.center


class Portfolio:
    def fit(self, x, y, choice="auto"):
        self.x, self.y = x, y
        self.center, self.scale = float(np.mean(y)), max(float(np.std(y)), 1e-12)
        # Deterministic held-out split; every candidate sees the same split.
        valid = np.arange(len(x)) % 5 == 0
        train = ~valid
        pd = distances(x[train],x[train])
        nonzero = pd[pd > 1e-10]
        length = max(float(np.median(nonzero)) if len(nonzero) else 1., .05)
        candidates = []
        if choice in {"auto", "rbf"}:
            candidates.extend(Regressor("rbf", length*k) for k in [.5,1.,2.])
        if choice in {"auto", "quadratic"}:
            candidates.append(Regressor("quadratic"))
        scored = []
        for candidate in candidates:
            candidate.fit(x[train], y[train])
            error = np.sqrt(np.mean((candidate.predict(x[valid])-y[valid])**2))/self.scale
            scored.append(float(error))
        self.cv_error = min(scored)
        self.model = candidates[int(np.argmin(scored))].fit(x, y)
        self.kind = self.model.kind
        self.length = self.model.length
        return self

    def acquisition(self, z, beta):
        # Distance is a coverage heuristic, not a calibrated posterior stddev.
        novelty = np.min(distances(z,self.x), axis=1)/np.sqrt(z.shape[1])
        mu = (self.model.predict(z)-self.center)/self.scale
        return mu-beta*novelty
