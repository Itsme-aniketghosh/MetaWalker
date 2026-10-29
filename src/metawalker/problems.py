"""Simulated test problems with a known optimum (value 0), the same generator the benchmark used.

    fn, space = make_problem('rastrigin', n=50, seed=0, noise=0.01)
    fn(params) -> noisy value to minimize;   fn.true(params) -> noise-free value

Families:
  sphere, ellipsoid, rastrigin, rosenbrock  (smooth to rugged; the learned policies trained on these)
  ackley, levy                              (held out from training)
  plateau   0 signal on ~95% of the space, one small region that depends on 3 hidden directions, measured
            like a rate (k successes out of 100 trials): a stand-in for honeypot / sparse-reward searches
Inputs are rotated and shifted, so the optimum is never on an axis or at the centre.
"""
import time

import numpy as np

HP_DIMS, HP_TRIALS, HP_FRAC = 3, 100, 0.05  # honeypot: relevant hidden directions, episodes per measurement, live fraction
FAMILIES = ('sphere', 'ellipsoid', 'rastrigin', 'rosenbrock', 'ackley', 'levy', 'plateau')
PLATEAU_DIMS, PLATEAU_TRIALS, PLATEAU_LIVE = 3, 100, 0.05


def raw_f(name, z, cond):
    n = len(z)
    if name == 'sphere': return z @ z
    if name == 'ellipsoid': return cond @ z ** 2
    if name == 'rastrigin': return 10 * n + z @ z - 10 * np.cos(2 * np.pi * z).sum()
    if name == 'rosenbrock':
        z = z + 1  # optimum at z = 0
        return (100 * (z[1:] - z[:-1] ** 2) ** 2 + (1 - z[:-1]) ** 2).sum()
    if name == 'ackley':
        return (-20 * np.exp(-0.2 * np.sqrt(z @ z / n)) - np.exp(np.cos(2 * np.pi * z).mean())
                + 20 + np.e)
    if name == 'levy':
        w = 1 + z / 4
        return (np.sin(np.pi * w[0]) ** 2 + ((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2)).sum()
                + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2))
    raise ValueError(name)


class TimeLimit(Exception):
    pass


class MeanBest:
    """Best point = lowest *mean* over repeated evaluations of the same x, so re-checking a lucky point can demote it.
    Needs self.obs = {} and self.best_y = inf; call record(x, y) once per evaluation. A subclass may override key(x)
    to pool equivalent points (e.g. two x that decode to the same config); reps keeps one x per key."""
    best_k = None

    def key(self, x):
        return np.asarray(x, float).tobytes()

    def record(self, x, y):
        k, reps = self.key(x), self.__dict__.setdefault('reps', {})
        reps.setdefault(k, np.array(x, float))
        s, c = self.obs.get(k, (0.0, 0)); self.obs[k] = (s + y, c + 1)
        if (s + y) / (c + 1) < self.best_y:
            self.best_y, self.best_x, self.best_k = (s + y) / (c + 1), np.array(x, float), k
        elif k == self.best_k:  # the incumbent got worse on a re-check: rescan (min keeps the earliest on ties)
            self.best_k = min(self.obs, key=lambda q: self.obs[q][0] / self.obs[q][1])
            self.best_y = self.obs[self.best_k][0] / self.obs[self.best_k][1]
            self.best_x = reps[self.best_k].copy()

    def stats(self, x):
        s, c = self.obs[self.key(x)]
        return s / c, c


class Problem(MeanBest):
    """Counts evals, adds noise, and remembers the best *observed* point (winner's curse included; repeats averaged)."""

    def __init__(self, name, n, seed, noise, budget):
        rng = np.random.default_rng(seed)
        self.name, self.n, self.budget = name, n, budget
        self.shift = rng.uniform(-4, 4, n)
        self.R = np.linalg.qr(rng.standard_normal((n, n)))[0]
        self.cond = 1e6 ** (np.arange(n) / max(n - 1, 1))
        if name == 'honeypot':  # exploit rate is 0 on ~95% of the space and rises to 1 in one small region
            Y = 10 * rng.uniform(0, 1, (512, n)) - 5 - self.shift
            self.rho = float(np.quantile(np.linalg.norm(Y @ self.R[:min(n, HP_DIMS)].T, axis=1), HP_FRAC))
        self.scale = float(np.median([self.true(x) for x in rng.uniform(0, 1, (16, n))]))
        self.noise_sd = noise * self.scale
        self.rng = np.random.default_rng(seed + 10_000)
        self.evals, self.best_y, self.best_x, self.obs = 0, np.inf, None, {}

    def exploit_rate(self, x):
        z = self.R[:min(self.n, HP_DIMS)] @ (10 * np.asarray(x) - 5 - self.shift)
        return max(0.0, 1 - np.linalg.norm(z) / self.rho) ** 2

    def true(self, x):
        if self.name == 'honeypot':
            return 1 - self.exploit_rate(x)
        return raw_f(self.name, self.R @ (10 * np.asarray(x) - 5 - self.shift), self.cond)

    deadline = None  # wall-clock cap (time.time()), set by run_one for slow optimizers

    def __call__(self, x):
        if self.evals >= self.budget:
            return np.inf  # over budget: ignored (lets CMA-ES finish a generation)
        if self.deadline is not None and time.time() > self.deadline:
            raise TimeLimit
        self.evals += 1
        if self.name == 'honeypot' and self.noise_sd > 0:  # measured like a honeypot: exploits out of N episodes
            y = 1 - self.rng.binomial(HP_TRIALS, self.exploit_rate(x)) / HP_TRIALS
        else:
            y = self.true(x) + self.noise_sd * self.rng.standard_normal()
        self.record(x, y)
        return y

    def score(self):
        """log10 of true regret at the recommended point, relative to a typical random point. Lower is better."""
        return float(np.log10(max(self.true(self.best_x), 1e-10) / self.scale))


class TestProblem:
    """Callable on a params dict {'x0': .., 'x1': ..} with values in [0, 1]. Lower is better; optimum is 0."""

    def __init__(self, family, n, seed=0, noise=0.0):
        if family not in FAMILIES:
            raise ValueError(f'unknown family {family!r}; choose from {FAMILIES}')
        if family == 'rosenbrock' and n < 2:
            raise ValueError('rosenbrock needs n >= 2')
        rng = np.random.default_rng(seed)
        self.family, self.n, self.noise = family, n, noise
        self.shift = rng.uniform(-4, 4, n)
        self.R = np.linalg.qr(rng.standard_normal((n, n)))[0]
        self.cond = 1e6 ** (np.arange(n) / max(n - 1, 1))
        if family == 'plateau':
            Y = 10 * rng.uniform(0, 1, (512, n)) - 5 - self.shift
            self.rho = float(np.quantile(np.linalg.norm(Y @ self.R[:min(n, PLATEAU_DIMS)].T, axis=1), PLATEAU_LIVE))
        self.scale = float(np.median([self._true_x(x) for x in rng.uniform(0, 1, (16, n))]))  # typical random value
        self._rng = np.random.default_rng(seed + 10_000)
        self.space = {f'x{i}': (0.0, 1.0) for i in range(n)}

    def _x(self, params):
        return np.array([params[f'x{i}'] for i in range(self.n)], float)

    def _rate(self, x):
        z = self.R[:min(self.n, PLATEAU_DIMS)] @ (10 * x - 5 - self.shift)
        return max(0.0, 1 - np.linalg.norm(z) / self.rho) ** 2

    def _true_x(self, x):
        if self.family == 'plateau':
            return 1 - self._rate(x)
        return float(raw_f(self.family, self.R @ (10 * x - 5 - self.shift), self.cond))

    def true(self, params):
        return self._true_x(self._x(params))

    def __call__(self, params):
        x = self._x(params)
        if self.noise <= 0:
            return self._true_x(x)
        if self.family == 'plateau':  # measured as k successes out of N trials
            return 1 - self._rng.binomial(PLATEAU_TRIALS, self._rate(x)) / PLATEAU_TRIALS
        return self._true_x(x) + self.noise * self.scale * self._rng.standard_normal()

    def regret(self, params):
        """log10(true value at params / typical random value): the benchmark's score. Lower is better."""
        return float(np.log10(max(self.true(params), 1e-10) / self.scale))


def make_problem(family, n, seed=0, noise=0.0):
    """Returns (fn, space): fn(params) -> value to minimize, space = {'x0': (0, 1), ...}."""
    p = TestProblem(family, n, seed, noise)
    return p, p.space
