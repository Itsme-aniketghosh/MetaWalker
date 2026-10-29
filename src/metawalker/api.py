"""minimize(): one entry point for every optimizer in the package."""
import math
from dataclasses import dataclass, field

import numpy as np

from .baselines import optuna_search, random_search
from .cma import cmaes
from .policy import POLICIES, load_policy
from .space import Space
from .walker import WALKER_RULE, navigate

METHODS = ('auto', 'rlcma', 'walker', 'walker_hp', 'walker_hp_warm', 'walker_rule', 'cmaes', 'cmaes_rule', 'gp', 'tpe',
           'random', 'panel')
WALKERS = ('walker', 'walker_hp', 'walker_rule')
CMAS = ('rlcma', 'cmaes', 'cmaes_rule')
PANEL = ('rlcma', 'walker_hp', 'cmaes')
AUTO_HIGH_DIM = 20   # benchmark: CMA-ES wins below ~20 dims, RL-controlled CMA-ES wins from ~50 up
AUTO_GP_MAX_DIM = 200  # benchmark: Optuna GP is strongest at 50-200 dims when its ~1 h/run overhead is affordable


@dataclass
class Result:
    best_params: dict
    best_value: float
    n_evals: int
    method: str
    history: list = field(repr=False, default_factory=list)  # [(params, value)] in evaluation order
    n_failed: int = 0
    recheck_values: list = field(default_factory=list)  # fresh re-evaluations of best_params (noisy objectives)
    maximize: bool = False  # direction of the search; certify() and importance() read it


class _Objective:
    """Budgeted, sign-normalized view of the user's function on the unit box."""

    def __init__(self, fn, space, budget, maximize, cache=True):
        self.fn, self.space, self.budget, self.sign = fn, space, budget, (-1.0 if maximize else 1.0)
        self.n, self.evals = space.n, 0
        self.history, self.n_failed = [], 0
        self.best_y, self.best_params = math.inf, None
        self.cache, self.n_cached = ({} if cache else None), 0

    def __call__(self, x):
        if self.evals >= self.budget:
            return math.inf  # over budget: ignored (lets CMA-ES finish a generation)
        params = self.space.decode(x)
        key = repr(sorted(params.items())) if self.cache is not None else None
        if key is not None and key in self.cache and self.n_cached < 50 * self.budget:  # cap: no endless loop
            self.n_cached += 1
            return self.cache[key]
        self.evals += 1
        try:
            v = float(self.fn(params))
        except Exception:  # an invalid configuration is a failed trial, not a crashed search
            v = math.nan
        if not math.isfinite(v):
            self.n_failed += 1
            self.history.append((params, v))
            if key:
                self.cache[key] = math.inf
            return math.inf
        self.history.append((params, v))
        y = self.sign * v
        if y < self.best_y:
            self.best_y, self.best_params = y, params
        if key:
            self.cache[key] = y
        return y


def search_rng(seed):
    """The optimizer's random stream. Salted so that seed=k never replays default_rng(k), which a test problem
    built with the same k uses to place its optimum (unsalted, the first random point lands near it)."""
    return np.random.default_rng(None if seed is None else [int(seed), 0x6E6176])


def _has_optuna():
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


class _PadTo2:
    """pycma needs >= 2 dimensions: add a dummy coordinate the objective ignores."""

    def __init__(self, obj):
        self.obj, self.n = obj, 2

    budget = property(lambda self: self.obj.budget)
    evals = property(lambda self: self.obj.evals)

    def __call__(self, x):
        return self.obj(x[:1])


def _warmup(obj, rng, k):
    """Evaluate k Latin-hypercube points; return them sorted best first, with their values."""
    X = (rng.permuted(np.tile(np.arange(k), (obj.n, 1)), axis=1).T + rng.uniform(size=(k, obj.n))) / k
    F = np.array([obj(x) for x in X])
    order = np.argsort(F, kind='stable')
    return X[order], F[order]


def _run(method, obj, rng, warmup=0.0):
    if method == 'walker_hp_warm':
        method, warmup = 'walker_hp', warmup or 0.2
    if method not in METHODS[1:-1]:
        raise ValueError(f'unknown method {method!r}; choose from {METHODS}')
    if not 0 <= warmup < 1:
        raise ValueError('warmup must be in [0, 1)')
    if obj.n == 1 and method in CMAS:
        obj = _PadTo2(obj)
    kw = {}
    if warmup > 0:
        k = max(1, round(warmup * obj.budget))
        if method in WALKERS:
            k = max(k, 4)  # every walker needs a start point; M = 4 for all shipped walkers
        X, F = _warmup(obj, rng, min(k, obj.budget))
        if method in WALKERS and len(X) >= 4:
            kw = dict(x0=X[:4], f0=F[:4])
        elif method in CMAS:
            kw = dict(x0=X[0])
    if method in POLICIES:
        theta, cfg = load_policy(method)
        (navigate if cfg['kind'] == 'walker' else cmaes)(obj, rng, theta, cfg, **kw)
    elif method == 'walker_rule':
        navigate(obj, rng, rule=WALKER_RULE, **kw)
    elif method in ('cmaes', 'cmaes_rule'):
        cmaes(obj, rng, rule=method == 'cmaes_rule', **kw)
    elif method in ('tpe', 'gp'):
        optuna_search(obj, rng, method)
    else:
        random_search(obj, rng)


def minimize(fn, space, budget=200, method='auto', maximize=False, seed=None, recheck=0, expensive=False,
             warmup=0.0, cache=True):
    """Optimize fn(params: dict) -> float over `space` with `budget` evaluations.

    method: 'rlcma', 'walker', 'walker_hp' (walker trained with plateau-shaped problems too), 'cmaes',
            'walker_rule' / 'cmaes_rule' (hand rules distilled from the learned walker / rlcma),
            'walker_hp_warm' (walker_hp with warmup=0.2: for rare-signal objectives),
            'gp' / 'tpe' (Optuna; needs metawalker[optuna]), 'random';
            'auto': CMA-ES below 20 parameters, RL-controlled CMA-ES from 20 up, or Optuna GP up to 200
                    parameters when expensive=True (each evaluation costs far more than GP's own overhead);
            a list of methods, e.g. ['rlcma', 'gp', 'cmaes']: your own panel; the budget is split across them
                    and the best result is kept; 'panel' = ['rlcma', 'walker_hp', 'cmaes'].
    fn may raise or return nan/inf for invalid configurations; those count as failed trials.
    warmup: fraction of the budget spent first on Latin-hypercube points (counted in the budget and history);
            walkers then start at the best 4 of them, CMA-based methods at the best one, other methods just use
            the rest of the budget. Helps when the signal is rare (a small live region on a flat objective).
    recheck: re-evaluate the best configuration this many extra times (outside the budget) and report the
             values, because on noisy objectives the best observed value is biased upward (winner's curse).
    cache: a configuration that decodes to one already tried (common with Int / Categorical parameters) reuses its
           value instead of calling fn again, and costs no budget. Pass cache=False to re-sample a noisy fn.
    """
    if budget < 1:
        raise ValueError('budget must be >= 1')
    sp = space if isinstance(space, Space) else Space(space)
    rng = search_rng(seed)
    if method == 'auto':
        if expensive and sp.n <= AUTO_GP_MAX_DIM and _has_optuna():
            method = 'gp'
        else:
            method = 'rlcma' if sp.n >= AUTO_HIGH_DIM else 'cmaes'
    panel = PANEL if method == 'panel' else method if isinstance(method, (list, tuple)) else None
    if panel is not None:
        bad = [m for m in panel if m not in METHODS or m in ('auto', 'panel')]
        if not panel or bad:
            raise ValueError(f'panel methods must be from {METHODS[1:-1]}; got {list(panel)}')
        # shortcut: equal split, no racing; upgrade to successive halving across methods if budgets get large
        results = []
        for i, m in enumerate(panel):
            share = budget // len(panel) + (1 if i < budget % len(panel) else 0)
            results.append(minimize(fn, sp, max(share, 1), m, maximize, int(rng.integers(2 ** 31)), warmup=warmup,
                                    cache=cache))
        best = min(results, key=lambda r: (-r.best_value if maximize else r.best_value)
                   if r.best_params is not None else math.inf)
        out = Result(best.best_params, best.best_value, sum(r.n_evals for r in results), f'panel:{best.method}',
                     [h for r in results for h in r.history], sum(r.n_failed for r in results))
    else:
        obj = _Objective(fn, sp, budget, maximize, cache)
        _run(method, obj, rng, warmup)
        out = Result(obj.best_params, obj.sign * obj.best_y if obj.best_params is not None else math.nan,
                     obj.evals, method, obj.history, obj.n_failed)
    out.maximize = maximize
    if recheck and out.best_params is not None:
        out.recheck_values = [float(fn(out.best_params)) for _ in range(recheck)]
    return out
