"""minimize(): one entry point for every optimizer in the package."""
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import rlbo as _rlbo
from .baselines import optuna_search, random_search
from .cma import cmaes
from .policy import POLICIES, load_policy, nparam
from .space import Space
from .walker import WALKER_RULE, navigate

METHODS = ('auto', 'rlcma', 'walker', 'walker_hp', 'walker_hp_warm', 'walker_rule', 'cmaes', 'cmaes_rule', 'rlbo', 'gp',
           'tpe', 'random', 'panel')
WALKERS = ('walker', 'walker_hp', 'walker_rule')
CMAS = ('rlcma', 'cmaes', 'cmaes_rule')
PANEL = ('rlcma', 'walker_hp', 'cmaes')
CUSTOM = {}  # your own optimizers by method name: see register()
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
    allocation: dict = field(default_factory=dict)  # panels: evaluations each method got
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


def _rlbo_policy_path():
    return Path(__file__).parent / 'policies' / 'rlbo_s0.npy'


def _load_rlbo():
    path = _rlbo_policy_path()
    if not path.exists():
        raise FileNotFoundError(f"method='rlbo' needs the trained policy {path.name}, which this build of "
                                "metawalker does not include yet")
    theta = np.load(path)
    assert theta.shape == (_rlbo.NPARAM,), ('rlbo', theta.shape)
    return theta


RLBO_CFG = {'kind': 'rlbo'}
KINDS = ('walker', 'rlcma', 'rlbo')  # the learned policies train() can fine-tune and register() accepts


def _policy_size(cfg):
    return _rlbo.NPARAM if cfg.get('kind') == 'rlbo' else nparam(cfg)


def _run_policy(obj, rng, theta, cfg, **kw):
    """Run a learned policy of any kind: a walker, RL-controlled CMA-ES, or rlbo."""
    if cfg['kind'] == 'walker':
        navigate(obj, rng, theta, cfg, **kw)
    elif cfg['kind'] == 'rlcma':
        cmaes(obj if obj.n > 1 else _PadTo2(obj), rng, theta, cfg, **kw)
    else:
        obj.cache = None  # rlbo indexes its own history by evaluation count, so every call must be an evaluation
        _rlbo.rlbo(obj, rng, theta)


def register(name, optimizer):
    """Add your own optimizer under a method name. After register('mine', opt), method='mine' works everywhere a
    built-in name does: minimize(), panels such as ['mine', 'cmaes'], compare(), MetaWalkerCV and AskTell.

    optimizer is either
      - a policy you trained with train() (a walker, rlcma or rlbo): the TrainedPolicy, or the .npz path it
        was saved to; or
      - any function optimizer(f, n, budget, rng) that calls f(x) on points x in the unit cube [0, 1]^n until
        f.evals reaches budget. Lower is better, and calls past the budget return inf. Decoding into your space,
        failed trials, caching, warmup and recheck are handled for you. For example, a 5-level grid search:

            def grid(f, n, budget, rng):
                for x in itertools.product(np.linspace(0, 1, 5), repeat=n):
                    if f.evals >= budget:
                        break
                    f(np.array(x))

    compare(n_jobs > 1) sends registered optimizers to its worker processes, so functions must be defined at the
    top level of a module (picklable), like the objective itself.
    """
    from .train import TrainedPolicy  # train imports this module, so import here
    if not isinstance(name, str) or name in METHODS:
        raise ValueError(f'{name!r} must be a new name, not one of the built-in methods {METHODS}')
    if isinstance(optimizer, (str, os.PathLike)):
        optimizer = TrainedPolicy.load(optimizer)
    if isinstance(optimizer, TrainedPolicy):
        if optimizer.cfg.get('kind') not in KINDS or optimizer.theta.shape != (_policy_size(optimizer.cfg),):
            raise ValueError(f'a registered policy must be one of {KINDS} from train()')
    elif not callable(optimizer):
        raise TypeError('optimizer must be a TrainedPolicy, a path to one, or a function optimizer(f, n, budget, rng)')
    CUSTOM[name] = optimizer


def _set_custom(custom):
    """compare(n_jobs > 1) worker initializer: the parent's registered optimizers."""
    CUSTOM.update(custom)


def _run(method, obj, rng, warmup=0.0):
    if method == 'walker_hp_warm':
        method, warmup = 'walker_hp', warmup or 0.2
    custom = CUSTOM.get(method)
    if method not in METHODS[1:-1] and custom is None:
        raise ValueError(f'unknown method {method!r}; choose from {METHODS + tuple(CUSTOM)} (or register() your own)')
    kind = custom.cfg['kind'] if hasattr(custom, 'theta') else None
    walker, cma = method in WALKERS or kind == 'walker', method in CMAS or kind == 'rlcma'
    M = custom.cfg['M'] if kind == 'walker' else 4  # walkers per search; 4 for every shipped walker
    if not 0 <= warmup < 1:
        raise ValueError('warmup must be in [0, 1)')
    if obj.n == 1 and cma:
        obj = _PadTo2(obj)
    kw = {}
    if warmup > 0:
        k = max(1, round(warmup * obj.budget))
        if walker:
            k = max(k, M)  # every walker needs a start point
        X, F = _warmup(obj, rng, min(k, obj.budget))
        if walker and len(X) >= M:
            kw = dict(x0=X[:M], f0=F[:M])
        elif cma:
            kw = dict(x0=X[0])
    if custom is not None:
        if kind:
            _run_policy(obj, rng, custom.theta, custom.cfg, **kw)
        else:
            custom(obj, obj.n, obj.budget, rng)
    elif method in POLICIES:
        _run_policy(obj, rng, *load_policy(method), **kw)
    elif method == 'walker_rule':
        navigate(obj, rng, rule=WALKER_RULE, **kw)
    elif method in ('cmaes', 'cmaes_rule'):
        cmaes(obj, rng, rule=method == 'cmaes_rule', **kw)
    elif method == 'rlbo':
        _run_policy(obj, rng, _load_rlbo(), RLBO_CFG)
    elif method in ('tpe', 'gp'):
        optuna_search(obj, rng, method)
    else:
        random_search(obj, rng)


def minimize(fn, space, budget=200, method='auto', maximize=False, seed=None, recheck=0, expensive=False,
             warmup=0.0, cache=True, n_jobs=1, allocate='split'):
    """Optimize fn(params: dict) -> float over `space` with `budget` evaluations.

    method: 'rlcma', 'walker', 'walker_hp' (walker trained with plateau-shaped problems too), 'cmaes',
            'walker_rule' / 'cmaes_rule' (hand rules distilled from the learned walker / rlcma),
            'walker_hp_warm' (walker_hp with warmup=0.2: for rare-signal objectives),
            'rlbo' (learned acquisition over cheap surrogates; needs policies/rlbo_s0.npy),
            or any name you added with register() (your own trained walker, grid search, ...),
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
    allocate: how a panel shares compute.
            'split' (default): the budget is split equally, so a panel costs what one method does.
            'each': every method gets the whole budget and the best result is kept: never worse than the panel's
                    best method, but it costs len(panel) x budget evaluations (with n_jobs, the same wall-clock).
            'race': successive halving: all methods start with an equal slice, the worse half is dropped after
                    each round and the survivors share the rest, so the better methods get more of the budget
                    (3 methods, budget 200: the winner gets ~123). The first round is short (~22 each here), so
                    it favors methods that start fast; a population-based method (CMA-ES) may be dropped before
                    it gets going. On a flat objective where nothing has scored yet, ties keep the earlier-listed
                    method.
    n_jobs: with a panel, run up to this many of its methods at the same time, in threads. Each method keeps its own
            share, seed and history, so the result is the same as with n_jobs=1. It speeds things up when fn
            spends its time outside Python (training with torch/jax, a subprocess, a remote job); fn must then be
            safe to call from several threads at once.
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
        bad = [m for m in panel if (m not in METHODS and m not in CUSTOM) or m in ('auto', 'panel')]
        if not panel or bad:
            raise ValueError(f'panel methods must be from {METHODS[1:-1] + tuple(CUSTOM)}; got {list(panel)}')
        if allocate not in ('split', 'each', 'race'):
            raise ValueError(f"allocate must be 'split', 'each' or 'race', got {allocate!r}")
        seeds = [int(rng.integers(2 ** 31)) for _ in panel]  # drawn up front: results don't depend on n_jobs
        if allocate == 'race':
            from .asktell import race
            params, value, alloc, hist = race(fn, sp, budget, list(panel), maximize, seeds, n_jobs, warmup=warmup,
                                              cache=cache)
            winner = max(panel, key=lambda m: alloc[m])  # the method that got the most evaluations
            out = Result(params, value, sum(alloc.values()), f'race:{winner}', [h for hs in hist for h in hs],
                         sum(1 for hs in hist for _, v in hs if not math.isfinite(v)), allocation=alloc)
        else:
            def member(i):
                share = budget if allocate == 'each' else budget // len(panel) + (1 if i < budget % len(panel) else 0)
                return minimize(fn, sp, max(share, 1), panel[i], maximize, seeds[i], warmup=warmup, cache=cache)

            if n_jobs > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(min(n_jobs, len(panel))) as ex:
                    results = list(ex.map(member, range(len(panel))))
            else:
                results = [member(i) for i in range(len(panel))]
            best = min(results, key=lambda r: (-r.best_value if maximize else r.best_value)
                       if r.best_params is not None else math.inf)
            out = Result(best.best_params, best.best_value, sum(r.n_evals for r in results), f'panel:{best.method}',
                         [h for r in results for h in r.history], sum(r.n_failed for r in results),
                         allocation={m: r.n_evals for m, r in zip(panel, results)})
    else:
        obj = _Objective(fn, sp, budget, maximize, cache)
        _run(method, obj, rng, warmup)
        out = Result(obj.best_params, obj.sign * obj.best_y if obj.best_params is not None else math.nan,
                     obj.evals, method, obj.history, obj.n_failed)
    out.maximize = maximize
    if recheck and out.best_params is not None:
        out.recheck_values = [float(fn(out.best_params)) for _ in range(recheck)]
    return out
