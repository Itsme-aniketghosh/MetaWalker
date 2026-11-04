"""train(): fine-tune a learned optimizer (walker, rlcma or rlbo), or train a walker from scratch, on your own
cheap simulated problems.

    from metawalker import make_problem
    from metawalker.train import train, TrainedPolicy

    def plateau40():                      # a top-level function, so n_jobs > 1 can pickle what it returns
        return make_problem('plateau', 40, seed=int(np.random.default_rng().integers(1e9)))

    pol = train([plateau40], generations=50)
    pol.save('my_walker.npz')
    r = TrainedPolicy.load('my_walker.npz').minimize(real_fn, real_space, budget=300)

Training is OpenAI-ES over the policy weights, ported from the benchmark trainer (experiments.benchmark.train, via metawalker.es): antithetic
pairs, centered ranks, Adam, and common random numbers (every candidate in a generation sees the same problem
instances with the same walker seed).

Score of one candidate on one instance (lower is better):
  fn.regret(best_params) if fn has .regret (e.g. metawalker TestProblem: log10 of true value / typical value),
  else the best value found, z-scored per instance across the population so instances of different scale count
  equally. Fitness = -mean score over the generation's instances. maximize=True flips the sign of the value.
"""
import copy
import json
import math
from multiprocessing import Pool

import numpy as np

from .es import es_step
from .policy import load_policy, nparam, unpack
from . import rlbo as _rlbo
from .api import RLBO_CFG, Result, _Objective, _load_rlbo, _run_policy, search_rng
from .space import Space


class TrainedPolicy:
    def __init__(self, theta, cfg, history=()):
        self.theta, self.cfg, self.history = np.asarray(theta, float), dict(cfg), list(history)

    def save(self, path):
        np.savez(path, theta=self.theta, cfg=json.dumps(self.cfg), history=np.array(self.history, float))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(z['theta'], json.loads(str(z['cfg'])), z['history'].tolist())

    def minimize(self, fn, space, budget=200, seed=None, maximize=False):
        """Same contract as metawalker.minimize, using this policy as the optimizer."""
        if budget < 1:
            raise ValueError('budget must be >= 1')
        obj = _Objective(fn, space if isinstance(space, Space) else Space(space), budget, maximize)
        _run_policy(obj, search_rng(seed), self.theta, self.cfg)
        return Result(obj.best_params, obj.sign * obj.best_y if obj.best_params is not None else math.nan,
                      obj.evals, f"trained_{self.cfg['kind']}", obj.history, obj.n_failed, maximize=maximize)


def _init(init, dirs, hidden, walkers, rng):
    if init in ('walker', 'walker_hp'):
        theta, cfg = load_policy(init)
        return theta.copy(), dict(cfg, M=walkers)
    if init == 'rlcma':
        theta, cfg = load_policy(init)
        return theta.copy(), dict(cfg)
    if init == 'rlbo':
        return _load_rlbo().copy(), dict(RLBO_CFG)
    if init != 'scratch':
        raise ValueError(f"init must be one of 'walker_hp', 'walker', 'rlcma', 'rlbo', 'scratch', got {init!r}")
    if dirs not in ('cov', 'mom'):
        raise ValueError(f"dirs must be 'cov' or 'mom', got {dirs!r}")
    cfg = dict(kind='walker', dirs=dirs, H=hidden, M=walkers)
    theta = np.zeros(nparam(cfg))
    W1 = unpack(theta, cfg)['W1']  # a view into theta
    W1[:] = 0.5 * rng.standard_normal(W1.shape)  # so ES perturbations of the later layers matter
    return theta, cfg


def _mask(cfg, size):
    """Which weights ES moves. rlbo's first layer stays at its random init, as in its original training."""
    if cfg['kind'] != 'rlbo':
        return 1.0
    name, shape = _rlbo.SHAPES[0]
    assert name == 'W1', name
    m = np.ones(size)
    m[:int(np.prod(shape))] = 0.0
    return m


def _scores(args):
    """Top-level (picklable) so multiprocessing workers can run it. Returns (scores, has_regret) per instance."""
    theta, cfg, instances, budget, maximize = args
    out = []
    for fn, space, seed in instances:
        fn = copy.deepcopy(fn)  # each candidate starts from the same instance state (incl. its noise rng)
        obj = _Objective(fn, space if isinstance(space, Space) else Space(space), budget, maximize)
        _run_policy(obj, np.random.default_rng(seed), theta, cfg)
        if obj.best_params is None:
            out.append((math.inf, hasattr(fn, 'regret')))
        elif hasattr(fn, 'regret'):
            out.append((float(fn.regret(obj.best_params)), True))
        else:
            out.append((obj.best_y, False))
    return out


def train(problems, dirs='cov', hidden=8, walkers=4, generations=100, pairs=20, instances_per_gen=8, budget=300,
          sigma=0.1, lr=0.05, seed=0, n_jobs=1, init='walker_hp', callback=None, maximize=False):
    """Train a learned optimizer on fresh problem instances; returns a TrainedPolicy.

    problems: a list of zero-arg factories, each returning a fresh (fn, space), e.g.
              [lambda: make_problem('plateau', 40, seed=random.randrange(10**9))]; each instance picks one
              factory at random. Or a single function rng -> (fn, space) (reproducible with `seed`).
    init: which shipped policy to fine-tune: 'walker_hp', 'walker', 'rlcma' (learned CMA-ES) or 'rlbo'. Fine-tuning
          is much faster and more stable than starting over; dirs, hidden and walkers then come from that policy
          (walkers still sets the walker count). 'scratch' starts a walker from random weights with your dirs/hidden.
          rlbo costs ~20x a walker per evaluation, so keep its training budgets small.
    maximize: True if your functions return something to maximize (a reward, an accuracy).
    budget: evaluations per instance; train with roughly the budget you will search with.
    n_jobs > 1: evaluates the population in a multiprocessing.Pool. The factories run in this process, but the
          (fn, space) they return are pickled to the workers, so fn must be picklable: a TestProblem, an
          instance of a top-level class, or a top-level function; not a lambda or closure. On Windows/macOS
          (spawn), call train() under `if __name__ == '__main__':` in scripts.
    callback(gen, policy): called after every generation with the current TrainedPolicy.
    policy.history: per-generation mean of -score over population and instances (before z-scoring; noisy,
          since every generation draws new instances).
    """
    rng = np.random.default_rng(seed)
    theta, cfg = _init(init, dirs, hidden, walkers, rng)
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    mask = _mask(cfg, len(theta))
    draw = (lambda: problems[rng.integers(len(problems))]()) if isinstance(problems, (list, tuple)) \
        else (lambda: problems(rng))
    history = []
    pool = Pool(n_jobs) if n_jobs > 1 else None
    try:
        for gen in range(generations):
            instances = [(*draw(), int(rng.integers(2 ** 31))) for _ in range(instances_per_gen)]
            eps = rng.standard_normal((pairs, len(theta))) * mask
            tasks = [(c, cfg, instances, budget, maximize) for c in np.concatenate([theta + sigma * eps, theta - sigma * eps])]
            res = pool.map(_scores, tasks, chunksize=1) if pool else list(map(_scores, tasks))
            S = np.array([[s for s, _ in r] for r in res])                   # (candidates, instances)
            has_regret = np.array([h for _, h in res[0]])
            for j in range(S.shape[1]):  # failed runs (no valid evaluation) score worst
                col, ok = S[:, j], np.isfinite(S[:, j])
                col[~ok] = col[ok].max() + 1 if ok.any() else 0.0
            history.append(-float(S.mean()))
            Z = S.copy()
            Z[:, ~has_regret] = (S[:, ~has_regret] - S[:, ~has_regret].mean(0)) / (S[:, ~has_regret].std(0) + 1e-12)
            fit = -Z.mean(1)
            theta, m, v = es_step(theta, m, v, fit, eps, sigma, lr, gen)
            if callback:
                callback(gen, TrainedPolicy(theta, cfg, history))
    finally:
        if pool:
            pool.close(); pool.join()
    return TrainedPolicy(theta, cfg, history)
