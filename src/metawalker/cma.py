"""CMA-ES (IPOP restarts, pycma), optionally steered by a learned policy or the distilled rule, and the two
walker / CMA-ES combinations (handoff: walker first; cma_then_walk: CMA-ES first)."""
import numpy as np

from .policy import STEP_MULT, pick, unpack
from .walker import navigate


def cmaes(prob, rng, theta=None, cfg=None, rule=False, x0=None, sigma0=0.25, inject=None):
    """CMA-ES with IPOP restarts; with theta, a learned policy also sets the step size and restarts.
    rule=True: the distilled rlcma rule, sigma *= 0.8 after a generation that did not improve on this restart's
    best, unless stalled >= 5 generations or the population is flat; never grows, never restarts early.
    x0 / sigma0 / inject (points tried in the first generations) set the first restart only."""
    import cma
    P = unpack(theta, cfg) if theta is not None else None
    popsize, default_pop = None, 4 + int(3 * np.log(prob.n))
    while prob.evals < prob.budget:  # IPOP restarts
        seed = int(rng.integers(1, 2 ** 31))
        # a private RandomState with pycma's own seeding: the same samples as np.random.seed(seed) + np.random.randn,
        # without sharing numpy's global generator, so two CMA-ES runs in parallel threads don't interleave draws
        opts = {'bounds': [0, 1], 'seed': seed, 'randn': np.random.RandomState(seed).randn, 'verbose': -9,
                'maxfevals': prob.budget - prob.evals}
        if popsize: opts['popsize'] = popsize
        es = cma.CMAEvolutionStrategy(rng.uniform(0, 1, prob.n) if x0 is None else x0, sigma0, opts)
        if inject is not None:
            es.inject(list(inject), force=True)
        x0, inject, sigma0 = None, None, 0.25  # later IPOP restarts start fresh
        best, stall = np.inf, 0
        while not es.stop() and prob.evals < prob.budget:
            X = es.ask()
            F = [prob(xi) for xi in X]
            es.tell(X, F)
            if P is None and not rule:
                continue
            Ff = np.array([f for f in F if np.isfinite(f)]) if np.isfinite(F).any() else np.array([best])
            improved = Ff.min() < best
            best = min(best, Ff.min()); stall = 0 if improved else stall + 1
            spread = np.log10((np.median(Ff) - Ff.min()) / (abs(Ff.min()) + 1e-12) + 1e-12)
            g = np.array([1.0, np.log(es.sigma / es.sigma0) / np.log(1e-6), prob.evals / prob.budget,
                          float(improved), min(stall / 10, 3), np.clip(spread / 6, -2, 2),
                          np.log(es.popsize / default_pop)])
            if rule:
                es.sigma = float(np.clip(es.sigma * (0.8 if not improved and stall < 5 and g[5] > -1.2 else 1.0),
                                         1e-12, 1.0))
                continue
            h = np.tanh(P['W1'] @ g + P['b1']) if 'W1' in P else g
            es.sigma = float(np.clip(es.sigma * STEP_MULT[pick(rng, P['Vs'] @ h + P['cs'])], 1e-12, 1.0))
            if pick(rng, P['Vr'] @ h + P['cr']) == 1:
                break  # learned restart
        popsize = 2 * es.popsize


def handoff(prob, rng, theta, cfg, frac=0.3, k=10):
    """The walker explores the first `frac` of the budget, then CMA-ES starts from its experience: mean at the
    best point (mean over repeats), step size from the spread of the k best points, and those points injected."""
    full, prob.budget = prob.budget, round(frac * prob.budget)
    navigate(prob, rng, theta, cfg)
    prob.budget = full
    top = sorted(prob.obs.items(), key=lambda kv: kv[1][0] / kv[1][1])[:k]
    X = np.array([prob.reps[q] for q, _ in top])
    cmaes(prob, rng, x0=prob.best_x, sigma0=float(np.clip(X.std(0).mean(), 0.02, 0.3)), inject=X)


def cma_then_walk(prob, rng, theta, cfg, frac=0.5):
    """CMA-ES explores the first `frac` of the budget, then the walkers start at its M best points (mean over repeats)."""
    full, prob.budget = prob.budget, round(frac * prob.budget)
    cmaes(prob, rng)
    prob.budget = full
    top = sorted(prob.obs.items(), key=lambda kv: kv[1][0] / kv[1][1])[:cfg.get('M', 4)]
    navigate(prob, rng, theta, cfg, x0=[prob.reps[q] for q, _ in top], f0=[t / c for _, (t, c) in top])
