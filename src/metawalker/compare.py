"""compare(): run several methods on your own losses (and/or simulated problems) and see which works.

    from metawalker import compare, make_problem
    res = compare({'my_loss': (my_fn, my_space), 'stand-in': make_problem('plateau', 50)},
                  methods=['rlcma', 'walker_hp', 'cmaes', 'gp'], budget=200, seeds=5)
    print(res.summary())         # average rank, win rates, time; then pick your panel
"""
import math, time
from dataclasses import dataclass

import numpy as np

from .api import minimize


@dataclass
class Comparison:
    rows: list      # one dict per (problem, method, seed): best, score, evals, failed, seconds
    methods: list
    maximize: bool

    def _instances(self):
        inst = {}
        for r in self.rows:
            inst.setdefault((r['problem'], r['seed']), {})[r['method']] = r['score']
        return inst

    def ranks(self):
        """Average rank per method over (problem, seed) instances; 1 = best, ties share the average rank."""
        out = {m: [] for m in self.methods}
        for scores in self._instances().values():
            vals = sorted(scores.values())
            for m, v in scores.items():
                first = vals.index(v)
                out[m].append(first + 1 + (vals.count(v) - 1) / 2)
        return {m: float(np.mean(v)) for m, v in out.items() if v}

    def win_rate(self, a, b):
        """Share of instances where method a beats method b (ties count 1/2)."""
        pairs = [s for s in self._instances().values() if a in s and b in s]
        if not pairs:
            return math.nan
        return float(np.mean([1.0 if s[a] < s[b] else 0.5 if s[a] == s[b] else 0.0 for s in pairs]))

    @property
    def best_method(self):
        r = self.ranks()
        return min(r, key=r.get)

    def summary(self):
        r = self.ranks()
        order = sorted(r, key=r.get)
        lines = [f"{'method':12s} {'avg rank':>9s} {'median score':>13s} {'failed':>7s} {'s/run':>8s}"]
        for m in order:
            rs = [x for x in self.rows if x['method'] == m]
            lines.append(f"{m:12s} {r[m]:9.2f} {np.median([x['score'] for x in rs]):13.4g} "
                         f"{sum(x['failed'] for x in rs):7d} {np.mean([x['seconds'] for x in rs]):8.2f}")
        lines.append('\nwin rate of row vs column (ties 1/2):')
        lines.append(' ' * 12 + ''.join(f'{m[:10]:>11s}' for m in order))
        for a in order:
            lines.append(f'{a:12s}' + ''.join(f'{"":>11s}' if a == b else f'{self.win_rate(a, b):11.2f}' for b in order))
        lines.append('\nscore = log10 regret for simulated problems (known optimum), else the best value found'
                     + (' (sign-flipped so lower is better)' if self.maximize else ''))
        return '\n'.join(lines)


def _job(name, fn, space, m, seed, budget, maximize):
    t0 = time.time()
    res = minimize(fn, space, budget=budget, method=m, maximize=maximize, seed=seed)
    if res.best_params is None:
        score = math.inf  # every evaluation failed
    elif hasattr(fn, 'regret'):
        score = fn.regret(res.best_params)
    else:
        score = -res.best_value if maximize else res.best_value
    return dict(problem=name, method=m, seed=seed, best=res.best_value, score=score,
                evals=res.n_evals, failed=res.n_failed, seconds=time.time() - t0)


def compare(problems, methods=('rlcma', 'walker', 'walker_hp', 'cmaes', 'random'), budget=200, seeds=3,
            maximize=False, n_jobs=1):
    """problems: {name: (fn, space)}; fn(params) -> value. Every method gets the same problems, budget and seeds.

    For simulated problems (from make_problem) the score is the true log10 regret at the best point found,
    so noisy luck isn't rewarded. For your own functions it is the best observed value.

    n_jobs > 1 runs the (problem, seed, method) jobs in that many processes (ProcessPoolExecutor). Then fn and
    space must be picklable (module-level functions or classes; make_problem's problems are), and on Windows /
    macOS the calling script needs an `if __name__ == "__main__":` guard. Each job gets its own copy of fn, so
    results equal n_jobs=1 exactly for deterministic functions; a stateful fn (e.g. a noisy make_problem, whose
    noise stream advances across jobs when run sequentially) sees fresh state in every job.
    """
    if not problems:
        raise ValueError('give at least one problem: {name: (fn, space)}')
    methods = list(methods)
    jobs = [(name, fn, space, m, seed, budget, maximize)
            for name, (fn, space) in problems.items() for seed in range(seeds) for m in methods]
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs))) as ex:
            rows = list(ex.map(_job, *zip(*jobs)))
    else:
        rows = [_job(*j) for j in jobs]
    return Comparison(rows, methods, maximize)
