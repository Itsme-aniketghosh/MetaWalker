"""certify(): re-evaluate the top candidates of a search and pick by mean, with a bootstrap CI.

On noisy objectives the single best observed value is biased (the winner's curse): the search keeps whichever
reading was luckiest. Re-measuring the top few candidates and choosing by mean undoes most of that bias.
"""
import math
from dataclasses import dataclass, field

import numpy as np

N_BOOT = 2000


@dataclass
class Certified:
    best_params: dict
    mean: float
    ci_low: float
    ci_high: float
    n_extra_evals: int
    shrinkage: float  # original best observed value - certified mean (> 0 when maximizing means it was inflated)
    maximize: bool
    confidence: float
    n_failed: int = 0
    table: list = field(repr=False, default_factory=list)  # one dict per candidate, best observed first

    def summary(self):
        lines = [f'certified best ({"max" if self.maximize else "min"}imize): mean {self.mean:.4g}, '
                 f'{self.confidence:.0%} CI [{self.ci_low:.4g}, {self.ci_high:.4g}], '
                 f'shrinkage {self.shrinkage:+.4g}, {self.n_extra_evals} extra evals ({self.n_failed} failed)',
                 f'  {"observed":>10} {"mean":>10} {"std":>10} {"ci":>23} {"n":>4} {"fail":>4}  params']
        for row in self.table:
            mark = '*' if row['params'] == self.best_params else ' '
            ci = f'[{row["ci"][0]:.4g}, {row["ci"][1]:.4g}]'
            lines.append(f'{mark} {row["observed"]:>10.4g} {row["mean"]:>10.4g} {row["std"]:>10.4g} {ci:>23} '
                         f'{row["n"]:>4} {row["n_failed"]:>4}  {row["params"]}')
        return '\n'.join(lines)


def _eval(fn, params):
    try:
        v = float(fn(dict(params)))
    except Exception:  # a crash on re-evaluation is a failed repeat, as in minimize()
        return math.nan
    return v if math.isfinite(v) else math.nan


def _stats(vals, rng, confidence):
    v = np.array([x for x in vals if not math.isnan(x)])
    if v.size == 0:
        return 0, math.nan, math.nan, (math.nan, math.nan)
    std = float(v.std(ddof=1)) if v.size > 1 else 0.0
    boot = rng.choice(v, (N_BOOT, v.size)).mean(axis=1)
    lo, hi = np.percentile(boot, [50 * (1 - confidence), 50 * (1 + confidence)])
    return int(v.size), float(v.mean()), std, (float(lo), float(hi))


def certify(fn, result, top_k=5, repeats=5, maximize=None, seed=None, confidence=0.95, adaptive=False):
    """Re-evaluate the top_k distinct params of result.history `repeats` times each; pick the best mean.

    maximize: None reads result.maximize if the result has that attribute, otherwise assumes minimization,
              so pass maximize=True for a search run with minimize(..., maximize=True).
    adaptive: successive halving with the same budget of top_k * repeats extra evaluations: every candidate
              starts at 2 repeats, the worse half is dropped, and survivors are topped up to double the
              repeats each round until the budget is spent. The winner is picked among the final survivors.
    Failed repeats (an exception or nan/inf) are excluded from the mean and counted in n_failed.
    seed seeds only the bootstrap; fn's own noise is up to fn.
    """
    if top_k < 1 or repeats < 1:
        raise ValueError('top_k and repeats must be >= 1')
    if not 0 < confidence < 1:
        raise ValueError('confidence must be in (0, 1)')
    if maximize is None:
        maximize = bool(getattr(result, 'maximize', False))
    sign = -1.0 if maximize else 1.0
    seen, cands = set(), []
    for params, v in sorted((h for h in result.history if math.isfinite(h[1])), key=lambda h: sign * h[1]):
        key = repr(sorted(params.items()))  # repr: categorical values need not be hashable
        if key not in seen:
            seen.add(key)
            cands.append((params, v))
            if len(cands) == top_k:
                break
    if not cands:
        raise ValueError('result.history has no finite values to certify')
    rng = np.random.default_rng(seed)
    vals = [[] for _ in cands]
    budget = top_k * repeats
    if not adaptive:
        for i, (p, _) in enumerate(cands):
            vals[i] = [_eval(fn, p) for _ in range(repeats)]
        alive = list(range(len(cands)))
    else:
        alive, target, spent = list(range(len(cands))), 2, 0
        while spent < budget:
            for i in alive:
                while len(vals[i]) < target and spent < budget:
                    vals[i].append(_eval(fn, cands[i][0]))
                    spent += 1
            if len(alive) > 1:
                alive = sorted(alive, key=lambda i: _rank(vals[i], sign))[:(len(alive) + 1) // 2]
            target *= 2
    table = []
    for (p, obs), vs in zip(cands, vals):
        n, mean, std, ci = _stats(vs, rng, confidence)
        table.append({'params': p, 'observed': obs, 'n': n, 'n_failed': len(vs) - n, 'mean': mean, 'std': std,
                      'ci': ci, 'values': vs})
    best = table[min(alive, key=lambda i: _rank(vals[i], sign))]
    return Certified(best['params'], best['mean'], best['ci'][0], best['ci'][1], sum(len(v) for v in vals),
                     cands[0][1] - best['mean'], maximize, confidence, sum(r['n_failed'] for r in table), table)


def _rank(vs, sign):
    # best mean first, ties broken by lower std; candidates with no valid repeat go last
    v = [x for x in vs if not math.isnan(x)]
    return (sign * float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) if v else (math.inf, math.inf)
