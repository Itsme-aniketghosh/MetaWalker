"""recommend(): probe an objective cheaply, then pick the optimizer the benchmark says suits it.

Decision rules (from our benchmark, checked in this order):
  plateau fraction > 0.5        -> 'walker_hp_warm'  learned walkers beat CMA-ES/TPE/GP/SMAC on plateaus; the random
                                   warm-up finds rare live regions (9/10 vs 1-5/10 without, at 0.5% live)
  seconds/eval >= 1.0, <= 200 params, optuna installed
                                -> 'gp'         Optuna GP tied our best overall and was better at 100-200 dims,
                                                but costs ~1 h of its own compute per 1000 evaluations
  >= 20 params                  -> 'rlcma'      beat CMA-ES on 72-99% of problems from 50 dims up, TPE/SMAC on 90%+
  otherwise                     -> 'cmaes'      best below ~20 noisy dims
"""
import math
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .api import AUTO_GP_MAX_DIM, AUTO_HIGH_DIM, _has_optuna, _Objective, minimize, search_rng
from .space import Space

PLATEAU_FRAC = 0.5    # benchmark: plateau objectives are where the learned walkers win
EXPENSIVE_SEC = 1.0   # seconds per evaluation from which GP's own overhead stops mattering
MIN_PROBE = 12


@dataclass
class Recommendation:
    method: str
    reasons: list
    profile: dict
    probe_history: list = field(repr=False, default_factory=list)  # [(params, value)] of the probe evaluations


def _probe_size(budget, probe_frac):
    return min(budget, max(MIN_PROBE, int(probe_frac * budget)))


def recommend(fn, space, budget, probe_frac=0.08, seed=None, maximize=False):
    """Spend at most max(12, probe_frac * budget) evaluations of fn probing it, then pick a method.

    The probe measures, on random points: the plateau fraction (share of values within a tiny tolerance of the
    most common value), the spread (std) and the failure rate (exceptions / nan / inf); on 2-3 points evaluated
    again: the noise level relative to the spread; on every call: seconds per evaluation.
    The method is then chosen by the rules our benchmark established (see the module docstring).
    """
    if budget < 1:
        raise ValueError('budget must be >= 1')
    sp = space if isinstance(space, Space) else Space(space)
    rng = search_rng(seed)
    n_probe = _probe_size(budget, probe_frac)
    n_rep = min(6, n_probe // 4)  # re-evaluations, spread over 3 of the random points
    X = rng.uniform(0, 1, (n_probe - n_rep, sp.n))
    obj, secs = _Objective(fn, sp, n_probe, maximize, cache=False), []  # the repeats measure noise
    for x in list(X) + [X[i % 3] for i in range(n_rep)]:
        t = perf_counter()
        obj(x)
        secs.append(perf_counter() - t)

    vals = np.array([v for _, v in obj.history])
    rand, rep = vals[:len(X)], vals[len(X):]
    ok = rand[np.isfinite(rand)]
    if len(ok):
        tol = 1e-9 * max(1.0, float(np.abs(ok).max()))
        counts = (np.abs(ok[:, None] - ok[None, :]) <= tol).sum(1)
        plateau, mode, spread = counts.max() / len(ok), float(ok[counts.argmax()]), float(ok.std())
    else:
        plateau, mode, spread = 0.0, math.nan, math.nan
    diffs = [abs(rep[i] - rand[i % 3]) for i in range(n_rep) if math.isfinite(rep[i]) and math.isfinite(rand[i % 3])]
    noise = float(np.mean(diffs)) / math.sqrt(2) if diffs else math.nan  # |a - b| / sqrt(2) ~ noise std
    sec = float(np.median(secs))
    profile = dict(n_params=sp.n, n_probe=obj.evals, plateau_frac=float(plateau), mode=mode, spread=spread,
                   failure_rate=obj.n_failed / obj.evals, noise_rel=noise / spread if spread > 0 else 0.0,
                   sec_per_eval=sec)

    reasons = []
    if plateau > PLATEAU_FRAC:
        method = 'walker_hp_warm'
        reasons.append(f'{plateau:.0%} of {len(ok)} random probes returned the same value ({mode:g}): a plateau '
                       f'objective, where learned walkers beat CMA-ES/TPE/GP/SMAC in our benchmark')
    else:
        reasons.append(f'only {plateau:.0%} of random probes share one value: not a plateau')
        if sec >= EXPENSIVE_SEC and sp.n <= AUTO_GP_MAX_DIM and _has_optuna():
            method = 'gp'
            reasons.append(f'{sec:.2f} s per evaluation with {sp.n} parameters (<= {AUTO_GP_MAX_DIM}): evaluations '
                           f'are expensive enough that Optuna GP (~1 h own compute per 1000 evals) pays off')
        elif sp.n >= AUTO_HIGH_DIM:
            method = 'rlcma'
            reasons.append(f'{sp.n} parameters (>= {AUTO_HIGH_DIM}) at {sec:.3g} s per evaluation: RL-controlled '
                           f'CMA-ES beat CMA-ES on 72-99% and TPE/SMAC on 90%+ of benchmark problems from 50 dims up')
        else:
            method = 'cmaes'
            reasons.append(f'{sp.n} parameters (< {AUTO_HIGH_DIM}) at {sec:.3g} s per evaluation: CMA-ES was best '
                           f'below ~20 dims in our benchmark')
    return Recommendation(method, reasons, profile, obj.history)


def minimize_auto(fn, space, budget, probe_frac=0.08, seed=None, maximize=False, recheck=0, **kw):
    """recommend(), then minimize() with the chosen method on the rest of the budget.

    Returns the Result over all `budget` evaluations (probe included), with the Recommendation as .recommendation.
    Extra keyword arguments go to minimize().
    """
    rec = recommend(fn, space, budget, probe_frac, seed, maximize)
    rest = budget - rec.profile['n_probe']
    if rest < 1:
        raise ValueError(f'budget {budget} leaves nothing after the {rec.profile["n_probe"]}-evaluation probe')
    res = minimize(fn, space, rest, rec.method, maximize=maximize, seed=seed, **kw)
    sign = -1.0 if maximize else 1.0
    for params, v in rec.probe_history:  # the probe may have found the best point
        if math.isfinite(v) and (res.best_params is None or sign * v < sign * res.best_value):
            res.best_params, res.best_value = params, v
    res.n_evals += rec.profile['n_probe']
    res.history = rec.probe_history + res.history
    res.n_failed += sum(1 for _, v in rec.probe_history if not math.isfinite(v))
    if recheck and res.best_params is not None:
        res.recheck_values = [float(fn(res.best_params)) for _ in range(recheck)]
    res.recommendation = rec
    return res
