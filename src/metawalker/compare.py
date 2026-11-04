"""compare(): run several methods on your own losses (and/or simulated problems) and see which works.

    from metawalker import compare, make_problem
    res = compare({'my_loss': (my_fn, my_space), 'stand-in': make_problem('plateau', 50)},
                  methods=['rlcma', 'walker_hp', 'cmaes', 'gp'], budget=200, seeds=5)
    print(res.summary())          # average rank, win rates (* = clear win), time
    print(res.success_table())    # how often and how fast each method reaches a target
    print(res.budget_table())     # rankings at smaller budgets, from the same runs
    panel = res.suggest_panel(3); print(panel.summary())
    minimize(my_fn, my_space, budget=600, method=panel)
    res.plot()                    # share of runs at the target vs evaluations and vs seconds (needs matplotlib)
"""
import math, os, time
from dataclasses import dataclass

import numpy as np

from . import api
from .api import minimize

APPROX = ('approximate: a learned or CMA-based method plans for its full budget, so its prefix of a longer run '
          'is not the same as a run with the smaller budget')


def _boot_ci(values, stat=np.mean, n_boot=1000, seed=0):
    """95% percentile bootstrap interval of stat(values)."""
    values = np.asarray(values, float)
    if len(values) < 2:
        return (math.nan, math.nan)
    idx = np.random.default_rng(seed).integers(0, len(values), (n_boot, len(values)))
    s = np.array([stat(values[i]) for i in idx])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


class PanelSuggestion(list):
    """The methods suggest_panel() picked. It is a list, so pass it straight to minimize(method=...)."""

    def __init__(self, methods, covered, n, best_single, best_single_covered, k, budget):
        super().__init__(methods)
        self.covered, self.n, self.k, self.budget = covered, n, k, budget
        self.best_single, self.best_single_covered = best_single, best_single_covered

    def summary(self):
        return (f'suggested panel: {list(self)}\n'
                f'  reaches the target on {self.covered}/{self.n} runs, each method using {self.budget // self.k} '
                f'evaluations (budget {self.budget} split {self.k} ways)\n'
                f'  best single method with the full budget: {self.best_single} on '
                f'{self.best_single_covered}/{self.n} runs\n'
                f'  ({APPROX})')


@dataclass
class Comparison:
    rows: list      # one dict per (problem, method, seed): best, score, evals, failed, seconds, curve, times
    methods: list
    maximize: bool

    # ---------- final-score views ----------
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

    def win_rate(self, a, b, ci=False):
        """Share of instances where method a beats method b (ties count 1/2). ci=True also returns a 95%
        bootstrap interval over instances: (rate, low, high)."""
        pairs = [s for s in self._instances().values() if a in s and b in s]
        if not pairs:
            return (math.nan, math.nan, math.nan) if ci else math.nan
        w = [1.0 if s[a] < s[b] else 0.5 if s[a] == s[b] else 0.0 for s in pairs]
        return (float(np.mean(w)), *_boot_ci(w)) if ci else float(np.mean(w))

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
        lines.append('\nwin rate of row vs column (ties 1/2; * = clear: the 95% bootstrap interval excludes 0.5):')
        lines.append(' ' * 12 + ''.join(f'{m[:10]:>11s}' for m in order))
        for a in order:
            cells = []
            for b in order:
                if a == b:
                    cells.append(f'{"":>11s}')
                    continue
                w, lo, hi = self.win_rate(a, b, ci=True)
                cells.append(f"{w:10.2f}{'*' if lo > 0.5 or hi < 0.5 else ' '}")
            lines.append(f'{a:12s}' + ''.join(cells))
        lines.append('\nscore = log10 regret for simulated problems (known optimum), else the best value found'
                     + (' (sign-flipped so lower is better)' if self.maximize else ''))
        return '\n'.join(lines)

    # ---------- time to success ----------
    def targets(self, target=None, quantile=0.25):
        """Per-problem target score. Default: the score the best `quantile` of all runs on that problem reached.
        Or pass one number for every problem, or {problem: number}. Scores are lower-is-better."""
        probs = sorted({r['problem'] for r in self.rows})
        if target is None:
            return {p: float(np.quantile([r['score'] for r in self.rows if r['problem'] == p], quantile))
                    for p in probs}
        if isinstance(target, dict):
            return {p: float(target[p]) for p in probs}
        return {p: float(target) for p in probs}

    def _hits(self, target):
        """Per row: index of the evaluation from which the run's pick stays at or below the problem's target
        (None if it ends above it). With noise, a better-looking reading can be a worse point, so a run can dip
        below the target and come back up; it only counts once it stays there, which matches its final score."""
        tg = self.targets(target)
        out = []
        for r in self.rows:
            ok = np.asarray(r['curve']) <= tg[r['problem']]
            if not len(ok) or not ok[-1]:
                out.append(None)
                continue
            bad = np.nonzero(~ok)[0]
            out.append(int(bad[-1]) + 1 if len(bad) else 0)
        return out

    def success(self, target=None):
        """Per method: share of runs that reach the target (with a 95% bootstrap interval), and among those
        runs the median number of evaluations and seconds it took (seconds include the optimizer's own time)."""
        hits = self._hits(target)
        out = {}
        for m in self.methods:
            idx = [i for i, r in enumerate(self.rows) if r['method'] == m]
            ok = [i for i in idx if hits[i] is not None]
            rate = len(ok) / len(idx) if idx else math.nan
            out[m] = dict(rate=rate, ci=_boot_ci([hits[i] is not None for i in idx]), runs=len(idx), hits=len(ok),
                          evals=float(np.median([hits[i] + 1 for i in ok])) if ok else math.nan,
                          seconds=float(np.median([self.rows[i]['times'][hits[i]] for i in ok])) if ok else math.nan)
        return out

    def success_table(self, target=None):
        s = self.success(target)
        order = sorted(s, key=lambda m: (-s[m]['rate'], s[m]['evals'] if s[m]['hits'] else math.inf))
        lines = [f"{'method':12s} {'reached':>9s} {'95% interval':>14s} {'median evals':>13s} {'median s':>9s}"]
        for m in order:
            x = s[m]
            lo, hi = x['ci']
            ci = f'[{lo:.0%}, {hi:.0%}]' if not math.isnan(lo) else ''
            ev = f"{x['evals']:.0f}" if x['hits'] else 'never'
            sec = f"{x['seconds']:.2f}" if x['hits'] else ''
            lines.append(f"{m:12s} {x['hits']:>4d}/{x['runs']:<4d} {ci:>14s} {ev:>13s} {sec:>9s}")
        lines.append('\ntarget: ' + ('the score the best quarter of all runs reached on each problem' if target is None
                                     else str(target)) + '; evals and seconds are medians over the runs that got there')
        return '\n'.join(lines)

    # ---------- smaller budgets, from the same runs ----------
    @property
    def budget(self):
        return max(len(r['curve']) for r in self.rows)

    def at_budget(self, b):
        """The same comparison scored after the first b evaluations of every run (see APPROX)."""
        if not 1 <= b <= self.budget:
            raise ValueError(f'budget must be in [1, {self.budget}]')
        rows = []
        for r in self.rows:
            c, t = r['curve'], r['times']
            j = min(b, len(c)) - 1
            rows.append(dict(r, score=float(c[j]) if j >= 0 else math.inf, evals=j + 1, seconds=t[j] if j >= 0 else 0,
                             curve=c[:j + 1], times=t[:j + 1]))
        return Comparison(rows, self.methods, self.maximize)

    def budget_table(self, budgets=None):
        """Average rank of each method at several budgets (default: 1/8, 1/4, 1/2 and all of the budget)."""
        B = self.budget
        budgets = sorted({max(1, round(B * f)) for f in (1 / 8, 1 / 4, 1 / 2, 1)} if budgets is None else budgets)
        ranks = {b: self.at_budget(b).ranks() for b in budgets}
        order = sorted(self.methods, key=lambda m: ranks[budgets[-1]].get(m, math.inf))
        lines = ['average rank (1 = best) after this many evaluations:',
                 f"{'method':12s}" + ''.join(f'{b:>8d}' for b in budgets)]
        for m in order:
            lines.append(f'{m:12s}' + ''.join(f'{ranks[b].get(m, math.nan):8.2f}' for b in budgets))
        lines.append(f'\n({APPROX})')
        return '\n'.join(lines)

    # ---------- which methods to put in a panel ----------
    def suggest_panel(self, k=3, target=None):
        """Greedy pick of k methods for minimize(method=[...]). A real panel gives each method budget // k
        evaluations, so coverage is scored on each run's first budget // k evaluations: at each step, add the
        method that reaches the target on the most (problem, seed) runs that no chosen method reaches yet
        (ties: better average rank). Returns a PanelSuggestion, a list you can pass to minimize()."""
        if not 1 <= k <= len(self.methods):
            raise ValueError(f'k must be between 1 and {len(self.methods)}')
        tg = self.targets(target)
        share = self.at_budget(max(1, self.budget // k))
        inst = sorted({(r['problem'], r['seed']) for r in self.rows})
        reach = {m: {(r['problem'], r['seed']) for r in share.rows if r['method'] == m and r['score'] <= tg[r['problem']]}
                 for m in self.methods}
        full = {m: {(r['problem'], r['seed']) for r in self.rows if r['method'] == m and r['score'] <= tg[r['problem']]}
                for m in self.methods}
        rank = share.ranks()
        chosen, covered = [], set()
        for _ in range(k):
            m = max((m for m in self.methods if m not in chosen),
                    key=lambda m: (len(reach[m] - covered), -rank.get(m, math.inf)))
            chosen.append(m); covered |= reach[m]
        single = max(self.methods, key=lambda m: (len(full[m]), -self.ranks().get(m, math.inf)))
        return PanelSuggestion(chosen, len(covered), len(inst), single, len(full[single]), k, self.budget)

    # ---------- picture ----------
    def plot(self, target=None, axes=None):
        """Share of runs that have reached the target, against evaluations (left) and seconds (right).
        Needs matplotlib. Returns the figure."""
        import matplotlib.pyplot as plt
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        else:
            fig = axes[0].figure
        hits = self._hits(target)
        for m in self.methods:
            idx = [i for i, r in enumerate(self.rows) if r['method'] == m]
            ev = np.sort([hits[i] + 1 for i in idx if hits[i] is not None])
            sec = np.sort([self.rows[i]['times'][hits[i]] for i in idx if hits[i] is not None])
            frac = np.arange(1, len(ev) + 1) / len(idx)
            for ax, x, end in ((axes[0], ev, self.budget), (axes[1], sec, None)):
                if len(x):
                    xs = np.r_[x, end if end else x[-1]]
                    ax.step(xs, np.r_[frac, frac[-1]], where='post', linewidth=2, label=m)
        axes[0].set(xlabel='evaluations', ylabel='share of runs at the target', ylim=(0, 1.02), xlim=(0, self.budget))
        axes[1].set(xlabel='seconds (including the optimizer\'s own time)')
        axes[1].set_xscale('log')
        axes[0].legend(frameon=False)
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig


class _Timed:
    """Wraps fn to stamp the time of every call (one call per evaluation)."""

    def __init__(self, fn, t0):
        self.fn, self.t0, self.times = fn, t0, []

    def __call__(self, params):
        try:
            return self.fn(params)
        finally:
            self.times.append(time.perf_counter() - self.t0)


def _curve(fn, history, maximize):
    """Best-so-far score after each evaluation (lower is better; true log10 regret for simulated problems)."""
    sign = -1.0 if maximize else 1.0
    best_y, best_score, out = math.inf, math.inf, []
    for params, v in history:
        if math.isfinite(v) and sign * v < best_y:
            best_y = sign * v
            best_score = fn.regret(params) if hasattr(fn, 'regret') else best_y
        out.append(best_score)
    return np.array(out, float)


def _job(name, fn, space, m, seed, budget, maximize):
    t0 = time.perf_counter()
    timed = _Timed(fn, t0)
    res = minimize(timed, space, budget=budget, method=m, maximize=maximize, seed=seed)
    seconds = time.perf_counter() - t0
    if res.best_params is None:
        score = math.inf  # every evaluation failed
    elif hasattr(fn, 'regret'):
        score = fn.regret(res.best_params)
    else:
        score = -res.best_value if maximize else res.best_value
    curve = _curve(fn, res.history, maximize)
    times = np.array(timed.times[:len(curve)], float)
    return dict(problem=name, method=m, seed=seed, best=res.best_value, score=score,
                evals=res.n_evals, failed=res.n_failed, seconds=seconds, curve=curve, times=times)


THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
_limits = None


def _worker_init(custom):
    """One BLAS thread per worker: n_jobs workers that each start a full thread pool fight over the cores and can
    run many times slower. The env vars cover spawned workers; threadpoolctl (installed with scikit-learn) also
    covers forked ones, which inherit numpy already loaded."""
    global _limits
    api._set_custom(custom)
    try:
        from threadpoolctl import threadpool_limits
        _limits = threadpool_limits(1)  # kept referenced so the limit holds for the worker's lifetime
    except ImportError:
        pass


def compare(problems, methods=('rlcma', 'walker', 'walker_hp', 'cmaes', 'random'), budget=200, seeds=3,
            maximize=False, n_jobs=1):
    """problems: {name: (fn, space)}; fn(params) -> value. Every method gets the same problems, budget and seeds.

    For simulated problems (from make_problem) the score is the true log10 regret at the best point found,
    so noisy luck isn't rewarded. For your own functions it is the best observed value.

    Besides final scores, every run keeps its best-so-far score after each evaluation and the time of each
    evaluation, which success(), success_table(), budget_table(), suggest_panel() and plot() use.

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
        saved = {v: os.environ.get(v) for v in THREAD_VARS}
        os.environ.update({v: '1' for v in THREAD_VARS})  # spawned workers read these when they load numpy
        try:
            with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs)), initializer=_worker_init,
                                     initargs=(dict(api.CUSTOM),)) as ex:
                rows = list(ex.map(_job, *zip(*jobs)))
        finally:
            for v, old in saved.items():
                if old is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = old
    else:
        rows = [_job(*j) for j in jobs]
    return Comparison(rows, methods, maximize)
