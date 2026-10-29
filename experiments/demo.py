"""Race metawalker's learned optimizers against standard tuners, every (scenario, problem, seed, method) run
as an independent job in a process pool.

  python -m experiments.demo --quick                          # laptop: 4 workers, a few minutes
  python -m experiments.demo --n-jobs 32                      # full race on one machine
  python -m experiments.demo --shard 3/16 --out results/demo  # one Slurm array task (cluster/demo.sbatch)
  python -m experiments.demo --merge results/demo             # shards -> results.json, results.md, plot_*.png

Scenarios: 'plateau' = fixed evaluation budget on honeypot-like plateau problems;
           'wallclock' = every method evaluates until a per-run time limit runs out.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')  # one thread per job: parallelism comes from the pool, and timing stays fair
import argparse, glob, json, math, time
from multiprocessing import Pool

import numpy as np
from metawalker import Comparison, make_problem
from metawalker.api import _run  # the dispatch minimize() uses; we skip minimize() only for its per-eval history

OURS, BASELINES = ('walker_hp', 'walker', 'rlcma'), ('cmaes', 'tpe', 'gp', 'random')
METHODS = OURS + BASELINES
NOISE = 0.01  # plateau: binomial (k of 100 trials); others: Gaussian, 1% of a typical value
CONFIGS = {  # scenario: (problems, eval budget, seconds per run, seeds)
    'full': {'plateau': ([('plateau', 50), ('plateau', 100), ('plateau', 200)], 1000, 7200, 10),  # walker_hp's training budget
             'wallclock': ([('plateau', 50), ('plateau', 100), ('plateau', 200), ('rastrigin', 200),
                            ('ellipsoid', 200)], 10 ** 9, 60, 10)},  # wall-clock race: budget = never
    'quick': {'plateau': ([('plateau', 20), ('plateau', 50)], 500, 60, 3),
              'wallclock': ([('plateau', 50), ('rastrigin', 100), ('ellipsoid', 100)], 10 ** 9, 8, 3)},
}


class TimeUp(BaseException):  # BaseException, so no optimizer (or Optuna) swallows it as a failed trial
    pass


class Objective:
    """What an optimizer sees: x in [0,1]^n -> noisy value. Tracks the best point and stops at the deadline.
    Unlike minimize() it keeps no per-eval history, which would be GBs after 10^5 evals at 200 dims."""

    def __init__(self, prob, budget, seconds):
        self.prob, self.n, self.budget, self.evals = prob, prob.n, budget, 0
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + seconds
        self.best, self.best_params, self.trace = math.inf, None, []  # trace: [(seconds, log10 regret)]
        self.next_log = 0.0

    def __call__(self, x):
        if self.evals >= self.budget:
            return math.inf  # over budget: ignored (lets CMA-ES finish a generation)
        if time.perf_counter() > self.deadline:
            raise TimeUp
        self.evals += 1
        params = {f'x{i}': float(v) for i, v in enumerate(x)}
        y = self.prob(params)
        if y < self.best:  # the optimizer's pick is its best *observed* point; we score its true regret
            self.best, self.best_params = y, params
            t = time.perf_counter() - self.t0
            if t >= self.next_log:  # log-spaced logging, so scoring doesn't slow the fast methods down
                self.trace.append((t, self.prob.regret(params)))
                self.next_log = 1.05 * t + 1e-3
        return y


def run_job(job):
    prob = make_problem(job['family'], job['n'], job['seed'], NOISE)[0]
    obj = Objective(prob, job['budget'], job['time_limit'])
    # own seed stream: the same int seed as make_problem would replay its shift draws as the first point
    rng, stopped = np.random.default_rng((job['seed'], 1)), False
    try:
        _run(job['method'], obj, rng)
    except TimeUp:
        stopped = True
    secs, score = time.perf_counter() - obj.t0, prob.regret(obj.best_params) if obj.best_params else math.inf
    return dict(job, problem=f"{job['family']}-{job['n']}", score=score, evals=obj.evals, seconds=secs,
                timed_out=stopped, trace=obj.trace + [(secs, score)])


def warm_up():
    """Import heavy dependencies before any clock starts, so no method pays import time inside its race."""
    import cma, optuna  # noqa: F401
    try:
        import torch  # Optuna's GPSampler imports it lazily
        torch.set_num_threads(1)
    except ImportError:
        pass


def make_jobs(mode):
    jobs = [dict(scenario=sc, family=f, n=n, seed=s, method=m, budget=b, time_limit=t)
            for sc, (probs, b, t, seeds) in CONFIGS[mode].items() for f, n in probs for s in range(seeds)
            for m in METHODS]
    return sorted(jobs, key=lambda j: j['method'] != 'gp')  # slowest first, and spread evenly over shards


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def capped(rows, scenario):
    """Fixed-budget runs stopped early by the safety time cap (it makes 'evals' fall short of the budget)."""
    k = sum(r['timed_out'] for r in rows)
    return f' ({k} hit the {rows[0]["time_limit"]} s cap)' if scenario == 'plateau' and k else ''


def report(rows, out):
    md = []
    for sc in dict.fromkeys(r['scenario'] for r in rows):
        rs = [r for r in rows if r['scenario'] == sc]
        cmp = Comparison(rs, list(METHODS), False)
        ranks, probs = cmp.ranks(), list(dict.fromkeys(r['problem'] for r in rs))
        best = min(OURS, key=ranks.get)
        j = rs[0]
        md += [f"### Scenario `{sc}`: " + (f"{j['budget']} evaluations per run" if sc == 'plateau'
                                          else f"{j['time_limit']} s of wall-clock per run"), '',
               'Median log10 regret (lower is better; 0 = a typical random point), average rank over '
               f"{len(cmp._instances())} problem instances, and median evaluations and seconds per run.", '',
               '| method | ' + ' | '.join(probs) + ' | avg rank | evals | s/run |',
               '|---' * (len(probs) + 4) + '|']
        for m in sorted(METHODS, key=ranks.get):
            mr = [r for r in rs if r['method'] == m]
            med = [np.median([r['score'] for r in mr if r['problem'] == p]) for p in probs]
            md.append(f'| {m} | ' + ' | '.join(f'{v:.2f}' for v in med) + f' | {ranks[m]:.2f} | '
                      f"{np.median([r['evals'] for r in mr]):.0f}{capped(mr, sc)} | {np.median([r['seconds'] for r in mr]):.1f} |")
        md += ['', f'Win rate of **{best}** (our best by average rank) against each baseline, ties count 1/2:', '',
               '| vs | win rate | 95% Wilson CI | instances |', '|---|---|---|---|']
        for b in BASELINES:
            p, n = cmp.win_rate(best, b), sum(1 for s in cmp._instances().values() if best in s and b in s)
            lo, hi = wilson(p, n)
            md.append(f'| {b} | {p:.2f} | [{lo:.2f}, {hi:.2f}] | {n} |')
        md.append('')
        plot(rs, probs, f'{out}/plot_{sc}.png', sc)
    with open(f'{out}/results.md', 'w') as f:
        f.write('\n'.join(md))
    print('\n'.join(md))


def plot(rows, probs, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(probs), figsize=(4 * len(probs), 3.4), squeeze=False)
    grid = np.logspace(-3, np.log10(max(r['seconds'] for r in rows)), 200)
    for ax, p in zip(axes[0], probs):
        for m in METHODS:
            curves = []
            for r in (r for r in rows if r['problem'] == p and r['method'] == m):
                t, v = np.array(r['trace'] or [(np.inf, np.nan)]).T
                i = np.searchsorted(t, grid, side='right') - 1  # best-so-far at each grid time (nan before 1st eval)
                curves.append(np.where(i >= 0, v[np.maximum(i, 0)], np.nan))
            with np.errstate(all='ignore'):
                ax.plot(grid, np.nanmedian(curves, 0), label=m, lw=2 if m in OURS else 1.2,
                        ls='-' if m in OURS else '--')
        ax.set(xscale='log', title=p, xlabel='wall-clock seconds', ylabel='median best log10 regret')
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f'{title}: best-so-far true regret vs time (median over seeds)')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def merge(out):
    shards = [json.load(open(f)) for f in sorted(glob.glob(f'{out}/race_*.json'))]
    if not shards:
        raise SystemExit(f'no race_*.json in {out}')
    mode, rows = shards[0]['mode'], [r for s in shards for r in s['rows']]
    key = lambda j: (j['scenario'], j['family'], j['n'], j['seed'], j['method'])
    missing = {key(j) for j in make_jobs(mode)} - {key(r) for r in rows}
    if missing or len(rows) != len(make_jobs(mode)):  # fail loudly: a partial table looks plausible
        raise SystemExit(f'{len(missing)} jobs missing, {len(rows)} rows for {len(make_jobs(mode))} jobs: '
                         f'{sorted(missing)[:5]}')
    with open(f'{out}/results.json', 'w') as f:
        json.dump(dict(mode=mode, rows=rows), f)
    report(rows, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='small laptop version (4 workers)')
    ap.add_argument('--n-jobs', type=int, default=None, help='pool size (default: CPUs, or 4 with --quick)')
    ap.add_argument('--shard', default='0/1', help='i/N: run every N-th job starting at i (a Slurm array task)')
    ap.add_argument('--out', default=None, help='output dir (default results/demo, or results/demo_quick)')
    ap.add_argument('--merge', metavar='DIR', help='combine race_*.json in DIR into the table and plots')
    a = ap.parse_args()
    if a.merge:
        return merge(a.merge)
    mode = 'quick' if a.quick else 'full'
    out = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results',
                                f'demo{"_quick" * a.quick}')
    os.makedirs(out, exist_ok=True)
    i, n = map(int, a.shard.split('/'))
    jobs = make_jobs(mode)[i::n]
    n_jobs = a.n_jobs or (4 if a.quick else os.cpu_count())
    t0, rows = time.time(), []
    with Pool(n_jobs, initializer=warm_up) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            rows.append(r)
            print(f"[{len(rows)}/{len(jobs)} {time.time() - t0:5.0f}s] {r['scenario']} {r['problem']} s{r['seed']} "
                  f"{r['method']}: regret {r['score']:.2f}, {r['evals']} evals, {r['seconds']:.1f}s", flush=True)
    with open(f'{out}/race_{i}.json', 'w') as f:
        json.dump(dict(mode=mode, rows=rows), f)
    if n == 1:
        merge(out)


if __name__ == '__main__':
    main()
