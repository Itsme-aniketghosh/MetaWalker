"""Time-matched comparison with Optuna GP: give our optimizers the wall-clock GP spent, then compare.

Claim tested: "given the same compute time that GP uses, our RL optimizers reach at least the same result".
GP ran 1000 evaluations per instance (results_test_hpo, results_hp_hpo; results_gpfull replaces the 3-hour-
capped runs). Each of our runs gets time_limit = GP's recorded `sec` on that instance, capped at --cap, and an
unlimited evaluation count. A cap below GP's time is conservative for us: we get less time than GP had.

  python -m experiments.timematched list [--cap 3600] [--task-hours 12]  -> timematched_jobs.json, CPU-hours, sbatch command
  python -m experiments.timematched run TASK [NTASKS]                    -> results_timematched/part_<job>.json for every job
                                                                in jobs[TASK::NTASKS] (NTASKS defaults to the one
                                                                `list` chose; finished jobs are skipped on resubmit)
  python -m experiments.timematched report                               -> paper-ready tables + verdict per dimension count

Methods (one job per instance x method):
  cmaes                      plain IPOP CMA-ES, unlimited budget (reference; uses no budget feature)
  rl:<v>:s0+restarts         rlcma_mlp, rlcma, mom: the policies read t/budget, which an unlimited budget pins near
                             0, so run budgeted episodes of 1000 evals (the trained budget) until time runs out.
                             Episode 1 is exactly the benchmark run; later episodes start one walker / the CMA mean
                             at the best point so far (cold restarts otherwise).
  rl:<v>:s0+calib            one run whose budget = evals/s measured in a 2 s calibration run on the instance x the
                             remaining time (then repeated with the leftover time if it finishes early). The 2 s
                             count against our time limit.
  ns:cmaes_rule, ns:walker_rule (+restarts, +calib): the distilled metawalker rules, only if
                             'cmaes_rule' / 'walker_rule' in metawalker.METHODS; cold restarts.
Scoring is Problem.score(): true regret at the best observed point, as in the benchmark. A deadline-wrapped
Problem raises a BaseException at the deadline (metawalker catches Exception from objectives), keeping the best.

Threads: our runs are single-threaded (-c 1, BLAS threads = 1). GP in results_test_hpo / results_hp_hpo ran torch
with 1 thread (8 runs per 8-core node); the results_gpfull reruns ran torch on 8 threads, so on those instances
GP's wall-clock bought it ~8x our CPU, again conservative for us.
With --cap <= 10800 s (the old GP cap) the time limits are the same before and after results_gpfull lands, so
`list` can run now; `report` re-reads the GP records, so run it after the reruns finish.

Cluster (from a work dir, $NAV_OUT, holding the results_* dirs and policy_*.npy):
  .venv/bin/python -m experiments.timematched list --cap 3600           # prints NTASKS and the exact sbatch line
  sbatch -A <account>--array=0-<NTASKS-1> -t <T> timematched.sbatch
  (split over accounts: two sbatch calls with --array ranges and --export=ALL,NTASKS=<NTASKS>)
  .venv/bin/python -m experiments.timematched report > report_timematched.txt
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')  # single-threaded, before numpy loads

import argparse, json, math, time
from pathlib import Path

import numpy as np

from metawalker.cma import cmaes
from metawalker.policy import nparam
from metawalker.problems import Problem
from experiments import benchmark as nb

ROOT = Path(os.environ.get('TM_ROOT', nb.OUT))  # TM_ROOT: only for testing on fake data
OUT = ROOT / 'results_timematched'
JOBS = ROOT / 'timematched_jobs.json'
SUITES = ('test_hpo', 'hp_hpo')
POLICIES = ['rl:rlcma_mlp:s0', 'rl:rlcma:s0', 'rl:mom:s0']
NOMINAL, UNLIMITED, CALIB_SEC = nb.BUDGET, 10 ** 15, 2.0

try:
    import metawalker
    from metawalker.api import _run as ns_run  # unit-box optimizer dispatch, without minimize()'s per-eval history
    NS = [f'ns:{m}' for m in ('cmaes_rule', 'walker_rule') if m in metawalker.METHODS]
except ImportError:
    NS = []


class Stop(BaseException):
    pass


class Timed(Problem):
    """The benchmark Problem with an unlimited budget and a wall-clock stop; keeps the best observed point."""

    def __init__(self, fn, n, seed, noise, stop_at):
        super().__init__(fn, n, seed, noise, UNLIMITED)
        self.stop_at = stop_at

    def __call__(self, x):
        if time.time() > self.stop_at:
            raise Stop
        return super().__call__(x)


class Episode:
    """A fresh evaluation budget on top of a shared problem (which keeps the clock and the overall best)."""

    def __init__(self, prob, budget):
        self.prob, self.budget, self.evals = prob, budget, 0

    def __getattr__(self, k):  # n, scale, noise_sd, best_y
        return getattr(self.prob, k)

    def __call__(self, x):
        if self.evals >= self.budget:
            return np.inf  # over budget: ignored, as in Problem
        self.evals += 1
        return self.prob(x)


class WarmStart:
    """rng proxy: the first uniform start point(s) an episode draws begin at the incumbent instead.
    shortcut: relies on metawalker navigate / cmaes drawing start points with rng.uniform(0, 1, n or (M, n));
    not used for metawalker methods, whose rule code draws other uniforms."""

    def __init__(self, rng, x):
        self.rng, self.x = rng, x

    def __getattr__(self, k):
        return getattr(self.rng, k)

    def uniform(self, low=0.0, high=1.0, size=None):
        u = self.rng.uniform(low, high, size)
        if self.x is not None and np.shape(u)[-1:] == self.x.shape:
            u.reshape(-1, len(self.x))[0] = self.x
            self.x = None
        return u


def algo(method):
    if method == 'cmaes':
        return cmaes
    if method.startswith('ns:'):
        return lambda p, r: ns_run(method[3:], p, r)
    cfg = nb.VARIANTS[method.split(':')[1]]
    theta = np.load(ROOT / nb.policy_path(method).name)
    assert theta.shape == (nparam(cfg),), (method, theta.shape)
    return lambda p, r: nb.run_policy(p, r, theta, cfg)


def episodes(prob, run, rng, budget, warm):
    k = 0
    try:
        while time.time() < prob.stop_at:
            run(Episode(prob, budget()), WarmStart(rng, prob.best_x) if warm and prob.best_x is not None else rng)
            k += 1
    except Stop:
        pass
    return k


def run_job(job):
    method, fn, n, seed, noise, limit, mode = job
    t0 = time.time()
    run, rate, err = algo(method), None, None
    if mode == 'calib':  # evals/s of this method on this instance, in budgeted episodes like its training
        cal = Timed(fn, n, seed, noise, t0 + CALIB_SEC)
        episodes(cal, run, np.random.default_rng(seed + 2), lambda: NOMINAL, False)
        rate = cal.evals / (time.time() - t0)
    prob = Timed(fn, n, seed, noise, t0 + limit)
    budget = {'unlimited': lambda: UNLIMITED, 'restarts': lambda: NOMINAL,
              'calib': lambda: max(NOMINAL, int(rate * (prob.stop_at - time.time())))}[mode]
    try:
        k = episodes(prob, run, np.random.default_rng(seed + 1), budget, mode == 'restarts' and method.startswith('rl:'))
    except MemoryError as e:  # e.g. IPOP popsize blow-up: keep the best so far instead of losing the run
        k, err = None, repr(e)[:300]
    return dict(method=method if mode == 'unlimited' else f'{method}+{mode}', fn=fn, n=n, seed=seed, noise=noise,
                score=prob.score() if prob.best_x is not None else None, sec=time.time() - t0, evals=prob.evals,
                timeout=True, error=err, time_limit=limit, episodes=k, calib_rate=rate)


def gp_records():
    key = lambda r: (r['fn'], r['n'], r['seed'], r['noise'])
    full = {key(r): r for f in sorted((ROOT / 'results_gpfull').glob('part_*.json')) for r in json.load(open(f))
            if r['method'] == 'optuna_gp'}  # absent dir: no replacements yet
    recs = [r for s in SUITES for f in sorted((ROOT / f'results_{s}').glob('part_*.json')) for r in json.load(open(f))
            if r['method'] == 'optuna_gp']
    assert recs, f'no optuna_gp records under {ROOT}/results_{{{",".join(SUITES)}}}'
    out = {key(r): full.get(key(r), r) for r in recs}
    return {k: r for k, r in out.items() if r.get('score') is not None}, len(full), len(out)


def limit_memory():  # MemoryError (caught, best kept) instead of a Slurm OOM kill that loses the whole run
    mb = os.environ.get('SLURM_MEM_PER_NODE')
    try:
        import resource
        if mb:
            lim = int(0.9 * int(mb) * 2 ** 20)
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    except (ImportError, ValueError):
        pass


def hms(s):
    s = int(math.ceil(s)); return f'{s // 86400}-{s % 86400 // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}'


def cmd_list(cap, task_hours):
    gp, n_full, n_all = gp_records()
    methods = [(m, mode) for m in POLICIES + NS for mode in ('restarts', 'calib')] + [('cmaes', 'unlimited')]
    for m in POLICIES:
        assert (ROOT / nb.policy_path(m).name).exists(), f'missing policy for {m}'
    for m in NS:  # smoke-test the metawalker dispatch so an API change fails here, not in 400 array tasks
        p = Timed('sphere', 5, 0, 0.0, time.time() + 60)  # first call pays import and setup; a busy node can take >5 s
        ns_run(m[3:], Episode(p, 50), np.random.default_rng(0))
        assert p.evals == 50, (m, p.evals)
    stale = [r for r in gp.values() if r.get('timeout') and r['sec'] < cap and not r.get('error')]
    if stale:
        print(f'WARNING: {len(stale)} GP runs were capped at {max(r["sec"] for r in stale):.0f}s < --cap and have '
              'no results_gpfull record yet: their time limits will be too short')
    jobs = sorted(([m, *k, min(r['sec'], cap), mode] for k, r in gp.items() for m, mode in methods),
                  key=lambda j: -j[5])  # longest first, so the strided split balances tasks
    tot = sum(j[5] + (CALIB_SEC if j[6] == 'calib' else 0) for j in jobs)
    ntasks = max(1, min(len(jobs), math.ceil(tot / (task_hours * 3600))))
    load = max(sum(j[5] + CALIB_SEC + 10 for j in jobs[t::ntasks]) for t in range(ntasks))
    json.dump(dict(cap=cap, ntasks=ntasks, jobs=jobs), open(JOBS, 'w'))
    print(f'{len(gp)} GP instances ({n_full} from results_gpfull; {n_all - len(gp)} crashed GP runs skipped)')
    print(f'methods: {[m if mo == "unlimited" else f"{m}+{mo}" for m, mo in methods]}'
          f'{"" if NS else "  (metawalker rule methods not available: skipped)"}')
    print(f'{len(jobs)} jobs, cap {cap:.0f}s, total {tot / 3600:.0f} CPU-hours; {ntasks} tasks, '
          f'max {load / 3600:.1f} h per task')
    if load > 47 * 3600:
        print('WARNING: tasks exceed the short partition limit: lower --task-hours')
    print(f'sbatch --array=0-{ntasks - 1} -t {hms(load * 1.1 + 900)} timematched.sbatch')


def cmd_run(task, ntasks):
    limit_memory()
    spec = json.load(open(JOBS))
    ntasks = ntasks or spec['ntasks']
    OUT.mkdir(exist_ok=True)
    for i in range(task, len(spec['jobs']), ntasks):
        f = OUT / f'part_{i}.json'
        if f.exists():
            continue
        res = run_job(spec['jobs'][i])
        json.dump([res], open(f, 'w'))
        print(res, flush=True)


def verdict(lo, hi):
    return 'matches or beats GP' if lo >= 0.5 else 'inconclusive' if hi >= 0.5 else 'loses'


def cmd_report():
    spec = json.load(open(JOBS))
    gp, n_full, _ = gp_records()
    ours = [r for f in sorted(OUT.glob('part_*.json')) for r in json.load(open(f))]
    if len(ours) < len(spec['jobs']):
        print(f'WARNING: {len(ours)} of {len(spec["jobs"])} jobs have results: tables are partial')
    bad = [r for r in ours if r['score'] is None or r['error']]
    if bad:
        print(f'note: {len(bad)} runs ended in an error (best-so-far kept where it exists): '
              f'{sorted({r["method"] for r in bad})}')
    ours = [r for r in ours if r['score'] is not None and (r['fn'], r['n'], r['seed'], r['noise']) in gp]
    methods = list(dict.fromkeys(r['method'] for r in ours))
    print(f'Time-matched comparison with Optuna GP (1000 evaluations; {n_full} uncapped reruns). Each of our runs got '
          f'the wall-clock GP spent on that instance, capped at {spec["cap"]:.0f} s, single-threaded, with unlimited '
          f'evaluations. Score: log10(true regret at the best observed point / random-point value), lower is better. '
          f'Win = lower score than GP on the same instance; ties count half. CI: 95% Wilson.')
    groups = [('Standard functions (sphere, ellipsoid, rastrigin, rosenbrock, ackley, levy)', lambda f: f != 'honeypot'),
              ('Honeypot-like plateau family', lambda f: f == 'honeypot')]
    for title, inf in groups:
        rs = [r for r in ours if inf(r['fn'])]
        if not rs:
            continue
        print(f'\n== {title} ==')
        verdicts = []
        for m in methods:
            mr = [r for r in rs if r['method'] == m]
            print(f'\n{m}\n      n  noise    N   win  tie  loss  win rate [95% CI]   median regret ours / GP'
                  f'   median sec ours / GP')
            for n in nb.DIMS:
                for noise in (0.0, 0.01, None):  # None: both noise levels pooled -> verdict
                    c = [r for r in mr if r['n'] == n and (noise is None or r['noise'] == noise)]
                    if not c:
                        continue
                    g = [gp[(r['fn'], r['n'], r['seed'], r['noise'])] for r in c]
                    w = sum(r['score'] < x['score'] for r, x in zip(c, g)) / len(c)
                    t = sum(r['score'] == x['score'] for r, x in zip(c, g)) / len(c)
                    lo, hi = nb.wilson((w + t / 2) * len(c), len(c))
                    print(f'  {n:5d}  {"both" if noise is None else noise:>5}  {len(c):3d}  {w:.2f}  {t:.2f}  '
                          f'{1 - w - t:.2f}  {w + t / 2:.2f} [{lo:.2f},{hi:.2f}]'
                          f'  {np.median([r["score"] for r in c]):9.2f} / {np.median([x["score"] for x in g]):6.2f}'
                          f'  {np.median([r["sec"] for r in c]):9.0f} / {np.median([x["sec"] for x in g]):6.0f}')
                    if noise is None:
                        verdicts.append(f'  {m:28s} n={n:<4d} {w + t / 2:.2f} [{lo:.2f},{hi:.2f}]  {verdict(lo, hi)}')
        print(f'\nVerdict ({title}): win rate vs GP, both noise levels pooled; "matches or beats GP" if the lower '
              f'95% bound >= 0.5, "inconclusive" if the interval contains 0.5, "loses" otherwise')
        print('\n'.join(verdicts))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['list', 'run', 'report'])
    ap.add_argument('task', nargs='*', type=int, help='run: TASK [NTASKS]')
    ap.add_argument('--cap', type=float, default=3600.0, help='max seconds per run (below GP time: conservative)')
    ap.add_argument('--task-hours', type=float, default=12.0, help='list: target hours of runs per array task')
    a = ap.parse_args()
    if a.cmd == 'list':
        cmd_list(a.cap, a.task_hours)
    elif a.cmd == 'run':
        cmd_run(a.task[0], a.task[1] if len(a.task) > 1 else None)
    else:
        cmd_report()
