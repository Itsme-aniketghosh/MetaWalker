"""Training, evaluation and report for rlbo (the optimizer itself: metawalker/rlbo.py).

Usage:  python -m experiments.rlbo_bench train SEED [GENS] -> rlbo_policy_s<SEED>.npy (checkpoint + resume:
                                                             rlbo_ckpt_s<SEED>.pkl; stops at $RLBO_WALL seconds)
        python -m experiments.rlbo_bench eval TASK NTASKS  -> results_rlbo/part_TASK.json (test_hpo instances)
        python -m experiments.rlbo_bench report            -> win rates vs optuna_gp, rl:rlcma_mlp:s0, cmaes per n
Files go to $NAV_OUT (default: the current directory), next to the benchmark's results_* dirs.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')  # one BLAS thread per pool worker
import json, pickle, sys, time, warnings
from multiprocessing import Pool

import numpy as np

from metawalker.rlbo import NPARAM, SHAPES, init_theta, rlbo
from metawalker.problems import Problem
from experiments import benchmark as nb

HERE = nb.OUT
CFG = dict(pairs=24, n_inst=8, gens=700, budget=1000, dims=(10, 20, 50, 100), sigma=0.5, lr=0.02,
           wall=float(os.environ.get('RLBO_WALL', 46 * 3600)))  # ~225 s/gen on 32 CPUs -> ~44 h; at `wall` it checkpoints and stops, a rerun resumes
FROZEN = ('W1',)  # the MLP's first layer stays at its random init (random tanh features): ES on ~115 params, not ~435


def train_mask():
    m = np.ones(NPARAM, bool); i = 0
    for name, shape in SHAPES:
        k = int(np.prod(shape))
        if name in FROZEN: m[i:i + k] = False
        i += k
    return m



def run(job):
    """job = (method, fn, n, seed, noise, budget) as benchmark.run_one; method 'rlbo:s<k>' or 'rlbo:init'."""
    method, fn, n, seed, noise, budget = job
    warnings.filterwarnings('ignore')
    prob = Problem(fn, n, seed, noise, budget)
    tag = method.split(':')[1]
    theta = None if tag == 'init' else np.load(HERE / f'rlbo_policy_{tag}.npy')
    assert theta is None or theta.shape == (NPARAM,), (method, theta.shape)
    t0 = time.time()
    rlbo(prob, np.random.default_rng(seed + 1), theta)
    assert prob.evals == budget, (job, prob.evals)
    return dict(method=method, fn=fn, n=n, seed=seed, noise=noise, score=prob.score(), sec=time.time() - t0,
                evals=prob.evals, timeout=False, error=None, budget=budget)


# ---------- training ----------
def episode(args):
    theta, inst = args
    warnings.filterwarnings('ignore')
    p = Problem(*inst)
    # same rng per candidate (common random numbers). seed + 1 as benchmark.run_one: default_rng(seed) is the
    # stream Problem draws its shift from, so the first random point would land next to the optimum
    rlbo(p, np.random.default_rng(inst[2] + 1), theta)
    return p.score()


def train(seed, gens=None, pairs=None, n_inst=None, budget=None, dims=None, workers=None, tag=None, lr=None, sigma=None):
    c = dict(CFG, **{k: v for k, v in dict(gens=gens, pairs=pairs, n_inst=n_inst, budget=budget, dims=dims, lr=lr,
                                           sigma=sigma).items()
                     if v is not None})
    tag = tag or f's{seed}'
    out, ck = HERE / f'rlbo_policy_{tag}.npy', HERE / f'rlbo_ckpt_{tag}.pkl'
    if ck.exists():
        st = pickle.load(open(ck, 'rb'))
        assert st['theta'].shape == (NPARAM,), 'checkpoint from another layout'
        theta, m, gen0, log = st['theta'], st['m'], st['gen'], st['log']
        rng = np.random.default_rng(); rng.bit_generator.state = st['rng']
        print(f'resuming {tag} at gen {gen0}', flush=True)
    else:
        rng = np.random.default_rng(4242 + seed)
        theta = init_theta(rng); m = np.zeros(NPARAM); gen0, log = 0, []

    def save(gen):
        tmp = ck.with_suffix('.tmp')
        pickle.dump(dict(theta=theta, m=m, gen=gen, log=log, rng=rng.bit_generator.state), open(tmp, 'wb'))
        os.replace(tmp, ck); np.save(out, theta)
        json.dump(log, open(HERE / f'rlbo_train_log_{tag}.json', 'w'))

    t0 = time.time()
    sig, P2, mask = c['sigma'], c['pairs'], train_mask()
    with Pool(workers or nb.ncpu()) as pool:
        for gen in range(gen0, c['gens']):
            if time.time() - t0 > c['wall']:
                print(f'wall limit at gen {gen}: checkpointed, resubmit to resume', flush=True)
                save(gen); return
            inst = [(str(rng.choice(nb.TRAIN_FNS)), int(rng.choice(c['dims'])), int(rng.integers(1e9)),
                     float(rng.choice([0.0, 0.01])), c['budget']) for _ in range(c['n_inst'])]
            eps = rng.standard_normal((P2, NPARAM)) * mask
            cands = np.concatenate([theta + sig * eps, theta - sig * eps])
            sc = np.array(pool.map(episode, [(th, i) for th in cands for i in inst], chunksize=1))
            S = sc.reshape(len(cands), len(inst))
            assert np.isfinite(S).all(), S
            fit = -S.mean(1)
            # centred ranks within each instance, then averaged: families whose scores spread widely (sphere)
            # would otherwise decide every update
            ranks = ((-S).argsort(0).argsort(0) / (len(cands) - 1) - 0.5).mean(1)
            grad = ((ranks[:P2] - ranks[P2:]) @ eps) / (2 * P2 * sig)
            k = gen + 1
            m = 0.9 * m + 0.1 * grad  # momentum SGD, step RMS (over trained params) = lr
            theta = theta + c['lr'] * m / (np.sqrt(np.mean(m[mask] ** 2)) + 1e-12)
            log.append(float(fit.mean()))
            if k % 10 == 0 or k == c['gens']:
                print(f'rlbo {tag} gen {gen:4d}  mean fitness {fit.mean():+.3f}  best {fit.max():+.3f}'
                      f'  ({time.time() - t0:.0f}s)', flush=True)
                save(k)
    save(c['gens'])


# ---------- evaluation on the test_hpo instances ----------
def test_jobs(seeds=range(nb.N_SEEDS)):
    S = nb.SUITES['test_hpo']
    jobs = [(f'rlbo:s{k}', fn, n, s, noise, nb.BUDGET) for k in seeds for fn in S['fns'] for n in nb.DIMS
            for s in S['seeds'] for noise in (0.0, 0.01)]
    return sorted(jobs, key=lambda j: -j[2])  # costly first, so the stride spreads them across tasks


def evaluate(task, ntasks):
    jobs = test_jobs()[task::ntasks]
    missing = sorted({j[0] for j in jobs if not (HERE / f'rlbo_policy_{j[0].split(":")[1]}.npy').exists()})
    assert not missing, f'missing policies: {missing}'
    with Pool(nb.ncpu()) as pool:
        res = pool.map(run, jobs, chunksize=1)
    out = HERE / 'results_rlbo'; out.mkdir(exist_ok=True)
    json.dump(res, open(out / f'part_{task}.json', 'w'))
    print(f'rlbo eval task {task}/{ntasks}: {len(res)} runs done')


def head_to_head(res, methods, opps, dims, pool=None):
    """pool = (label, [methods]): one extra row over all (method, instance) pairs of those methods."""
    by = {}
    for r in res:
        by.setdefault((r['fn'], r['n'], r['seed'], r['noise']), {})[r['method']] = r['score']
    for m, group in [(m, [m]) for m in methods] + ([pool] if pool else []):
        for opp in opps:
            cells = []
            for n in dims:
                pairs = [(d[g], d[opp]) for k, d in by.items() if k[1] == n and opp in d for g in group if g in d]
                if not pairs:
                    cells.append(f'n={n}: -'); continue
                w = sum(nb.beats(a, b) for a, b in pairs)
                lo, hi = nb.wilson(w, len(pairs))
                cells.append(f'n={n}: {w / len(pairs):.2f} [{lo:.2f},{hi:.2f}] ({len(pairs)})')
            print(f'  {m} vs {opp}:  ' + '  '.join(cells))


def summary(res, names, dims):
    print('median log10(regret / random-point value), lower is better; last row = mean s/run')
    print('      n ' + ''.join(f'{m[-14:]:>15}' for m in names))
    for n in dims:
        print(f'  {n:5d} ' + ''.join(
            f'{np.median([r["score"] for r in res if r["method"] == m and r["n"] == n] or [np.nan]):15.2f}' for m in names))
    print('  s/run ' + ''.join(f'{np.mean([r["sec"] for r in res if r["method"] == m] or [np.nan]):15.1f}' for m in names))


def report():
    from experiments import compare_hpo as compare  # its loader merges results_test + results_test_hpo and swaps in the uncapped results_gpfull GP runs
    seeds = set(nb.SUITES['test_hpo']['seeds'])
    base = [r for r in compare.load('results_test', 'results_test_hpo', seeds=seeds)
            if r['method'] in ('optuna_gp', 'rl:rlcma_mlp:s0', 'cmaes')]
    ours = [r for f in sorted((HERE / 'results_rlbo').glob('part_*.json')) for r in json.load(open(f))]
    assert len(ours) == len(test_jobs()), f'have {len(ours)} rlbo runs, expected {len(test_jobs())}: a task is missing'
    res = base + ours
    key = lambda r: (r['fn'], r['n'], r['seed'], r['noise'])
    gp = [r for r in base if r['method'] == 'optuna_gp']
    full = {key(r) for r in gp if not r.get('timeout') and r.get('evals', nb.BUDGET) >= nb.BUDGET}
    print(f'{len(ours)} rlbo runs; {len(gp)} optuna_gp runs, {len(gp) - len(full)} of them stopped before '
          f'{nb.BUDGET} evals (not equal-evaluation; excluded in the last table)')
    rl = [f'rlbo:s{k}' for k in range(nb.N_SEEDS)]
    for k in range(nb.N_SEEDS):
        lg = HERE / f'rlbo_train_log_s{k}.json'
        print(f'  rlbo:s{k} trained {len(json.load(open(lg))) if lg.exists() else "?"} of {CFG["gens"]} generations')
    summary(res, ['optuna_gp', 'cmaes', 'rl:rlcma_mlp:s0'] + rl, nb.DIMS)
    print('\nwin rate (ties 1/2) [95% Wilson CI] (pairs). rlbo:all pools the 3 seed policies (pairs not independent).')
    head_to_head(res, rl, ['optuna_gp', 'rl:rlcma_mlp:s0', 'cmaes'], nb.DIMS, pool=('rlbo:all', rl))
    if len(full) < len(gp):
        print('\nequal evaluations only (instances where GP reached the full budget):')
        head_to_head([r for r in res if key(r) in full], rl, ['optuna_gp'], nb.DIMS, pool=('rlbo:all', rl))



if __name__ == '__main__':
    cmd, a = (sys.argv[1] if len(sys.argv) > 1 else ''), sys.argv[2:]
    if cmd == 'train': train(int(a[0]), *(int(x) for x in a[1:]))
    elif cmd == 'eval': evaluate(int(a[0]), int(a[1]))
    elif cmd == 'report': report()
    else: sys.exit(__doc__)
