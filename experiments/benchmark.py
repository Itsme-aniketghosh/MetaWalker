"""Benchmark of learned optimizers vs standard ones on n-dim landscapes with a known optimum.

Defines the policy variants (VARIANTS) and their ES training, the evaluation suites, and the reports. The
optimizers themselves are in the metawalker library (walker, cma, rlbo, baselines). Policies are trained with
OpenAI-ES at the test budget; variants are picked on a validation suite (seeds 5000+) and only the picked ones run
on the test suite (seeds 1000+). Files (policies, results, logs) go to $NAV_OUT (default: the current directory).

Usage:  python -m experiments.benchmark train TASK            -> policy_<variant>_s<seed>.npy  (TASK = variant x seed)
        python -m experiments.benchmark val TASK              -> trained vs untrained vs hand, train-like instances
        python -m experiments.benchmark eval SUITE TASK NTASKS-> results_<suite>/part_TASK.json
        python -m experiments.benchmark select                -> selected.json (top variants on the val suite)
        python -m experiments.benchmark report SUITE          -> tables
        python -m experiments.benchmark smoke | smoke_vb | smoke_hpx   -> quick self-checks
"""
import json, os, sys, time, warnings
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

# one BLAS thread per process: training runs a 32-process pool, and 32 x 32 threads stalled pycma ~300x
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

from metawalker import problems
from metawalker.baselines import ngopt, optuna_search, random_search, smac_hpo
from metawalker.cma import cmaes
from metawalker.es import centered_ranks, es_step
from metawalker.policy import init_theta, nparam, unpack, widen
from metawalker.problems import MeanBest, Problem, TimeLimit
from metawalker.walker import navigate, snap_moves

OUT = Path(os.environ.get('NAV_OUT', '.'))
TRAIN_FNS = ['sphere', 'ellipsoid', 'rastrigin', 'rosenbrock']
HELDOUT_FNS = ['ackley', 'levy']
TRAIN_HP = TRAIN_FNS + ['honeypot']
BUDGET = 1000
N_SEEDS = 3

VARIANTS = {
    'mom':        dict(kind='walker', dirs='mom', H=8, M=4),
    'cov':        dict(kind='walker', dirs='cov', H=8, M=4),
    'cov_h32':    dict(kind='walker', dirs='cov', H=32, M=4),
    'cov_m1':     dict(kind='walker', dirs='cov', H=8, M=1),
    'cov_m8':     dict(kind='walker', dirs='cov', H=8, M=8),
    'cov_long':   dict(kind='walker', dirs='cov', H=8, M=4, gens=500),
    'cov_es05':   dict(kind='walker', dirs='cov', H=8, M=4, es_sigma=0.05),
    # pycma runs ~20x slower per generation than a walker, so rlcma trains on a 4x smaller ES batch
    'rlcma':      dict(kind='rlcma', H=0, pairs=20, n_inst=8),
    'rlcma_mlp':  dict(kind='rlcma', H=16, pairs=20, n_inst=8),
    # RL algorithms (rlalgos.py: branching DQN, PPO) with the same moves and features; appended last so the
    # TRAIN_TASKS indices of the jobs already running stay the same
    'q_mom':      dict(kind='walker', dirs='mom', H=32, M=4, learner='dqn', steps=20_000_000),
    'q_cov':      dict(kind='walker', dirs='cov', H=32, M=4, learner='dqn', steps=20_000_000),
    'q_cov_big':  dict(kind='walker', dirs='cov', H=64, M=4, learner='dqn', steps=40_000_000),
    'q_cov_n8':   dict(kind='walker', dirs='cov', H=32, M=4, learner='dqn', nstep=8, steps=20_000_000),
    # PPO trains the same stochastic policy as ES, so its weights run on the ES (sampling) path at eval
    'ppo_mom':    dict(kind='walker', dirs='mom', H=32, M=4, learner='ppo', steps=20_000_000),
    'ppo_cov':    dict(kind='walker', dirs='cov', H=32, M=4, learner='ppo', steps=20_000_000),
    'ppo_cov_big': dict(kind='walker', dirs='cov', H=64, M=4, learner='ppo', steps=60_000_000),
    # round B (2026-10-21): dense vs sparse reward at gamma=1, trained with the honeypot-like family included
    'ppo_dense_hp':  dict(kind='walker', dirs='cov', H=32, M=4, learner='ppo', steps=20_000_000,
                          reward='dense', gamma=1.0, train_fns=TRAIN_HP),
    'ppo_sparse_hp': dict(kind='walker', dirs='cov', H=32, M=4, learner='ppo', steps=20_000_000,
                          reward='sparse', gamma=1.0, train_fns=TRAIN_HP),
    'qn8_dense_hp':  dict(kind='walker', dirs='cov', H=32, M=4, learner='dqn', nstep=8, steps=20_000_000,
                          reward='dense', gamma=1.0, train_fns=TRAIN_HP),
    'qn8_sparse_hp': dict(kind='walker', dirs='cov', H=32, M=4, learner='dqn', nstep=8, steps=20_000_000,
                          reward='sparse', gamma=1.0, train_fns=TRAIN_HP),
    'cov_hp':        dict(kind='walker', dirs='cov', H=8, M=4, train_fns=TRAIN_HP),  # ES: always episodic
}
ROUND_AB = list(VARIANTS)  # what the older suites, select() and smoke iterate over, so their job lists stay the same
# round C (2026-10-22): budget-aware fine-tuning. Each training instance draws its budget log-uniformly from
# `budgets`, so the t/budget feature is seen at 300..10000 evals; init_from warm-starts from policy_<init_from>.npy
VB = ['rlcma_vb', 'cov_vb', 'mom_vb']
VARIANTS.update(
    rlcma_vb=dict(kind='rlcma', H=16, pairs=20, n_inst=8, gens=70, budgets=(300, 10000), init_from='rlcma_mlp_s0'),  # ~1200 s/gen on 32 CPUs: 250 would exceed the 2-day limit,
    cov_vb=dict(kind='walker', dirs='cov', H=8, M=4, train_fns=TRAIN_HP, gens=250, budgets=(300, 10000),
                init_from='cov_hp_s0'),
    mom_vb=dict(kind='walker', dirs='mom', H=8, M=4, gens=250, budgets=(300, 10000), init_from='mom_s0'))
# round D (2026-10-23): fine-tune on the honeypot family alone, at the real search budgets, closer to the real
# 169-param space. Train these ONLY via `python -m experiments.honeypot.search train TASK`: it swaps in
# HPFAM_DIMS / HPFAM_NOISE / HPFAM_FRAC; plain `benchmark train` would use the default dims and noise mix.
HP_FAMILY = ['honeypot']
HPFAM = ['hpfam_walker', 'hpfam_rlcma']
HPFAM_DIMS, HPFAM_NOISE, HPFAM_FRAC = (20, 50, 100, 169), 0.01, 0.01  # real 'manyh': ~1.5% of random levels pass the rules
VARIANTS.update(
    # measured locally: ~0.06 s per training run for both (pycma is cheap at <= 600 evals), so both keep a full
    # ES batch with 2x the instances (the plateau signal is near-binary): ~5 s/gen, ~10 min for 100 gens on 32 CPUs
    hpfam_walker=dict(kind='walker', dirs='cov', H=8, M=4, train_fns=HP_FAMILY, pairs=40, n_inst=32, gens=100,
                      budgets=(300, 600), init_from='cov_hp_s0'),
    hpfam_rlcma=dict(kind='rlcma', H=16, train_fns=HP_FAMILY, pairs=40, n_inst=32, gens=100, budgets=(300, 600),
                     init_from='rlcma_mlp_s0'))
# round E (2026-10-23): new walker actions (extra: re-check the best point / jump to a random point; long: x4, x8
# steps), trained on the honeypot family at the real search's noise (5 PPO seeds per evaluation ~ 5 trials).
# hpx_base is the same training without the new actions, so the comparison isolates them.
HPX = ['hpx_base', 'hpx_walker']
HPFAM += HPX + ['hpfam_rlcma_t']
VARIANTS.update(hpx_base=dict(VARIANTS['hpfam_walker'], hp_trials=5, gens=200),
                hpx_walker=dict(VARIANTS['hpfam_walker'], hp_trials=5, gens=200, extra=True, long=True))
# hpfam_rlcma retrained with tie-aware ranks (centered_ranks): the first one collapsed on the real honeypot
VARIANTS['hpfam_rlcma_t'] = dict(VARIANTS['hpfam_rlcma'])
TRAIN_TASKS = [(v, s) for v in VARIANTS for s in range(N_SEEDS)]
# interpretability ablations of hpx_walker (same layout; one new action's logit set to -1e9 in the saved
# weights, see experiments/analysis/policies.py). Added after TRAIN_TASKS: eval-only, never trained.
VARIANTS.update(hpx_norecheck=VARIANTS['hpx_walker'], hpx_nojump=VARIANTS['hpx_walker'])


def ncpu():
    if os.environ.get('NAV_CPUS'): return int(os.environ['NAV_CPUS'])  # e.g. 1 for a RAM-tight local smoke run
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()


TIME_CAP = float(os.environ.get('NAV_TIME_CAP', 3 * 3600))  # seconds per run for the slow HPO baselines
HPO_METHODS = ['optuna_gp', 'optuna_tpe_mv', 'smac', 'ngopt']


def run_policy(prob, rng, theta, cfg):
    (navigate if cfg['kind'] == 'walker' else cmaes)(prob, rng, theta, cfg)


# ---------- training (OpenAI-ES over policy weights) ----------
def train_fitness(args):
    theta, cfg, instances = args
    warnings.filterwarnings('ignore')
    out = []
    for inst in instances:
        p = Problem(*inst)
        # same rng per candidate (common random numbers); +1 as in run_one: default_rng(seed) is the stream Problem
        # draws its shift from, so it would start every walker next to the optimum
        run_policy(p, np.random.default_rng(inst[2] + 1), theta, cfg)
        out.append(p.score())
    return -float(np.mean(out))


def train_instances(rng, k, budget=BUDGET, fns=TRAIN_FNS, budgets=None):
    inst = [(str(rng.choice(fns)), int(rng.choice([5, 10, 20, 50, 100])), int(rng.integers(1e9)),
             float(rng.choice([0.0, 0.01])), budget) for _ in range(k)]
    if budgets:  # log-uniform budget per instance, drawn after the rest so fixed-budget rng streams are unchanged
        lo, hi = np.log(budgets)
        inst = [i[:4] + (int(round(np.exp(rng.uniform(lo, hi)))),) for i in inst]
    return inst


def train(variant, seed, gens=None, pairs=None, lr=0.05, n_inst=None, budget=BUDGET, budgets=None):
    cfg = VARIANTS[variant]
    pairs = pairs or cfg.get('pairs', 40)
    n_inst = n_inst or cfg.get('n_inst', 16)
    gens = gens or cfg.get('gens', 200)
    budgets = budgets or cfg.get('budgets')
    sigma = cfg.get('es_sigma', 0.1)
    rng = np.random.default_rng(1000 * seed + list(VARIANTS).index(variant))
    theta = init_theta(cfg, rng); m = np.zeros_like(theta); v = np.zeros_like(theta)
    if cfg.get('init_from'):  # warm start: fine-tune an existing policy with the same layout
        theta = np.load(OUT / f'policy_{cfg["init_from"]}.npy')
        src = VARIANTS[cfg['init_from'].rsplit('_s', 1)[0]]
        if theta.shape != (nparam(cfg),) and src['kind'] == 'walker':
            theta = widen(theta, src, cfg)
        assert theta.shape == (nparam(cfg),), (variant, cfg['init_from'], theta.shape)
    out = OUT / f'policy_{variant}_s{seed}.npy'
    log = []
    t0 = time.time()
    with Pool(ncpu()) as pool:
        for gen in range(gens):
            instances = train_instances(rng, n_inst, budget, cfg.get('train_fns', TRAIN_FNS), budgets)
            eps = rng.standard_normal((pairs, len(theta)))
            cands = np.concatenate([theta + sigma * eps, theta - sigma * eps])
            fit = np.array(pool.map(train_fitness, [(c, cfg, instances) for c in cands], chunksize=1))
            assert np.isfinite(fit).all(), fit
            theta, m, v = es_step(theta, m, v, fit, eps, sigma, lr, gen)
            log.append(float(fit.mean()))
            if gen % 10 == 0 or gen == gens - 1:
                print(f'{variant} s{seed} gen {gen:3d}  mean fitness {fit.mean():+.3f}  best {fit.max():+.3f}'
                      f'  ({time.time() - t0:.0f}s)', flush=True)
                np.save(out, theta)  # checkpoint
                json.dump(log, open(OUT / f'train_log_{variant}_s{seed}.json', 'w'))
    np.save(out, theta)


def hand_score(args):
    inst, kw = args
    p = Problem(*inst)
    navigate(p, np.random.default_rng(inst[2] + 1), **kw)
    return p.score()


def val(variant, seed):
    cfg = VARIANTS[variant]
    inst = train_instances(np.random.default_rng(123), 64)
    with Pool(ncpu()) as pool:
        for name, th in [('untrained', init_theta(cfg, np.random.default_rng(0))),
                         ('trained', np.load(OUT / f'policy_{variant}_s{seed}.npy'))]:
            print(f'{variant} s{seed} {name:10s}',
                  -np.mean(pool.map(train_fitness, [(th, cfg, [i]) for i in inst], chunksize=1)))
        for kw in ({'hand_beta': 1.0}, {'hand_beta': 1.0, 'hand_kappa': 1.0}):
            print(f'hand {kw}', np.mean(pool.map(hand_score, [(i, kw) for i in inst], chunksize=1)))


# ---------- evaluation ----------
TEST_BASE = ['random', 'walker_hand', 'walker_hand_mom', 'walker_hand_cov', 'cmaes', 'tpe']
ROUND_B = ['ppo_dense_hp', 'ppo_sparse_hp', 'qn8_dense_hp', 'qn8_sparse_hp', 'cov_hp']
SUITES = {'val': dict(seeds=range(5000, 5005), fns=TRAIN_FNS + HELDOUT_FNS,
                      base=['cmaes', 'walker_hand_mom', 'walker_hand_cov']),
          'test': dict(seeds=range(1000, 1010), fns=TRAIN_FNS + HELDOUT_FNS, base=TEST_BASE),
          # round B: every method on the honeypot-like family; dense vs sparse reward on the standard families
          'hp': dict(seeds=range(2000, 2010), fns=['honeypot'], base=TEST_BASE),
          'rwd': dict(seeds=range(5000, 5005), fns=TRAIN_FNS + HELDOUT_FNS, base=['cmaes', 'walker_hand_cov'],
                      variants=ROUND_B + ['ppo_cov', 'q_cov_n8', 'cov_long']),
          # current HPO optimizers on a subset of the test and honeypot-family seeds (compare.py merges results)
          'test_hpo': dict(seeds=range(1000, 1005), fns=TRAIN_FNS + HELDOUT_FNS, base=HPO_METHODS, variants=[]),
          'hp_hpo': dict(seeds=range(2000, 2005), fns=['honeypot'], base=HPO_METHODS, variants=[])}
# how flat the honeypot family is: the same instances with 5% (default) and 1% of the space live
HP_LIVE_BASE = ['random', 'walker_hand_cov', 'cmaes', 'tpe', 'rl:mom:s0', 'rl:cov_hp:s0', 'rl:rlcma_mlp:s0',
                'rl:cov_vb:s0', 'rl:mom_vb:s0']
for _name, _frac in (('hp_live5', 0.05), ('hp_live1', 0.01)):
    SUITES[_name] = dict(seeds=range(2000, 2005), fns=['honeypot'], base=HP_LIVE_BASE, variants=[], hp_frac=_frac)
DIMS = (5, 20, 50, 100, 200, 500)


def warm_method(variant):
    """The policy a variant was warm-started from, as an eval method name ('rlcma_mlp_s0' -> 'rl:rlcma_mlp:s0')."""
    name, s = VARIANTS[variant]['init_from'].rsplit('_s', 1)
    return f'rl:{name}:s{s}'


# round C: every budget-aware policy vs CMA-ES and the policy it was fine-tuned from, at several budgets.
# Optuna GP only where it can finish (budget <= 1000, n <= 100), under NAV_TIME_CAP.
SUITES['budgets'] = dict(seeds=range(3000, 3005), fns=TRAIN_FNS + HELDOUT_FNS + ['honeypot'], dims=(20, 100, 500),
                         budgets=(300, 1000, 3000, 10000), base=['cmaes', 'optuna_gp'] + [warm_method(v) for v in VB],
                         variants=VB, keep=lambda m, n, b: m != 'optuna_gp' or (b <= 1000 and n <= 100))


def suite_methods(suite):
    if suite == 'test':
        variants = json.load(open(OUT / 'selected.json'))['selected']
    else:
        variants = SUITES[suite].get('variants', ROUND_AB)
    methods = SUITES[suite]['base'] + [f'rl:{v}:s{s}' for v in variants for s in range(N_SEEDS)]
    missing = [p for p in methods if p.startswith('rl:') and not policy_path(p).exists()]
    assert not missing, f'missing policies: {missing}'
    return methods


def policy_path(method):
    _, v, s = method.split(':')
    return OUT / f'policy_{v}_{s}.npy'


def run_one(job):
    method, fn, n, seed, noise, budget = job
    warnings.filterwarnings('ignore')
    prob = Problem(fn, n, seed, noise, budget)
    rng = np.random.default_rng(seed + 1)
    t0 = time.time()
    timeout, error = False, None
    if method in HPO_METHODS:
        try:
            import torch  # Optuna GP uses torch: one thread per pool worker unless NAV_TORCH_THREADS says otherwise
            torch.set_num_threads(int(os.environ.get('NAV_TORCH_THREADS', 1)))
        except ImportError:
            pass
        prob.deadline = t0 + TIME_CAP
        fn_ = {'optuna_gp': lambda p, r: optuna_search(p, r, 'gp'),
               'optuna_tpe_mv': lambda p, r: optuna_search(p, r, 'tpe', multivariate=True),
               'smac': smac_hpo, 'ngopt': ngopt}[method]
        try:
            fn_(prob, rng)
        except TimeLimit:
            timeout = True
        except Exception as e:  # a third-party optimizer crash costs this run only, not the whole task
            if prob.best_x is None:
                return dict(method=method, fn=fn, n=n, seed=seed, noise=noise, score=None, sec=time.time() - t0,
                            evals=prob.evals, timeout=False, error=repr(e)[:300])
            timeout, error = True, repr(e)[:300]
    elif method.startswith('rl:'):
        cfg = VARIANTS[method.split(':')[1]]
        theta = np.load(policy_path(method))
        assert theta.shape == (nparam(cfg),), (method, theta.shape)
        run_policy(prob, rng, theta, cfg)
    elif method == 'walker_hand': navigate(prob, rng)
    elif method == 'walker_hand_mom': navigate(prob, rng, hand_beta=1.0)
    elif method == 'walker_hand_cov': navigate(prob, rng, hand_beta=1.0, hand_kappa=1.0)
    else: {'random': random_search, 'cmaes': cmaes, 'tpe': optuna_search}[method](prob, rng)
    assert timeout or prob.evals == budget, (job, prob.evals)  # every method must spend exactly the budget
    return dict(method=method, fn=fn, n=n, seed=seed, noise=noise, score=prob.score(), sec=time.time() - t0,
                evals=prob.evals, timeout=timeout, error=error, budget=budget)


def all_jobs(suite):
    S = SUITES[suite]
    keep = S.get('keep', lambda m, n, b: True)
    jobs = [(m, fn, n, s, noise, b) for m in suite_methods(suite) for fn in S['fns']
            for n in S.get('dims', DIMS) for s in S['seeds'] for noise in (0.0, 0.01)
            for b in S.get('budgets', (BUDGET,)) if keep(m, n, b)]
    slow = ('tpe', 'optuna_gp', 'smac')
    return sorted(jobs, key=lambda j: (j[0] not in slow, -j[2] * j[5]))  # slow, costly jobs first


def set_hp_frac(frac):
    problems.HP_FRAC = frac  # a Pool initializer, so it reaches forked and spawned workers alike


def evaluate(suite, task=0, ntasks=1):
    jobs = all_jobs(suite)[task::ntasks]
    with Pool(ncpu(), initializer=set_hp_frac, initargs=(SUITES[suite].get('hp_frac', problems.HP_FRAC),)) as pool:
        res = pool.map(run_one, jobs, chunksize=1)
    out = OUT / f'results_{suite}'
    out.mkdir(exist_ok=True)
    json.dump(res, open(out / f'part_{task}.json', 'w'))
    print(f'{suite} task {task}: {len(res)} runs done')


def load(suite):
    res = [r for f in sorted((OUT / f'results_{suite}').glob('part_*.json')) for r in json.load(open(f))]
    expected = len(all_jobs(suite))
    assert len(res) == expected, f'{suite}: have {len(res)} results, expected {expected}: an eval task is missing'
    return res


def mean_ranks(res, names):
    by_inst = defaultdict(dict)
    for r in res:
        by_inst[(r['fn'], r['n'], r['seed'], r['noise'])][r['method']] = r['score']
    ranks = {m: [] for m in names}
    for d in by_inst.values():
        for i, m in enumerate(sorted(names, key=d.get)): ranks[m].append(i + 1)
    return {m: float(np.mean(v)) for m, v in ranks.items()}, by_inst


def select(top=3):
    res = load('val')
    names = suite_methods('val')
    mr, by_inst = mean_ranks(res, names)
    score = {v: np.mean([mr[f'rl:{v}:s{s}'] for s in range(N_SEEDS)]) for v in ROUND_AB}
    win = {v: np.mean([beats(d[f'rl:{v}:s{s}'], d['cmaes']) for d in by_inst.values() for s in range(N_SEEDS)])
           for v in ROUND_AB}
    order = sorted(ROUND_AB, key=score.get)
    print('Validation leaderboard (mean rank over seeds; lower is better), win rate vs CMA-ES:')
    for m in sorted(SUITES['val']['base'], key=mr.get):
        print(f'  {m:18s} {mr[m]:6.2f}')
    for v in order:
        print(f'  rl:{v:15s} {score[v]:6.2f}   beats cmaes on {win[v]:.0%}')
    json.dump({'selected': order[:top], 'val_rank': score, 'val_win_vs_cmaes': win},
              open(OUT / 'selected.json', 'w'), indent=1)
    print('selected:', order[:top])


def beats(a, b):
    return 1.0 if a < b else 0.5 if a == b else 0.0  # ties count half


def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def report_budgets(suite):
    S = SUITES[suite]
    res = [r for r in load(suite) if r['score'] is not None]
    names = suite_methods(suite)
    print(f'{suite} suite: {len(res)} runs. Median log10(regret / random-point value) over families, seeds and both '
          f'noise levels; lower is better. optuna_gp only at budget <= 1000, n <= 100 (nan = not run).\n')
    for m in names:
        rs = [r for r in res if r['method'] == m]
        if any(r['timeout'] for r in rs):
            print(f'  note: {m} stopped early (NAV_TIME_CAP) on {sum(r["timeout"] for r in rs)}/{len(rs)} runs')
    groups = [('trained families', TRAIN_FNS), ('held-out families', HELDOUT_FNS), ('honeypot family', ['honeypot'])]
    for subset, fns in [(g, f) for g, f in groups if set(f) & set(S['fns'])]:
        print(f'\n{subset}')
        print(' budget     n ' + ''.join(f'{m[-14:]:>15}' for m in names))
        for b in S['budgets']:
            for n in S['dims']:
                row = [np.median([r['score'] for r in res if r['method'] == m and r['n'] == n and r['budget'] == b
                                  and r['fn'] in fns] or [np.nan]) for m in names]
                print(f'{b:7d} {n:5d} ' + ''.join(f'{v:15.2f}' for v in row))
    by_inst = defaultdict(dict)
    for r in res:
        by_inst[(r['fn'], r['n'], r['seed'], r['noise'], r['budget'])][r['method']] = r['score']
    print('\nHead-to-head: fraction of paired instances where the budget-aware policy is better [95% Wilson CI]')
    for rl in [m for m in names if m.startswith('rl:') and m.split(':')[1] in VB]:
        for opp in ('cmaes', warm_method(rl.split(':')[1])):
            cells = []
            for b in S['budgets']:
                pairs = [d for k, d in by_inst.items() if k[4] == b and rl in d and opp in d]
                wins = sum(beats(d[rl], d[opp]) for d in pairs)
                lo, hi = wilson(wins, len(pairs))
                cells.append(f'B={b}: {wins / len(pairs):.2f} [{lo:.2f},{hi:.2f}]')
            print(f'  {rl} vs {opp}:  ' + '  '.join(cells))


def report(suite):
    if 'budgets' in SUITES[suite]:
        return report_budgets(suite)
    res = load(suite)
    names = suite_methods(suite)
    print(f'{suite} suite: {len(res)} runs. Median log10(regret / random-point value), lower is better. '
          f'Budget = {BUDGET} evals.\n')
    groups = [('trained families', TRAIN_FNS), ('held-out families', HELDOUT_FNS), ('honeypot family', ['honeypot'])]
    for subset, fns in [(g, f) for g, f in groups if set(f) & set(SUITES[suite]['fns'])]:
        for noise in (0.0, 0.01):
            print(f'{subset}, noise={noise}')
            print('    n ' + ''.join(f'{m[-14:]:>15}' for m in names))
            for n in DIMS:
                row = [np.median([r['score'] for r in res if r['method'] == m and r['n'] == n
                                  and r['noise'] == noise and r['fn'] in fns]) for m in names]
                print(f'{n:5d} ' + ''.join(f'{v:15.2f}' for v in row))
            print()
    mr, by_inst = mean_ranks(res, names)
    print('Average rank over all instances (1 = best):')
    for m in sorted(names, key=mr.get):
        sec = np.mean([r['sec'] for r in res if r['method'] == m])
        print(f'  {m:22s} {mr[m]:.2f}   ({sec:.2f} s/run incl. cheap f)')
    print('\nHead-to-head: fraction of paired instances where the learned method is better [95% Wilson CI]')
    for rl in [m for m in names if m.startswith('rl:')]:
        for opp in ('cmaes', 'tpe', 'walker_hand_cov'):
            if opp not in names: continue
            cells = []
            for n in DIMS:
                pairs = [d for k, d in by_inst.items() if k[1] == n]
                wins = sum(beats(d[rl], d[opp]) for d in pairs)
                lo, hi = wilson(wins, len(pairs))
                cells.append(f'n={n}: {wins / len(pairs):.2f} [{lo:.2f},{hi:.2f}]')
            print(f'  {rl} vs {opp}:  ' + '  '.join(cells))


# job count and sha1(repr(all_jobs)) of each older suite, from the file before round C ('test' needs selected.json)
OLD_JOBS = {'val': (23760, '464ff3899816'), 'hp': (8280, '3bc61932817d'), 'rwd': (9360, '818b695b03db'),
            'test_hpo': (1440, '7a943968be54'), 'hp_hpo': (240, 'edb4833a01ca')}


def smoke_vb():
    """Round C check: older suites unchanged, then 2 tiny generations per *_vb variant and a tiny 'budgets' eval."""
    import hashlib, shutil
    global policy_path
    real_pp, policy_path = policy_path, lambda m: OUT  # pretend every policy exists, to list jobs
    for s, (n_jobs, sha) in OLD_JOBS.items():
        j = all_jobs(s)
        assert (len(j), hashlib.sha1(repr(j).encode()).hexdigest()[:12]) == (n_jobs, sha), f'{s} jobs changed'
    policy_path = real_pp
    assert TRAIN_TASKS[:63] == [(v, s) for v in ROUND_AB for s in range(N_SEEDS)] and len(TRAIN_TASKS) >= 72
    print('older suites and train tasks unchanged; round C train tasks 63-71:', TRAIN_TASKS[63:])
    assert {i[4] for i in train_instances(np.random.default_rng(0), 50)} == {BUDGET}
    b = np.array([i[4] for i in train_instances(np.random.default_rng(0), 4000, budgets=(300, 10000))])
    assert b.min() >= 300 and b.max() <= 10000 and abs(np.log(np.median(b) / np.sqrt(300 * 10000))) < 0.1, b
    for v in VB:  # tiny warm-start source at seed 99, then 2 fine-tuning generations from it
        src = VARIANTS[v]['init_from'].rsplit('_s', 1)[0]
        train(src, 99, gens=1, pairs=2, n_inst=2, budget=200)
        VARIANTS[v] = dict(VARIANTS[v], init_from=f'{src}_s99')
        train(v, 99, gens=2, pairs=2, n_inst=2, budgets=(50, 400))
        th0, th = np.load(OUT / f'policy_{src}_s99.npy'), np.load(OUT / f'policy_{v}_s99.npy')
        assert th.shape == th0.shape and np.abs(th - th0).max() < 0.2, (v, np.abs(th - th0).max())  # 2 Adam steps
    SUITES['budgets_smoke'] = dict(SUITES['budgets'], seeds=[3000], fns=['rastrigin', 'honeypot'], dims=(5, 150),
                                   budgets=(60, 1200), variants=[], base=['cmaes', 'optuna_gp']
                                   + [warm_method(v) for v in VB] + [f'rl:{v}:s99' for v in VB])
    jobs = all_jobs('budgets_smoke')
    assert len(jobs) == 7 * 2 * 2 * 2 * 2 + 2 * 2, len(jobs)  # GP only at n=5, budget 60
    assert jobs[0][0] == 'optuna_gp' and {j[5] for j in jobs} == {60, 1200}
    try:
        evaluate('budgets_smoke')
        report('budgets_smoke')
    finally:
        shutil.rmtree(OUT / 'results_budgets_smoke', ignore_errors=True)
        for p in OUT.glob('*_s99.*'): p.unlink()
    print('smoke_vb ok')


def smoke_hpx(src=None):
    """Round E check: mean-of-repeats best, widen() keeps the source policy, and the new actions run."""
    class Dummy(MeanBest):
        def __init__(self): self.obs, self.best_y, self.best_x = {}, np.inf, None
    d, a, b = Dummy(), np.zeros(2), np.ones(2)
    d.record(a, 0.5); d.record(b, 0.3); assert d.best_y == 0.3 and (d.best_x == b).all()
    d.record(b, 0.9); assert d.best_y == 0.5 and (d.best_x == a).all() and d.stats(b) == (0.6, 2)
    d.record(b, 0.0); assert abs(d.best_y - 0.4) < 1e-12 and (d.best_x == b).all()
    src_cfg, cfg = VARIANTS['cov_hp'], VARIANTS['hpx_walker']
    old = np.load(src) if src else init_theta(src_cfg, np.random.default_rng(0))
    th = widen(old, src_cfg, cfg)
    P0, P = unpack(old, src_cfg), unpack(th, cfg)
    assert all((P[k][tuple(slice(0, s) for s in v.shape)] == v).all() for k, v in P0.items())
    problems.HP_TRIALS = 5
    for theta, c in [(th, cfg), (old, VARIANTS['hpx_base'])]:
        p = Problem('honeypot', 50, 7, 0.01, 300)
        navigate(p, np.random.default_rng(1), theta, c)
        reps = sum(v[1] > 1 for v in p.obs.values())
        assert p.evals == 300 and sum(v[1] for v in p.obs.values()) == 300, p.evals
        print('extra' if c.get('extra') else 'base ', 'score', round(p.score(), 2), 'points re-checked', reps)
    r = centered_ranks(np.array([3., 1, 1, 2])); assert np.allclose(r, [0.5, -1 / 3, -1 / 3, 1 / 6]), r
    g = [np.array([0.8]), None, np.array([0.25, 0.5, 0.75])]
    z = snap_moves(np.array([0.3, 0.3, 0.3]), np.array([0.35, 0.35, 0.2]), g)  # wall gene stays shut -> flips open
    assert np.allclose(z, [0.9, 0.35, 0.2]), z  # coordinate 3 already changed cell: left alone
    assert np.allclose(snap_moves(np.array([0.9, .5, .6]), np.array([0.95, .5, .9]), g), [0.9, .5, .9])  # top cell: stays
    for ek, ak in ((5, 0), (0, 5), (5, 5)):  # forced exploration / forced acceptance run to budget
        p = Problem('honeypot', 50, 7, 0.01, 300); p.explore_every, p.accept_every = ek, ak
        navigate(p, np.random.default_rng(1), th, cfg)
        assert p.evals == 300, (ek, ak, p.evals)
    print('smoke_hpx ok')


if __name__ == '__main__':
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else 'smoke'), sys.argv[2:]
    if cmd == 'train':
        variant, seed = TRAIN_TASKS[int(args[0])]
        if VARIANTS[variant].get('learner'):
            from experiments import rlalgos
            rlalgos.train(variant, seed, *(int(a) for a in args[1:]))
        else:
            train(variant, seed, *(int(a) for a in args[1:]))
    elif cmd == 'val': val(*TRAIN_TASKS[int(args[0])])
    elif cmd == 'eval': evaluate(args[0], *(int(a) for a in args[1:]))
    elif cmd == 'select': select()
    elif cmd == 'report': report(args[0])
    elif cmd == 'smoke':  # 1 tiny ES generation per variant, then every method once
        for v in ROUND_AB:
            if VARIANTS[v].get('learner'):
                from experiments import rlalgos
                rlalgos.WARMUP = 300  # so the smoke run exercises the update step
                rlalgos.train(v, 99, steps=2000, budget=200)
            else:
                train(v, 99, gens=1, pairs=2, n_inst=2, budget=200)
        for m in SUITES['test']['base'] + [f'rl:{v}:s99' for v in ROUND_AB]:
            print(run_one((m, 'rastrigin', 10, 1, 0.01, 200)))
        for p in OUT.glob('*_s99.*'): p.unlink()
    elif cmd == 'smoke_vb': smoke_vb()
    elif cmd == 'smoke_hpx': smoke_hpx(*args)
    else: sys.exit(__doc__)
