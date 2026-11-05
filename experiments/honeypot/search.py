"""Outer search over honeypot levels: find the level / PPO settings that maximize PPO's exploit rate.

Uses the metawalker optimizers (CMA-ES, TPE, GP, random, hand walker, trained policies, rlbo) on the real
objective f(theta) = 1 - exploit_rate (invalid levels score their penalty >= 1.05, worse than any valid level).
Every evaluation is logged; the top candidates are then re-measured on fresh PPO seeds and the best of them is the
certified result (winner's-curse check).

Methods: random, cmaes, tpe, optuna_gp, smac, walker_hand_cov, rl:VARIANT:SEED (a trained benchmark policy; add
':warm' to start from the best points of a 20% Latin-hypercube warm-up), handoff:VARIANT:SEED (walker 30%, then
CMA-ES), cmawalk:VARIANT:SEED (CMA-ES 50%, then walkers), rlbo:sK, ns:METHOD (any metawalker.minimize method), and
panel (the package's original default panel, HONEYPOT_PANEL: the budget split equally, run in turn), and
panel:K (member K of that panel alone, so the members run as parallel jobs; then `combine` certifies the search).
Options (env): HP_REEVAL re-measurement seeds (10); HP_FREE=1 structural rejections cost no budget; HP_CACHE=1
re-proposed levels are free; HP_GRID / HP_STRAT walker grid moves / Latin-hypercube jumps; HP_CERT10 certify every
tie at the top; HP_EXPLORE / HP_ACCEPT forced walker exploration. Files go to $NAV_OUT/runs.

Usage: python -m experiments.honeypot.search SPACE METHOD SEARCH_SEED [BUDGET]   (SPACE: 4 | many | 4h | manyh)
       -> runs/<space>_<method>_s<seed>.jsonl + .json
       python -m experiments.honeypot.search combine SPACE SEARCH_SEED   (after panel:0..2 -> runs/<space>_panel_s<seed>.json)
       python -m experiments.honeypot.search train TASK [GENS PAIRS]   (hpfam_* / hpx_* variants: benchmark.TRAIN_TASKS)
"""
import json, os, sys, time

# one BLAS thread per process: training runs a 32-process pool, and 32 x 32 threads stalled pycma ~300x
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

from metawalker import problems
from metawalker.baselines import optuna_search, random_search, smac_hpo
from metawalker.cma import cma_then_walk, cmaes, handoff
from metawalker.policy import nparam
from metawalker.problems import MeanBest
from metawalker.walker import navigate
from experiments import benchmark as nb

HERE = nb.OUT
INVALID = 1.05
REEVAL_SEEDS = int(os.environ.get('HP_REEVAL', 10))
CERT_TOP = 3
WARMUP = 0.2
FREE, FREE_CAP = os.environ.get('HP_FREE') == '1', 50
CACHE = os.environ.get('HP_CACHE') == '1'  # an already-evaluated level + PPO settings returns its mean, free
GRID = os.environ.get('HP_GRID') == '1'    # walkers step to the next decoded cell (metawalker.walker.snap_moves)
STRAT = os.environ.get('HP_STRAT') == '1'  # walker jumps form a streaming Latin hypercube
CERT_TIES = os.environ.get('HP_CERT10') == '1'  # certify every candidate tied with the best, up to 10, not just 3
EXPLORE, ACCEPT = int(os.environ.get('HP_EXPLORE', 0)), int(os.environ.get('HP_ACCEPT', 0))  # metawalker.walker.navigate
PPO_REJECT = 'honest path not learnable'  # the only rejection that costs PPO runs (env.evaluate_many)


def hpfam_instances(rng, k, budget=nb.BUDGET, fns=None, budgets=None):
    """benchmark.train_instances for the hpfam_* variants: honeypot family at HPFAM_DIMS, binomial noise only."""
    assert fns == nb.HP_FAMILY and budgets, (fns, budgets)
    lo, hi = np.log(budgets)
    return [('honeypot', int(rng.choice(nb.HPFAM_DIMS)), int(rng.integers(1e9)), nb.HPFAM_NOISE,
             int(round(np.exp(rng.uniform(lo, hi))))) for _ in range(k)]


TRAINING = sys.argv[1:2] == ['train']
COMBINE = sys.argv[1:2] == ['combine']  # reads the members' summaries only: no PPO, no jax
if TRAINING:  # at module level: spawn-started Pool workers re-run this file as __mp_main__ and need HP_FRAC too
    problems.HP_FRAC, nb.train_instances = nb.HPFAM_FRAC, hpfam_instances
    problems.HP_TRIALS = nb.VARIANTS[nb.TRAIN_TASKS[int(sys.argv[2])][0]].get('hp_trials', problems.HP_TRIALS)
elif not COMBINE:
    from experiments.honeypot import env as E  # jax: kept out of the training processes, which fork a pool


def level_grid(space):
    """Cut points of env.decode_many per search coordinate: wall genes, position picks, door row; None = continuous."""
    def levels(cells, axis):
        return len({c[axis] for c in cells})
    cuts = lambda m: np.arange(1, m) / m
    g = [np.array([E.WALL_IF_ABOVE])] * len(E.CELLS)
    for cells in (E.LEFT, E.LEFT, E.LEFT, E.RIGHT):  # start, key, terminal, goal: (row, col) picks
        g += [cuts(levels(cells, 0)), cuts(levels(cells, 1))]
    g += [cuts(E.H - 2), None if space == 'manyh' else np.array([0.5])] + [None] * 5  # door row, visibility, PPO hps
    assert len(g) == E.N_MANY, len(g)
    return g


class HoneypotProblem(MeanBest):
    """Same interface the metawalker optimizers expect (call, budget, evals, best_x, scale, noise_sd, stats)."""

    def __init__(self, space, search_seed, budget, log):
        self.space, self.n, self.budget, self.seed = space, E.SPACES[space][0], budget, search_seed
        self.cap_budget = budget  # free rejections / cache hits allowed: FREE_CAP x this
        self.scale, self.noise_sd = 1.0, 0.05
        self.evals, self.best_y, self.best_x, self.obs = 0, np.inf, None, {}
        self.strat_jumps = STRAT
        self.explore_every, self.accept_every = EXPLORE, ACCEPT
        if GRID and space in ('many', 'manyh'):
            self.grid = level_grid(space)
        self.log, self.hist = log, []

    offset = 0  # panel:K members start at their share's place in the panel, so PPO seeds match the sequential panel

    free = 0  # HP_FREE=1: levels failing the structural checks (no PPO run) cost no budget, up to FREE_CAP x budget

    cached = 0

    def key(self, x):
        """With HP_CACHE=1: the decoded level plus the PPO settings in 2% bins, so moves too small to change the
        level pool with it (the walkers repeated a level on ~half their PPO runs in round D)."""
        if not (CACHE and self.space in ('many', 'manyh')):
            return super().key(x)
        lv, _, vis = E.decode_many(x, hard=self.space == 'manyh')
        return (lv['walls'].tobytes() + np.array([*lv['start'], *lv['key'], *lv['door'], *lv['goal'], *lv['term'], vis]).tobytes()
                + np.round(np.clip(np.asarray(x, float)[-5:], 0, 1) * 50).astype(int).tobytes())

    def recheck(self, x):
        self._fresh = True
        try:
            return self(x)
        finally:
            self._fresh = False

    _fresh = False

    def __call__(self, x):
        if self.evals >= self.budget:
            return np.inf
        k = self.key(x) if CACHE and not self._fresh else None
        if k in self.obs and self.cached < FREE_CAP * self.cap_budget:
            self.cached += 1
            y = self.obs[k][0] / self.obs[k][1]
            self.log.write(json.dumps(dict(x=[float(v) for v in x], y=y, eval=self.evals, cached=True, free=True)) + '\n')
            return y
        r = E.evaluate(x, seed=1_000_003 * self.seed + self.offset + self.evals + 1, space=self.space)  # fresh PPO seeds each eval
        y = 1.0 - r['exploit'] if r['valid'] else (r.get('penalty') or INVALID)
        free = FREE and not r['valid'] and r['reason'] != PPO_REJECT and self.free < FREE_CAP * self.cap_budget
        self.free += free
        if not free:
            self.evals += 1
            self.hist.append((y, np.array(x, float)))
        self.log.write(json.dumps(dict(r, x=[float(v) for v in x], y=y, eval=self.evals, free=bool(free))) + '\n')
        self.log.flush()
        self.record(x, y)
        return y


def run_warm_policy(prob, rng, theta, cfg):
    """metawalker.minimize(warmup=0.2) semantics for a benchmark policy: Latin-hypercube warm-up inside the budget,
    then walkers start at the best 4 points, CMA-ES at the best one."""
    from metawalker.api import _warmup
    X, F = _warmup(prob, rng, min(max(4, round(WARMUP * prob.budget)), prob.budget))
    if cfg['kind'] == 'walker':
        navigate(prob, rng, theta, cfg, x0=X[:cfg['M']], f0=F[:cfg['M']])
    else:
        cmaes(prob, rng, theta, cfg, x0=X[0])


class Share:
    """One panel member's view of the search: its own share of the budget, on the shared problem (same history,
    cache and certification pool), as metawalker.minimize(method='panel') gives each method a share."""

    def __init__(self, prob, budget):
        self.prob, self.budget, self.start = prob, budget, prob.evals

    evals = property(lambda self: self.prob.evals - self.start)

    def __call__(self, x):
        return self.prob(x) if self.evals < self.budget else np.inf

    def __getattr__(self, name):  # n, best_x, stats, recheck, ...: the shared problem's
        return getattr(self.prob, name)


PANEL = HONEYPOT_PANEL = ('rlcma', 'walker_hp', 'cmaes')  # metawalker's default panel up to 0.2.6: the honeypot runs


def run_panel(prob, rng, methods):
    """Equal split, no racing, each method seeded from rng: metawalker.minimize's panel, on the honeypot."""
    from metawalker.api import _run
    for i, m in enumerate(methods):
        share = prob.budget // len(methods) + (1 if i < prob.budget % len(methods) else 0)
        _run(m, Share(prob, share), np.random.default_rng(int(rng.integers(2 ** 31))))


def panel_member(budget, rng, k):
    """(method, share, offset, seed) of member k of HONEYPOT_PANEL: the same split and seeds as run_panel."""
    shares = [budget // len(PANEL) + (1 if i < budget % len(PANEL) else 0) for i in range(len(PANEL))]
    seeds = [int(rng.integers(2 ** 31)) for _ in PANEL]
    return PANEL[k], shares[k], sum(shares[:k]), seeds[k]


def combine_panel(space, search_seed):
    """One panel search from its members' runs, certified as run() certifies a single search: the CERT_TOP best
    candidates by search score over all members, then the best of those on the re-measurements. Every member
    re-measured its own top CERT_TOP on the same fresh seeds, so the overall top CERT_TOP are already measured."""
    out = HERE / 'runs'
    parts = [json.load(open(out / f'{space}_panel-{k}_s{search_seed}.json')) for k in range(len(PANEL))]
    pool = [dict(t, member=PANEL[k]) for k, p in enumerate(parts) for t in p['cert_table']]
    top = sorted(pool, key=lambda t: -t['search_exploit'])[:CERT_TOP]  # stable: ties keep member order, then rank
    cert = max(top, key=lambda t: t['reeval_exploit'])
    summary = dict(space=space, method='panel', seed=search_seed, budget=sum(p['budget'] for p in parts),
                   reeval_seeds=parts[0]['reeval_seeds'], search_exploit=max(p['search_exploit'] for p in parts),
                   cert_exploit=cert['reeval_exploit'], cert_member=cert['member'], cert_params=cert['params'],
                   members={PANEL[k]: p['cert_exploit'] for k, p in enumerate(parts)},
                   sec=max(p['sec'] for p in parts), cert_table=top)
    json.dump(summary, open(out / f'{space}_panel_s{search_seed}.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'cert_table'}))


def remeasure(x, space, seeds):
    rs = [E.evaluate(x, seed=s, space=space) for s in seeds]
    return rs, float(np.mean([r['exploit'] if r['valid'] else 0.0 for r in rs]))


def run(space, method, search_seed, budget):
    out = HERE / 'runs'; out.mkdir(exist_ok=True)
    stem = out / f"{space}_{method.replace(':', '-')}_s{search_seed}"
    rng = np.random.default_rng(search_seed + 1)
    t0 = time.time()
    full_budget = budget
    if method.startswith('panel:'):
        member, budget, offset, member_seed = panel_member(budget, rng, int(method.split(':')[1]))
    with open(f'{stem}.jsonl', 'w') as log:
        prob = HoneypotProblem(space, search_seed, budget, log)
        if method.startswith('panel:'):
            # a member's free allowance is a full search's, as for any single method; capped at its own share, the
            # walker spent its PPO budget on structural rejections once the cap ran out
            prob.offset, prob.cap_budget = offset, full_budget
        if method.startswith('rl:'):
            pol = method.removesuffix(':warm')
            cfg = nb.VARIANTS[pol.split(':')[1]]
            theta = np.load(nb.policy_path(pol))
            assert theta.shape == (nparam(cfg),), (method, theta.shape)
            (run_warm_policy if method.endswith(':warm') else nb.run_policy)(prob, rng, theta, cfg)
        elif method.startswith('handoff:'):  # 'handoff:VARIANT:SEED': walker explores 30%, then CMA-ES (metawalker.cma.handoff)
            pol = 'rl:' + method.removeprefix('handoff:')
            handoff(prob, rng, np.load(nb.policy_path(pol)), nb.VARIANTS[pol.split(':')[1]])
        elif method.startswith('rlbo:'):  # 'rlbo:sK': the surrogate-feature RL optimizer (metawalker.rlbo)
            from metawalker import rlbo
            rlbo.rlbo(prob, rng, np.load(HERE / f"rlbo_policy_{method.split(':')[1]}.npy"))
        elif method.startswith('cmawalk:'):  # 'cmawalk:VARIANT:SEED': CMA-ES 50%, then walkers (metawalker.cma.cma_then_walk)
            pol = 'rl:' + method.removeprefix('cmawalk:')
            cma_then_walk(prob, rng, np.load(nb.policy_path(pol)), nb.VARIANTS[pol.split(':')[1]])
        elif method.startswith('ns:'):  # the metawalker package's optimizers, e.g. walker_hp_warm, cmaes_rule
            from metawalker.api import _run
            _run(method[3:], prob, rng)
        elif method == 'panel':
            run_panel(prob, rng, PANEL)
        elif method.startswith('panel:'):
            from metawalker.api import _run
            _run(member, prob, np.random.default_rng(member_seed))
        elif method == 'walker_hand_cov': navigate(prob, rng, hand_beta=1.0, hand_kappa=1.0)
        elif method == 'optuna_gp':  # no time cap: a PPO evaluation costs far more than the GP's own overhead
            optuna_search(prob, rng, 'gp')
        elif method == 'smac': smac_hpo(prob, rng)
        else: {'random': random_search, 'cmaes': cmaes, 'tpe': optuna_search}[method](prob, rng)
    assert prob.evals == budget == len(prob.hist), (method, prob.evals, len(prob.hist))
    seeds = [10 ** 9 + 7919 * k + search_seed for k in range(REEVAL_SEEDS)]  # same fresh seeds for every candidate
    # distinct points by mean over repeats; stable sort keeps first-seen order on ties, as MeanBest does for best_x
    ranked = sorted(prob.obs.items(), key=lambda kv: kv[1][0] / kv[1][1])
    if CERT_TIES:  # ties at the top (a saturated search score) broken by more measurements, then first seen
        ranked = sorted(ranked, key=lambda kv: (kv[1][0] / kv[1][1], -kv[1][1]))
        best = ranked[0][1][0] / ranked[0][1][1]
        ranked = ranked[:max(CERT_TOP, min(10, sum(t / c == best for _, (t, c) in ranked)))]
    else:
        ranked = ranked[:CERT_TOP]
    top = [(s / c, prob.reps[k].copy()) for k, (s, c) in ranked]
    assert CERT_TIES or np.array_equal(top[0][1], prob.best_x)
    table = []
    for rank, (y, x) in enumerate(top):
        rs, m = remeasure(x, space, seeds)
        table.append(dict(rank=rank, search_exploit=1 - y, reeval_exploit=m, x=x.tolist(), params=rs[0]['params'],
                          reeval_valid=float(np.mean([r['valid'] for r in rs]))))
    cert = max(table, key=lambda t: t['reeval_exploit'])  # max is first-wins on ties: prefers the search's best
    summary = dict(space=space, method=method, seed=search_seed, budget=budget, reeval_seeds=REEVAL_SEEDS,
                   best_x=prob.best_x.tolist(), best_params=table[0]['params'], search_exploit=1 - prob.best_y,
                   reeval_exploit=table[0]['reeval_exploit'], reeval_valid=table[0]['reeval_valid'],
                   cert_exploit=cert['reeval_exploit'], cert_rank=cert['rank'], cert_params=cert['params'],
                   cert_x=cert['x'], free_evals=prob.free, cached_evals=prob.cached, cert_table=[{k: v for k, v in t.items() if k != 'x'} for t in table],
                   sec=time.time() - t0)
    json.dump(summary, open(f'{stem}.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith('_x')}))


if __name__ == '__main__':
    if TRAINING:
        variant, seed = nb.TRAIN_TASKS[int(sys.argv[2])]
        assert variant in nb.HPFAM, f'task {sys.argv[2]} is {variant}: train other variants with experiments.benchmark'
        nb.train(variant, seed, *(int(a) for a in sys.argv[3:]))
    elif COMBINE:
        combine_panel(sys.argv[2], int(sys.argv[3]))
    else:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 300)
