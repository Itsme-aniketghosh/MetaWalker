"""How much of the 169-parameter honeypot space does each searcher actually touch in one 600-evaluation search?

On the honeypot-like training family (n = 169, 1% live, 5 trials), every evaluated point is decoded with the real
honeypot cut points (walls flip at 0.8; start / key / terminal on an 11x11 grid, goal 11x3, door row 11):
- marginal: share of the 169 x 16 (coordinate, value-bin) cells that were sampled;
- positions: share of start / key / terminal / goal cells and door rows tried;
- walls: share of the 154 wall cells seen both open and shut.
Methods: random, CMA-ES, hpx_walker, and hpx_walker with grid-aware moves + Latin-hypercube jumps.

Usage: python -m experiments.analysis.coverage   -> prints the table, writes results/analysis/coverage_report.txt
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from metawalker import problems
from metawalker.baselines import random_search
from metawalker.cma import cmaes
from metawalker.problems import Problem
from metawalker.walker import navigate
from experiments import benchmark as nb

N, K = 169, 154
CUTS = ([np.array([0.8])] * K + [np.arange(1, 11) / 11] * 7 + [np.arange(1, 3) / 3] + [np.arange(1, 11) / 11]
        + [None] * 6)  # same layout as honeypot.search.level_grid('manyh')
assert len(CUTS) == N


class Rec(Problem):
    def __init__(self, *a, grid=None, strat=False):
        super().__init__(*a); self.X = []
        if grid is not None:
            self.grid = grid
        self.strat_jumps = strat

    def __call__(self, x):
        y = super().__call__(x)
        if np.isfinite(y):
            self.X.append(np.array(x, float))
        return y


def coverage(X):
    X = np.clip(np.array(X), 0, 1)
    bins = np.minimum((X * 16).astype(int), 15)
    marg = np.mean([len(set(bins[:, i])) / 16 for i in range(N)])
    cell = lambda u, m: np.minimum((u * m).astype(int), m - 1)
    pos = {}
    for name, j, (nr, nc) in (('start', K, (11, 11)), ('key', K + 2, (11, 11)), ('term', K + 4, (11, 11)),
                              ('goal', K + 6, (11, 3))):
        pos[name] = len(set(zip(cell(X[:, j], nr), cell(X[:, j + 1], nc)))) / (nr * nc)
    pos['door'] = len(set(cell(X[:, K + 8], 11))) / 11
    w = X[:, :K] > 0.8
    walls = np.mean(w.any(0) & (~w).any(0))
    return marg, pos, walls


def main():
    problems.HP_FRAC, problems.HP_TRIALS = nb.HPFAM_FRAC, 5
    th = np.load(ROOT / 'results' / 'policies' / 'policy_hpx_walker_s0.npy')
    cfg = nb.VARIANTS['hpx_walker']
    runs = {'random': lambda p, r: random_search(p, r), 'CMA-ES': lambda p, r: cmaes(p, r),
            'hpx_walker': lambda p, r: navigate(p, r, th, cfg),
            'hpx_walker + grid + LHS jumps': lambda p, r: navigate(p, r, th, cfg)}
    lines = ['method                         marginal  start   key    term   goal   door   walls(both)  repeats']
    for name, f in runs.items():
        rows = []
        for seed in range(4):
            opt = name.endswith('jumps')
            p = Rec('honeypot', N, 500 + seed, 0.01, 600, grid=CUTS if opt else None, strat=opt)
            f(p, np.random.default_rng(seed + 1))
            m, pos, w = coverage(p.X)
            keys = {tuple(np.minimum((np.clip(x[:K + 9], 0, 1) * [*([1.25] * K), *([11] * 7), 3, 11]).astype(int), 10))
                    for x in p.X}  # decoded level, ignoring PPO settings
            rows.append([m, pos['start'], pos['key'], pos['term'], pos['goal'], pos['door'], w, 1 - len(keys) / len(p.X)])
        r = np.mean(rows, 0)
        lines.append(f'{name:30s} ' + '  '.join(f'{v:5.2f}' for v in r[:6]) + f'   {r[6]:5.2f}       {r[7]:5.2f}')
    lines.append('marginal = share of 169 x 16 (coordinate, value-bin) cells sampled; repeats = share of evaluations '
                 'whose decoded level was already tried')
    out = '\n'.join(lines)
    print(out)
    (ROOT / 'results' / 'analysis' / 'coverage_report.txt').write_text(out + '\n')


if __name__ == '__main__':
    main()
