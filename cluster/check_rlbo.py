"""Release check for minimize(method='rlbo'). Run it where the trained rlbo policies live (the cluster work dir).

Usage: python cluster/check_rlbo.py PATH/TO/rlbo_policy_s0.npy [--quick]

1. Copies the policy to src/metawalker/policies/rlbo_s0.npy, after checking its shape.
2. Runs the package's rlbo against CMA-ES on the benchmark's synthetic families (training families plus held-out
   Ackley and Levy) at 20 and 50 dimensions, budget 1000, 3 seeds, scored by true log10 regret.
3. Exits non-zero unless rlbo beats CMA-ES on at least 70% of instances. The research runs had 91-100% from 20
   dimensions up, so a lower number means the package wiring does not reproduce the research optimizer.
--quick: 10 dimensions, budget 200, 1 seed (checks the script runs; the threshold is not applied).
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import metawalker as mw  # noqa: E402
from metawalker.rlbo import NPARAM  # noqa: E402

FAMILIES = ('sphere', 'ellipsoid', 'rastrigin', 'rosenbrock', 'ackley', 'levy')
THRESHOLD = 0.7


def main(policy, quick=False):
    theta = np.load(policy)
    assert theta.shape == (NPARAM,), f'{policy}: shape {theta.shape}, expected ({NPARAM},): not an rlbo policy'
    dest = ROOT / 'src' / 'metawalker' / 'policies' / 'rlbo_s0.npy'
    shutil.copyfile(policy, dest)
    print(f'copied {policy} -> {dest.relative_to(ROOT)}')

    dims, budget, seeds = ((10,), 200, 1) if quick else ((20, 50), 1000, 3)
    problems = {f'{f}-{n}': mw.make_problem(f, n) for f in FAMILIES for n in dims}
    res = mw.compare(problems, methods=['rlbo', 'cmaes'], budget=budget, seeds=seeds,
                     n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    print(res.summary())
    win = res.win_rate('rlbo', 'cmaes')
    print(f'rlbo beats CMA-ES on {win:.0%} of {len(problems) * seeds} instances (research runs: 91-100%)')
    if quick:
        print('quick mode: threshold not applied')
    elif win < THRESHOLD:
        sys.exit(f'FAIL: below {THRESHOLD:.0%}; do not release this policy')
    else:
        print('PASS: ready to release')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--quick']
    if len(args) != 1:
        sys.exit(__doc__)
    main(args[0], quick='--quick' in sys.argv)
