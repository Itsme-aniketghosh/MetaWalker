"""Re-measure chosen levels on brand-new PPO seeds: a winner's-curse check when a result is picked as the best of
several certified ones (e.g. a full-budget panel keeping its best member). The certified scores already used seeds
10**9 + ...; these use 2 * 10**9 + ..., so no level is scored on a seed that helped pick it.

Usage: python -m experiments.honeypot.recheck MANIFEST.json INDEX   -> recheck_<INDEX>.json next to the manifest
MANIFEST is a list of {"key", "space", "seed", "x"}; on the cluster, one array task per entry:
  sbatch --array=0-<N-1> -c 4 --mem=8G -t 2:00:00 --wrap 'module load python/3.13.5; export PYTHONPATH=$REPO/src:$REPO
         XLA_FLAGS=--xla_force_host_platform_device_count=1; $REPO/.venv/bin/python -m experiments.honeypot.recheck
         manifest.json $SLURM_ARRAY_TASK_ID'
"""
import json
import sys
from pathlib import Path

from experiments.honeypot.search import remeasure

N_SEEDS = 10

if __name__ == '__main__':
    path, i = Path(sys.argv[1]), int(sys.argv[2])
    item = json.load(open(path))[i]
    seeds = [2 * 10 ** 9 + 7919 * k + item['seed'] for k in range(N_SEEDS)]
    rs, exploit = remeasure(item['x'], item['space'], seeds)
    out = dict(item, exploit=exploit, valid=sum(r['valid'] for r in rs) / len(rs), seeds=seeds)
    json.dump(out, open(path.parent / f'recheck_{i}.json', 'w'))
    print(json.dumps({k: v for k, v in out.items() if k != 'x'}))
