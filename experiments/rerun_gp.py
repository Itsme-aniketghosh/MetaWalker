"""Give Optuna GP all the time it can get: rerun every GP run that hit the 3-hour cap, uncapped, all cores.

  python -m experiments.rerun_gp list      -> gp_reruns.json  (the capped GP jobs from results_test_hpo + results_hp_hpo)
  python -m experiments.rerun_gp run IDX   -> results_gpfull/part_IDX.json   (one job per Slurm task; set NAV_TIME_CAP and
                                  NAV_TORCH_THREADS in the environment, e.g. 1e9 and the task's CPU count)
compare.py swaps these in for the capped records.
"""
import json, sys

from experiments import benchmark as nb

HERE = nb.OUT
SUITES = ('test_hpo', 'hp_hpo')


def capped():
    jobs = []
    for suite in SUITES:
        for f in sorted((HERE / f'results_{suite}').glob('part_*.json')):
            for r in json.load(open(f)):
                if r['method'] == 'optuna_gp' and r.get('timeout') and not r.get('error'):
                    jobs.append([r['method'], r['fn'], r['n'], r['seed'], r['noise'], nb.BUDGET])
    return sorted(jobs, key=lambda j: -j[2])  # longest (highest-dimensional) first


if __name__ == '__main__':
    if sys.argv[1] == 'list':
        jobs = capped()
        json.dump(jobs, open(HERE / 'gp_reruns.json', 'w'))
        print(len(jobs), 'capped GP runs; dims:', sorted({j[2] for j in jobs}))
    elif sys.argv[1] == 'run':
        i = int(sys.argv[2])
        job = tuple(json.load(open(HERE / 'gp_reruns.json'))[i])
        res = nb.run_one(job)
        (HERE / 'results_gpfull').mkdir(exist_ok=True)
        json.dump([res], open(HERE / 'results_gpfull' / f'part_{i}.json', 'w'))
        print(res)
