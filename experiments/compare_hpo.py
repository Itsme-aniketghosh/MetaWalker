"""Learned optimizers vs current HPO optimizers on the same instances.

Merges results_test + results_test_hpo (seeds of the 'test_hpo' suite) and results_hp + results_hp_hpo
(seeds of 'hp_hpo'). Usage: python -m experiments.compare_hpo > report_hpo.txt
"""
import json

import numpy as np

from experiments import benchmark as nb

HERE = nb.OUT


def load(*dirs, seeds):
    res = [r for d in dirs for f in sorted((HERE / d).glob('part_*.json')) for r in json.load(open(f))]
    # uncapped GP reruns (rerun_gp.py) replace the matching capped records
    key = lambda r: (r['method'], r['fn'], r['n'], r['seed'], r['noise'])
    full = {key(r): r for f in sorted((HERE / 'results_gpfull').glob('part_*.json')) for r in json.load(open(f))}
    res = [full.get(key(r), r) for r in res]
    bad = [r for r in res if r.get('score') is None]
    if bad:
        print(f'note: {len(bad)} runs crashed before their first evaluation and are excluded: '
              f'{sorted({r["method"] for r in bad})}')
    return [r for r in res if r['seed'] in seeds and r.get('score') is not None]


def table(res, names, fns, title):
    print(f'\n{title}: median log10(regret / random-point value), lower is better')
    for noise in (0.0, 0.01):
        print(f'  noise={noise}')
        print('      n ' + ''.join(f'{m[-13:]:>14}' for m in names))
        for n in nb.DIMS:
            row = [np.median([r['score'] for r in res if r['method'] == m and r['n'] == n and r['noise'] == noise
                              and r['fn'] in fns] or [np.nan]) for m in names]
            print(f'  {n:5d} ' + ''.join(f'{v:14.2f}' for v in row))


def compare(res, names, champions, label):
    mr, by_inst = nb.mean_ranks(res, names)
    print(f'\n{label}: average rank over {len(by_inst)} instances (1 = best of {len(names)})')
    for m in sorted(names, key=mr.get):
        rs = [r for r in res if r['method'] == m]
        to = sum(r.get('timeout', False) for r in rs)
        er = sum(bool(r.get('error')) for r in rs)
        extra = (f'  stopped early {to}/{len(rs)} (of which crashed {er}), '
                 f'mean evals {np.mean([r.get("evals", nb.BUDGET) for r in rs]):.0f}') if to else ''
        print(f'  {m:22s} {mr[m]:6.2f}   {np.mean([r["sec"] for r in rs]):8.1f} s/run{extra}')
    print(f'\n{label}: share of instances where the learned method beats each HPO optimizer (ties 1/2) [95% CI]')
    for champ in champions:
        for opp in nb.HPO_METHODS + ['cmaes', 'tpe']:
            cells = []
            for n in nb.DIMS:
                pairs = [d for k, d in by_inst.items() if k[1] == n and champ in d and opp in d]
                if not pairs:
                    continue
                w = sum(nb.beats(d[champ], d[opp]) for d in pairs)
                lo, hi = nb.wilson(w, len(pairs))
                cells.append(f'n={n}: {w / len(pairs):.2f} [{lo:.2f},{hi:.2f}]')
            print(f'  {champ} vs {opp}:  ' + '  '.join(cells))


if __name__ == '__main__':
    sel = json.load(open(HERE / 'selected.json'))['selected']
    learned = [f'rl:{v}:s{s}' for v in sel for s in range(nb.N_SEEDS)]
    res = load('results_test', 'results_test_hpo', seeds=set(nb.SUITES['test_hpo']['seeds']))
    names = ['random', 'cmaes', 'tpe'] + nb.HPO_METHODS + learned
    table(res, names, nb.TRAIN_FNS, 'Standard functions, trained families')
    table(res, names, nb.HELDOUT_FNS, 'Standard functions, held-out families')
    compare(res, names, [f'rl:{v}:s0' for v in sel], 'Standard functions')
    res = load('results_hp', 'results_hp_hpo', seeds=set(nb.SUITES['hp_hpo']['seeds']))
    learned = ['rl:mom:s0', 'rl:mom:s1', 'rl:mom:s2', 'rl:cov_long:s0', 'rl:cov_long:s1', 'rl:cov_long:s2',
               'rl:rlcma_mlp:s0', 'rl:rlcma:s0', 'rl:cov_hp:s0']
    names = ['random', 'cmaes', 'tpe'] + nb.HPO_METHODS + learned
    table(res, names, ['honeypot'], 'Honeypot-like plateau family')
    compare(res, names, ['rl:mom:s0', 'rl:cov_long:s2'], 'Honeypot-like family')
