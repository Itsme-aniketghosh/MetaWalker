"""The refactored code must reproduce the pre-refactor code bit for bit: tests/golden/<platform>.json holds best
values from the old navbench.py / navsearch._core (same problems, same seeds), recorded before the split into
modules. Floating-point results differ across platforms and library builds, so each file is only compared on
the platform and library versions it was recorded with; anywhere else the tests skip."""
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments import benchmark
from metawalker import Float, Int, minimize, rlbo
from metawalker.baselines import optuna_search, random_search
from metawalker.cma import cma_then_walk, cmaes, handoff
from metawalker.policy import load_policy
from metawalker.problems import Problem
from metawalker.walker import WALKER_RULE, navigate

ROOT = Path(__file__).resolve().parents[1]
_gold = ROOT / 'tests' / 'golden' / f'{sys.platform}.json'
GOLD = json.load(open(_gold)) if _gold.exists() else {'runs': {}, 'train': {}, 'package': {}, 'env': {}}


def _version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


ENV = dict(platform=sys.platform, python=platform.python_version(),
           **{k: _version(k) for k in ('numpy', 'cma', 'scipy', 'optuna')})
CORE = ('platform', 'python', 'numpy', 'cma', 'scipy')
pytestmark = pytest.mark.skipif(any(GOLD['env'].get(k) != ENV[k] for k in CORE),
                                reason=f"golden outputs recorded with {GOLD['env'] or 'nothing'}; running {ENV}")
V = benchmark.VARIANTS
mom, cov, rlc = (load_policy(k)[0] for k in ('walker', 'walker_hp', 'rlcma'))
hpx = np.load(ROOT / 'results' / 'policies' / 'policy_hpx_walker_s0.npy')


def opts(p):  # every optional walker behaviour switched on
    p.grid = [np.array([0.5])] * (p.n // 2) + [None] * (p.n - p.n // 2)
    p.strat_jumps, p.explore_every, p.accept_every = True, 5, 5


CASES = {
    'hand': lambda p, r: navigate(p, r),
    'hand_cov': lambda p, r: navigate(p, r, hand_beta=1.0, hand_kappa=1.0),
    'walker_rule': lambda p, r: navigate(p, r, rule=WALKER_RULE),
    'mom': lambda p, r: navigate(p, r, mom, V['mom']),
    'cov_hp': lambda p, r: navigate(p, r, cov, V['cov_hp']),
    'hpx': lambda p, r: navigate(p, r, hpx, V['hpx_walker']),
    'hpx_opts': lambda p, r: (opts(p), navigate(p, r, hpx, V['hpx_walker'])),
    'cmaes': lambda p, r: cmaes(p, r),
    'cmaes_rule': lambda p, r: cmaes(p, r, rule=True),
    'rlcma': lambda p, r: cmaes(p, r, rlc, V['rlcma_mlp']),
    'handoff': lambda p, r: handoff(p, r, hpx, V['hpx_walker']),
    'cma_then_walk': lambda p, r: cma_then_walk(p, r, cov, V['cov_hp']),
    'random': lambda p, r: random_search(p, r),
    'tpe': lambda p, r: optuna_search(p, r),
    'rlbo_init': lambda p, r: rlbo.rlbo(p, r, None),
}


@pytest.mark.parametrize('key', sorted(GOLD['runs']))
def test_optimizer_runs(key):
    name, fn, n, seed, noise, budget = key.split('|')
    if name == 'tpe' and GOLD['env'].get('optuna') != ENV['optuna']:
        pytest.skip('optuna version differs from the recording')
    p = Problem(fn, int(n), int(seed), float(noise), int(budget))
    CASES[name](p, np.random.default_rng(int(seed) + 1))
    assert [repr(float(p.best_y)), p.evals] == GOLD['runs'][key]


@pytest.mark.parametrize('variant', sorted(GOLD['train']))
def test_es_training(variant, tmp_path, monkeypatch):
    monkeypatch.setenv('NAV_CPUS', '1')
    monkeypatch.setattr(benchmark, 'OUT', tmp_path)
    benchmark.train(variant, 99, gens=2, pairs=2, n_inst=2, budget=80)
    assert np.array_equal(np.load(tmp_path / f'policy_{variant}_s99.npy'), np.array(GOLD['train'][variant]))


@pytest.mark.parametrize('method', sorted(GOLD['package']))
def test_minimize(method):
    if method == 'tpe' and GOLD['env'].get('optuna') != ENV['optuna']:
        pytest.skip('optuna version differs from the recording')
    space = {f'x{i}': (0.0, 1.0) for i in range(6)} | {'k': Int(0, 5), 'lr': Float(1e-4, 1e-1, log=True)}
    f = lambda q: sum((q[f'x{i}'] - 0.3) ** 2 for i in range(6)) + (q['k'] - 2) ** 2 + (np.log10(q['lr']) + 2) ** 2
    r = minimize(f, space, budget=90, method=method, seed=4)
    assert [repr(float(r.best_value)), r.n_evals] == GOLD['package'][method]
