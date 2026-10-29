"""The distilled hand rules (walker_rule, cmaes_rule) and the random warm-up."""
import sys
from pathlib import Path

import numpy as np
import pytest

import metawalker.problems as PR
from metawalker import minimize
from metawalker.cma import cmaes
from metawalker.walker import WALKER_RULE, navigate
from metawalker.problems import TestProblem

POLICY = Path(__file__).resolve().parents[2] / 'analysis' / 'policy'


@pytest.mark.skipif(not (POLICY / 'common.py').exists(), reason='analysis code not present (installed package)')
@pytest.mark.parametrize('fn,n', [('rastrigin', 10), ('plateau', 12)])
def test_rules_identical_to_distillation_study(fn, n):
    """The package rules must be the exact rules the first distillation study measured, run for run."""
    sys.path.insert(0, str(POLICY))
    import common  # noqa: E402

    rule = dict(gamma=0.5, p_move=0.6, step=lambda s, ok, succ, tf, n_, mn: s * np.exp(0.5 * (succ - 0.35)))
    a, b = common.Prob(fn, n, 3, budget=400), common.Prob(fn, n, 3, budget=400)
    common.navigate(a, np.random.default_rng(5), rule=rule)
    navigate(b, np.random.default_rng(5), rule=WALKER_RULE)
    assert a.evals == b.evals == 400 and a.best_y == b.best_y

    shrink = lambda g: 0.8 if (g[3] == 0 and g[4] < 0.45 and g[5] > -1.2) else 1.0  # distill.rlcma_rule
    a, b = common.Prob(fn, n, 3, budget=400), common.Prob(fn, n, 3, budget=400)
    common.cmaes(a, np.random.default_rng(5), sigma_rule=shrink)
    cmaes(b, np.random.default_rng(5), rule=True)
    assert a.evals == b.evals == 400 and a.best_y == b.best_y


@pytest.mark.parametrize('method', ['walker_rule', 'cmaes_rule', 'walker_hp_warm'])
def test_new_methods_run_in_budget(method):
    fn = TestProblem('sphere', 5, 0)
    r = minimize(fn, fn.space, budget=120, method=method, seed=0)
    assert r.n_evals == len(r.history) == 120 and r.method == method
    assert fn.regret(r.best_params) < -1


@pytest.mark.parametrize('method', ['walker_hp', 'cmaes', 'rlcma', 'random'])
def test_warmup_counts_toward_budget_and_starts_from_it(method):
    fn = TestProblem('ellipsoid', 6, 1)
    r = minimize(fn, fn.space, budget=150, method=method, seed=2, warmup=0.2)
    assert r.n_evals == len(r.history) == 150
    warm = [v for _, v in r.history[:30]]
    assert r.best_value <= min(warm)
    assert minimize(fn, fn.space, budget=150, method=method, seed=2, warmup=0.2).best_value == r.best_value


def test_warmup_finds_rare_live_region(monkeypatch):
    """0.5% live plateau: the walkers shrink and restart instead of exploring; the warm-up fixes that."""
    monkeypatch.setattr(PR, 'PLATEAU_LIVE', 0.005)
    hits = {'walker_hp': 0, 'walker_hp_warm': 0}
    for sd in range(10):
        fn = TestProblem('plateau', 10, sd)
        for m in hits:
            r = minimize(fn, fn.space, budget=1000, method=m, seed=sd)
            hits[m] += r.best_value < 1
    assert hits['walker_hp_warm'] >= 8 and hits['walker_hp_warm'] > hits['walker_hp'], hits
