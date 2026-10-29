import importlib.util

import pytest

import importlib
R = importlib.import_module('metawalker.recommend')  # the module; `metawalker.recommend` itself is the exported function
from metawalker import make_problem


def sphere(params):
    return sum((v - 0.3) ** 2 for v in params.values())


def space(n):
    return {f'x{i}': (0.0, 1.0) for i in range(n)}


def test_plateau_problem_gets_walker_hp_warm():
    fn, sp = make_problem('plateau', n=10, seed=0, noise=0.01)
    rec = R.recommend(fn, sp, budget=200, seed=0)
    assert rec.method == 'walker_hp_warm' and rec.profile['plateau_frac'] > 0.5, rec


def test_cheap_50d_sphere_gets_rlcma():
    rec = R.recommend(sphere, space(50), budget=200, seed=0)
    assert rec.method == 'rlcma' and rec.profile['plateau_frac'] < 0.5, rec


def test_5d_sphere_gets_cmaes():
    rec = R.recommend(sphere, space(5), budget=200, seed=0)
    assert rec.method == 'cmaes' and rec.reasons, rec


@pytest.mark.skipif(importlib.util.find_spec('optuna') is None, reason='needs optuna')
def test_expensive_objective_gets_gp(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(R, 'perf_counter', lambda: clock[0])

    def slow(params):  # "sleeps" 1.1 s on the fake clock
        clock[0] += 1.1
        return sphere(params)

    rec = R.recommend(slow, space(5), budget=100, seed=0)
    assert rec.profile['n_probe'] == 12 and rec.method == 'gp', rec
    assert abs(rec.profile['sec_per_eval'] - 1.1) < 1e-9


@pytest.mark.parametrize('budget', [5, 12, 50, 200, 1000])
def test_probe_stays_within_its_share(budget):
    calls = []
    rec = R.recommend(lambda p: calls.append(1) or sphere(p), space(3), budget=budget, seed=1)
    share = min(budget, max(12, int(0.08 * budget)))
    assert len(calls) == len(rec.probe_history) == rec.profile['n_probe'] <= share


def test_failures_are_counted():
    def flaky(params):
        if params['x0'] < 0.5:
            raise ValueError('invalid config')
        return sphere(params)

    rec = R.recommend(flaky, space(5), budget=200, seed=0)
    assert 0 < rec.profile['failure_rate'] < 1


def test_minimize_auto_spends_total_budget():
    calls = []
    res = R.minimize_auto(lambda p: calls.append(1) or sphere(p), space(5), budget=100, seed=0)
    assert len(calls) == res.n_evals == len(res.history) == 100
    assert res.recommendation.method == res.method == 'cmaes'
    assert res.best_value == min(v for _, v in res.history)
