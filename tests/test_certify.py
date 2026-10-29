import math

import numpy as np
import pytest

from metawalker import minimize
from metawalker.api import Result
from metawalker.certify import certify

TRUE = {'lucky': 0.5, 'good': 0.8, 'meh': 0.4, 'bad': 0.2}
OBSERVED = {'lucky': 1.0, 'good': 0.85, 'meh': 0.6, 'bad': 0.3}  # 'lucky' hit 100% once, like the honeypot run


def _result(maximize=True):
    hist = [({'arm': a}, v) for a, v in OBSERVED.items()] + [({'arm': 'lucky'}, 0.7), ({'arm': 'x'}, math.nan)]
    r = Result({'arm': 'lucky'}, 1.0, len(hist), 'random', hist)
    r.maximize = maximize
    return r


def test_picks_the_truly_better_candidate_over_the_lucky_one():
    wins = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        c = certify(lambda p: TRUE[p['arm']] + rng.normal(0, 0.1), _result(), top_k=3, repeats=5, seed=seed)
        wins += c.best_params == {'arm': 'good'}
        assert c.n_extra_evals == 15 and len(c.table) == 3  # duplicates of 'lucky' and the nan trial are skipped
        assert c.table[0]['params'] == {'arm': 'lucky'} and c.table[0]['observed'] == 1.0
    assert wins >= 18
    assert c.shrinkage == pytest.approx(1.0 - c.mean)  # the search score was inflated by ~0.2
    assert 'certified best' in c.summary()


def test_maximize_is_read_from_the_result_and_defaults_to_minimize():
    f = lambda p: TRUE[p['arm']]  # noqa: E731
    assert certify(f, _result(), top_k=4, repeats=1).best_params == {'arm': 'good'}
    r = _result()
    del r.maximize
    c = certify(f, r, top_k=4, repeats=1)
    assert not c.maximize and c.best_params == {'arm': 'bad'}


def test_ci_contains_the_true_mean_under_gaussian_noise():
    hits = 0
    for seed in range(40):
        rng = np.random.default_rng(100 + seed)
        c = certify(lambda p: 1.0 + rng.normal(0, 0.5), Result({'a': 1}, 0.0, 1, 'random', [({'a': 1}, 0.0)]),
                    top_k=1, repeats=20, seed=seed)
        assert c.ci_low <= c.mean <= c.ci_high
        hits += c.ci_low <= 1.0 <= c.ci_high
    assert hits >= 34  # nominal 95%; percentile bootstrap undercovers a little at n=20


def test_failures_are_excluded_and_counted():
    calls = {'n': 0}

    def flaky(p):
        calls['n'] += 1
        if calls['n'] % 3 == 0:
            raise RuntimeError('crash')
        if calls['n'] % 3 == 1 and p['arm'] == 'good':
            return math.nan
        return TRUE[p['arm']]

    c = certify(flaky, _result(), top_k=3, repeats=6, seed=0)
    assert c.n_extra_evals == 18 and c.n_failed == sum(r['n_failed'] for r in c.table) > 0
    for row in c.table:
        assert row['n'] + row['n_failed'] == 6 and row['mean'] == TRUE[row['params']['arm']]
    assert c.best_params == {'arm': 'good'} and c.mean == 0.8


def test_adaptive_respects_its_extra_evaluation_budget():
    for top_k, repeats in [(4, 5), (3, 1), (1, 3), (4, 8)]:
        calls = {'n': 0}
        rng = np.random.default_rng(top_k)

        def f(p):
            calls['n'] += 1
            return TRUE[p['arm']] + rng.normal(0, 0.05)

        c = certify(f, _result(), top_k=top_k, repeats=repeats, seed=0, adaptive=True)
        assert calls['n'] == c.n_extra_evals == min(top_k, 4) * repeats
        if top_k >= 2 and repeats >= 5:
            assert c.best_params == {'arm': 'good'}
            assert max(r['n'] for r in c.table) > repeats  # survivors got more than an even share


def test_works_with_a_minimize_result():
    rng = np.random.default_rng(0)
    noisy = lambda p: (p['x'] - 0.3) ** 2 + (p['y'] - 0.7) ** 2 + rng.normal(0, 0.01)  # noqa: E731
    r = minimize(noisy, {'x': (0.0, 1.0), 'y': (0.0, 1.0)}, budget=40, method='random', seed=0)
    c = certify(noisy, r, top_k=3, repeats=3, seed=0)
    assert c.n_extra_evals == 9 and c.best_params in [p for p, _ in r.history]
    assert c.table[0]['observed'] == r.best_value and c.shrinkage == pytest.approx(r.best_value - c.mean)
    assert isinstance(c.summary(), str)
