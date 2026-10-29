import math

import numpy as np

from metawalker import Float, minimize
from metawalker.importance import _rank, importance


def _random_history(fn, n=300, seed=0, d=3):
    rng = np.random.default_rng(seed)
    out = []
    for row in rng.random((n, d)):
        p = {f'x{i}': float(v) for i, v in enumerate(row)}
        out.append((p, fn(p)))
    return out


def _objective(p):  # x0 matters a lot (lower is better), x1 a little (higher is better), x2 not at all
    return 10 * p['x0'] - 2 * p['x1'] + 0 * p['x2']


def test_ranking_and_direction():
    rng = np.random.default_rng(1)
    hist = [(p, v + rng.normal(0, 0.3)) for p, v in _random_history(_objective)]
    imp = importance(hist)
    s = imp.scores
    assert s['x0'] > s['x1'] > s['x2'] and s['x2'] < 0.03
    assert abs(sum(s.values()) - 1) < 1e-9
    d = imp.details
    assert d['x0'].direction == 'lower is better' and d['x1'].direction == 'higher is better'
    assert d['x2'].direction == 'no clear trend'
    assert d['x0'].best[1] < 0.3  # best bin is at the low end
    lo, hi = d['x0'].eta2_ci
    assert lo <= d['x0'].eta2 <= hi
    # maximize flips the direction
    assert importance(hist, maximize=True, n_boot=0).details['x0'].direction == 'higher is better'
    text = imp.summary()
    assert text.index('x0') < text.index('x1') < text.index('x2') and 'lower is better' in text


def test_best_category_and_failed_trials():
    rng = np.random.default_rng(0)
    effect = {'relu': 0.0, 'tanh': 1.0, 'gelu': 2.0}
    hist = []
    for i in range(200):
        p = {'act': ['relu', 'tanh', 'gelu'][i % 3], 'lr': float(rng.random())}
        hist.append((p, effect[p['act']] + 0.1 * rng.normal()))
    hist += [({'act': 'relu', 'lr': 0.5}, math.nan)] * 7
    imp = importance(hist, n_boot=50)
    assert imp.n_failed == 7 and imp.n_used == 200
    act = imp.details['act']
    assert act.kind == 'categorical' and act.best == 'relu' and imp.scores['act'] > 0.9
    assert importance(hist, maximize=True, n_boot=0).details['act'].best == 'gelu'
    assert '7 failed' in imp.summary()


def test_numeric_categorical_via_space_and_interaction():
    rng = np.random.default_rng(0)
    hist = []
    for _ in range(300):
        p = {'a': int(rng.choice([1, 2, 4])), 'b': float(rng.random()), 'c': float(rng.random())}
        hist.append((p, (p['b'] - 0.5) * (p['c'] - 0.5)))  # pure interaction, no main effects
    imp = importance(hist, space={'a': [1, 2, 4], 'b': (0.0, 1.0), 'c': (0.0, 1.0)}, n_boot=0)
    assert imp.details['a'].kind == 'categorical'
    assert max(imp.interactions, key=imp.interactions.get) == ('b', 'c') or \
        max(imp.interactions, key=imp.interactions.get) == ('c', 'b')


def test_rank_average_ties():
    assert _rank(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [2.5, 0.0, 2.5, 1.0]


def test_works_on_minimize_result():
    space = {'lr': Float(1e-4, 1e-1, log=True), 'x': (0.0, 1.0), 'act': ['relu', 'tanh']}

    def fn(p):
        if p['x'] > 0.95:
            raise ValueError('invalid')
        return (math.log10(p['lr']) + 3) ** 2 + 0.1 * p['x'] + (p['act'] == 'tanh')

    r = minimize(fn, space, budget=120, method='random', seed=0)
    imp = importance(r, space=space, n_boot=50)
    assert imp.n_failed == r.n_failed and imp.n_used + imp.n_failed == len(r.history)
    assert max(imp.scores, key=imp.scores.get) == 'lr'
    assert imp.details['act'].best == 'relu'
    assert 'lr' in imp.summary()
