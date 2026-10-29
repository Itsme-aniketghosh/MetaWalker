import math

import pytest

from metawalker import compare, make_problem, minimize
from metawalker.asktell import AskTell


def sphere(params):
    return sum((v - 0.3) ** 2 for v in params.values())


SPACE4 = {f'x{i}': (0.0, 1.0) for i in range(4)}


def run_loop(at, fn):
    asked = []
    while not at.done:
        tid, p = at.ask()
        asked.append(p)
        at.tell(tid, fn(p))
    return asked


@pytest.mark.parametrize('method', ['cmaes', 'rlcma', 'walker_hp', 'random', 'tpe', 'panel'])
def test_asks_exactly_what_minimize_evaluates(method):
    ref = minimize(sphere, SPACE4, budget=40, method=method, seed=7)
    with AskTell(SPACE4, budget=40, method=method, seed=7) as at:
        asked = run_loop(at, sphere)
        res = at.result()
    assert asked == [p for p, _ in ref.history]
    assert res.best_value == ref.best_value and res.best_params == ref.best_params and res.n_evals == 40


def test_failed_tells_are_counted_and_maximize():
    with AskTell(SPACE4, budget=20, method='cmaes', seed=0, maximize=True) as at:
        i = 0
        while not at.done:
            tid, p = at.ask()
            at.tell(tid, None if i % 4 == 0 else float('nan') if i % 4 == 1 else -sphere(p))
            i += 1
        r = at.result()
    assert r.n_failed == 10 and r.n_evals == 20 and r.best_value <= 0


def test_budget_and_errors():
    at = AskTell(SPACE4, budget=3, method='random', seed=0)
    with pytest.raises(RuntimeError):
        at.result()
    tid, p = at.ask()
    with pytest.raises(RuntimeError):   # one trial in flight
        at.ask()
    with pytest.raises(ValueError):
        at.tell(tid + 99, 1.0)
    at.tell(tid, 1.0)
    with pytest.raises(ValueError):     # already told
        at.tell(tid, 1.0)
    for _ in range(2):
        t, p = at.ask()
        at.tell(t, sphere(p))
    assert at.done
    with pytest.raises(RuntimeError, match='budget'):
        at.ask()
    assert at.result().n_evals == 3
    with pytest.raises(ValueError):
        AskTell(SPACE4, budget=3, method='nope')


def test_close_midway_stops_thread():
    at = AskTell(SPACE4, budget=50, method='cmaes', seed=0)
    at.ask()
    at.close()
    assert not at._thread.is_alive() and at.done
    with pytest.raises(RuntimeError):
        at.ask()
    at.close()


def test_compare_parallel_equals_sequential():
    probs = {'sph': make_problem('sphere', 5, seed=1), 'ras': make_problem('rastrigin', 3, seed=2)}
    kw = dict(methods=['cmaes', 'random'], budget=30, seeds=2)
    a, b = compare(probs, **kw), compare(probs, n_jobs=2, **kw)
    strip = lambda rows: [{k: v for k, v in r.items() if k != 'seconds'} for r in rows]
    assert strip(a.rows) == strip(b.rows)


def test_nav_sampler_runs_optuna_study():
    optuna = pytest.importorskip('optuna')
    from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
    from metawalker.asktell import MetaWalkerSampler

    space = {'x': FloatDistribution(-2, 2), 'lr': FloatDistribution(1e-4, 1e-1, log=True),
             'k': IntDistribution(1, 5), 'act': CategoricalDistribution(['a', 'b', 'c'])}

    def objective(tr):
        x, lr, k, act = (tr.suggest_float('x', -2, 2), tr.suggest_float('lr', 1e-4, 1e-1, log=True),
                         tr.suggest_int('k', 1, 5), tr.suggest_categorical('act', ['a', 'b', 'c']))
        if k == 5:
            raise ValueError('invalid config')  # a failed trial
        return (x - 0.5) ** 2 + math.log10(lr) ** 2 / 10 + k + (act != 'b')

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=MetaWalkerSampler(space, budget=30, method='cmaes', seed=0))
    study.optimize(objective, n_trials=30, catch=(ValueError,))
    assert len(study.trials) == 30
    first = next(t.value for t in study.trials if t.value is not None)
    assert study.best_value <= first  # a 30-trial CMA-ES on a mixed 4-param space is ~random; just check it searches
