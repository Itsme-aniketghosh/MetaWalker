import numpy as np

from metawalker import make_problem
from metawalker.train import TrainedPolicy, _init, train

BUDGET = 100


def rastrigin5(rng):
    return make_problem('rastrigin', 5, seed=int(rng.integers(10 ** 9)))


def plateau5():  # zero-arg factory form; top-level so its problems pickle for n_jobs > 1
    return make_problem('plateau', 5, seed=int(np.random.default_rng().integers(10 ** 9)))


def heldout_regret(pol, k=12):
    out = []
    for i in range(k):
        fn, space = make_problem('rastrigin', 5, seed=777 + i)
        out.append(fn.regret(pol.minimize(fn, space, budget=BUDGET, seed=i).best_params))
    return float(np.mean(out))


def test_training_from_scratch_beats_untrained():
    pol = train(rastrigin5, generations=8, pairs=10, instances_per_gen=6, budget=BUDGET, init='scratch', seed=0)
    untrained = TrainedPolicy(*_init('scratch', 'cov', 8, 4, np.random.default_rng(0)))
    assert len(pol.history) == 8 and np.isfinite(pol.history).all()
    # seeded, so deterministic; margin at seed 0 is ~0.09 decades (seeds 1, 2: 0.03, 0.12)
    assert heldout_regret(pol) < heldout_regret(untrained)


def test_save_load_roundtrip(tmp_path):
    pol = train([plateau5], generations=1, pairs=2, instances_per_gen=2, budget=30, init='walker_hp')
    pol.save(tmp_path / 'p.npz')
    back = TrainedPolicy.load(tmp_path / 'p.npz')
    assert back.cfg == pol.cfg and back.history == pol.history
    fn, space = make_problem('rastrigin', 5, seed=1)
    a, b = pol.minimize(fn, space, budget=60, seed=3), back.minimize(fn, space, budget=60, seed=3)
    assert a.best_value == b.best_value and a.best_params == b.best_params and a.n_evals == 60


def test_parallel_and_value_scored_problems():
    calls = []
    pol = train([plateau5, plateau5], generations=2, pairs=2, instances_per_gen=2, budget=30, init='walker',
                n_jobs=2, callback=lambda gen, p: calls.append(gen))
    assert calls == [0, 1] and pol.cfg['dirs'] == 'mom'
    # a plain function without .regret: scored by best value, z-scored per instance
    pol = train(lambda rng: (lambda p: (p['a'] - 0.3) ** 2, {'a': (0.0, 1.0), 'b': (0.0, 1.0)}),
                generations=2, pairs=2, instances_per_gen=2, budget=20)
    r = pol.minimize(lambda p: (p['a'] - 0.3) ** 2, {'a': (0.0, 1.0), 'b': (0.0, 1.0)}, budget=50, seed=0)
    assert r.n_evals == 50 and r.best_value < 0.01


def reward(p):  # something to maximize, peak 1 at a = 0.3
    return 1 - (p['a'] - 0.3) ** 2


def reward_problem(rng):
    return reward, {'a': (0.0, 1.0), 'b': (0.0, 1.0)}


def test_maximize_trains_on_the_reward_itself():
    # history is the mean of -score, and the score of a maximized function is -reward, so history ~ +reward
    pol = train(reward_problem, generations=2, pairs=2, instances_per_gen=2, budget=20, maximize=True)
    assert all(0.5 < h <= 1 for h in pol.history)
    assert pol.minimize(reward, reward_problem(None)[1], budget=50, seed=0, maximize=True).best_value > 0.99


def test_fine_tune_rlcma_and_rlbo(tmp_path):
    from metawalker import minimize, register
    from metawalker.api import CUSTOM, _load_rlbo
    for init, budget in (('rlcma', 60), ('rlbo', 25)):
        pol = train([plateau5], generations=2, pairs=2, instances_per_gen=2, budget=budget, init=init)
        assert pol.cfg['kind'] == init and len(pol.history) == 2 and np.isfinite(pol.history).all()
        pol.save(tmp_path / f'{init}.npz')
        try:
            register(f'my_{init}', tmp_path / f'{init}.npz')
            fn, space = make_problem('rastrigin', 5, seed=1)
            r = minimize(fn, space, budget=budget, method=f'my_{init}', seed=0, warmup=0.2)
            assert r.n_evals == budget
        finally:
            CUSTOM.pop(f'my_{init}', None)
    from metawalker.rlbo import F, H
    shipped = _load_rlbo()  # rlbo's first layer stays frozen; the rest moves
    assert np.array_equal(pol.theta[:H * F], shipped[:H * F]) and not np.array_equal(pol.theta, shipped)
