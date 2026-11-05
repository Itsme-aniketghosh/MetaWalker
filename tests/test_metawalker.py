import math

import numpy as np
import pytest

from metawalker import Categorical, Float, Int, Space, minimize


def sphere(params):
    return sum((v - 0.3) ** 2 for v in params.values())


SPACE5 = {f'x{i}': (0.0, 1.0) for i in range(5)}


def test_space_roundtrip_and_types():
    sp = Space({'lr': Float(1e-4, 1e-1, log=True), 'depth': (2, 12), 'kernel': ['rbf', 'linear'], 'c': (0.0, 2.0)})
    p = sp.decode([0.5, 0.5, 0.9, 0.25])
    assert isinstance(p['depth'], int) and 2 <= p['depth'] <= 12
    assert p['kernel'] == 'linear' and abs(p['c'] - 0.5) < 1e-12
    assert abs(p['lr'] - math.sqrt(1e-4 * 1e-1)) < 1e-12  # log midpoint
    q = sp.decode(sp.encode(p))
    assert q['kernel'] == p['kernel'] and q['depth'] == p['depth'] and abs(q['lr'] - p['lr']) < 1e-9
    with pytest.raises(TypeError):
        Space({'bad': 'not a spec'})


@pytest.mark.parametrize('method', ['rlcma', 'walker', 'walker_hp', 'cmaes', 'random', 'auto', 'panel'])
def test_every_method_spends_budget_and_improves(method):
    r = minimize(sphere, SPACE5, budget=150, method=method, seed=0)
    assert r.n_evals == 150 and len(r.history) == 150
    first = r.history[0][1]
    assert r.best_value <= first and r.best_value == min(v for _, v in r.history)
    # easy 5-D sphere: every real optimizer gets close; random search only needs to beat a typical random point
    assert r.best_value < (0.25 if method == 'random' else 0.05), (method, r.best_value)


def test_seed_makes_runs_reproducible():
    a = minimize(sphere, SPACE5, budget=60, method='walker', seed=3)
    b = minimize(sphere, SPACE5, budget=60, method='walker', seed=3)
    assert a.best_value == b.best_value and a.best_params == b.best_params


def test_maximize_and_auto_choice():
    r = minimize(lambda p: -sphere(p), SPACE5, budget=80, method='auto', maximize=True, seed=1)
    assert r.method == 'cmaes' and r.best_value > -0.1  # 5 params -> CMA-ES
    big = {f'x{i}': (0.0, 1.0) for i in range(30)}
    assert minimize(sphere, big, budget=40, method='auto', seed=1).method == 'rlcma'


def test_invalid_configurations_are_failed_trials_not_crashes():
    def f(p):
        if p['x0'] > 0.5:
            raise ValueError('invalid config')
        return sphere(p)
    r = minimize(f, SPACE5, budget=100, method='walker', seed=0)
    assert r.n_evals == 100 and r.n_failed > 0 and r.best_params['x0'] <= 0.5


def test_recheck_reports_fresh_values():
    rng = np.random.default_rng(0)
    r = minimize(lambda p: sphere(p) + 0.01 * rng.standard_normal(), SPACE5, budget=60, method='cmaes',
                 seed=0, recheck=4)
    assert len(r.recheck_values) == 4


def test_metawalkercv_is_a_drop_in_for_gridsearchcv():
    sklearn = pytest.importorskip('sklearn')
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from metawalker.sklearn import MetaWalkerCV
    X, y = load_iris(return_X_y=True)
    grid = {'C': [0.01, 0.1, 1.0, 10.0], 'fit_intercept': [True, False]}  # a GridSearchCV param_grid, unchanged
    gs = GridSearchCV(LogisticRegression(max_iter=500), grid, cv=3).fit(X, y)
    ns = MetaWalkerCV(LogisticRegression(max_iter=500), grid, n_iter=12, method='walker', cv=3,
                     random_state=0).fit(X, y)
    assert len(ns.cv_results_['params']) == 12 and set(ns.best_params_) == {'C', 'fit_intercept'}
    assert ns.best_score_ >= gs.best_score_ - 0.02 and hasattr(ns, 'best_estimator_')
    ns2 = MetaWalkerCV(LogisticRegression(max_iter=500), {'C': Float(1e-3, 1e2, log=True)}, n_iter=10,
                      method='cmaes', cv=3, random_state=0).fit(X, y)
    assert 1e-3 <= ns2.best_params_['C'] <= 1e2


def test_simulated_problems_have_known_optimum_and_plateau_is_flat():
    from metawalker import FAMILIES, TestProblem
    for fam in FAMILIES:
        p = TestProblem(fam, 10, seed=1)
        opt = {f'x{i}': float(v) for i, v in enumerate((p.shift + 5) / 10)}  # z = 0 there
        assert abs(p.true(opt)) < 1e-6, fam
        assert p.scale > 0
    p = TestProblem('plateau', 20, seed=0)
    xs = np.random.default_rng(0).uniform(0, 1, (400, 20))
    live = np.mean([p.true({f'x{i}': v for i, v in enumerate(x)}) < 1 for x in xs])
    assert 0.01 < live < 0.12  # ~95% of the space gives no signal


def test_user_panel_and_unknown_method():
    r = minimize(sphere, SPACE5, budget=90, method=['walker', 'cmaes', 'random'], seed=0)
    assert r.n_evals == 90 and r.method.startswith('panel:')
    with pytest.raises(ValueError):
        minimize(sphere, SPACE5, budget=10, method=['walker', 'nonsense'])


def test_parallel_panel_gives_the_same_result_from_several_threads():
    import threading, time
    barrier, threads = None, set()

    def slow_sphere(p):  # sleeps like a call into torch/jax or a subprocess would, releasing the GIL
        if barrier is not None and threading.get_ident() not in threads:
            threads.add(threading.get_ident())
            barrier.wait()  # opens only when all 3 members are running at once
        time.sleep(0.001)
        return sphere(p)

    panel = ['rlcma', 'walker', 'cmaes']  # two CMA-based methods: pycma must not share numpy's global generator
    seq = minimize(slow_sphere, SPACE5, budget=90, method=panel, seed=0)
    barrier = threading.Barrier(3, timeout=30)
    par = minimize(slow_sphere, SPACE5, budget=90, method=panel, seed=0, n_jobs=3)
    assert len(threads) == 3 and not barrier.broken  # every member ran in its own thread, at the same time
    assert (par.best_value, par.best_params, par.method, par.n_evals) == (seq.best_value, seq.best_params, seq.method, 90)
    assert par.history == seq.history  # members' histories in panel order, whatever order they finished in


def test_compare_on_own_loss_and_simulated_problem():
    from metawalker import compare, make_problem
    res = compare({'my_loss': (sphere, SPACE5), 'sim': make_problem('rastrigin', 8, seed=2)},
                  methods=['walker', 'cmaes', 'random'], budget=80, seeds=2)
    assert len(res.rows) == 2 * 2 * 3
    ranks = res.ranks()
    assert set(ranks) == {'walker', 'cmaes', 'random'} and abs(sum(ranks.values()) - 6.0) < 1e-9  # ranks 1..3
    assert ranks['random'] > min(ranks['walker'], ranks['cmaes'])
    assert 0 <= res.win_rate('cmaes', 'random') <= 1 and 'win rate' in res.summary()
    matrix = res.summary().split('win rate of row vs column')[1].splitlines()[2:5]
    assert [line.split()[0] for line in matrix] == sorted(ranks, key=ranks.get)  # row labels


def test_gp_and_auto_expensive():
    pytest.importorskip('optuna'); pytest.importorskip('torch')  # Optuna's GPSampler needs torch
    r = minimize(sphere, SPACE5, budget=25, method='gp', seed=0)
    assert r.n_evals == 25 and r.best_value < sphere({k: 0.5 for k in SPACE5})
    def flaky(p):  # failed trials must not break the GP model
        return math.nan if p['x0'] > 0.8 else sphere(p)
    assert minimize(flaky, SPACE5, budget=20, method='gp', seed=0).n_failed >= 0
    assert minimize(sphere, SPACE5, budget=12, method='auto', expensive=True, seed=0).method == 'gp'


def test_optimizer_seed_never_replays_a_test_problem_seed():
    # regression: unsalted, minimize(seed=k)'s first random point reproduced make_problem(seed=k)'s optimum draws
    from metawalker import make_problem
    fn, sp = make_problem('sphere', 20, seed=0)
    r = minimize(fn, sp, budget=4, method='random', seed=0)
    first = np.array([r.history[0][0][f'x{i}'] for i in range(20)])
    assert np.linalg.norm(first - (fn.shift + 5) / 10) > 1.0


def test_cache_skips_repeated_configs():
    import metawalker as ns
    calls = []

    def f(p):
        calls.append(p)
        return (p['a'] - 3) ** 2 + (p['b'] - 1) ** 2

    space = {'a': ns.Int(0, 6), 'b': ns.Int(0, 3)}  # 28 configs: a search revisits them
    r = ns.minimize(f, space, budget=20, method='walker', seed=0)
    keys = [tuple(sorted(p.items())) for p in calls]
    assert len(keys) == len(set(keys)) == r.n_evals == len(r.history)  # no config evaluated twice
    r2 = ns.minimize(f, space, budget=20, method='walker', seed=0, cache=False)
    assert r2.n_evals == 20


def test_metawalkercv_accepts_legacy_randomstate():
    pytest.importorskip('sklearn')
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from metawalker.sklearn import MetaWalkerCV
    X, y = load_iris(return_X_y=True)
    for rs in (np.random.RandomState(0), np.random.default_rng(0)):  # regression: RandomState has no .integers
        ns = MetaWalkerCV(LogisticRegression(max_iter=500), {'C': Float(1e-3, 1e2, log=True)}, n_iter=6,
                         method='cmaes', cv=3, random_state=rs).fit(X, y)
        assert len(ns.cv_results_['params']) == 6


def test_walker_runs_on_a_bare_problem():
    from metawalker.walker import navigate

    class Bare:  # only the documented contract: n, budget, evals, __call__ (regression: walker read prob.best_y)
        n, budget, evals = 4, 60, 0

        def __call__(self, x):
            self.evals += 1
            return float(((x - 0.3) ** 2).sum())

    p = Bare()
    navigate(p, np.random.default_rng(0))
    assert p.evals == 60


def test_rlbo_without_policy_file_says_so(monkeypatch, tmp_path):
    import metawalker.api as api
    monkeypatch.setattr(api, '_rlbo_policy_path', lambda: tmp_path / 'rlbo_s0.npy')
    with pytest.raises(FileNotFoundError, match='rlbo_s0.npy'):
        minimize(sphere, SPACE5, budget=20, method='rlbo')


def test_rlbo_runs_on_a_mixed_space_within_budget(monkeypatch, tmp_path):
    """Wiring check with an untrained policy: ints and categoricals repeat points, and rlbo must not overrun."""
    import metawalker.api as api
    from metawalker.rlbo import init_theta
    np.save(tmp_path / 'rlbo_s0.npy', init_theta(np.random.default_rng(0)))
    monkeypatch.setattr(api, '_rlbo_policy_path', lambda: tmp_path / 'rlbo_s0.npy')
    space = {'a': (0.0, 1.0), 'k': Int(0, 3), 'c': ['x', 'y'], **{f'z{i}': (0.0, 1.0) for i in range(4)}}
    f = lambda p: (p['a'] - 0.3) ** 2 + (p['k'] - 2) ** 2 + (p['c'] == 'x') + sum(p[f'z{i}'] ** 2 for i in range(4))
    r = minimize(f, space, budget=60, method='rlbo', seed=0)
    assert r.n_evals == 60 and len(r.history) == 60 and math.isfinite(r.best_value)
    assert r.best_value < 2  # better than the typical random point


def grid5(f, n, budget, rng):
    """A user-supplied optimizer: 5-level grid search over the unit cube."""
    import itertools
    for x in itertools.product(np.linspace(0, 1, 5), repeat=n):
        if f.evals >= budget:
            break
        f(np.array(x))


@pytest.fixture
def no_custom(monkeypatch):
    import metawalker.api as api
    monkeypatch.setattr(api, 'CUSTOM', {})


def test_register_a_function_optimizer_works_in_minimize_and_panels(no_custom):
    from metawalker import register
    register('grid', grid5)
    f = lambda p: (p['a'] - 0.25) ** 2 + (p['b'] - 0.75) ** 2  # both optima sit on the 5-level grid
    r = minimize(f, {'a': (0.0, 1.0), 'b': (0.0, 1.0)}, budget=25, method='grid')
    assert r.n_evals == 25 and r.best_value == 0 and r.method == 'grid'
    p = minimize(f, {'a': (0.0, 1.0), 'b': (0.0, 1.0)}, budget=50, method=['grid', 'cmaes'], seed=0)
    assert p.n_evals <= 50 and p.best_value < 1e-3


def test_register_a_trained_walker_by_path_with_warmup(no_custom, tmp_path):
    from metawalker import register
    from metawalker.policy import load_policy
    from metawalker.train import TrainedPolicy
    theta, cfg = load_policy('walker_hp')
    TrainedPolicy(theta, cfg).save(tmp_path / 'mine.npz')
    register('my_walker', tmp_path / 'mine.npz')
    r = minimize(sphere, SPACE5, budget=120, method='my_walker', warmup=0.2, seed=1)
    assert r.n_evals == 120 and r.best_value < 0.05
    # the same policy under the built-in name gives the same search
    assert r.best_value == minimize(sphere, SPACE5, budget=120, method='walker_hp', warmup=0.2, seed=1).best_value


def test_registered_optimizers_reach_compare_worker_processes(no_custom):
    from metawalker import compare, make_problem, register
    from metawalker.policy import load_policy
    from metawalker.train import TrainedPolicy
    register('my_walker', TrainedPolicy(*load_policy('walker_hp')))
    res = compare({'sphere': make_problem('sphere', 5)}, methods=['my_walker', 'random'], budget=60, seeds=2, n_jobs=2)
    assert {r['method'] for r in res.rows} == {'my_walker', 'random'}


def test_register_rejects_bad_names_and_objects(no_custom):
    from metawalker import register
    with pytest.raises(ValueError):
        register('cmaes', grid5)
    with pytest.raises(TypeError):
        register('x', 42)
    with pytest.raises(ValueError, match='register'):
        minimize(sphere, SPACE5, budget=10, method='not_registered')


def _small_comparison():
    from metawalker import compare, make_problem
    probs = {'sphere': make_problem('sphere', 8), 'plateau': make_problem('plateau', 8, noise=0.01)}
    return compare(probs, methods=['cmaes', 'walker_hp', 'random'], budget=80, seeds=3)


def test_compare_keeps_curves_and_times_for_every_run():
    res = _small_comparison()
    for r in res.rows:
        c, t = r['curve'], r['times']
        assert len(c) == len(t) == r['evals'] and c[-1] == r['score']
        assert np.all(np.diff(t) >= 0) and t[-1] <= r['seconds']
        if r['problem'] == 'sphere':  # noiseless: the true regret of the pick never gets worse
            assert np.all(np.diff(c) <= 0)


def test_success_counts_and_budget_slices_agree_with_the_runs():
    res = _small_comparison()
    tg = res.targets()
    s = res.success()
    for m in res.methods:
        runs = [r for r in res.rows if r['method'] == m]
        assert s[m]['hits'] == sum(r['score'] <= tg[r['problem']] for r in runs) and s[m]['runs'] == 6
        assert s[m]['hits'] == 0 or 1 <= s[m]['evals'] <= 80
    assert res.at_budget(80).ranks() == res.ranks()            # the full prefix is the run itself
    small = res.at_budget(20)
    assert all(r['evals'] <= 20 and r['score'] >= full['score'] for r, full in zip(small.rows, res.rows))
    for text in (res.success_table(), res.budget_table(), res.summary()):
        assert 'cmaes' in text


def test_suggest_panel_is_a_usable_method_list():
    res = _small_comparison()
    panel = res.suggest_panel(2)
    assert 1 <= len(panel) <= 2 and set(panel) <= set(res.methods) and 0 <= panel.covered <= panel.n == 6
    assert len(panel.options) == 2 and panel.covered == max(c for _, c in panel.options)  # the best size wins
    assert 'suggested' in panel.summary()
    r = minimize(sphere, SPACE5, budget=40, method=panel, seed=0)
    assert r.method.startswith('panel:')


def test_win_rate_interval_brackets_the_rate():
    res = _small_comparison()
    w, lo, hi = res.win_rate('cmaes', 'random', ci=True)
    assert lo <= w <= hi and res.win_rate('cmaes', 'random') == w


def test_plot_draws_both_axes():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    fig = _small_comparison().plot()
    assert len(fig.axes) == 2


def test_panel_allocate_each_and_race():
    panel = ['random', 'cmaes', 'rlcma']
    each = minimize(sphere, SPACE5, budget=60, method=panel, seed=0, allocate='each')
    assert each.allocation == {m: 60 for m in panel} and each.n_evals == 180
    assert each.best_value == min(v for _, v in each.history)  # the best of every method's full run
    race = minimize(sphere, SPACE5, budget=120, method=panel, seed=0, allocate='race')
    assert sum(race.allocation.values()) == race.n_evals == 120
    first_round = 120 // 3 // 3 + 1  # 3 rounds; the first gives each method ~13
    assert min(race.allocation.values()) <= first_round  # one method was dropped after the first round
    assert max(race.allocation.values()) > 120 / 3 and race.method == 'race:' + max(panel, key=race.allocation.get)
    assert race.best_value == min(v for _, v in race.history)
    par = minimize(sphere, SPACE5, budget=120, method=panel, seed=0, allocate='race', n_jobs=3)
    assert (par.best_value, par.allocation, par.history) == (race.best_value, race.allocation, race.history)
    with pytest.raises(ValueError):
        minimize(sphere, SPACE5, budget=30, method=panel, allocate='greedy')
