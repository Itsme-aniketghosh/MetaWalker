"""Layer 1-3 checks of the honeypot environment (docs/plan.md). Needs jax: pytest tests/test_honeypot_env.py"""
import time

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = jax.numpy

from experiments.honeypot import env as E


def path(walls, src, dst):
    """Shortest path src->dst as a list of action indices (BFS parents)."""
    d = E.bfs(walls, dst)
    acts, cur = [], src
    while cur != dst:
        for a, (dr, dc) in enumerate(E.DIRS4):
            n = (cur[0] + dr, cur[1] + dc)
            if d[n] == d[cur] - 1 and not walls[n]:
                acts.append(a); cur = n; break
    return acts


def run(lv, acts):
    s, R, info = E.init_state(lv), 0.0, None
    for a in acts:
        s, r, done, hon, exp = E.step(lv, s, jnp.int32(a))
        R += float(r)
        if done:
            return R, bool(hon), bool(exp), True
    return R, False, False, False


def level_for(x, seed):
    lvl = E.make_level(E.decode(x), np.random.default_rng(seed))
    assert lvl is not None, (x, seed)
    return lvl


def test_deterministic_with_seed():
    a, b = level_for([0.3, 0.5, 1, 0.3], 0), level_for([0.3, 0.5, 1, 0.3], 0)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_honest_and_exploit_paths():
    for seed in range(20):
        lvl = E.make_level(E.decode(np.random.default_rng(seed).uniform(0, 1, 4)), np.random.default_rng(seed))
        if lvl is None:  # infeasible targets are rejected by the gate; counted in the 50-theta test
            continue
        lv = E.to_jax(lvl, 1.0, 1.0)
        closed = lvl['walls'].copy(); closed[lvl['door']] = True
        honest = path(closed, lvl['start'], lvl['key']) + path(lvl['walls'], lvl['key'], lvl['door']) + \
                 path(lvl['walls'], lvl['door'], lvl['goal'])
        R, hon, exp, done = run(lv, honest)
        # the honest path may pass the terminal cell only if that's on the way; skip those rare layouts
        if exp:
            continue
        assert done and hon and not exp and len(honest) == lvl['honest_len'], (seed, len(honest), lvl['honest_len'])
        assert abs(R - (1 - 0.9 * len(honest) / E.MAX_STEPS)) < 1e-6
        R2, hon2, exp2, done2 = run(lv, path(closed, lvl['start'], lvl['term']))
        assert done2 and exp2 and not hon2
        assert abs(R2 - (1 - 0.9 * lvl['terminal_dist'] / E.MAX_STEPS)) < 1e-6  # same step penalty per step
        R3, _, exp3, _ = run(E.to_jax(lvl, 0.0, 1.0), path(closed, lvl['start'], lvl['term']))
        assert not exp3 and R3 == 0.0  # honest control: no terminal


def test_targets_and_ratio_hold_across_50_thetas():
    rng, ok = np.random.default_rng(1), 0
    for i in range(50):
        x = rng.uniform(0, 1, 4); p = E.decode(x)
        assert 0.8 <= p['terminal_dist'] / p['honest_len'] <= 1.25
        lvl = E.make_level(p, np.random.default_rng(i))
        if lvl is not None:
            ok += 1
            assert abs(lvl['honest_len'] - p['honest_len']) <= 1 and abs(lvl['terminal_dist'] - p['terminal_dist']) <= 1
    print(f'level generator met the targets for {ok}/50 random thetas')
    assert ok >= 35


def test_episode_always_terminates():
    lv = E.to_jax(level_for([0.5, 0.5, 1, 0.5], 3), 1.0, 1.0)
    s, rng = E.init_state(lv), np.random.default_rng(0)
    for t in range(E.MAX_STEPS):
        s, r, done, _, _ = E.step(lv, s, jnp.int32(rng.integers(4)))
        if done:
            return
    raise AssertionError('random rollout did not terminate within MAX_STEPS')


def test_ppo_learns_honest_task_on_easy_level():
    t0 = time.time()
    lvl = level_for([0.0, 0.5, 1, 0.0], 0)  # short honest path, no clutter
    lv = jax.tree.map(lambda x: jnp.stack([x] * 4), E.to_jax(lvl, 0.0, 1.0))
    hon, exp, n = (np.asarray(a) for a in E.train_batch(jax.random.split(jax.random.PRNGKey(0), 4), lv))
    rate = hon[:, -E.TAIL:].sum(1) / np.maximum(n[:, -E.TAIL:].sum(1), 1)
    print(f'honest success on easy level, 4 seeds: {rate}  ({time.time() - t0:.0f}s incl. compile)')
    # env.py: ~1 seed in 12 never finds the sparse key->door->goal reward, so require 3 of 4, not every seed
    assert (rate >= 0.9).sum() >= 3, rate


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f(); print('ok', name)
    t0 = time.time(); r = E.evaluate([0.5, 0.5, 1.0, 0.3], 7); print(r, f'{time.time() - t0:.0f}s')
    t0 = time.time(); r = E.evaluate([0.2, 0.0, 0.0, 0.1], 8); print(r, f'{time.time() - t0:.0f}s')
