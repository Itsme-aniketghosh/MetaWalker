"""The walker: M walkers share the evaluation budget; each step one walker proposes a move.

Per coordinate it picks -1 / 0 / +1 (a shared MLP over 8 features); globally it picks a step-size change,
momentum, pull toward the best walker and (dirs='cov') a step in the span of its past successful steps. Policies
with extra actions may instead re-check the best point or jump to a random point. theta=None runs the hand-coded
walker (1/5th success rule), optionally with a distilled rule. Problems may opt into extra behaviour through
attributes: grid (decoded cell boundaries), strat_jumps, explore_every, accept_every (see navigate()).
"""
import numpy as np

from .policy import BETAS, GAMMAS, KAPPAS, R_MEM, S_MAX, S_MIN, mults, pick, unpack

# distilled walker rule (results/analysis/REPORT.md): sparse moves, pull to best, high-success step target
WALKER_RULE = dict(p_move=0.6, gamma=0.5, target=0.35)


class GreedyQ:
    """Acts eps-greedily on a trained Q-walker's weights (same layout as the ES policy)."""

    def __init__(self, P, eps):
        self.P, self.eps = P, eps

    def act(self, feats, g, rng):
        P, eps = self.P, self.eps
        qc = np.tanh(feats @ P['W1'].T + P['b1']) @ P['W2'].T + P['b2']
        d = np.where(rng.random(len(qc)) < eps, rng.integers(3, size=len(qc)), qc.argmax(1))
        heads = [P[h] @ g for h in Q_HEADS if h in P]
        return d, [int(rng.integers(len(q))) if rng.random() < eps else int(q.argmax()) for q in heads]

    def observe(self, reward, done):
        pass


Q_HEADS = ['Vs', 'Vb', 'Vg', 'Vk', 'Va']


def navigate(prob, rng, theta=None, cfg=None, hand_beta=0.0, hand_kappa=0.0, M=4, rule=None, x0=None, f0=None,
             actor=None):
    """Multi-start walker on prob (n, budget, evals, __call__(x) -> value to minimize) over [0, 1]^n.
    theta=None: the hand-coded walker (1/5th success rule); rule (hand path only, e.g. WALKER_RULE): move each
        coordinate with prob p_move, pull non-best walkers by gamma toward the best one, and set
        s <- s*exp(0.5*(succ_EMA - target)) every step instead of the 1/5th rule.
    x0, f0: start the M walkers at these points with these known values instead of random points.
    actor (RL training, experiments/rlalgos.py): .act(feats, g, rng) -> (per-coord action idx, global head idxs);
        .observe(r, done).
    Optional problem attributes: grid (per-coordinate cut points: snap_moves), strat_jumps (Latin-hypercube jumps),
        explore_every / accept_every (force a non-top choice / accept a worse point after k greedy / stuck steps)."""
    n = prob.n
    cfg = cfg or {}
    M = cfg.get('M', M)
    P = unpack(theta, cfg) if theta is not None else None
    if P is not None and cfg.get('learner') == 'dqn' and actor is None:
        actor = GreedyQ(P, cfg.get('eps', 0.05))
    learned = P is not None or actor is not None
    use_cov = (learned and cfg['dirs'] == 'cov') or hand_kappa > 0
    extra, SM = learned and cfg.get('extra', False), mults(cfg)
    assert not (extra and actor is not None), 'extra actions are ES-only'
    if actor is not None:
        floor = max(1e-10 * prob.scale, 2 * prob.noise_sd)  # reward ignores "improvements" inside the noise
    x = rng.uniform(0, 1, (M, n)) if x0 is None else np.array(x0, float)  # x0, f0: M start points and their values
    fx = np.array([prob(xi) for xi in x]) if f0 is None else np.array(f0, float)
    assert x.shape == (M, n), (x.shape, M, n)
    s = np.full(M, 0.2)
    last_d = np.zeros((M, n)); mom_axis = np.zeros((M, n)); mvec = np.zeros((M, n))
    mem = np.zeros((M, R_MEM, n)); mem_k = np.zeros(M, int)
    last_ok = np.zeros(M); succ = np.full(M, 0.2)
    damp = np.sqrt(n + 1)
    grid = getattr(prob, 'grid', None)  # per-coordinate cut points of a discrete decoding (None: continuous)
    strat = getattr(prob, 'strat_jumps', False)  # jumps as a streaming Latin hypercube (16 bins per coordinate)
    jperm, jk = None, 0
    explore_k = getattr(prob, 'explore_every', 0)  # >0: after k top-choice-only steps, force one non-top choice
    accept_k = getattr(prob, 'accept_every', 0)    # >0: after k steps without moving, accept the next point anyway
    greedy_run, stall, fh = np.zeros(M, int), np.zeros(M, int), [-1]

    def choose(logits, h):
        """pick(), except head h == fh[0] may not take its top choice; any non-top choice resets the greedy run."""
        top = int(np.argmax(logits))
        a = pick(rng, np.where(np.arange(len(logits)) == top, -np.inf, logits) if h == fh[0] else logits)
        if a != top:
            greedy_run[w] = 0
        return a

    t = 0
    while prob.evals < prob.budget:
        w = t % M; t += 1
        b = int(np.argmin(fx))
        mnorm = np.linalg.norm(mvec[w])
        m_unit = mvec[w] / mnorm if mnorm > 0 else mvec[w]
        kappa, alpha = hand_kappa, 1.0
        prev_best = prob.best_y if actor is not None else None  # only the RL reward reads it
        if not learned:
            d = rng.choice([-1.0, 1.0], n)
            if rule:
                d = d * (rng.uniform(size=n) < rule['p_move'])
            beta, gamma = hand_beta, (rule or {}).get('gamma', 0.0)
        else:
            feats = np.stack([np.ones(n), last_d[w], last_d[w] * last_ok[w], mom_axis[w], 2 * x[w] - 1,
                              np.full(n, succ[w]), m_unit * np.sqrt(n),
                              np.clip((x[b] - x[w]) / s[w], -3, 3)], 1)                      # (n, K)
            g = np.array([1.0, succ[w], np.log(s[w]) / np.log(S_MIN), t / prob.budget,
                          min(mnorm / np.sqrt(n), 3), float(w == b)])
            if actor is None:
                nh = extra + 3 + 2 * use_cov
                fh[0] = int(rng.integers(nh)) if explore_k and greedy_run[w] >= explore_k else -1
                greedy_run[w] += 1
            if extra:
                g = np.append(g, np.log2(prob.stats(x[b])[1]) / 4)
                act = choose(P['Vx'] @ g, 0)
                if act == 1:  # re-check the best point: a lucky noisy best loses its lead
                    getattr(prob, 'recheck', prob)(x[b]); fx[b] = prob.stats(x[b])[0]  # recheck: bypass a cache
                    continue
                if act == 2:  # jump: probe a random point, move there only if it beats this walker
                    if strat:  # every 16 jumps hit each coordinate's 16 bins once each
                        if jk % 16 == 0:
                            jperm = rng.permuted(np.tile(np.arange(16), (n, 1)), axis=1)
                        xj = (jperm[:, jk % 16] + rng.uniform(0, 1, n)) / 16; jk += 1
                    else:
                        xj = rng.uniform(0, 1, n)
                    yj = prob(xj)
                    if yj < fx[w]:
                        x[w], fx[w], s[w] = xj, yj, 0.2
                        mom_axis[w] = 0; mvec[w] = 0; succ[w] = 0.2; mem_k[w] = 0
                    continue
            if actor is not None:
                di, hi = actor.act(feats, g, rng)
                d = di - 1.0
                s[w] = np.clip(s[w] * SM[hi[0]], S_MIN, S_MAX)
                beta, gamma = BETAS[hi[1]], GAMMAS[hi[2]]
                if use_cov:
                    kappa, alpha = KAPPAS[hi[3]], float(hi[4])
            else:
                logits = np.tanh(feats @ P['W1'].T + P['b1']) @ P['W2'].T + P['b2']         # (n, 3)
                d = np.argmax(logits + rng.gumbel(size=(n, 3)), 1) - 1.0
                e = int(extra)  # head numbers for choose(): Vx 0 (extra only), then Vs, Vb, Vg, Vk, Va
                s[w] = np.clip(s[w] * SM[choose(P['Vs'] @ g, e)], S_MIN, S_MAX)
                beta, gamma = BETAS[choose(P['Vb'] @ g, e + 1)], GAMMAS[choose(P['Vg'] @ g, e + 2)]
                if use_cov:
                    kappa, alpha = KAPPAS[choose(P['Vk'] @ g, e + 3)], float(choose(P['Va'] @ g, e + 4))
        step = alpha * d + beta * m_unit * np.sqrt(n)    # every term has the length scale of a full +-1 vector
        r = min(mem_k[w], R_MEM)
        if kappa > 0 and r > 0:
            step = step + kappa * (rng.standard_normal(r) @ mem[w, :r]) * np.sqrt(n / r)
        if not step.any() and (gamma == 0 or w == b):
            step[rng.integers(n)] = rng.choice([-1.0, 1.0])  # never propose a zero move
        x_new = np.clip(x[w] + s[w] * step + gamma * (x[b] - x[w]), 0, 1)
        if grid is not None:
            x_new = snap_moves(x[w], x_new, grid)
        y = prob(x_new)
        ok = float(y < fx[w])
        if ok:
            delta = x_new - x[w]
            mvec[w] = 0.7 * mvec[w] + 0.3 * delta / s[w]
            if use_cov and np.linalg.norm(delta) > 0:
                mem[w, mem_k[w] % R_MEM] = delta / np.linalg.norm(delta); mem_k[w] += 1
            x[w], fx[w] = x_new, y
            mom_axis[w] = 0.7 * mom_axis[w] + 0.3 * d
        else:
            mvec[w] *= 0.9
        stall[w] = 0 if ok else stall[w] + 1
        if accept_k and stall[w] >= accept_k and np.isfinite(y):  # forced: leave a point it can no longer beat
            x[w], fx[w], stall[w] = x_new, y, 0
        if rule:  # succ is the EMA before this step, as in the distillation study
            s[w] = np.clip(s[w] * np.exp(0.5 * (succ[w] - rule['target'])), S_MIN, S_MAX)
        elif not learned:  # 1/5th success rule
            s[w] = np.clip(s[w] * np.exp((0.8 if ok else -0.2) / damp), S_MIN, S_MAX)
        succ[w] = 0.8 * succ[w] + 0.2 * ok
        last_d[w], last_ok[w] = d, ok
        if s[w] <= S_MIN and prob.evals < prob.budget:  # converged: restart this walker elsewhere
            x[w] = rng.uniform(0, 1, n); fx[w] = prob(x[w]); s[w] = 0.2
            mom_axis[w] = 0; mvec[w] = 0; succ[w] = 0.2; mem_k[w] = 0
        if actor is not None:
            done = prob.evals >= prob.budget
            if cfg.get('reward') == 'sparse':  # one number per episode, at the end: the true final score gain
                r = -prob.score() if done else 0.0
            else:  # dense: per-step rewards sum to the log-improvement of the best observed value
                r = np.log10(max(prev_best, floor) / max(prob.best_y, floor)) if np.isfinite(prev_best) else 0.0
            actor.observe(r, done)


def snap_moves(x, x_new, grid):
    """A move too small to change a coordinate's decoded cell goes to the neighbouring cell's centre instead, so no
    evaluation re-tests the same config by accident. grid[i]: sorted cut points in (0, 1), or None (continuous)."""
    x_new = x_new.copy()
    for i in np.flatnonzero(x_new != x):
        cuts = grid[i]
        if cuts is None:
            continue
        c0, c1 = np.searchsorted(cuts, x[i], 'right'), np.searchsorted(cuts, x_new[i], 'right')
        if c0 == c1:
            c = min(max(c0 + (1 if x_new[i] > x[i] else -1), 0), len(cuts))
            x_new[i] = ((cuts[c - 1] if c else 0.0) + (cuts[c] if c < len(cuts) else 1.0)) / 2
    return x_new
