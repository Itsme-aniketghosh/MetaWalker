"""rlbo: an RL-trained acquisition policy over cheap surrogate predictions (cf. MetaBO, Volpp et al. 2020).

Each step:
  1. Surrogates. Targets = (y - median) / IQR with the bad tail above +1 log-compressed: affine-invariant
     (scale-free) and robust to outliers. Rank normal scores were tried first and bend the quadratic structure:
     on a 200-d sphere the global model's minimizer went from log10 regret -2.2 to -0.1 as good points arrived.
     local:  4 GPs on the NFIT evaluated points nearest the incumbent x_b, inputs centred at x_b. Kernels
             const+linear+diagonal-quadratic, and const+0.3 linear+RBF at 0.5/1/2 x the median distance to x_b.
             Mean = ensemble mean, spread = sqrt(mean variance + variance of means). O(NFIT^2 n) per step.
     global: Bayesian ridge on [1, |x|^2, x, x^2] (x - 0.5; per-coordinate curvature shrunk toward the shared
             one) over ALL points, Phi^T Phi updated incrementally, one (2n+2)^2 Cholesky per step. Gives a
             second mean/spread and the minimizer of the fitted separable quadratic.
  2. Candidates (C=48) from 6 generators: local Gaussian around x_b at radius rho*e^U(-1,1) (12), samples from
     the weighted mean/covariance of the top points (8), steps down the local mean gradient at 0.25..4 x rho
     (8), partial steps toward the global quadratic's minimizer (8), extrapolations x_b + t (x_b - x_j) away
     from other top points (6), uniform random points (6).
  3. Features per candidate (F=20): local mean robust z-score and rank among candidates, predicted
     improvement over x_b in sd units, log relative spread, spread, log(distance to x_b / rho), distance to x_b
     and to the nearest evaluated point (both / sqrt n), budget fraction, recent improvement rate, log rho,
     global mean z-score and predicted improvement, generator one-hot.
  4. Policy. A shared MLP (tanh, H=16; first layer frozen at random init) plus a linear skip scores each
     candidate, plus a per-generator bias from a global head on the state [1, budget fraction, improvement
     rate, last step improved, failures since the last radius change, log rho, explored], where explored =
     max(N / 2n, 2.5 * budget fraction): ~1 once the global quadratic is identifiable. That head is the learned
     epsilon-greedy: it decides when to sample far (uniform) points. A third head picks rho x {0.5, 1, 2}.
     Decisions are Gumbel-argmax (sampling from the softmax) at training AND at test, so ES optimizes exactly
     the evaluated policy. Argmax at test was tried on the warm start and was worse (n=100 sphere -1.61 vs
     -2.54): the sampling noise explores.
     Warm start (= 'rlbo:init'): uniform points until explored ~0.8 (N ~ 1.6 n or 32% of the budget), then the
     candidate with the lowest local + global mean z-score; rho doubles after an improvement and halves after
     ~7 failures in a row; the MLP output starts at 0.
  5. Training: OpenAI-ES as experiments.benchmark.train (antithetic pairs, centred ranks, common random numbers), on
     TRAIN_FNS, dims {10,20,50,100}, noise {0, 0.01}, budget 1000; fitness = -mean log10 regret. Changes, each
     measured locally: ranks within each instance; sigma 0.5 (at 0.1, gradients of independent batches were
     uncorrelated: episodes are chaotic); 115 trained params; momentum SGD with RMS-normalized steps instead of
     Adam; policy rng = default_rng(seed + 1) as benchmark.run_one. default_rng(seed), as in
     the leaked benchmark.train_fitness, replays the stream Problem draws its shift from, so the first random point lands
     within 0.1 per coordinate of the optimum and ES learns to exploit that.

Training, evaluation and the benchmark report live in experiments/rlbo_bench.py.
"""
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .policy import pick

GENS = {'local': 12, 'cma': 8, 'grad': 8, 'newton': 8, 'line': 6, 'far': 6}
C = sum(GENS.values())
GEN_ID = np.repeat(np.arange(len(GENS)), list(GENS.values()))
NFIT, N_INIT, NOISE, NOISE_G, QUAD_PREC = 256, 10, 0.05, 0.1, 30.0
KERNELS = [(1.0, 1.0, 0.0, 0.0), (0.3, 0.0, 1.0, 0.5), (0.3, 0.0, 1.0, 1.0), (0.3, 0.0, 1.0, 2.0)]  # lin, quad, rbf, ell
RHO0, RHO_MIN, RHO_MAX = 0.1, 1e-5, 0.5  # local radius: per-coordinate sd in [0,1]^n
RMULT = np.array([0.5, 1.0, 2.0])
WARM_V = [-1.5, -1.5]  # warm-start weight on the local and global mean z-scores
WARM_FAR = [27.0, -30.0]  # warm-start generator bias for uniform points: const, x 'explored'
F, GF, H = 14 + len(GENS), 7, 16
SHAPES = [('W1', (H, F)), ('b1', (H,)), ('w2', (H,)), ('v', (F,)), ('Vr', (3, GF)), ('Vg', (len(GENS), GF))]
NPARAM = sum(int(np.prod(s)) for _, s in SHAPES)


def unpack(theta):
    out, i = {}, 0
    for name, shape in SHAPES:
        k = int(np.prod(shape)); out[name] = theta[i:i + k].reshape(shape); i += k
    assert i == len(theta), (len(theta), i)
    return out


def init_theta(rng=None):
    th = np.zeros(NPARAM); P = unpack(th)
    P['W1'][:] = 0.5 * (rng or np.random.default_rng(0)).standard_normal((H, F))  # so ES perturbations of w2 matter
    P['v'][[1, 12]] = WARM_V                     # prefer the lowest local / global surrogate mean ...
    P['Vg'][list(GENS).index('far'), [0, 6]] = WARM_FAR  # uniform points until ~2n evals or 40% of the budget
    P['Vr'][0, [0, 4]] = [-2.0, 2.5]             # x0.5 once ~7 steps fail in a row
    P['Vr'][1, 0] = 1.0                          # x1 by default
    P['Vr'][2, [0, 3]] = [-1.0, 3.0]             # x2 after an improvement
    return th


# ---------- surrogates ----------
class LocalGP:
    """GP ensemble on the NFIT points nearest x_b, inputs centred at x_b, targets = normal scores of local ranks."""

    def __init__(self, U, yz):
        self.U, self.U2 = U, U * U
        sq = self.U2.sum(1)
        self.D = sq.mean() + 1e-12
        self.Q = (self.U2 ** 2).sum(1).mean() + 1e-24
        self.sq = sq
        G, G2 = U @ U.T, self.U2 @ self.U2.T
        d2 = np.maximum(sq[:, None] + sq[None] - 2 * G, 0)
        med = float(np.median(np.sqrt(sq[sq > 0]))) if (sq > 0).any() else 1e-3
        self.models = []
        for sl, sq_, sr, em in KERNELS:
            ell2 = (em * med) ** 2
            K = 1 + sl * G / self.D + sq_ * G2 / self.Q + (sr * np.exp(-d2 / (2 * ell2)) if sr else 0)
            K[np.diag_indices_from(K)] += NOISE
            L = cho_factor(K, lower=True)
            self.models.append((sl, sq_, sr, ell2, L, cho_solve(L, yz)))

    def predict(self, Uq):
        sq = (Uq * Uq).sum(1)
        G, G2 = Uq @ self.U.T, (Uq * Uq) @ self.U2.T
        d2 = np.maximum(sq[:, None] + self.sq[None] - 2 * G, 0)
        mus, vs = [], []
        for sl, sq_, sr, ell2, L, a in self.models:
            Kq = 1 + sl * G / self.D + sq_ * G2 / self.Q + (sr * np.exp(-d2 / (2 * ell2)) if sr else 0)
            kss = 1 + sl * sq / self.D + sq_ * ((Uq * Uq) ** 2).sum(1) / self.Q + sr
            w = solve_triangular(L[0], Kq.T, lower=True)
            mus.append(Kq @ a); vs.append(np.maximum(kss - (w * w).sum(0), 1e-12))
        mus = np.array(mus)
        return mus.mean(0), np.sqrt(np.mean(vs, 0) + mus.var(0))

    def grad0(self):
        """Gradient of the ensemble mean at the incumbent (u = 0); the quadratic term's gradient is 0 there."""
        beta = sum(sl / self.D * a + (sr / ell2 * a * np.exp(-self.sq / (2 * ell2)) if sr else 0)
                   for sl, sq_, sr, ell2, L, a in self.models)
        return self.U.T @ beta / len(self.models)


def fit_local(X, y, xb):
    d = ((X - xb) ** 2).sum(1)
    idx = np.argpartition(d, NFIT - 1)[:NFIT] if len(y) > NFIT else np.arange(len(y))
    return LocalGP(X[idx] - xb, robust_scale(y[idx]))


class GlobalQuad:
    """Bayesian ridge on [1, |u|^2, u, u^2] (u = x - 0.5) over ALL points; Phi^T Phi is updated incrementally."""

    def __init__(self, n, B):
        self.n, self.N = n, 0
        self.Phi = np.empty((B, 2 * n + 2))
        self.A0 = np.zeros((2 * n + 2, 2 * n + 2))
        # ~unit prior variance per block; the per-coordinate curvature is shrunk hard toward the shared one
        self.prec = np.concatenate([[0.1, 1.0], np.full(n, n / 12), np.full(n, QUAD_PREC * n / 80)])

    def phi(self, X):
        U = X - 0.5
        return np.hstack([np.ones((len(U), 1)), (U * U).sum(1, keepdims=True) * 12 / self.n, U, U * U])

    def add(self, x):
        p = self.phi(x[None])[0]
        if self.N == len(self.Phi):  # a problem may not count some calls (free / cached), so N can pass B
            self.Phi = np.vstack([self.Phi, np.empty_like(self.Phi)])
        self.Phi[self.N] = p; self.N += 1
        self.A0 += np.outer(p, p)

    def fit(self, yz):
        self.L = cho_factor(self.A0 / NOISE_G + np.diag(self.prec), lower=True)
        self.w = cho_solve(self.L, self.Phi[:self.N].T @ yz / NOISE_G)

    def predict(self, Xq):
        P = self.phi(Xq)
        v = solve_triangular(self.L[0], P.T, lower=True)
        return P @ self.w, np.sqrt((v * v).sum(0) + NOISE_G)

    def newton(self):
        """Minimizer of the fitted separable quadratic over the box, per coordinate."""
        n = self.n
        a, q = self.w[2:n + 2], self.w[n + 2:] + self.w[1] * 12 / n
        u = np.where(q > 1e-12, -a / (2 * np.maximum(q, 1e-12)), -0.5 * np.sign(a))
        return np.clip(u, -0.5, 0.5) + 0.5


def robust_scale(y):
    """Affine-invariant: (y - median) / IQR, with the bad tail above +1 log-compressed (ranks would bend the
    quadratic structure the global model relies on)."""
    s = np.subtract(*np.percentile(y, [75, 25]))
    s = s if s > 0 else (y.std() if y.std() > 0 else 1.0)  # plateaus: IQR can be 0
    z = (y - np.median(y)) / s
    return np.where(z > 1, 1 + np.log(np.maximum(z, 1)), z)


# ---------- candidates + features ----------
def candidates(rng, X, y, xb, rho, g, xn):
    n = len(xb)
    k = GENS['local']
    out = [xb + (rho * np.exp(rng.uniform(-1, 1, k)))[:, None] * rng.standard_normal((k, n))]
    order = np.argsort(y)
    mu = int(np.clip(len(y) // 4, 2, 20))
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    top = X[order[:mu]]; m = w @ top
    k = GENS['cma']
    out.append(m + (rng.standard_normal((k, mu)) * np.sqrt(w)) @ (top - m) + 0.3 * rho * rng.standard_normal((k, n)))
    k = GENS['grad']
    gu = -g / (np.linalg.norm(g) + 1e-300)
    t = np.exp(rng.uniform(np.log(0.25), np.log(4), k)) * rho * np.sqrt(n)
    out.append(xb + t[:, None] * gu + 0.2 * rho * rng.standard_normal((k, n)))
    k = GENS['newton']
    t = np.exp(rng.uniform(np.log(0.05), 0, k))
    out.append(xb + t[:, None] * (xn - xb) + 0.1 * rho * rng.standard_normal((k, n)))
    k = GENS['line']
    j = order[rng.integers(1, min(len(y), 11), k)]
    out.append(xb + rng.uniform(-0.5, 2, k)[:, None] * (xb - X[j]) + 0.1 * rho * rng.standard_normal((k, n)))
    out.append(rng.uniform(0, 1, (GENS['far'], n)))
    return np.clip(np.vstack(out), 0, 1)


def rz(mu):
    med = np.median(mu)
    return np.clip((mu - med) / (1.4826 * np.median(np.abs(mu - med)) + 1e-9), -5, 5)


def features(S, Gq, cands, X, xb, rho, frac, imp_rate):
    n = len(xb)
    Q = np.vstack([cands, xb[None]])
    mu, sd = S.predict(Q - xb)
    gm, gs = Gq.predict(Q)
    imp = np.clip((mu[:-1] - mu[-1]) / (sd[:-1] + 1e-9), -5, 5)
    gimp = np.clip((gm[:-1] - gm[-1]) / (gs[:-1] + 1e-9), -5, 5)
    mu, sd = mu[:-1], sd[:-1]
    rank = mu.argsort().argsort() / (len(mu) - 1)
    lsd = np.clip(np.log(sd / np.median(sd)), -3, 3)
    db = np.linalg.norm(cands - xb, axis=1) / np.sqrt(n)
    ldb = np.clip(np.log(db / rho + 1e-12), -5, 5)
    d2 = np.maximum((cands ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None] - 2 * cands @ X.T, 0)
    dnn = np.sqrt(d2.min(1) / n)
    c = len(mu)
    return np.column_stack([np.ones(c), rz(mu), rank, imp, lsd, np.minimum(sd, 3), ldb, db, dnn,
                            np.full(c, frac), np.full(c, imp_rate), np.full(c, np.log(rho) / np.log(RHO_MIN)),
                            rz(gm[:-1]), gimp, np.eye(len(GENS))[GEN_ID]])


# ---------- the optimizer ----------
def rlbo(prob, rng, theta=None, tlog=None):
    P = unpack(init_theta() if theta is None else theta)
    n, B = prob.n, prob.budget
    X = np.empty((B, n)); y = np.empty(B); N = 0
    Gq = GlobalQuad(n, B)
    for _ in range(min(N_INIT, B)):
        X[N] = rng.uniform(0, 1, n); y[N] = prob(X[N]); Gq.add(X[N]); N += 1
    rho, imp_rate, last_ok, stall = RHO0, 0.0, 0.0, 0
    while prob.evals < B:
        t0 = time.perf_counter()
        frac = prob.evals / B
        explored = min(max(N / (2 * n), 2.5 * frac), 2)  # ~1 once the global quadratic is identifiable
        gg = np.array([1.0, frac, imp_rate, last_ok, min(stall / 5, 2), np.log(rho) / np.log(RHO_MIN), explored])
        m = RMULT[pick(rng, P['Vr'] @ gg)]
        if m != 1:
            rho = float(np.clip(rho * m, RHO_MIN, RHO_MAX)); stall = 0
        Xh, yh = X[:N], y[:N]
        b = int(np.argmin(yh)); xb = Xh[b]
        S = fit_local(Xh, yh, xb)
        Gq.fit(robust_scale(yh))
        cands = candidates(rng, Xh, yh, xb, rho, S.grad0(), Gq.newton())
        f = features(S, Gq, cands, Xh, xb, rho, frac, imp_rate)
        logits = np.tanh(f @ P['W1'].T + P['b1']) @ P['w2'] + f @ P['v'] + (P['Vg'] @ gg)[GEN_ID]
        c = pick(rng, logits); x = cands[c]
        if tlog is not None: tlog.append((time.perf_counter() - t0, GEN_ID[c]))
        fx = prob(x)
        last_ok = float(fx < yh[b])
        imp_rate = 0.9 * imp_rate + 0.1 * last_ok
        stall = 0 if last_ok else stall + 1
        if N == len(X):  # more calls than budget: the problem did not count some of them
            X, y = np.vstack([X, np.empty_like(X)]), np.concatenate([y, np.empty_like(y)])
        X[N], y[N] = x, fx; Gq.add(x); N += 1
