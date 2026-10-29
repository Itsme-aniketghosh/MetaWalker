"""RL algorithms for the walker: branching double DQN (1-step or n-step) and PPO.

Same moves, features and network layout as the ES walker (metawalker.walker), so trained weights are saved as
policy_<variant>_s<seed>.npy and go through the same eval pipeline:
  dqn  acts eps-greedily on the weights as Q-values (metawalker.walker.GreedyQ)
  ppo  samples from the weights as logits, exactly like the ES policies
Action branches: one per coordinate (-1/0/+1, shared weights, averaged into one branch) plus one per
global head (step size, momentum, pull, subspace, axis), as in action-branching DQN (Tavakoli et al. 2018).
Reward per step = log10 improvement of the best observed value, so an episode's return is its score gain.
"""
import json, time
from collections import deque

import numpy as np
import torch

from metawalker.policy import G, K, init_theta, shapes, unpack
from metawalker.problems import Problem
from metawalker.walker import GreedyQ, Q_HEADS, navigate
from experiments import benchmark as nb

GAMMA = 0.995
# DQN
LR, BATCH, BUFFER, WARMUP = 1e-3, 128, 100_000, 10_000
TRAIN_EVERY, SYNC_ACTOR, SYNC_TARGET = 8, 100, 2000   # env steps per update; updates per copy
# PPO
PPO_LR, PPO_EPISODES, PPO_EPOCHS, PPO_MB, CLIP, LAMBDA, ENT = 3e-4, 8, 4, 1024, 0.2, 0.95, 0.01

hub = torch.nn.functional.smooth_l1_loss


class Net(torch.nn.Module):
    """The walker network (metawalker.policy.shapes layout): per-coordinate MLP + linear global heads."""

    def __init__(self, cfg, theta):
        super().__init__()
        self.names = [nm for nm, _ in shapes(cfg)]
        self.p = torch.nn.ParameterDict({nm: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))
                                         for nm, v in unpack(theta, cfg).items()})
        self.heads = [h for h in Q_HEADS if h in self.p]

    def forward(self, fc, g):
        qc = torch.tanh(fc @ self.p['W1'].T + self.p['b1']) @ self.p['W2'].T + self.p['b2']
        return qc, [g @ self.p[h].T for h in self.heads]

    def flat(self):
        return np.concatenate([self.p[nm].detach().numpy().ravel() for nm in self.names]).astype(float)


def seg_mean(v, seg, B):
    return torch.zeros(B).index_add_(0, seg, v) / torch.zeros(B).index_add_(0, seg, torch.ones_like(v))


def batch_states(states):
    """[(feats (n,K), g (G,))...] -> concatenated coord feats, segment ids, stacked globals."""
    seg = torch.tensor(np.repeat(np.arange(len(states)), [len(f) for f, _ in states]))
    fc = torch.tensor(np.concatenate([f for f, _ in states]))
    g = torch.tensor(np.stack([g for _, g in states]))
    return fc, seg, g


# ---------- DQN ----------
class DQN:
    """Plugs into metawalker.walker.navigate as `actor`: acts eps-greedily, stores (n-step) transitions, trains."""

    def __init__(self, cfg, rng, total_steps):
        self.cfg, self.rng, self.total = cfg, rng, total_steps
        self.nstep = cfg.get('nstep', 1)
        self.gamma = cfg.get('gamma', GAMMA)
        self.net = Net(cfg, init_theta(cfg, rng))
        self.tgt = Net(cfg, self.net.flat())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.buf = deque(maxlen=BUFFER)
        self.window = deque()  # this episode's last n steps: (feats, g, d, h, r)
        self.steps = self.updates = 0
        self.losses = []
        self.actor = GreedyQ(unpack(self.net.flat(), cfg), 1.0)

    def _emit(self, s2, done):
        """Oldest step in the window -> transition with its n-step discounted return."""
        R = sum(self.gamma ** k * w[4] for k, w in enumerate(self.window))
        f, g, d, h, _ = self.window.popleft()
        self.buf.append((f, g, d, h, R, s2[0], s2[1], float(done), self.gamma ** (len(self.window) + 1)))

    def act(self, feats, g, rng):
        s = (feats.astype(np.float32), g.astype(np.float32))
        if len(self.window) == self.nstep:
            self._emit(s, False)
        self.actor.eps = max(0.05, 1.0 - self.steps / (0.1 * self.total))  # linear decay over first 10%
        d, h = self.actor.act(feats, g, rng)
        self._last = s + (d.astype(np.int64), np.array(h))
        self.steps += 1
        if len(self.buf) >= WARMUP and self.steps % TRAIN_EVERY == 0:
            self.update()
        return d, h

    def observe(self, reward, done):
        self.window.append(self._last + (float(reward),))
        if done:  # flush: every remaining step bootstraps from nothing
            while self.window:
                self._emit(self._last[:2], True)

    def update(self):
        tr = [self.buf[i] for i in self.rng.integers(len(self.buf), size=BATCH)]
        fc, seg, g = batch_states([(t[0], t[1]) for t in tr])
        fc2, seg2, g2 = batch_states([(t[5], t[6]) for t in tr])
        dc = torch.tensor(np.concatenate([t[2] for t in tr]))
        ha = torch.tensor(np.stack([t[3] for t in tr]))
        r, done, disc = (torch.tensor([t[k] for t in tr], dtype=torch.float32) for k in (4, 7, 8))
        with torch.no_grad():  # double DQN: online net picks the action, target net values it
            qc_on, qh_on = self.net(fc2, g2)
            qc_tg, qh_tg = self.tgt(fc2, g2)
            v = [seg_mean(qc_tg.gather(1, qc_on.argmax(1, keepdim=True)).squeeze(1), seg2, BATCH)]
            v += [qt.gather(1, qo.argmax(1, keepdim=True)).squeeze(1) for qo, qt in zip(qh_on, qh_tg)]
            y = r + disc * (1 - done) * torch.stack(v, 1).mean(1)
        qc, qh = self.net(fc, g)
        loss = seg_mean(hub(qc.gather(1, dc[:, None]).squeeze(1), y[seg], reduction='none'), seg, BATCH).mean()
        loss = (loss + sum(hub(q.gather(1, ha[:, i:i + 1]).squeeze(1), y) for i, q in enumerate(qh))) / (1 + len(qh))
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        self.losses.append(float(loss))
        if self.updates % SYNC_ACTOR == 0:
            self.actor.P = unpack(self.net.flat(), self.cfg)
        if self.updates % SYNC_TARGET == 0:
            self.tgt.load_state_dict(self.net.state_dict())

    def after_episode(self):
        pass


# ---------- PPO ----------
class Critic(torch.nn.Module):
    """V(s) from the global features plus mean-pooled per-coordinate features (dimension-agnostic)."""

    def __init__(self, hc=32):
        super().__init__()
        self.coord = torch.nn.Linear(K, hc)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(G + hc, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))

    def forward(self, fc, seg, g):
        pooled = torch.zeros(len(g), self.coord.out_features).index_add_(0, seg, torch.tanh(self.coord(fc)))
        pooled = pooled / torch.zeros(len(g)).index_add_(0, seg, torch.ones(len(seg)))[:, None]
        return self.mlp(torch.cat([g, pooled], 1)).squeeze(1)


class PPO:
    """Plugs into metawalker.walker.navigate as `actor`: samples from the policy, collects episodes, updates."""

    def __init__(self, cfg, rng, total_steps):
        self.cfg, self.rng, self.total = cfg, rng, total_steps
        self.pi = Net(cfg, init_theta(cfg, rng))
        self.vf = Critic()
        self.gamma = cfg.get('gamma', GAMMA)
        self.opt = torch.optim.Adam(list(self.pi.parameters()) + list(self.vf.parameters()), lr=PPO_LR)
        self.P = unpack(self.pi.flat(), cfg)
        self.traj, self.episodes = [], []
        self.steps = self.updates = 0
        self.losses = []
        self.eps = 0.0  # logged only

    def act(self, feats, g, rng):
        P = self.P
        lc = np.tanh(feats @ P['W1'].T + P['b1']) @ P['W2'].T + P['b2']
        d = np.argmax(lc + rng.gumbel(size=lc.shape), 1)
        lh = [P[h] @ g for h in Q_HEADS if h in P]
        h = [int(np.argmax(l + rng.gumbel(size=len(l)))) for l in lh]
        lse = lambda l: l.max(-1) + np.log(np.exp(l - l.max(-1, keepdims=True)).sum(-1))
        lp_c = lc[np.arange(len(d)), d] - lse(lc)
        lp_h = np.array([l[a] - lse(l) for l, a in zip(lh, h)])
        self.traj.append([feats.astype(np.float32), g.astype(np.float32), d.astype(np.int64), np.array(h),
                          lp_c.astype(np.float32), lp_h.astype(np.float32), 0.0])
        self.steps += 1
        return d, h

    def observe(self, reward, done):
        self.traj[-1][6] = float(reward)

    def after_episode(self):
        self.episodes.append(self.traj); self.traj = []
        if len(self.episodes) >= PPO_EPISODES:
            self.update(); self.episodes = []

    def update(self):
        steps = [st for ep in self.episodes for st in ep]
        with torch.no_grad():  # values for every state, then GAE per episode (episodes end at the budget)
            V = torch.cat([self.vf(*batch_states([(s[0], s[1]) for s in steps[i:i + 2048]]))
                           for i in range(0, len(steps), 2048)]).numpy()
        adv, i0 = np.zeros(len(steps), np.float32), 0
        for ep in self.episodes:
            last = 0.0
            for t in reversed(range(len(ep))):
                v_next = V[i0 + t + 1] if t + 1 < len(ep) else 0.0
                delta = ep[t][6] + self.gamma * v_next - V[i0 + t]
                last = delta + self.gamma * LAMBDA * last
                adv[i0 + t] = last
            i0 += len(ep)
        ret = torch.tensor(adv + V)
        A = torch.tensor((adv - adv.mean()) / (adv.std() + 1e-8))
        for _ in range(PPO_EPOCHS):
            for mb in np.array_split(self.rng.permutation(len(steps)), max(1, len(steps) // PPO_MB)):
                ss = [steps[i] for i in mb]
                fc, seg, g = batch_states([(s[0], s[1]) for s in ss])
                dc = torch.tensor(np.concatenate([s[2] for s in ss]))
                ha = torch.tensor(np.stack([s[3] for s in ss]))
                old_c = torch.tensor(np.concatenate([s[4] for s in ss]))
                old_h = torch.tensor(np.stack([s[5] for s in ss]))
                a, B = A[mb], len(mb)
                lc, lh = self.pi(fc, g)
                logp_c = torch.log_softmax(lc, 1)
                new_c = logp_c.gather(1, dc[:, None]).squeeze(1)
                surr = lambda new, old, adv: torch.min((new - old).exp() * adv,
                                                       (new - old).exp().clamp(1 - CLIP, 1 + CLIP) * adv)
                # per-branch clipped objective (coords averaged into one branch) keeps ratios sane at large n
                branches = [seg_mean(surr(new_c, old_c, a[seg]), seg, B)]
                ent = [seg_mean(-(logp_c.exp() * logp_c).sum(1), seg, B)]
                for i, l in enumerate(lh):
                    lp = torch.log_softmax(l, 1)
                    branches.append(surr(lp.gather(1, ha[:, i:i + 1]).squeeze(1), old_h[:, i], a))
                    ent.append(-(lp.exp() * lp).sum(1))
                pl = -torch.stack(branches, 1).mean()
                vl = ((self.vf(fc, seg, g) - ret[mb]) ** 2).mean()
                loss = pl + 0.5 * vl - ENT * torch.stack(ent, 1).mean()
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.pi.parameters()) + list(self.vf.parameters()), 0.5)
                self.opt.step()
                self.losses.append(float(loss))
        self.updates += 1
        self.P = unpack(self.pi.flat(), self.cfg)

    @property
    def net(self):
        return self.pi


# ---------- training loop shared by both ----------
def train(variant, seed, steps=None, budget=nb.BUDGET):
    cfg = nb.VARIANTS[variant]
    total = steps or cfg['steps']
    rng = np.random.default_rng(1000 * seed + list(nb.VARIANTS).index(variant))
    torch.manual_seed(seed)
    torch.set_num_threads(nb.ncpu())
    L = {'dqn': DQN, 'ppo': PPO}[cfg['learner']](cfg, rng, total)
    out = nb.OUT / f'policy_{variant}_s{seed}.npy'
    log, scores, ep, t0 = [], [], 0, time.time()
    while L.steps < total:
        inst = nb.train_instances(rng, 1, budget, cfg.get('train_fns', nb.TRAIN_FNS))[0]
        p = Problem(*inst)
        navigate(p, np.random.default_rng(inst[2] + 1), None, cfg, actor=L)  # +1: see benchmark.train_fitness
        L.after_episode()
        scores.append(p.score()); ep += 1
        if ep % 200 == 0 or L.steps >= total:
            eps = getattr(L, 'actor', L).eps
            row = dict(ep=ep, steps=L.steps, eps=eps, score=float(np.mean(scores[-200:])),
                       loss=float(np.mean(L.losses[-1000:])) if L.losses else None, sec=time.time() - t0)
            log.append(row)
            print(f"{variant} s{seed} ep {ep:6d} steps {L.steps:9d} eps {eps:.2f} "
                  f"score(last200, lower=better) {row['score']:+.3f} loss {row['loss']} ({row['sec']:.0f}s)",
                  flush=True)
            np.save(out, L.net.flat())  # checkpoint
            json.dump(log, open(nb.OUT / f'train_log_{variant}_s{seed}.json', 'w'))
    np.save(out, L.net.flat())
