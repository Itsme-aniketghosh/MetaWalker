"""Policy weights: action sets, parameter layout (shapes / unpack), initialisation, widening, sampling.

A policy is a flat numpy vector theta; cfg says how to read it:
  kind='walker'  dirs='mom'|'cov', H hidden units, M walkers; extra=True adds the move/re-check/jump head and one
                 global feature; long=True adds x4 / x8 step multipliers.
  kind='rlcma'   H hidden units (0 = linear) over the CMA-ES generation features.
"""
from pathlib import Path

import numpy as np

STEP_MULT = np.array([0.5, 0.8, 1.0, 1.25, 2.0])
STEP_MULT_LONG = np.array([0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 8.0])  # 'long' variants: skip far along a working direction
BETAS = np.array([0.0, 1.0, 3.0])        # momentum weight
GAMMAS = np.array([0.0, 0.5])            # pull toward best walker
KAPPAS = np.array([0.0, 0.5, 1.0, 2.0])  # weight of the subspace (past successful steps) term
K, G, GC = 8, 6, 7                       # per-coordinate, walker-global, rlcma feature counts
R_MEM = 10                               # remembered successful steps per walker
S_MIN, S_MAX = 1e-7, 0.5

# shipped policies: (benchmark variant config, weights file)
POLICIES = {
    'rlcma': (dict(kind='rlcma', H=16), 'rlcma_mlp_s0.npy'),           # RL-controlled CMA-ES
    'walker': (dict(kind='walker', dirs='mom', H=8, M=4), 'mom_s0.npy'),  # learned walker
    'walker_hp': (dict(kind='walker', dirs='cov', H=8, M=4), 'cov_hp_s0.npy'),  # trained with plateau problems too
}


def mults(cfg):
    return STEP_MULT_LONG if cfg.get('long') else STEP_MULT


def shapes(cfg):
    H = cfg['H']
    if cfg['kind'] == 'walker':
        g = G + bool(cfg.get('extra'))  # extra: + log2(evaluations at the best point) / 4
        s = [('W1', (H, K)), ('b1', (H,)), ('W2', (3, H)), ('b2', (3,)),
             ('Vs', (len(mults(cfg)), g)), ('Vb', (len(BETAS), g)), ('Vg', (len(GAMMAS), g))]
        if cfg['dirs'] == 'cov':
            s += [('Vk', (len(KAPPAS), g)), ('Va', (2, g))]
        if cfg.get('extra'):
            s += [('Vx', (3, g))]  # move / re-check the best point / jump to a random point
        return s
    gin = H or GC
    return ([('W1', (H, GC)), ('b1', (H,))] if H else []) + \
        [('Vs', (len(STEP_MULT), gin)), ('cs', (len(STEP_MULT),)), ('Vr', (2, gin)), ('cr', (2,))]


def nparam(cfg):
    return sum(int(np.prod(s)) for _, s in shapes(cfg))


def unpack(theta, cfg):
    out, i = {}, 0
    for name, shape in shapes(cfg):
        size = int(np.prod(shape))
        out[name] = theta[i:i + size].reshape(shape); i += size
    assert i == len(theta), (cfg, len(theta), i)
    return out


def init_theta(cfg, rng):
    th = np.zeros(nparam(cfg))
    P = unpack(th, cfg)  # views into th
    if 'W1' in P:
        P['W1'][:] = 0.5 * rng.standard_normal(P['W1'].shape)  # so ES perturbations of later layers matter
    if cfg['kind'] == 'rlcma':
        P['cs'][list(STEP_MULT).index(1.0)] = 4.0  # start close to plain CMA-ES: keep sigma ...
        P['cr'][0] = 5.0                            # ... and don't restart
    return th


def widen(old, src_cfg, cfg):
    """Embed a trained walker in a layout with extra step sizes / feature / action head. New options start rare
    (logit -3, or +3 for 'move', through the constant feature g[0]) and the new feature at weight 0."""
    th = np.zeros(nparam(cfg))
    P = unpack(th, cfg)  # views into th
    for k, a in unpack(old, src_cfg).items():
        P[k][tuple(slice(0, d) for d in a.shape)] = a
    P['Vs'][len(STEP_MULT):, 0] = -3.0
    if 'Vx' in P:
        P['Vx'][0, 0] = 3.0
    return th


def pick(rng, logits):
    return int(np.argmax(logits + rng.gumbel(size=len(logits))))


def load_policy(name):
    cfg, fname = POLICIES[name]
    theta = np.load(Path(__file__).parent / 'policies' / fname)
    assert theta.shape == (nparam(cfg),), (name, theta.shape)
    return theta, cfg
