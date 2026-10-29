"""Policy interpretability, redone on the leak-free policies (results/policies/).

Part 0  seed robustness: action shares of each policy on 3 disjoint blocks of instances.
Part 1  hpx_walker (round E, new actions): how often it re-checks / jumps / takes x4-x8 steps, before and after
        it finds signal and by budget phase; then ablate each new action (its logit set to -1e9) and score.
Part 2  budget-aware walkers (round C, cov_vb): head choices by budget phase at 300 / 1000 / 10000 evals, and the
        best-value improvement rate per phase next to CMA-ES (why they win small budgets and lose large ones).
Head choices are read by wrapping metawalker.walker.pick; per-coordinate moves are not logged.

Usage: python -m experiments.analysis.policies [quick]   -> prints the report, writes results/analysis/redo_report.txt
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from metawalker import problems
from metawalker.cma import cmaes
from metawalker.policy import STEP_MULT, unpack
from metawalker.problems import Problem
from metawalker.walker import navigate
from experiments import benchmark as nb
from metawalker import walker

QUICK = 'quick' in sys.argv
POL = {p.stem.removeprefix('policy_'): p for p in (ROOT / 'results' / 'policies').glob('policy_*.npy')}
LOG, CUR = [], [None]
_pick = walker.pick


def logged_pick(rng, logits):
    a = _pick(rng, logits)
    p = CUR[0]
    LOG.append((len(logits), a, p.evals / p.budget, p.best_y))
    return a


walker.pick = logged_pick  # navigate() looks pick up in its own module


def steps(cfg):
    """Parse LOG into per-step dicts: act (0 move, 1 re-check, 2 jump), mult, beta, gamma, tfrac, signal."""
    out, i, extra = [], 0, cfg.get('extra')
    heads = ['mult', 'beta', 'gamma'] + (['kappa', 'alpha'] if cfg['dirs'] == 'cov' else [])
    while i < len(LOG):
        _, a, tf, by = LOG[i]
        st = dict(tfrac=tf, signal=by < 1.0, act=0)
        if extra:
            st['act'] = a; i += 1
            if a:
                out.append(st); continue
        for h in heads:
            st[h] = LOG[i][1]; i += 1
        out.append(st)
    return out


def run(theta, cfg, prob, seed):
    LOG.clear(); CUR[0] = prob
    navigate(prob, np.random.default_rng(seed), theta, cfg)
    return prob.score(), steps(cfg)


def hp_family(k, seed0, budget=600):
    """The round D/E training family: plateau, 1% live, 5 binomial trials per evaluation."""
    problems.HP_FRAC, problems.HP_TRIALS = nb.HPFAM_FRAC, 5
    rng = np.random.default_rng(seed0)
    return [('honeypot', int(rng.choice(nb.HPFAM_DIMS)), int(rng.integers(1e9)), nb.HPFAM_NOISE, budget)
            for _ in range(k)]


def shares(S, key='act', n=3):
    c = Counter(s.get(key) for s in S if s.get(key) is not None)
    tot = sum(c.values()) or 1
    return [c[j] / tot for j in range(n)]


def ablate(theta, cfg, what):
    th = theta.copy()
    P = unpack(th, cfg)  # views into th
    for w in what:
        if w == 'recheck': P['Vx'][1, 0] = -1e9
        if w == 'jump': P['Vx'][2, 0] = -1e9
        if w == 'long': P['Vs'][len(STEP_MULT):, 0] = -1e9
    return th


def part0_1(L):
    cfg, th = nb.VARIANTS['hpx_walker'], np.load(POL['hpx_walker_s0'])
    k = 8 if QUICK else 24
    L.append('== Part 0: seed robustness of hpx_walker (move / re-check / jump shares, x4+x8 share of moves)')
    allS, inst_all = [], []
    for block in range(3):
        inst = hp_family(k, 100 + block)
        inst_all += inst
        S = []
        for i, ins in enumerate(inst):
            S += run(th, cfg, Problem(*ins), i + 1)[1]
        sh = shares(S); mv = [s for s in S if s['act'] == 0]
        L.append(f'  block {block}: move {sh[0]:.2f}  re-check {sh[1]:.2f}  jump {sh[2]:.2f}  '
                 f'x4+x8 {np.mean([s["mult"] >= 5 for s in mv]):.2f}')
        allS += S
    L.append('== Part 1: hpx_walker behaviour (all blocks)')
    for name, sel in (('before any signal', lambda s: not s['signal']), ('after signal', lambda s: s['signal'])):
        S = [s for s in allS if sel(s)]
        sh = shares(S); mv = [s for s in S if s['act'] == 0]
        L.append(f'  {name:18s} ({len(S)} steps): move {sh[0]:.2f} re-check {sh[1]:.2f} jump {sh[2]:.2f}  '
                 f'x4+x8 {np.mean([s["mult"] >= 5 for s in mv]) if mv else float("nan"):.2f}  '
                 f'pull {np.mean([s["gamma"] for s in mv]) if mv else float("nan"):.2f}')
    for lo in (0.0, 0.25, 0.5, 0.75):
        S = [s for s in allS if lo <= s['tfrac'] < lo + 0.25]
        sh = shares(S)
        L.append(f'  budget {lo:.2f}-{lo + 0.25:.2f}: move {sh[0]:.2f} re-check {sh[1]:.2f} jump {sh[2]:.2f}')
    L.append('  ablations (same instances and rng; score = log10 regret, lower is better; win = better than full):')
    base = None
    for name, what in (('full', []), ('no re-check', ['recheck']), ('no jump', ['jump']), ('no long steps', ['long']),
                       ('none of the three', ['recheck', 'jump', 'long'])):
        t = ablate(th, cfg, what)
        sc = np.array([run(t, cfg, Problem(*ins), i + 1)[0] for i, ins in enumerate(inst_all)])
        if base is None:
            base = sc
        w = (sc < base).mean() + 0.5 * (sc == base).mean()
        L.append(f'    {name:18s} median {np.median(sc):+.2f}  mean {sc.mean():+.2f}  found region '
                 f'{(sc < -0.5).mean():.2f}  win vs full {w:.2f}')
    thb = np.load(POL['hpx_base_s0'])
    sc = np.array([run(thb, nb.VARIANTS['hpx_base'], Problem(*ins), i + 1)[0] for i, ins in enumerate(inst_all)])
    L.append(f'    {"hpx_base (control)":18s} median {np.median(sc):+.2f}  mean {sc.mean():+.2f}  found region '
             f'{(sc < -0.5).mean():.2f}  win vs full {(sc < base).mean() + 0.5 * (sc == base).mean():.2f}')


def improvement_by_phase(trace, q=4):
    """Share of the total log10 improvement of the best value made in each quarter of the budget."""
    t = np.log10(np.maximum(np.minimum.accumulate(trace), 1e-12))
    cuts = np.linspace(0, len(t) - 1, q + 1).astype(int)
    d = -np.diff(t[cuts])
    return d / max(d.sum(), 1e-12)


class Traced(Problem):
    def __init__(self, *a):
        super().__init__(*a); self.trace = []

    def __call__(self, x):
        y = super().__call__(x)
        if np.isfinite(y):
            self.trace.append(self.true(x))
        return y


def part2(L):
    L.append('== Part 2: budget-aware walker cov_vb (seeds 0-2): head choices and improvement by budget quarter')
    problems.HP_FRAC, problems.HP_TRIALS = 0.05, 100
    budgets = (300, 1000) if QUICK else (300, 1000, 10000)
    for fn, n in (('sphere', 100), ('rastrigin', 100), ('honeypot', 100)):
        for B in budgets:
            per_q, imp_w, imp_c, sc_w, sc_c = defaultdict(list), [], [], [], []
            for s in range(3):
                cfg, th = nb.VARIANTS['cov_vb'], np.load(POL[f'cov_vb_s{s}'])
                for seed in range(2 if QUICK else 4):
                    p = Traced(fn, n, 7000 + seed, 0.01, B)
                    sc, S = run(th, cfg, p, seed + 1)
                    sc_w.append(sc); imp_w.append(improvement_by_phase(p.trace))
                    for st in S:
                        q = min(int(st['tfrac'] * 4), 3)
                        per_q[q].append((np.log(STEP_MULT[st['mult']]), st['gamma'] > 0, st['beta']))
                    if s == 0:
                        c = Traced(fn, n, 7000 + seed, 0.01, B)
                        cmaes(c, np.random.default_rng(seed + 1))
                        sc_c.append(c.score()); imp_c.append(improvement_by_phase(c.trace))
            qs = ' '.join(f'Q{q + 1}: logmult {np.mean([v[0] for v in per_q[q]]):+.2f} pull {np.mean([v[1] for v in per_q[q]]):.2f}'
                          for q in range(4))
            L.append(f'  {fn:9s} n={n} B={B:5d}: walker {np.median(sc_w):+.2f} vs CMA-ES {np.median(sc_c):+.2f} | {qs}')
            L.append(f'      share of improvement per quarter: walker {np.round(np.mean(imp_w, 0), 2)}  '
                     f'CMA-ES {np.round(np.mean(imp_c, 0), 2)}')


def main():
    assert {'hpx_walker_s0', 'hpx_base_s0', 'cov_vb_s0', 'cov_vb_s1', 'cov_vb_s2'} <= set(POL), sorted(POL)
    L = [f'policies: {sorted(POL)}']
    part0_1(L)
    part2(L)
    out = '\n'.join(L)
    print(out)
    if not QUICK:
        (ROOT / 'results' / 'analysis' / 'redo_report.txt').write_text(out + '\n')


if __name__ == '__main__':
    main()
