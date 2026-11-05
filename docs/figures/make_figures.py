"""Figures for docs/blog_post.md, from the saved results. Usage, from the repo root:
    python docs/figures/make_figures.py
The plateau-flatness figure also reads scratch/hp_live/ (benchmark suites hp_live5 / hp_live1) and is skipped
without it."""
import json, sys
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
HP = ROOT / 'results' / 'honeypot'
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
for m in ('jax', 'jax.numpy', 'optax', 'flax', 'flax.linen'):  # decode_many is numpy-only; skip the JAX PPO stack
    sys.modules.setdefault(m, MagicMock())
from experiments.honeypot import env as E  # noqa: E402

# reference palette (dataviz skill): one highlight hue, neutral for the rest
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, GRAY = '#2a78d6', '#a3a29c'
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
                     'axes.titlelocation': 'left'})


def scores(pattern):
    return [json.load(open(f))['cert_exploit'] for f in sorted(HP.glob(pattern))]


def dot_rows(ax, rows, xlabel):
    """One row per method: a dot per search, a bar for the mean, the mean as a direct label."""
    rng = np.random.default_rng(0)
    for i, (label, vals, color) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.scatter(vals, y + rng.uniform(-0.12, 0.12, len(vals)), s=46, color=color, alpha=0.75,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        m = float(np.mean(vals))
        ax.plot([m, m], [y - 0.3, y + 0.3], color=INK, linewidth=2, zorder=4)
        ax.text(m, y + 0.36, f'{m:.2f}', ha='center', va='bottom', fontsize=10, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fresh(prefix):
    """Exploit rate of each search's chosen level on 10 brand-new PPO seeds (experiments/honeypot/recheck.py)."""
    out = []
    for f in sorted(HP.glob('recheck_*/recheck_*.json')):
        d = json.load(open(f))
        keys = d['key'] if isinstance(d['key'], list) else [d['key']]
        out += [d['exploit'] for k in keys if k.replace(chr(92), '/').startswith(prefix)]  # keys from Windows paths
    return out


def split_panel():
    """Equal-split panel (rlcma + walker_hp + CMA-ES, 1/3 budget each): per search, the member level with the best
    certified score, scored on brand-new seeds."""
    man = json.load(open(HP / 'recheck_split' / 'manifest.json'))
    fr = {json.load(open(f))['key']: json.load(open(f))['exploit'] for f in (HP / 'recheck_split').glob('recheck_*.json')}
    best = {}
    for it in man:
        if it['seed'] not in best or it['cert'] > best[it['seed']]['cert']:
            best[it['seed']] = it
    return [fr[best[s]['key']] for s in sorted(best)]


def fig_honeypot():
    rows = [('Panel, 3x budget (15)', fresh('panel:'), '#86b6ef'),  # rlbo + CMA-ES + walker, each the full budget
            ('Panel, same budget (5)', split_panel(), '#86b6ef'),
            ('rlbo (15)', [v for p in ('s0', 's1', 's2') for v in fresh(f'rlbo:{p}|')], BLUE),
            ('CMA-ES (10)', fresh('cmaes|'), GRAY),
            ('Walker (5)', fresh('walker|'), GRAY),
            ('GP Bayesian opt. (5)', fresh('gp|'), GRAY)]
    assert [len(r[1]) for r in rows] == [15, 5, 15, 10, 5, 5], [len(r[1]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    dot_rows(ax, rows, 'Exploit rate of the chosen level on 10 brand-new PPO seeds')
    ax.set_title('Honeypots scored on brand-new PPO seeds')
    fig.tight_layout(); fig.savefig(OUT / 'honeypot_results.png', dpi=200); plt.close(fig)


def fig_ablation():
    pat = 'round_E/manyh_rl-hpx_{}-s0_*.json'
    rows = [('Full walker', scores(pat.format('walker')), BLUE),
            ('Without jump', scores(pat.format('nojump')), GRAY),
            ('Without re-check', scores(pat.format('norecheck')), GRAY),
            ('Without any of the new actions', scores(pat.format('base')), GRAY)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    dot_rows(ax, rows, 'Exploit rate of the best level (budget 600, 5 searches each)')
    ax.set_title('Each new action pulls its weight')
    fig.tight_layout(); fig.savefig(OUT / 'walker_ablation.png', dpi=200); plt.close(fig)


def fig_speed():
    # mean seconds per 1,000-evaluation run on the standard test functions (results/reports/rlbo.txt,
    # hpo_baselines.txt); SMAC hit the time cap on 55 of 360 runs, so its true cost is higher
    rows = [('NGOpt', 16.7), ('rlbo', 23.0), ('CMA-ES', 35.2), ('RL-CMA', 35.5), ('TPE', 263.2),
            ('SMAC3', 3678.4), ('GP Bayesian opt.', 4296.4)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, (label, sec) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, sec, height=0.62, color=BLUE if label == 'rlbo' else GRAY, zorder=3)
        txt = f'{sec:,.0f} s' + (' (capped)' if label == 'SMAC3' else '')
        ax.text(sec * 1.12, y, txt, va='center', fontsize=10, color=INK)
    ax.set_xscale('log')
    ax.set_xlim(5, 30000)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Seconds of optimizer time per 1,000-evaluation run (log scale)')
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title('rlbo runs in seconds; GP takes over an hour')
    fig.tight_layout(); fig.savefig(OUT / 'speed.png', dpi=200); plt.close(fig)


def fig_time_to_target():
    """PPO runs until a search first logs a valid level scoring >= 0.8 (results/analysis/time_to_target.json,
    rounds D and F; search scores, not re-measured)."""
    ttt = json.load(open(ROOT / 'results' / 'analysis' / 'time_to_target.json'))
    groups = [('Learned walker\n+ warm-up', ('honeypot/ns-walker_hp_warm', 'honeypot/rl-hpfam_walker-s0-warm',
                                            'honeypotF/rl-hpfam_walker-s0-warm'), BLUE),
              ('CMA-ES', ('honeypot/cmaes', 'honeypotF/cmaes'), GRAY),
              ('TPE', ('honeypot/tpe',), GRAY),
              ('GP Bayesian opt.', ('honeypot/optuna_gp', 'honeypotF/optuna_gp'), GRAY),
              ('SMAC3', ('honeypot/smac',), GRAY),
              ('Random search', ('honeypot/random',), GRAY)]
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    rng = np.random.default_rng(0)
    for i, (label, prefixes, color) in enumerate(groups):
        y = len(groups) - 1 - i
        runs = [v for k, v in ttt.items() if k.rsplit('_s', 1)[0] in prefixes]
        hits = [v['first']['0.8'][0] for v in runs if '0.8' in v['first']]
        ax.scatter(hits, y + rng.uniform(-0.12, 0.12, len(hits)), s=46, color=color, alpha=0.75,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if len(hits) * 2 > len(runs):  # median over all searches, misses counted as never
            m = float(np.median(hits + [np.inf] * (len(runs) - len(hits))))
            ax.plot([m, m], [y - 0.3, y + 0.3], color=INK, linewidth=2, zorder=4)
        ax.text(260, y, f'{len(hits)}/{len(runs)} searches', va='center', fontsize=9.5, color=INK2)
    ax.set_xscale('log')
    ax.set_xlim(0.8, 600)
    ax.set_yticks(range(len(groups)), [g[0] for g in groups][::-1])
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('PPO runs until the search first finds a level scoring 0.8+ (log scale)')
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    ax.set_title('The learned walker gets there first')
    fig.tight_layout(); fig.savefig(OUT / 'time_to_target.png', dpi=200); plt.close(fig)


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fig_flatness():
    walkers = ('rl:mom:s0', 'rl:mom_vb:s0', 'rl:cov_vb:s0')
    cols = []
    for suite, label in (('hp_live5', '95% flat'), ('hp_live1', '~98.5% flat')):
        f = ROOT / 'scratch' / 'hp_live' / f'results_{suite}' / 'part_0.json'
        if not f.exists():
            return print(f'skip flatness figure: {f} missing')
        by = {}
        for r in json.load(open(f)):
            if r['n'] >= 20:
                by.setdefault((r['n'], r['seed'], r['noise']), {})[r['method']] = r['score']
        wins = [1.0 if d[w] < d['cmaes'] else 0.5 if d[w] == d['cmaes'] else 0.0
                for d in by.values() for w in walkers]
        cols.append((label, float(np.mean(wins)), wilson(sum(wins), len(wins))))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for i, (label, p, (lo, hi)) in enumerate(cols):
        ax.bar(i, p, width=0.5, color=BLUE, zorder=3)
        ax.plot([i, i], [lo, hi], color=INK, linewidth=1.5, zorder=4)
        ax.text(i + 0.29, p, f'{p:.0%}', va='center', fontsize=10, color=INK)
    ax.axhline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.5, 0.52, 'coin flip', fontsize=9, color=INK2, ha='center', va='bottom')
    ax.set_xticks([0, 1], [c[0] for c in cols])
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel('Learned walkers beat CMA-ES')
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title('The flatter the problem, the smaller the edge')
    fig.tight_layout(); fig.savefig(OUT / 'flatness.png', dpi=200); plt.close(fig)


def fig_level():
    best = max((json.load(open(f)) for f in HP.glob('round_J/*.json')), key=lambda s: s['cert_exploit'])
    level, hp, _ = E.decode_many(best['cert_x'], hard=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.imshow(level['walls'], cmap=matplotlib.colors.ListedColormap([SURFACE, '#3d3c38']))
    marks = [('start', 'S', BLUE), ('key', 'K', '#eda100'), ('door', 'D', '#6250d6'), ('goal', 'G', '#1baf7a'),
             ('term', 'X', '#eb6834')]
    for name, letter, color in marks:
        r, c = level[name]
        ax.add_patch(plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9, color=color, zorder=3))
        ax.text(c, r, letter, ha='center', va='center', color='white', fontsize=12, fontweight='bold', zorder=4)
    ax.set_xticks(np.arange(-0.5, E.W, 1), minor=True); ax.set_yticks(np.arange(-0.5, E.H, 1), minor=True)
    ax.grid(which='minor', color=GRID, linewidth=0.5)
    ax.tick_params(which='both', length=0, labelbottom=False, labelleft=False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f'A level rlbo found: PPO takes the grader {best["cert_exploit"]:.0%} of the time')
    legend = '  '.join(f'{l} {n}' for n, l, _ in
                       [('start', 'S', 0), ('key', 'K', 0), ('door', 'D', 0), ('goal', 'G', 0), ('grader', 'X', 0)])
    fig.text(0.02, 0.03, f'{legend}    PPO: lr {hp["lr"]:.1e}, entropy {hp["ent"]:.3f}, gamma {hp["gamma"]:.3f}',
             fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT / 'honeypot_level.png', dpi=200); plt.close(fig)


if __name__ == '__main__':
    for f in (fig_level, fig_honeypot, fig_speed, fig_ablation, fig_time_to_target, fig_flatness):
        f()
    print('figures written to', OUT)
