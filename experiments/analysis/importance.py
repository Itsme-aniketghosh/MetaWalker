"""Which honeypot level / PPO settings make PPO take the exploit? Uses every evaluation the searches already ran
(results/honeypot/evals.csv.gz, pulled from ~/honeypot*/runs/manyh_*.jsonl), no new PPO runs.

Two targets: P(level valid), and exploit rate given valid. A random forest per target, scored with GroupKFold over
search runs (points from one search are correlated, so a random split would leak). Exact interventional Shapley
values over the 9 decoded settings (2^9 coalitions), plus permutation importance over the 169 raw search
coordinates, which is what an importance-guided optimizer would act on.

Usage: python -m experiments.analysis.importance   -> prints a report, writes results/analysis/importance_report.txt
"""
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
P = ['p_lr', 'p_ent', 'p_gamma', 'p_lam', 'p_clip', 'p_terminal_visible', 'p_honest_len', 'p_terminal_dist', 'p_wall_frac']


def shapley(f, X, bg):
    """Exact interventional Shapley values of f at rows X against background rows bg: phi (len(X), d)."""
    d = X.shape[1]
    v = {}
    for k in range(d + 1):
        for S in combinations(range(d), k):
            Z = np.repeat(bg[None], len(X), 0)            # (n, b, d)
            Z[:, :, list(S)] = X[:, None, list(S)]
            v[S] = f(Z.reshape(-1, d)).reshape(len(X), -1).mean(1)
    phi = np.zeros(X.shape)
    for i in range(d):
        for S, val in v.items():
            if i in S:
                continue
            T = tuple(sorted(S + (i,)))
            phi[:, i] += factorial(len(S)) * factorial(d - len(S) - 1) / factorial(d) * (v[T] - val)
    return phi


def cv_score(model, X, y, groups, metric):
    out = []
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = model.fit(X[tr], y[tr])
        out.append(metric(y[te], m.predict_proba(X[te])[:, 1] if hasattr(m, 'predict_proba') else m.predict(X[te])))
    return np.mean(out), np.std(out)


def main():
    t = np.random.default_rng(1).standard_normal((5, 3)); b = np.random.default_rng(2).standard_normal((7, 3))
    lin = lambda Z: Z @ [1.0, -2.0, 0.5] + Z[:, 0] * 0  # additive: phi_i = w_i (x_i - mean bg_i)
    assert np.allclose(shapley(lin, t, b), (t - b.mean(0)) * [1.0, -2.0, 0.5])
    df = pd.read_csv(ROOT / 'results' / 'honeypot' / 'evals.csv.gz')
    df['p_terminal_visible'] = df['p_terminal_visible'].astype(str).eq('True').astype(float)
    xs = [c for c in df if c.startswith('x')]
    lines = [f'{len(df)} evaluations from {df.run.nunique()} searches; valid {df.valid.mean():.1%}; '
             f'exploit>0.5 among valid {(df.exploit[df.valid == 1] > 0.5).mean():.1%}']
    rng = np.random.default_rng(0)
    for target in ('valid', 'exploit|valid'):
        d = df if target == 'valid' else df[df.valid == 1]
        y = d['valid' if target == 'valid' else 'exploit'].to_numpy(float)
        groups = d.run.to_numpy()
        clf = target == 'valid'
        mk = lambda: (RandomForestClassifier if clf else RandomForestRegressor)(
            n_estimators=200, min_samples_leaf=5, n_jobs=2, random_state=0)
        metric = roc_auc_score if clf else r2_score
        lines += ['', f'== {target}  (n={len(d)}, mean {y.mean():.3f})']
        for name, cols in (('9 settings', P), ('169 raw coords', xs)):
            X = d[cols].to_numpy(float)
            m, s = cv_score(mk(), X, y, groups, metric)
            lines.append(f'{name}: held-out-search {"AUC" if clf else "R2"} {m:.3f} +- {s:.3f}')
        X = d[P].to_numpy(float)
        model = mk().fit(X, y)
        f = (lambda Z: model.predict_proba(Z)[:, 1]) if clf else model.predict
        idx = rng.choice(len(X), min(150, len(X)), replace=False)
        phi = shapley(f, X[idx], X[rng.choice(len(X), 30, replace=False)])
        imp = np.abs(phi).mean(0)
        lines.append('Shapley mean |phi| (and correlation of the setting with its phi: + means higher helps):')
        for i in np.argsort(-imp):
            c = pd.Series(X[idx, i]).corr(pd.Series(phi[:, i]))  # NaN-aware: honest_len is NaN when unreachable
            lines.append(f'  {P[i][2:]:18s} {imp[i]:.4f}   dir {c:+.2f}')
        Xr = d[xs].to_numpy(float)
        tr, te = next(GroupKFold(5).split(Xr, y, groups))
        mr = mk().fit(Xr[tr], y[tr])
        pi = permutation_importance(mr, Xr[te], y[te], n_repeats=5, random_state=0, n_jobs=2).importances_mean
        top = np.argsort(-pi)[:15]
        share = np.sort(np.clip(pi, 0, None))[::-1]
        k80 = int(np.searchsorted(np.cumsum(share) / share.sum(), 0.8)) + 1 if share.sum() > 0 else 0
        lines.append(f'raw coords: {k80} of 169 (0 = no single coordinate matters) carry 80% of the permutation importance; top 15: '
                     + ', '.join(f'x{j} {pi[j]:.3f}' for j in top))
    out = '\n'.join(lines)
    print(out)
    (ROOT / 'results' / 'analysis' / 'importance_report.txt').write_text(out + '\n')


if __name__ == '__main__':
    main()
